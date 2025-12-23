//! Streaming tensor import pipeline using producer-consumer pattern.

use anyhow::Result;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use transform_storage::{BlockFormat, GmatMetadata, GraphMatrix};

use crate::common::bytes_to_f32;
use crate::config::import_config::ImportConfig;

/// Tensor data extracted from safetensor, ready for conversion.
pub struct ExtractedTensor {
    pub name: String,
    pub target_uuid: String,
    pub shape: (usize, usize),
    pub dtype_str: String,
    pub f32_data: Vec<f32>,
}

/// Result of converting and saving a tensor.
pub struct SavedTensor {
    pub name: String,
    pub uuid: String,
}

/// Run the producer-consumer streaming import pipeline.
pub fn run_streaming_import(
    safetensor_files: &[PathBuf],
    tensor_map: &HashMap<String, String>,
    tensors_dir: &Path,
    config: &ImportConfig,
    block_format: BlockFormat,
    total: usize,
) -> Result<Vec<SavedTensor>> {
    let buffer_size = num_cpus().saturating_mul(2).max(4);
    let (tx, rx) = mpsc::sync_channel::<Result<ExtractedTensor>>(buffer_size);

    let tensors_dir = tensors_dir.to_path_buf();
    let config = config.clone();
    let counter = AtomicUsize::new(0);

    let consumer = thread::spawn(move || {
        consume_tensors(rx, &tensors_dir, &config, block_format, &counter, total)
    });

    // Producer: extract tensors in parallel
    safetensor_files.par_iter().for_each(|file_path| {
        if let Err(e) = extract_and_send_tensors(file_path, tensor_map, &tx) {
            let _ = tx.send(Err(e));
        }
    });

    drop(tx);
    consumer.join().map_err(|_| anyhow::anyhow!("Consumer thread panicked"))?
}

/// Consumer: convert and save tensors as they arrive.
fn consume_tensors(
    rx: mpsc::Receiver<Result<ExtractedTensor>>,
    tensors_dir: &Path,
    config: &ImportConfig,
    block_format: BlockFormat,
    counter: &AtomicUsize,
    total: usize,
) -> Result<Vec<SavedTensor>> {
    let mut saved = Vec::new();

    for result in rx {
        let tensor = result?;
        let (rows, cols) = tensor.shape;

        let mut matrix = GraphMatrix::from_dense(&tensor.f32_data, (rows, cols), block_format);
        matrix.set_metadata(build_tensor_metadata(&tensor, config));

        let out_file = tensors_dir.join(format!("{}.gmat", tensor.target_uuid));
        matrix.save(&out_file)?;

        let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
        println!(
            "[{}/{}] {} [{rows}x{cols}] -> {} (nnz={}, sparsity={:.1}%)",
            count, total, tensor.name,
            out_file.file_name().unwrap().to_string_lossy(),
            matrix.nnz(), matrix.sparsity() * 100.0
        );

        saved.push(SavedTensor { name: tensor.name, uuid: tensor.target_uuid });
    }

    Ok(saved)
}

/// Build GMAT metadata for a tensor.
fn build_tensor_metadata(tensor: &ExtractedTensor, config: &ImportConfig) -> GmatMetadata {
    let mut meta = GmatMetadata::new();
    meta.set_str("source_tensor_name", &tensor.name);
    meta.set_str("tensor_uuid", &tensor.target_uuid);
    meta.set_str("original_dtype", &tensor.dtype_str);
    meta.set_str("block_format", &config.block_format);
    meta.set_str("source_format", &config.source_format);

    if let Some(ref arch) = config.metadata.architecture {
        meta.set_str("model_architecture", arch);
    }
    if let Some(v) = config.metadata.vocab_size {
        meta.set_u64("vocab_size", v);
    }
    if let Some(v) = config.metadata.hidden_size {
        meta.set_u64("hidden_size", v);
    }
    if let Some(v) = config.metadata.num_layers {
        meta.set_u64("num_layers", v);
    }
    meta
}

/// Extract tensors from a safetensor file and send each to the channel.
fn extract_and_send_tensors(
    file_path: &PathBuf,
    tensor_map: &HashMap<String, String>,
    tx: &mpsc::SyncSender<Result<ExtractedTensor>>,
) -> Result<()> {
    let data = fs::read(file_path)?;
    let st = safetensors::SafeTensors::deserialize(&data)?;

    for (tensor_name, tensor_view) in st.tensors() {
        let target_uuid = match tensor_map.get(&tensor_name) {
            Some(uuid) => uuid.clone(),
            None => continue,
        };

        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();

        if shape.len() != 2 {
            tx.send(Err(anyhow::anyhow!(
                "Tensor '{}' must be 2D, got {:?}",
                tensor_name, shape
            )))?;
            continue;
        }

        let (rows, cols) = (shape[0], shape[1]);
        let f32_data = match bytes_to_f32(tensor_view.data(), dtype, rows * cols) {
            Ok(data) => data,
            Err(e) => {
                tx.send(Err(e))?;
                continue;
            }
        };

        tx.send(Ok(ExtractedTensor {
            name: tensor_name.to_string(),
            target_uuid,
            shape: (rows, cols),
            dtype_str: format!("{:?}", dtype),
            f32_data,
        }))?;
    }

    Ok(())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_cpus_returns_positive() {
        assert!(num_cpus() > 0);
    }

    #[test]
    fn test_extracted_tensor_creation() {
        let tensor = ExtractedTensor {
            name: "model.layers.0.weight".to_string(),
            target_uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            shape: (4096, 4096),
            dtype_str: "F16".to_string(),
            f32_data: vec![1.0, 2.0, 3.0, 4.0],
        };

        assert_eq!(tensor.name, "model.layers.0.weight");
        assert_eq!(tensor.shape, (4096, 4096));
    }

    #[test]
    fn test_saved_tensor_creation() {
        let saved = SavedTensor {
            name: "model.layers.0.weight".to_string(),
            uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        };

        assert_eq!(saved.name, "model.layers.0.weight");
        assert_eq!(saved.uuid, "550e8400-e29b-41d4-a716-446655440000");
    }
}
