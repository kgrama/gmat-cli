//! Streaming tensor import pipeline with parallel processing.
//!
//! Uses async pipeline with tokio-rayon:
//! - Producer: async, reads safetensor files with mmap, sends tensors
//! - Workers: CPU-bound tensor conversion on rayon pool
//! - Writer: async, saves .gmat files and collects results

use anyhow::Result;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use transform_storage::{BlockFormat, GmatMetadata, GraphMatrix};

use crate::common::bytes_to_f32;
use crate::config::import_config::ImportConfig;
use crate::workqueue::{PipelineState, run_pipeline};

/// Tensor data extracted from safetensor, ready for conversion.
pub struct ExtractedTensor {
    pub name: String,
    pub target_uuid: String,
    pub shape: (usize, usize),
    pub dtype_str: String,
    pub f32_data: Vec<f32>,
    /// Original shape before normalization to 2D
    pub original_shape: Vec<usize>,
    /// Plane index if multi-plane tensor
    pub plane_index: Option<usize>,
    /// Total number of planes
    pub num_planes: usize,
}

/// Result of converting and saving a tensor.
pub struct SavedTensor {
    pub name: String,
    pub uuid: String,
    /// Original shape (for N-D tensors split into planes)
    pub original_shape: Option<Vec<usize>>,
    /// Matrix shape (rows, cols) - the 2D shape of this plane
    pub matrix_shape: (usize, usize),
    /// Plane index if this is part of an N-D tensor
    pub plane_index: Option<usize>,
    /// Total planes if this is part of an N-D tensor
    pub num_planes: Option<usize>,
}

/// Get the buffer size for the processing pipeline.
/// Can be configured via GMAT_BUFFER_SIZE environment variable.
fn get_buffer_size() -> usize {
    std::env::var("GMAT_BUFFER_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| num_cpus().saturating_mul(2).max(4))
}

/// Run the streaming import pipeline with async workqueue.
///
/// Architecture:
/// - Producer: async, reads files with mmap, sends tensors
/// - Workers: CPU-bound conversion on rayon pool
/// - Writer: async, saves .gmat files
pub fn run_streaming_import(
    safetensor_files: &[PathBuf],
    tensor_map: &HashMap<String, String>,
    tensors_dir: &Path,
    config: &ImportConfig,
    block_format: BlockFormat,
    total: usize,
) -> Result<Vec<SavedTensor>> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(run_streaming_import_async(
        safetensor_files,
        tensor_map,
        tensors_dir,
        config,
        block_format,
        total,
    ))
}

/// Producer task: extract tensors from safetensor files.
async fn produce_tensors_task(
    files: Vec<PathBuf>,
    tensor_map: HashMap<String, String>,
    tx: mpsc::Sender<ExtractedTensor>,
    state: Arc<PipelineState>,
) -> Result<()> {
    for file_path in &files {
        if let Err(e) = extract_and_send_tensors(file_path, &tensor_map, &tx, &state).await {
            eprintln!("Warning: Failed to process {}: {}", file_path.display(), e);
        }
    }
    Ok(())
}

/// Writer task: collect results and print progress.
async fn collect_results_task(
    mut rx: mpsc::Receiver<Result<SavedTensor>>,
    state: Arc<PipelineState>,
    results: Arc<std::sync::Mutex<Vec<SavedTensor>>>,
    total: usize,
) -> Result<()> {
    while let Some(result) = rx.recv().await {
        match result {
            Ok(saved) => {
                let completed = state.inc_completed();
                println!(
                    "[{}/{}] {} -> {}.gmat",
                    completed, total, saved.name, saved.uuid
                );
                results.lock().unwrap().push(saved);
            }
            Err(e) => {
                eprintln!("Warning: {}", e);
            }
        }
    }
    Ok(())
}

/// Async implementation of streaming import.
async fn run_streaming_import_async(
    safetensor_files: &[PathBuf],
    tensor_map: &HashMap<String, String>,
    tensors_dir: &Path,
    config: &ImportConfig,
    block_format: BlockFormat,
    total: usize,
) -> Result<Vec<SavedTensor>> {
    let buffer_size = get_buffer_size();

    // Clone for closures
    let files = safetensor_files.to_vec();
    let tensor_map_owned = tensor_map.clone();
    let tensors_dir_owned = tensors_dir.to_path_buf();
    let config_owned = config.clone();

    // Shared result collector
    let results = Arc::new(std::sync::Mutex::new(Vec::new()));
    let results_for_writer = Arc::clone(&results);

    run_pipeline::<ExtractedTensor, SavedTensor, anyhow::Error, _, _, _, _, _>(
        buffer_size,
        buffer_size,
        // Producer: extract tensors from safetensor files
        move |tx, state| produce_tensors_task(files, tensor_map_owned, tx, state),
        // Worker: convert tensor to GMAT format (runs on rayon)
        move |tensor: ExtractedTensor| -> Result<SavedTensor> {
            convert_tensor(&tensor, &tensors_dir_owned, &config_owned, block_format)
        },
        // Writer: collect results and print progress
        move |rx, state| collect_results_task(rx, state, results_for_writer, total),
    )
    .await?;

    let saved = Arc::try_unwrap(results)
        .map_err(|_| anyhow::anyhow!("Failed to unwrap results"))?
        .into_inner()
        .unwrap();

    Ok(saved)
}

/// Normalize tensor shape to 2D representation.
/// Returns (num_planes, rows, cols) where planes represent leading dimensions.
fn normalize_tensor_shape(shape: &[usize]) -> (usize, usize, usize) {
    match shape.len() {
        0 => (1, 1, 1),               // scalar
        1 => (1, 1, shape[0]),        // 1D -> 1xN
        2 => (1, shape[0], shape[1]), // 2D -> 1 plane
        _ => {
            // N-D: leading dims are planes, last 2 dims are matrix
            let planes: usize = shape[..shape.len() - 2].iter().product();
            let rows = shape[shape.len() - 2];
            let cols = shape[shape.len() - 1];
            (planes, rows, cols)
        }
    }
}

/// Send a single tensor plane to the processing channel.
#[allow(clippy::too_many_arguments)]
async fn send_tensor_plane(
    tx: &mpsc::Sender<ExtractedTensor>,
    state: &PipelineState,
    tensor_name: &str,
    target_uuid: &str,
    plane_idx: usize,
    num_planes: usize,
    rows: usize,
    cols: usize,
    f32_data: &[f32],
    original_shape: Vec<usize>,
    dtype_str: String,
) -> Result<bool> {
    let plane_size = rows * cols;
    let start = plane_idx * plane_size;
    let end = start + plane_size;

    let plane_uuid = if num_planes > 1 {
        format!("{}_{}", target_uuid, plane_idx)
    } else {
        target_uuid.to_string()
    };

    let plane_name = if num_planes > 1 {
        format!("{}[{}]", tensor_name, plane_idx)
    } else {
        tensor_name.to_string()
    };

    let plane_data = f32_data[start..end].to_vec();

    state.inc_produced();
    Ok(tx
        .send(ExtractedTensor {
            name: plane_name,
            target_uuid: plane_uuid,
            shape: (rows, cols),
            dtype_str,
            f32_data: plane_data,
            original_shape,
            plane_index: if num_planes > 1 {
                Some(plane_idx)
            } else {
                None
            },
            num_planes,
        })
        .await
        .is_ok())
}

/// Extract tensors from a safetensor file and send to channel.
async fn extract_and_send_tensors(
    file_path: &PathBuf,
    tensor_map: &HashMap<String, String>,
    tx: &mpsc::Sender<ExtractedTensor>,
    state: &PipelineState,
) -> Result<()> {
    let file = File::open(file_path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let st = safetensors::SafeTensors::deserialize(&mmap)?;

    for (tensor_name, tensor_view) in st.tensors() {
        let target_uuid = match tensor_map.get(&tensor_name) {
            Some(uuid) => uuid.clone(),
            None => continue,
        };

        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();
        let numel: usize = shape.iter().product();

        let f32_data = match bytes_to_f32(tensor_view.data(), dtype, numel) {
            Ok(data) => data,
            Err(e) => {
                eprintln!("Warning: Failed to convert {}: {}", tensor_name, e);
                continue;
            }
        };

        let (num_planes, rows, cols) = normalize_tensor_shape(shape);
        let dtype_str = format!("{:?}", dtype);

        for plane_idx in 0..num_planes {
            let send_ok = send_tensor_plane(
                tx,
                state,
                &tensor_name,
                &target_uuid,
                plane_idx,
                num_planes,
                rows,
                cols,
                &f32_data,
                shape.to_vec(),
                dtype_str.clone(),
            )
            .await?;

            if !send_ok {
                return Ok(());
            }
        }
    }

    Ok(())
}

/// Convert a tensor to GMAT format (sync, runs on rayon).
fn convert_tensor(
    tensor: &ExtractedTensor,
    tensors_dir: &Path,
    config: &ImportConfig,
    block_format: BlockFormat,
) -> Result<SavedTensor> {
    let (rows, cols) = tensor.shape;

    let mut matrix = GraphMatrix::from_dense(&tensor.f32_data, (rows, cols), block_format);
    matrix.set_metadata(build_tensor_metadata(
        &tensor.name,
        &tensor.target_uuid,
        &tensor.dtype_str,
        config,
    ));

    let out_file = tensors_dir.join(format!("{}.gmat", tensor.target_uuid));
    matrix.save(&out_file)?;

    let (original_shape, plane_index, num_planes) = if tensor.num_planes > 1 {
        (
            Some(tensor.original_shape.clone()),
            tensor.plane_index,
            Some(tensor.num_planes),
        )
    } else {
        (None, None, None)
    };

    Ok(SavedTensor {
        name: tensor.name.clone(),
        uuid: tensor.target_uuid.clone(),
        original_shape,
        matrix_shape: tensor.shape,
        plane_index,
        num_planes,
    })
}

/// Build GMAT metadata for a tensor.
fn build_tensor_metadata(
    name: &str,
    uuid: &str,
    dtype_str: &str,
    config: &ImportConfig,
) -> GmatMetadata {
    let mut meta = GmatMetadata::new();
    meta.set_str("source_tensor_name", name);
    meta.set_str("tensor_uuid", uuid);
    meta.set_str("original_dtype", dtype_str);
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
    fn test_get_buffer_size_default() {
        // Without env var, should return num_cpus * 2 or at least 4
        let size = get_buffer_size();
        assert!(size >= 4);
    }

    #[test]
    fn test_extracted_tensor_2d() {
        let tensor = ExtractedTensor {
            name: "model.layers.0.weight".to_string(),
            target_uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            shape: (4096, 4096),
            dtype_str: "F16".to_string(),
            f32_data: vec![1.0, 2.0, 3.0, 4.0],
            original_shape: vec![4096, 4096],
            plane_index: None,
            num_planes: 1,
        };

        assert_eq!(tensor.name, "model.layers.0.weight");
        assert_eq!(tensor.shape, (4096, 4096));
        assert_eq!(tensor.num_planes, 1);
    }

    #[test]
    fn test_extracted_tensor_1d_becomes_1xn() {
        let tensor = ExtractedTensor {
            name: "model.bias".to_string(),
            target_uuid: "550e8400-e29b-41d4-a716-446655440001".to_string(),
            shape: (1, 64),
            dtype_str: "F32".to_string(),
            f32_data: vec![0.1; 64],
            original_shape: vec![64],
            plane_index: None,
            num_planes: 1,
        };

        assert_eq!(tensor.shape, (1, 64));
    }

    #[test]
    fn test_extracted_tensor_3d_plane() {
        let tensor = ExtractedTensor {
            name: "conv.weight[2]".to_string(),
            target_uuid: "550e8400-e29b-41d4-a716-446655440002_2".to_string(),
            shape: (32, 64),
            dtype_str: "F32".to_string(),
            f32_data: vec![0.1; 32 * 64],
            original_shape: vec![4, 32, 64],
            plane_index: Some(2),
            num_planes: 4,
        };

        assert_eq!(tensor.shape, (32, 64));
        assert_eq!(tensor.plane_index, Some(2));
        assert_eq!(tensor.num_planes, 4);
    }

    #[test]
    fn test_saved_tensor_simple() {
        let saved = SavedTensor {
            name: "model.layers.0.weight".to_string(),
            uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            original_shape: None,
            matrix_shape: (4096, 4096),
            plane_index: None,
            num_planes: None,
        };

        assert_eq!(saved.name, "model.layers.0.weight");
        assert_eq!(saved.uuid, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(saved.matrix_shape, (4096, 4096));
    }

    #[test]
    fn test_saved_tensor_nd_plane() {
        let saved = SavedTensor {
            name: "conv.weight[0]".to_string(),
            uuid: "550e8400-e29b-41d4-a716-446655440002_0".to_string(),
            original_shape: Some(vec![4, 32, 64]),
            matrix_shape: (32, 64),
            plane_index: Some(0),
            num_planes: Some(4),
        };

        assert_eq!(saved.original_shape, Some(vec![4, 32, 64]));
        assert_eq!(saved.matrix_shape, (32, 64));
        assert_eq!(saved.num_planes, Some(4));
    }
}
