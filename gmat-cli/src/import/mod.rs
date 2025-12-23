//! Import module - SafeTensors/GGUF to GMAT conversion.
//!
//! Uses streaming processing to minimize memory pressure:
//! each tensor is extracted, converted, and written to disk
//! before processing the next tensor.

use anyhow::{Context, Result};
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use transform_storage::{BlockFormat, GmatMetadata, GraphMatrix};
use uuid::Uuid;

use crate::common::{bytes_to_f32, discover_safetensor_files, load_config};
use crate::config::import_config::{ImportConfig, ModelMetadata, TensorMapping};

/// Tensor data extracted from safetensor, ready for conversion.
struct ExtractedTensor {
    name: String,
    target_uuid: String,
    shape: (usize, usize),
    dtype_str: String,
    f32_data: Vec<f32>,
}

/// Result of converting and saving a tensor.
struct SavedTensor {
    name: String,
    uuid: String,
}

/// Generate an import config template from a model file.
pub fn generate_config_template(model_path: &str) -> Result<()> {
    let path = Path::new(model_path);
    let safetensor_files = discover_safetensor_files(path)?;

    println!("Found {} safetensors file(s)", safetensor_files.len());

    // Process files in parallel, collect tensor mappings
    let tensor_mappings: Vec<TensorMapping> = safetensor_files
        .par_iter()
        .map(|file_path| -> Result<Vec<TensorMapping>> {
            let data = fs::read(file_path)
                .with_context(|| format!("Failed to read: {}", file_path.display()))?;

            let st = SafeTensors::deserialize(&data)
                .with_context(|| format!("Failed to parse: {}", file_path.display()))?;

            let mappings: Vec<TensorMapping> = st
                .tensors()
                .into_iter()
                .map(|(name, tensor_view)| {
                    let shape = tensor_view.shape();
                    println!("  - {} (shape: {:?})", name, shape);
                    TensorMapping {
                        source: name.to_string(),
                        target: Uuid::new_v4().to_string(),
                        include: true,
                    }
                })
                .collect();

            Ok(mappings)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    let config = ImportConfig {
        source_format: "safetensors".to_string(),
        block_format: "B8x8".to_string(),
        tensor_map: tensor_mappings,
        metadata: ModelMetadata::default(),
    };

    let json = serde_json::to_string_pretty(&config)?;
    fs::write("import_config.json", json)?;

    println!("\n=== Generated import_config.json ===");
    println!("Tensors: {}", config.tensor_map.len());
    println!("\nEdit the config, then run:");
    println!("  gmat import --model {} --config import_config.json", model_path);

    Ok(())
}

/// Run the import process with streaming tensor conversion.
///
/// Uses a producer-consumer pattern to minimize memory pressure:
/// - Producer: Iterates through safetensor files, extracts tensors one at a time
/// - Consumer: Converts and saves each tensor to disk immediately
///
/// This ensures only a bounded number of tensors are in memory at once.
pub fn run(model_path: &str, config_path: Option<&str>, output_path: Option<&str>) -> Result<()> {
    let config: ImportConfig = load_config(config_path, "import_config.json")?;

    let block_format = parse_block_format(&config.block_format)
        .with_context(|| format!("Invalid block format: {}", config.block_format))?;

    let output_dir = Path::new(output_path.unwrap_or("output")).join("model.gmat");
    let tensors_dir = output_dir.join("tensors");
    fs::create_dir_all(&tensors_dir)?;

    println!("Output: {}", output_dir.display());
    println!("Block format: {:?}", block_format);

    let path = Path::new(model_path);
    let safetensor_files = discover_safetensor_files(path)?;

    println!("Found {} safetensors file(s)", safetensor_files.len());

    let tensor_map: HashMap<String, String> = config
        .tensor_map
        .iter()
        .filter(|tm| tm.include)
        .map(|tm| (tm.source.clone(), tm.target.clone()))
        .collect();

    // Count total tensors for progress display
    let total = tensor_map.len();
    println!("Processing {} tensors...", total);

    // Bounded channel limits memory to ~buffer_size tensors at a time
    let buffer_size = num_cpus().saturating_mul(2).max(4);
    let (tx, rx) = mpsc::sync_channel::<Result<ExtractedTensor>>(buffer_size);

    // Clone what consumer needs
    let tensors_dir_clone = tensors_dir.clone();
    let config_clone = config.clone();
    let counter = AtomicUsize::new(0);

    // Consumer thread: convert and save tensors as they arrive
    let consumer = thread::spawn(move || -> Result<Vec<SavedTensor>> {
        let mut saved = Vec::new();

        for result in rx {
            let tensor = result?;
            let (rows, cols) = tensor.shape;

            // Convert to GMAT format
            let mut matrix = GraphMatrix::from_dense(&tensor.f32_data, (rows, cols), block_format);

            // Build metadata
            let mut meta = GmatMetadata::new();
            meta.set_str("source_tensor_name", &tensor.name);
            meta.set_str("tensor_uuid", &tensor.target_uuid);
            meta.set_str("original_dtype", &tensor.dtype_str);
            meta.set_str("block_format", &config_clone.block_format);
            meta.set_str("source_format", &config_clone.source_format);

            if let Some(ref arch) = config_clone.metadata.architecture {
                meta.set_str("model_architecture", arch);
            }
            if let Some(v) = config_clone.metadata.vocab_size {
                meta.set_u64("vocab_size", v);
            }
            if let Some(v) = config_clone.metadata.hidden_size {
                meta.set_u64("hidden_size", v);
            }
            if let Some(v) = config_clone.metadata.num_layers {
                meta.set_u64("num_layers", v);
            }

            matrix.set_metadata(meta);

            // Write to disk immediately
            let out_file = tensors_dir_clone.join(format!("{}.gmat", tensor.target_uuid));
            matrix.save(&out_file)?;

            let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
            println!(
                "[{}/{}] {} [{rows}x{cols}] -> {} (nnz={}, sparsity={:.1}%)",
                count,
                total,
                tensor.name,
                out_file.file_name().unwrap().to_string_lossy(),
                matrix.nnz(),
                matrix.sparsity() * 100.0
            );

            saved.push(SavedTensor {
                name: tensor.name,
                uuid: tensor.target_uuid,
            });
        }

        Ok(saved)
    });

    // Producer: extract tensors from files and send to consumer
    // Process files in parallel, but each tensor is sent immediately
    safetensor_files.par_iter().for_each(|file_path| {
        if let Err(e) = extract_and_send_tensors(file_path, &tensor_map, &tx) {
            let _ = tx.send(Err(e));
        }
    });

    // Drop sender to signal completion
    drop(tx);

    // Wait for consumer
    let saved = consumer
        .join()
        .map_err(|_| anyhow::anyhow!("Consumer thread panicked"))??;

    // Write metadata.json
    let metadata_json = serde_json::json!({
        "config": config,
        "tensor_name_map": saved.iter()
            .map(|s| (s.name.as_str(), s.uuid.as_str()))
            .collect::<HashMap<_, _>>(),
        "total_tensors": saved.len(),
    });

    fs::write(
        output_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata_json)?,
    )?;

    println!("\n=== Import Complete: {} tensors ===", saved.len());
    Ok(())
}

/// Get number of CPUs for buffer sizing.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Extract tensors from a safetensor file and send each to the channel immediately.
/// This streams tensors one at a time to limit memory usage.
fn extract_and_send_tensors(
    file_path: &PathBuf,
    tensor_map: &HashMap<String, String>,
    tx: &mpsc::SyncSender<Result<ExtractedTensor>>,
) -> Result<()> {
    let data = fs::read(file_path)?;
    let st = SafeTensors::deserialize(&data)?;

    for (tensor_name, tensor_view) in st.tensors() {
        // Skip tensors not in our map
        let target_uuid = match tensor_map.get(&tensor_name) {
            Some(uuid) => uuid.clone(),
            None => continue,
        };

        let shape = tensor_view.shape();
        let dtype = tensor_view.dtype();

        if shape.len() != 2 {
            tx.send(Err(anyhow::anyhow!(
                "Tensor '{}' must be 2D, got {:?}",
                tensor_name,
                shape
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

        // Send tensor immediately - bounded channel provides backpressure
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

/// Parse block format string to BlockFormat enum.
fn parse_block_format(s: &str) -> Result<BlockFormat> {
    match s {
        "B8x4" => Ok(BlockFormat::B8x4),
        "B8x8" => Ok(BlockFormat::B8x8),
        "B16x4" => Ok(BlockFormat::B16x4),
        "B16x8" => Ok(BlockFormat::B16x8),
        "DualRow8x4" => Ok(BlockFormat::DualRow8x4),
        "DualRow8x8" => Ok(BlockFormat::DualRow8x8),
        "DualRow16x4" => Ok(BlockFormat::DualRow16x4),
        "DualRow16x8" => Ok(BlockFormat::DualRow16x8),
        _ => anyhow::bail!(
            "Unknown block format: {}. Valid formats: B8x4, B8x8, B16x4, B16x8, DualRow8x4, DualRow8x8, DualRow16x4, DualRow16x8",
            s
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== parse_block_format tests ====================

    #[test]
    fn test_parse_block_format_b8x4() {
        let result = parse_block_format("B8x4").unwrap();
        assert!(matches!(result, BlockFormat::B8x4));
    }

    #[test]
    fn test_parse_block_format_b8x8() {
        let result = parse_block_format("B8x8").unwrap();
        assert!(matches!(result, BlockFormat::B8x8));
    }

    #[test]
    fn test_parse_block_format_b16x4() {
        let result = parse_block_format("B16x4").unwrap();
        assert!(matches!(result, BlockFormat::B16x4));
    }

    #[test]
    fn test_parse_block_format_b16x8() {
        let result = parse_block_format("B16x8").unwrap();
        assert!(matches!(result, BlockFormat::B16x8));
    }

    #[test]
    fn test_parse_block_format_dualrow8x4() {
        let result = parse_block_format("DualRow8x4").unwrap();
        assert!(matches!(result, BlockFormat::DualRow8x4));
    }

    #[test]
    fn test_parse_block_format_dualrow8x8() {
        let result = parse_block_format("DualRow8x8").unwrap();
        assert!(matches!(result, BlockFormat::DualRow8x8));
    }

    #[test]
    fn test_parse_block_format_dualrow16x4() {
        let result = parse_block_format("DualRow16x4").unwrap();
        assert!(matches!(result, BlockFormat::DualRow16x4));
    }

    #[test]
    fn test_parse_block_format_dualrow16x8() {
        let result = parse_block_format("DualRow16x8").unwrap();
        assert!(matches!(result, BlockFormat::DualRow16x8));
    }

    #[test]
    fn test_parse_block_format_unknown() {
        let result = parse_block_format("B32x32");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown block format"));
        assert!(err.contains("B32x32"));
    }

    #[test]
    fn test_parse_block_format_case_sensitive() {
        // Block format parsing is case-sensitive
        let result = parse_block_format("b8x8");
        assert!(result.is_err());
    }

    // ==================== num_cpus tests ====================

    #[test]
    fn test_num_cpus_returns_positive() {
        let cpus = num_cpus();
        assert!(cpus > 0);
    }

    // ==================== ExtractedTensor tests ====================

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
        assert_eq!(tensor.target_uuid, "550e8400-e29b-41d4-a716-446655440000");
        assert_eq!(tensor.shape, (4096, 4096));
        assert_eq!(tensor.dtype_str, "F16");
        assert_eq!(tensor.f32_data.len(), 4);
    }

    // ==================== SavedTensor tests ====================

    #[test]
    fn test_saved_tensor_creation() {
        let saved = SavedTensor {
            name: "model.layers.0.weight".to_string(),
            uuid: "550e8400-e29b-41d4-a716-446655440000".to_string(),
        };

        assert_eq!(saved.name, "model.layers.0.weight");
        assert_eq!(saved.uuid, "550e8400-e29b-41d4-a716-446655440000");
    }

    // ==================== Tensor map filtering tests ====================

    #[test]
    fn test_tensor_map_filters_excluded() {
        let mappings = vec![
            TensorMapping {
                source: "tensor1".to_string(),
                target: "uuid1".to_string(),
                include: true,
            },
            TensorMapping {
                source: "tensor2".to_string(),
                target: "uuid2".to_string(),
                include: false,
            },
            TensorMapping {
                source: "tensor3".to_string(),
                target: "uuid3".to_string(),
                include: true,
            },
        ];

        let tensor_map: HashMap<String, String> = mappings
            .iter()
            .filter(|tm| tm.include)
            .map(|tm| (tm.source.clone(), tm.target.clone()))
            .collect();

        assert_eq!(tensor_map.len(), 2);
        assert!(tensor_map.contains_key("tensor1"));
        assert!(!tensor_map.contains_key("tensor2"));
        assert!(tensor_map.contains_key("tensor3"));
    }
}
