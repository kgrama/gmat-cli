//! Export module - GMAT to GGUF/SafeTensors conversion.

mod shard;
mod util;

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use transform_storage::conversions::gguf_quant::{
    compute_tensor_importance, quantize_to_gguf, GgufQuantType, ScaleOptimization,
};
use transform_storage::GraphMatrix;

/// Represents a tensor entry from metadata - either simple 2D or N-D with planes.
#[derive(Debug, Clone)]
enum TensorEntry {
    /// Simple 2D tensor: single UUID
    Simple { uuid: String },
    /// N-D tensor split into planes
    NdPlanes {
        original_shape: Vec<usize>,
        matrix_shape: (usize, usize),
        plane_uuids: Vec<String>,
    },
}

impl TensorEntry {
    /// Parse a tensor entry from metadata JSON value.
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        if let Some(uuid) = value.as_str() {
            // Simple 2D tensor
            Some(TensorEntry::Simple { uuid: uuid.to_string() })
        } else if let Some(obj) = value.as_object() {
            // N-D tensor with planes
            let original_shape: Vec<usize> = obj.get("original_shape")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();

            let matrix_arr = obj.get("matrix_shape")?.as_array()?;
            let matrix_shape = (
                matrix_arr.first()?.as_u64()? as usize,
                matrix_arr.get(1)?.as_u64()? as usize,
            );

            let plane_uuids: Vec<String> = obj.get("plane_uuids")?
                .as_array()?
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();

            Some(TensorEntry::NdPlanes {
                original_shape,
                matrix_shape,
                plane_uuids,
            })
        } else {
            None
        }
    }

}

/// Find a tensor entry by UUID in the name_to_entry map.
/// The config uses UUID as source, but metadata maps name -> entry.
/// Returns a QuantizeEntry with validated paths.
fn find_tensor_entry(
    name_to_entry: &HashMap<String, TensorEntry>,
    source_uuid: &str,
    tensors_dir: &std::path::Path,
) -> Option<QuantizeEntry> {
    // First check if source_uuid directly matches a simple tensor's UUID
    for entry in name_to_entry.values() {
        match entry {
            TensorEntry::Simple { uuid } if uuid == source_uuid => {
                let tensor_path = tensors_dir.join(format!("{}.gmat", uuid));
                if !tensor_path.exists() {
                    eprintln!("Warning: missing tensor: {}", tensor_path.display());
                    return None;
                }
                return Some(QuantizeEntry::Simple { tensor_path });
            }
            TensorEntry::NdPlanes { plane_uuids, original_shape, matrix_shape } => {
                // Check if source_uuid matches any plane UUID (for per-plane export)
                // or the base UUID pattern (first plane UUID without suffix)
                let base_uuid = plane_uuids.first()?.strip_suffix("_0")?;
                if source_uuid == base_uuid || plane_uuids.contains(&source_uuid.to_string()) {
                    // Build paths for all planes
                    let plane_paths: Vec<PathBuf> = plane_uuids
                        .iter()
                        .map(|uuid| tensors_dir.join(format!("{}.gmat", uuid)))
                        .collect();

                    // Validate all paths exist
                    for path in &plane_paths {
                        if !path.exists() {
                            eprintln!("Warning: missing tensor plane: {}", path.display());
                            return None;
                        }
                    }

                    return Some(QuantizeEntry::NdPlanes {
                        original_shape: original_shape.clone(),
                        matrix_shape: *matrix_shape,
                        plane_paths,
                    });
                }
            }
            _ => {}
        }
    }

    // Fallback: try as a simple tensor path directly
    let tensor_path = tensors_dir.join(format!("{}.gmat", source_uuid));
    if tensor_path.exists() {
        return Some(QuantizeEntry::Simple { tensor_path });
    }

    eprintln!("Warning: tensor not found: {}", source_uuid);
    None
}

use crate::common::{load_config, load_gmat_model};
use crate::config::export_config::{ExportConfig, QuantizationConfig, TensorExportMapping};

use shard::{run_sharded_consumer, run_single_file_consumer, ProcessedTensor, ShardResult};
use util::{num_cpus, parse_quant_type, recommend_quant_type, safetensor_to_gguf_name};

/// Tensor analysis result for config generation.
struct TensorAnalysis {
    source_name: String,
    uuid: String,
    gguf_name: String,
    rows: usize,
    cols: usize,
    importance: f32,
    quant_type: String,
}

/// Tensor processing job with all info needed for quantization.
struct QuantizeJob {
    source: String,
    target: String,
    quant_type: GgufQuantType,
    /// For simple 2D tensors: single path. For N-D: multiple plane paths.
    tensor_entry: QuantizeEntry,
}

/// Entry type for quantization job.
#[derive(Debug, Clone)]
enum QuantizeEntry {
    /// Simple 2D tensor
    Simple { tensor_path: PathBuf },
    /// N-D tensor with multiple planes to reassemble
    NdPlanes {
        original_shape: Vec<usize>,
        matrix_shape: (usize, usize),
        plane_paths: Vec<PathBuf>,
    },
}

/// Generate an export config template from a GMAT model.
pub fn generate_config_template(model_path: &str) -> Result<()> {
    let (model_dir, metadata) = load_gmat_model(model_path)?;

    let tensor_map = metadata["tensor_name_map"]
        .as_object()
        .context("tensor_name_map not found in metadata.json")?;

    println!("Analyzing {} tensors...", tensor_map.len());

    let tensors_dir = model_dir.join("tensors");

    // Collect tensor entries for parallel processing
    let tensor_entries: Vec<(&String, &serde_json::Value)> = tensor_map.iter().collect();

    // Analyze tensors in parallel
    let analyses: Vec<TensorAnalysis> = tensor_entries
        .par_iter()
        .filter_map(|(source_name, json_value)| {
            let entry = TensorEntry::from_json(json_value)?;

            // Get the primary UUID and path for analysis
            let (uuid, tensor_path, shape) = match &entry {
                TensorEntry::Simple { uuid } => {
                    let path = tensors_dir.join(format!("{}.gmat", uuid));
                    if !path.exists() {
                        eprintln!("Warning: missing tensor: {}", path.display());
                        return None;
                    }
                    let matrix = GraphMatrix::load(&path).ok()?;
                    let shape = matrix.shape();
                    (uuid.clone(), path, shape)
                }
                TensorEntry::NdPlanes { plane_uuids, matrix_shape, original_shape, .. } => {
                    // For N-D tensors, analyze first plane and compute total size
                    let first_uuid = plane_uuids.first()?;
                    let path = tensors_dir.join(format!("{}.gmat", first_uuid));
                    if !path.exists() {
                        eprintln!("Warning: missing tensor plane: {}", path.display());
                        return None;
                    }
                    // Use original shape's last 2 dims times num_planes for total size estimation
                    let total_elements: usize = original_shape.iter().product();
                    let approx_rows = total_elements / matrix_shape.1;
                    (first_uuid.clone(), path, (approx_rows, matrix_shape.1))
                }
            };

            let matrix = GraphMatrix::load(&tensor_path).ok()?;
            let importance = compute_tensor_importance(&matrix);
            let (rows, cols) = shape;

            let gguf_name = safetensor_to_gguf_name(source_name);
            let quant_type = recommend_quant_type(source_name, importance, rows * cols);

            Some(TensorAnalysis {
                source_name: source_name.to_string(),
                uuid,
                gguf_name,
                rows,
                cols,
                importance,
                quant_type,
            })
        })
        .collect();

    // Print results and build config
    let mut tensor_mappings = Vec::with_capacity(analyses.len());
    let mut per_tensor_quant = HashMap::with_capacity(analyses.len());

    for analysis in &analyses {
        println!(
            "  {} -> {} [{}x{}] imp={:.3} quant={}",
            analysis.source_name,
            analysis.gguf_name,
            analysis.rows,
            analysis.cols,
            analysis.importance,
            analysis.quant_type
        );

        tensor_mappings.push(TensorExportMapping {
            source: analysis.uuid.clone(),
            target: analysis.gguf_name.clone(),
        });
        per_tensor_quant.insert(analysis.uuid.clone(), analysis.quant_type.clone());
    }

    let config = ExportConfig {
        target_format: "gguf".to_string(),
        quantization: Some(QuantizationConfig {
            default_type: "q4_k_m".to_string(),
            scale_optimization: "trellis".to_string(),
            trellis_lambda: 0.3,
            per_tensor: per_tensor_quant,
        }),
        tensor_map: tensor_mappings,
        shard_size: None,
    };

    fs::write("export_config.json", serde_json::to_string_pretty(&config)?)?;

    println!(
        "\nGenerated export_config.json ({} tensors)",
        tensor_map.len()
    );
    println!(
        "Run: gmat export --model {} --config export_config.json -o model.gguf",
        model_path
    );

    Ok(())
}

/// Run the export process with parallel quantization and incremental building.
///
/// Uses a producer-consumer pattern:
/// - Producer threads (rayon pool): Load and quantize tensors in parallel
/// - Consumer thread: Adds tensors to GGUF builder as they complete
///
/// This limits memory usage to ~buffer_size tensors at a time instead of all tensors.
///
/// If `shard_size_override` is provided, it overrides the config file setting.
pub fn run(
    model_path: &str,
    config_path: Option<&str>,
    output_path: Option<&str>,
    shard_size_override: Option<u64>,
) -> Result<()> {
    let config: ExportConfig = load_config(config_path, "export_config.json")?;

    if config.target_format.to_lowercase() != "gguf" {
        anyhow::bail!("Only 'gguf' target format is currently supported");
    }

    let (model_dir, metadata) = load_gmat_model(model_path)?;
    let output_file = output_path.unwrap_or("model.gguf").to_string();

    // CLI override takes precedence over config file
    let shard_size = shard_size_override.or(config.shard_size);

    if let Some(size) = shard_size {
        println!(
            "Exporting to GGUF (sharded, max {} MB per shard): {}",
            size / 1_000_000,
            output_file
        );
    } else {
        println!("Exporting to GGUF: {}", output_file);
    }

    let quant_config = config
        .quantization
        .as_ref()
        .context("quantization config is required")?;
    let default_quant = parse_quant_type(&quant_config.default_type)?;

    // Parse scale optimization setting
    let scale_opt = match quant_config.scale_optimization.to_lowercase().as_str() {
        "trellis" => ScaleOptimization::Trellis {
            lambda: quant_config.trellis_lambda,
        },
        "standard" | _ => ScaleOptimization::Standard,
    };

    println!("Scale optimization: {:?}", scale_opt);

    let tensors_dir = model_dir.join("tensors");

    // Parse tensor entries from metadata
    let tensor_name_map = metadata["tensor_name_map"]
        .as_object()
        .context("tensor_name_map not found in metadata.json")?;

    // Build a lookup from source tensor name to entry
    let mut name_to_entry: HashMap<String, TensorEntry> = HashMap::new();
    for (name, value) in tensor_name_map {
        if let Some(entry) = TensorEntry::from_json(value) {
            name_to_entry.insert(name.clone(), entry);
        }
    }

    // Build job list with validation
    let jobs: Vec<QuantizeJob> = config
        .tensor_map
        .iter()
        .filter_map(|mapping| {
            let quant_type = quant_config
                .per_tensor
                .get(&mapping.source)
                .map(|s| parse_quant_type(s))
                .transpose()
                .ok()?
                .unwrap_or(default_quant);

            // Look up tensor entry by source UUID or name
            // The config maps UUID -> GGUF name, but metadata maps name -> UUID/entry
            // We need to find the entry that matches this source UUID
            let tensor_entry = find_tensor_entry(&name_to_entry, &mapping.source, &tensors_dir)?;

            Some(QuantizeJob {
                source: mapping.source.clone(),
                target: mapping.target.clone(),
                quant_type,
                tensor_entry,
            })
        })
        .collect();

    let total = jobs.len();
    println!("\nProcessing {} tensors...", total);

    // Bounded channel to limit memory usage
    let buffer_size = num_cpus().saturating_mul(2).max(4);
    let (tx, rx) = mpsc::sync_channel::<Result<ProcessedTensor>>(buffer_size);

    let counter = AtomicUsize::new(0);

    // Clone metadata for consumer thread
    let metadata_clone = metadata.clone();
    let output_file_clone = output_file.clone();

    // Spawn consumer thread that builds GGUF incrementally (with sharding support)
    let consumer = thread::spawn(move || -> Result<ShardResult> {
        if let Some(max_shard_size) = shard_size {
            run_sharded_consumer(rx, &metadata_clone, &output_file_clone, max_shard_size)
        } else {
            run_single_file_consumer(rx, &metadata_clone, &output_file_clone)
        }
    });

    // Producer: quantize tensors in parallel and send to consumer
    jobs.into_par_iter().for_each(|job| {
        let result = (|| -> Result<ProcessedTensor> {
            match &job.tensor_entry {
                QuantizeEntry::Simple { tensor_path } => {
                    // Simple 2D tensor - load and quantize directly
                    let matrix = GraphMatrix::load(tensor_path)
                        .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", job.source, e))?;

                    let (rows, cols) = matrix.shape();

                    let quant_data = quantize_to_gguf(&matrix, job.quant_type, scale_opt, None)
                        .map_err(|e| anyhow::anyhow!("Quantize failed for {}: {}", job.target, e))?;

                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    println!(
                        "[{}/{}] {} [{}x{}] -> {:?}",
                        count, total, job.target, rows, cols, job.quant_type
                    );

                    Ok(ProcessedTensor {
                        target_name: job.target,
                        shape: (rows, cols),
                        quant_data,
                    })
                }
                QuantizeEntry::NdPlanes { original_shape, matrix_shape, plane_paths } => {
                    // N-D tensor - quantize each plane and concatenate the quantized data
                    let (_rows, cols) = *matrix_shape;
                    let mut all_quant_data: Vec<u8> = Vec::new();

                    for (plane_idx, path) in plane_paths.iter().enumerate() {
                        let matrix = GraphMatrix::load(path)
                            .map_err(|e| anyhow::anyhow!("Failed to load plane {} of {}: {}", plane_idx, job.source, e))?;

                        let plane_quant = quantize_to_gguf(&matrix, job.quant_type, scale_opt, None)
                            .map_err(|e| anyhow::anyhow!("Quantize failed for plane {} of {}: {}", plane_idx, job.target, e))?;

                        all_quant_data.extend(plane_quant.data);
                    }

                    // Build combined quant data with original shape info
                    let total_elements: usize = original_shape.iter().product();
                    let combined_rows = total_elements / cols;

                    let quant_data = transform_storage::conversions::gguf_quant::GgufQuantizedData {
                        data: all_quant_data,
                        quant_type: job.quant_type,
                        shape: (combined_rows, cols),
                    };

                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    println!(
                        "[{}/{}] {} {:?} ({} planes) -> {:?}",
                        count, total, job.target, original_shape, plane_paths.len(), job.quant_type
                    );

                    Ok(ProcessedTensor {
                        target_name: job.target,
                        shape: (combined_rows, cols),
                        quant_data,
                    })
                }
            }
        })();

        // Send result (ok or error) to consumer
        let _ = tx.send(result);
    });

    // Drop sender to signal completion
    drop(tx);

    // Wait for consumer and get results
    let result = consumer
        .join()
        .map_err(|_| anyhow::anyhow!("Consumer thread panicked"))??;

    match result {
        ShardResult::Single {
            tensor_count,
            bytes_written,
        } => {
            println!("Done! {} tensors, {} bytes", tensor_count, bytes_written);
        }
        ShardResult::Sharded {
            shard_count,
            total_tensors,
            total_bytes,
        } => {
            println!(
                "Done! {} shards, {} tensors, {} bytes total",
                shard_count, total_tensors, total_bytes
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    // ==================== TensorEntry::from_json tests ====================

    #[test]
    fn test_tensor_entry_from_json_simple() {
        let json = json!("550e8400-e29b-41d4-a716-446655440000");
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::Simple { uuid } => {
                assert_eq!(uuid, "550e8400-e29b-41d4-a716-446655440000");
            }
            _ => panic!("Expected Simple variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_1d_tensor() {
        // 1D tensor [64] becomes matrix_shape [1, 64]
        let json = json!({
            "original_shape": [64],
            "matrix_shape": [1, 64],
            "num_planes": 1,
            "plane_uuids": ["uuid_0"]
        });
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::NdPlanes { original_shape, matrix_shape, plane_uuids } => {
                assert_eq!(original_shape, vec![64]);
                assert_eq!(matrix_shape, (1, 64));
                assert_eq!(plane_uuids.len(), 1);
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_3d_planes() {
        let json = json!({
            "original_shape": [4, 32, 64],
            "matrix_shape": [32, 64],
            "num_planes": 4,
            "plane_uuids": ["uuid_0", "uuid_1", "uuid_2", "uuid_3"]
        });
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::NdPlanes { original_shape, matrix_shape, plane_uuids } => {
                assert_eq!(original_shape, vec![4, 32, 64]);
                assert_eq!(matrix_shape, (32, 64));
                assert_eq!(plane_uuids.len(), 4);
                assert_eq!(plane_uuids[0], "uuid_0");
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_4d_tensor() {
        let json = json!({
            "original_shape": [2, 4, 32, 64],
            "matrix_shape": [32, 64],
            "num_planes": 8,
            "plane_uuids": ["u_0", "u_1", "u_2", "u_3", "u_4", "u_5", "u_6", "u_7"]
        });
        let entry = TensorEntry::from_json(&json).unwrap();

        match entry {
            TensorEntry::NdPlanes { original_shape, matrix_shape, plane_uuids } => {
                assert_eq!(original_shape, vec![2, 4, 32, 64]);
                assert_eq!(matrix_shape, (32, 64));
                assert_eq!(plane_uuids.len(), 8);
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_tensor_entry_from_json_invalid() {
        let json = json!(123); // number, not string or object
        assert!(TensorEntry::from_json(&json).is_none());
    }

    #[test]
    fn test_tensor_entry_from_json_missing_fields() {
        let json = json!({
            "original_shape": [4, 32, 64],
            // missing matrix_shape and plane_uuids
        });
        assert!(TensorEntry::from_json(&json).is_none());
    }

    // ==================== find_tensor_entry tests ====================

    #[test]
    fn test_find_tensor_entry_simple() {
        let dir = TempDir::new().unwrap();
        let uuid = "550e8400-e29b-41d4-a716-446655440000";

        // Create a dummy tensor file
        std::fs::write(dir.path().join(format!("{}.gmat", uuid)), b"dummy").unwrap();

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "model.weight".to_string(),
            TensorEntry::Simple { uuid: uuid.to_string() },
        );

        let result = find_tensor_entry(&name_to_entry, uuid, dir.path());
        assert!(result.is_some());

        match result.unwrap() {
            QuantizeEntry::Simple { tensor_path } => {
                assert!(tensor_path.exists());
            }
            _ => panic!("Expected Simple variant"),
        }
    }

    #[test]
    fn test_find_tensor_entry_nd_planes() {
        let dir = TempDir::new().unwrap();
        let base_uuid = "550e8400-e29b-41d4-a716-446655440000";

        // Create plane files
        for i in 0..4 {
            std::fs::write(dir.path().join(format!("{}_{}.gmat", base_uuid, i)), b"dummy").unwrap();
        }

        let plane_uuids: Vec<String> = (0..4).map(|i| format!("{}_{}", base_uuid, i)).collect();

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "conv.weight".to_string(),
            TensorEntry::NdPlanes {
                original_shape: vec![4, 32, 64],
                matrix_shape: (32, 64),
                plane_uuids,
            },
        );

        // Find by base UUID
        let result = find_tensor_entry(&name_to_entry, base_uuid, dir.path());
        assert!(result.is_some());

        match result.unwrap() {
            QuantizeEntry::NdPlanes { original_shape, matrix_shape, plane_paths } => {
                assert_eq!(original_shape, vec![4, 32, 64]);
                assert_eq!(matrix_shape, (32, 64));
                assert_eq!(plane_paths.len(), 4);
                for path in &plane_paths {
                    assert!(path.exists());
                }
            }
            _ => panic!("Expected NdPlanes variant"),
        }
    }

    #[test]
    fn test_find_tensor_entry_missing_file() {
        let dir = TempDir::new().unwrap();
        let uuid = "nonexistent-uuid";

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "model.weight".to_string(),
            TensorEntry::Simple { uuid: uuid.to_string() },
        );

        let result = find_tensor_entry(&name_to_entry, uuid, dir.path());
        assert!(result.is_none());
    }

    #[test]
    fn test_find_tensor_entry_fallback_direct_path() {
        let dir = TempDir::new().unwrap();
        let uuid = "direct-uuid";

        // Create file but don't add to name_to_entry
        std::fs::write(dir.path().join(format!("{}.gmat", uuid)), b"dummy").unwrap();

        let name_to_entry: HashMap<String, TensorEntry> = HashMap::new();

        // Should find via fallback
        let result = find_tensor_entry(&name_to_entry, uuid, dir.path());
        assert!(result.is_some());
    }

    #[test]
    fn test_find_tensor_entry_missing_plane() {
        let dir = TempDir::new().unwrap();
        let base_uuid = "550e8400-e29b-41d4-a716-446655440000";

        // Create only some plane files (missing plane 2)
        std::fs::write(dir.path().join(format!("{}_0.gmat", base_uuid)), b"dummy").unwrap();
        std::fs::write(dir.path().join(format!("{}_1.gmat", base_uuid)), b"dummy").unwrap();
        // plane 2 missing
        std::fs::write(dir.path().join(format!("{}_3.gmat", base_uuid)), b"dummy").unwrap();

        let plane_uuids: Vec<String> = (0..4).map(|i| format!("{}_{}", base_uuid, i)).collect();

        let mut name_to_entry = HashMap::new();
        name_to_entry.insert(
            "conv.weight".to_string(),
            TensorEntry::NdPlanes {
                original_shape: vec![4, 32, 64],
                matrix_shape: (32, 64),
                plane_uuids,
            },
        );

        // Should fail because plane 2 is missing
        let result = find_tensor_entry(&name_to_entry, base_uuid, dir.path());
        assert!(result.is_none());
    }
}
