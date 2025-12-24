//! Import module - SafeTensors/GGUF to GMAT conversion.

mod metadata;
mod streaming;

use anyhow::{Context, Result};
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;
use transform_storage::BlockFormat;
use uuid::Uuid;

use crate::common::{discover_safetensor_files, load_config};
use crate::config::import_config::{ImportConfig, TensorMapping};

use metadata::extract_model_metadata;
use streaming::{run_streaming_import, SavedTensor};

/// Generate an import config template from a model file.
pub fn generate_config_template(model_path: &str) -> Result<()> {
    let path = Path::new(model_path);
    let safetensor_files = discover_safetensor_files(path)?;

    println!("Found {} safetensors file(s)", safetensor_files.len());

    let metadata = extract_model_metadata(&safetensor_files)?;
    let tensor_mappings = collect_tensor_mappings(&safetensor_files)?;

    let config = ImportConfig {
        source_format: "safetensors".to_string(),
        block_format: "B8x8".to_string(),
        tensor_map: tensor_mappings,
        metadata,
    };

    let json = serde_json::to_string_pretty(&config)?;
    fs::write("import_config.json", json)?;

    println!("\n=== Generated import_config.json ===");
    println!("Tensors: {}", config.tensor_map.len());
    if let Some(ref arch) = config.metadata.architecture {
        println!("Architecture: {}", arch);
    }
    println!("\nEdit the config, then run:");
    println!("  gmat import --model {} --config import_config.json", model_path);

    Ok(())
}

/// Run the import process with streaming tensor conversion.
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

    let total = tensor_map.len();
    println!("Processing {} tensors...", total);

    let saved = run_streaming_import(&safetensor_files, &tensor_map, &tensors_dir, &config, block_format, total)?;

    write_import_metadata(&output_dir, &config, &saved)?;
    println!("\n=== Import Complete: {} tensors ===", saved.len());
    Ok(())
}

/// Collect tensor mappings from safetensor files using memory-mapped I/O.
///
/// This only reads the safetensor header metadata, not the full tensor data,
/// making it much faster for large model files.
fn collect_tensor_mappings(safetensor_files: &[std::path::PathBuf]) -> Result<Vec<TensorMapping>> {
    safetensor_files
        .par_iter()
        .map(|file_path| -> Result<Vec<TensorMapping>> {
            // Use memory-mapped I/O - only the header pages will be read from disk
            let file = File::open(file_path)
                .with_context(|| format!("Failed to open: {}", file_path.display()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("Failed to mmap: {}", file_path.display()))?;

            // read_metadata only parses the header, not tensor data
            let (header_size, metadata) = SafeTensors::read_metadata(&mmap)
                .with_context(|| format!("Failed to parse header: {}", file_path.display()))?;

            let mappings: Vec<TensorMapping> = metadata
                .tensors()
                .iter()
                .map(|(name, info)| {
                    println!("  - {} (shape: {:?})", name, info.shape);
                    TensorMapping {
                        source: name.to_string(),
                        target: Uuid::new_v4().to_string(),
                        include: true,
                    }
                })
                .collect();

            // Log header size for debugging large files
            if header_size > 1_000_000 {
                println!("  (header: {} KB)", header_size / 1024);
            }

            Ok(mappings)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .pipe(Ok)
}

/// Write the final metadata.json file.
fn write_import_metadata(output_dir: &Path, config: &ImportConfig, saved: &[SavedTensor]) -> Result<()> {
    // Build tensor_name_map with nested structure for N-D tensors
    let mut tensor_name_map = serde_json::Map::new();
    let mut nd_tensors: HashMap<String, Vec<&SavedTensor>> = HashMap::new();

    for s in saved {
        if s.num_planes.is_some() {
            // N-D tensor plane - group by base name (without [idx])
            let base_name = s.name.split('[').next().unwrap_or(&s.name);
            nd_tensors.entry(base_name.to_string()).or_default().push(s);
        } else {
            // Regular 2D tensor - simple string mapping
            tensor_name_map.insert(s.name.clone(), serde_json::json!(s.uuid));
        }
    }

    // Add N-D tensors as nested blocks
    for (base_name, planes) in nd_tensors {
        // Sort by plane index
        let mut planes = planes;
        planes.sort_by_key(|p| p.plane_index.unwrap_or(0));

        let first = planes.first().unwrap();
        let original_shape = first.original_shape.as_ref().unwrap();
        let num_planes = first.num_planes.unwrap();
        let (rows, cols) = first.matrix_shape;

        let plane_uuids: Vec<&str> = planes.iter().map(|p| p.uuid.as_str()).collect();

        tensor_name_map.insert(base_name, serde_json::json!({
            "original_shape": original_shape,
            "matrix_shape": [rows, cols],
            "num_planes": num_planes,
            "plane_uuids": plane_uuids,
        }));
    }

    let metadata_json = serde_json::json!({
        "config": config,
        "tensor_name_map": tensor_name_map,
        "total_tensors": saved.len(),
    });

    fs::write(
        output_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata_json)?,
    )?;
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

/// Pipe trait for fluent syntax.
trait Pipe: Sized {
    fn pipe<F, R>(self, f: F) -> R where F: FnOnce(Self) -> R { f(self) }
}
impl<T> Pipe for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_block_format_b8x8() {
        assert!(matches!(parse_block_format("B8x8").unwrap(), BlockFormat::B8x8));
    }

    #[test]
    fn test_parse_block_format_all_variants() {
        assert!(parse_block_format("B8x4").is_ok());
        assert!(parse_block_format("B16x4").is_ok());
        assert!(parse_block_format("B16x8").is_ok());
        assert!(parse_block_format("DualRow8x4").is_ok());
        assert!(parse_block_format("DualRow8x8").is_ok());
        assert!(parse_block_format("DualRow16x4").is_ok());
        assert!(parse_block_format("DualRow16x8").is_ok());
    }

    #[test]
    fn test_parse_block_format_unknown() {
        let result = parse_block_format("B32x32");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown block format"));
    }

    #[test]
    fn test_parse_block_format_case_sensitive() {
        assert!(parse_block_format("b8x8").is_err());
    }

    #[test]
    fn test_tensor_map_filters_excluded() {
        let mappings = vec![
            TensorMapping { source: "t1".into(), target: "u1".into(), include: true },
            TensorMapping { source: "t2".into(), target: "u2".into(), include: false },
            TensorMapping { source: "t3".into(), target: "u3".into(), include: true },
        ];

        let tensor_map: HashMap<String, String> = mappings
            .iter()
            .filter(|tm| tm.include)
            .map(|tm| (tm.source.clone(), tm.target.clone()))
            .collect();

        assert_eq!(tensor_map.len(), 2);
        assert!(tensor_map.contains_key("t1"));
        assert!(!tensor_map.contains_key("t2"));
    }
}
