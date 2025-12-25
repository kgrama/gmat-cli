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
///
/// Uses tokio-rayon for parallel metadata extraction and tensor mapping.
pub fn generate_config_template(model_path: &str) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(generate_config_template_async(model_path))
}

/// Async implementation of import config generation.
async fn generate_config_template_async(model_path: &str) -> Result<()> {
    use std::time::Instant;

    let start_time = Instant::now();
    let path = Path::new(model_path);
    let safetensor_files = discover_safetensor_files(path)?;

    println!("Found {} safetensors file(s)", safetensor_files.len());

    // Run CPU-bound metadata extraction on rayon pool
    let files_for_metadata = safetensor_files.clone();
    let metadata_start = Instant::now();
    let metadata = tokio_rayon::spawn(move || extract_model_metadata(&files_for_metadata)).await?;
    let metadata_elapsed = metadata_start.elapsed();

    // Run CPU-bound tensor mapping and stats collection on rayon pool
    let files_for_mappings = safetensor_files.clone();
    let mapping_start = Instant::now();
    let (tensor_mappings, stats) = tokio_rayon::spawn(move || collect_tensor_mappings_with_stats(&files_for_mappings)).await?;
    let mapping_elapsed = mapping_start.elapsed();

    let config = ImportConfig {
        source_format: "safetensors".to_string(),
        block_format: "B8x8".to_string(),
        tensor_map: tensor_mappings,
        metadata,
    };

    let json = serde_json::to_string_pretty(&config)?;
    fs::write("import_config.json", json)?;

    // Print statistics
    stats.print_summary();

    let total_elapsed = start_time.elapsed();

    println!("\n=== Generated import_config.json ===");
    println!("Tensors: {}", config.tensor_map.len());
    if let Some(ref arch) = config.metadata.architecture {
        println!("Architecture: {}", arch);
    }

    println!("\n=== Timing ===");
    println!("Metadata extraction: {:.2}s", metadata_elapsed.as_secs_f64());
    println!("Tensor mapping: {:.2}s", mapping_elapsed.as_secs_f64());
    println!("Total: {:.2}s", total_elapsed.as_secs_f64());

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

/// Statistics collected during tensor mapping.
#[derive(Debug, Default)]
struct TensorStats {
    total_tensors: usize,
    total_elements: u64,
    total_bytes: u64,
    dtype_counts: HashMap<String, usize>,
    shape_distribution: ShapeDistribution,
}

/// Distribution of tensor shapes by dimension count.
#[derive(Debug, Default)]
struct ShapeDistribution {
    dim_1d: usize,
    dim_2d: usize,
    dim_3d: usize,
    dim_4d_plus: usize,
    min_elements: u64,
    max_elements: u64,
}

impl TensorStats {
    fn merge(&mut self, other: TensorStats) {
        self.total_tensors += other.total_tensors;
        self.total_elements += other.total_elements;
        self.total_bytes += other.total_bytes;
        for (dtype, count) in other.dtype_counts {
            *self.dtype_counts.entry(dtype).or_insert(0) += count;
        }
        self.shape_distribution.dim_1d += other.shape_distribution.dim_1d;
        self.shape_distribution.dim_2d += other.shape_distribution.dim_2d;
        self.shape_distribution.dim_3d += other.shape_distribution.dim_3d;
        self.shape_distribution.dim_4d_plus += other.shape_distribution.dim_4d_plus;
        if other.shape_distribution.min_elements > 0 {
            if self.shape_distribution.min_elements == 0 {
                self.shape_distribution.min_elements = other.shape_distribution.min_elements;
            } else {
                self.shape_distribution.min_elements = self.shape_distribution.min_elements.min(other.shape_distribution.min_elements);
            }
        }
        self.shape_distribution.max_elements = self.shape_distribution.max_elements.max(other.shape_distribution.max_elements);
    }

    fn print_summary(&self) {
        println!("\n=== Tensor Statistics ===");
        println!("Total tensors: {}", self.total_tensors);
        println!("Total elements: {} ({:.2} B)", self.total_elements, self.total_elements as f64 / 1e9);
        println!("Total size: {} ({:.2} GB)", format_bytes(self.total_bytes), self.total_bytes as f64 / 1e9);

        println!("\nShape distribution:");
        println!("  1D tensors: {}", self.shape_distribution.dim_1d);
        println!("  2D tensors: {}", self.shape_distribution.dim_2d);
        println!("  3D tensors: {}", self.shape_distribution.dim_3d);
        println!("  4D+ tensors: {}", self.shape_distribution.dim_4d_plus);
        println!("  Min elements: {}", self.shape_distribution.min_elements);
        println!("  Max elements: {} ({:.2} M)", self.shape_distribution.max_elements, self.shape_distribution.max_elements as f64 / 1e6);

        println!("\nData types:");
        let mut dtypes: Vec<_> = self.dtype_counts.iter().collect();
        dtypes.sort_by(|a, b| b.1.cmp(a.1));
        for (dtype, count) in dtypes {
            println!("  {}: {}", dtype, count);
        }
    }
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Get byte size for a dtype.
fn dtype_byte_size(dtype: safetensors::Dtype) -> u64 {
    match dtype {
        safetensors::Dtype::BOOL | safetensors::Dtype::U8 | safetensors::Dtype::I8 => 1,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 | safetensors::Dtype::I16 | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        _ => 4, // default assumption
    }
}

/// Collect tensor mappings and stats from safetensor files using memory-mapped I/O.
///
/// This only reads the safetensor header metadata, not the full tensor data,
/// making it much faster for large model files.
fn collect_tensor_mappings_with_stats(safetensor_files: &[std::path::PathBuf]) -> Result<(Vec<TensorMapping>, TensorStats)> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let total_files = safetensor_files.len();
    let processed_files = AtomicUsize::new(0);

    let results: Vec<(Vec<TensorMapping>, TensorStats)> = safetensor_files
        .par_iter()
        .map(|file_path| -> Result<(Vec<TensorMapping>, TensorStats)> {
            // Use memory-mapped I/O - only the header pages will be read from disk
            let file = File::open(file_path)
                .with_context(|| format!("Failed to open: {}", file_path.display()))?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("Failed to mmap: {}", file_path.display()))?;

            // read_metadata only parses the header, not tensor data
            let (header_size, metadata) = SafeTensors::read_metadata(&mmap)
                .with_context(|| format!("Failed to parse header: {}", file_path.display()))?;

            let mut stats = TensorStats::default();

            let mappings: Vec<TensorMapping> = metadata
                .tensors()
                .iter()
                .map(|(name, info)| {
                    let shape = &info.shape;
                    let numel: u64 = shape.iter().map(|&x| x as u64).product();
                    let dtype = info.dtype;
                    let bytes = numel * dtype_byte_size(dtype);

                    // Update stats
                    stats.total_tensors += 1;
                    stats.total_elements += numel;
                    stats.total_bytes += bytes;
                    *stats.dtype_counts.entry(format!("{:?}", dtype)).or_insert(0) += 1;

                    // Shape distribution
                    match shape.len() {
                        1 => stats.shape_distribution.dim_1d += 1,
                        2 => stats.shape_distribution.dim_2d += 1,
                        3 => stats.shape_distribution.dim_3d += 1,
                        _ => stats.shape_distribution.dim_4d_plus += 1,
                    }
                    if stats.shape_distribution.min_elements == 0 || numel < stats.shape_distribution.min_elements {
                        stats.shape_distribution.min_elements = numel;
                    }
                    if numel > stats.shape_distribution.max_elements {
                        stats.shape_distribution.max_elements = numel;
                    }

                    TensorMapping {
                        source: name.to_string(),
                        target: Uuid::new_v4().to_string(),
                        include: true,
                    }
                })
                .collect();

            // Log header size for debugging large files
            if header_size > 1_000_000 {
                eprintln!("  (header: {} KB for {})", header_size / 1024, file_path.display());
            }

            // Progress update
            let count = processed_files.fetch_add(1, Ordering::Relaxed) + 1;
            eprint!("\rProcessed {}/{} files ({} tensors)...", count, total_files, stats.total_tensors);
            use std::io::Write;
            let _ = std::io::stderr().flush();

            Ok((mappings, stats))
        })
        .collect::<Result<Vec<_>>>()?;

    eprintln!(); // New line after progress

    // Merge all results
    let mut all_mappings = Vec::new();
    let mut total_stats = TensorStats::default();

    for (mappings, stats) in results {
        all_mappings.extend(mappings);
        total_stats.merge(stats);
    }

    Ok((all_mappings, total_stats))
}

/// Collect tensor mappings from safetensor files using memory-mapped I/O.
///
/// This only reads the safetensor header metadata, not the full tensor data,
/// making it much faster for large model files.
#[allow(dead_code)]
fn collect_tensor_mappings(safetensor_files: &[std::path::PathBuf]) -> Result<Vec<TensorMapping>> {
    let (mappings, _) = collect_tensor_mappings_with_stats(safetensor_files)?;
    Ok(mappings)
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
#[allow(dead_code)]
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
