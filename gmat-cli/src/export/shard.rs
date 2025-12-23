//! Sharded GGUF output - write tensors across multiple files.

use anyhow::Result;
use gguf_rs_lib::builder::GGUFBuilder;
use gguf_rs_lib::format::MetadataValue;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use transform_storage::conversions::gguf_quant::GgufQuantizedData;

use super::util::quant_type_to_gguf_tensor_type;

/// Quantized tensor ready for GGUF output.
pub struct ProcessedTensor {
    pub target_name: String,
    pub shape: (usize, usize),
    pub quant_data: GgufQuantizedData,
}

/// Result from the consumer thread.
pub enum ShardResult {
    Single {
        tensor_count: usize,
        bytes_written: u64,
    },
    Sharded {
        shard_count: usize,
        total_tensors: usize,
        total_bytes: u64,
    },
}

/// Consumer that writes all tensors to a single GGUF file.
pub fn run_single_file_consumer(
    rx: mpsc::Receiver<Result<ProcessedTensor>>,
    metadata: &serde_json::Value,
    output_file: &str,
) -> Result<ShardResult> {
    let mut builder = GGUFBuilder::new();

    if let Some(arch) = metadata.get("architecture").and_then(|v| v.as_str()) {
        builder =
            builder.add_metadata("general.architecture", MetadataValue::String(arch.to_string()));
    }
    if let Some(name) = metadata.get("name").and_then(|v| v.as_str()) {
        builder = builder.add_metadata("general.name", MetadataValue::String(name.to_string()));
    }

    let mut tensor_count = 0;

    // Receive tensors as they're processed and add to builder
    for result in rx {
        let tensor = result?;
        let (rows, cols) = tensor.shape;

        builder = builder.add_quantized_tensor(
            tensor.target_name,
            vec![cols as u64, rows as u64],
            quant_type_to_gguf_tensor_type(&tensor.quant_data.quant_type),
            tensor.quant_data.data,
        );
        tensor_count += 1;
    }

    println!("\nWriting: {}", output_file);
    let result = builder
        .build_to_file(output_file)
        .map_err(|e| anyhow::anyhow!("Failed to write GGUF: {}", e))?;

    Ok(ShardResult::Single {
        tensor_count,
        bytes_written: result.total_bytes_written as u64,
    })
}

/// Consumer that writes tensors to multiple sharded GGUF files.
pub fn run_sharded_consumer(
    rx: mpsc::Receiver<Result<ProcessedTensor>>,
    metadata: &serde_json::Value,
    output_file: &str,
    max_shard_size: u64,
) -> Result<ShardResult> {
    let output_path = Path::new(output_file);
    let stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let parent = output_path.parent().unwrap_or(Path::new("."));

    let arch = metadata
        .get("architecture")
        .and_then(|v| v.as_str())
        .map(String::from);
    let name = metadata
        .get("name")
        .and_then(|v| v.as_str())
        .map(String::from);

    let mut shard_index = 0;
    let mut current_shard_size: u64 = 0;
    let mut builder = new_builder_with_metadata(&arch, &name);
    let mut tensors_in_shard = 0;

    let mut total_tensors = 0;
    let mut total_bytes: u64 = 0;

    // Collect all tensors first (we need to know total count for shard naming)
    let mut all_tensors = Vec::new();
    for result in rx {
        all_tensors.push(result?);
    }

    for tensor in all_tensors {
        let (rows, cols) = tensor.shape;
        let tensor_size = tensor.quant_data.data.len() as u64;

        // Check if adding this tensor would exceed shard size
        // (but always add at least one tensor per shard)
        if tensors_in_shard > 0 && current_shard_size + tensor_size > max_shard_size {
            // Write current shard
            let shard_file = shard_filename(parent, stem, shard_index);
            println!("\nWriting shard: {}", shard_file.display());

            let result = builder
                .build_to_file(&shard_file)
                .map_err(|e| anyhow::anyhow!("Failed to write shard {}: {}", shard_index, e))?;

            total_bytes += result.total_bytes_written as u64;

            // Start new shard
            shard_index += 1;
            builder = new_builder_with_metadata(&arch, &name);
            current_shard_size = 0;
            tensors_in_shard = 0;
        }

        // Add tensor to current shard
        builder = builder.add_quantized_tensor(
            tensor.target_name,
            vec![cols as u64, rows as u64],
            quant_type_to_gguf_tensor_type(&tensor.quant_data.quant_type),
            tensor.quant_data.data,
        );

        current_shard_size += tensor_size;
        tensors_in_shard += 1;
        total_tensors += 1;
    }

    // Write final shard if it has tensors
    if tensors_in_shard > 0 {
        let shard_file = shard_filename(parent, stem, shard_index);
        println!("\nWriting shard: {}", shard_file.display());

        let result = builder
            .build_to_file(&shard_file)
            .map_err(|e| anyhow::anyhow!("Failed to write final shard: {}", e))?;

        total_bytes += result.total_bytes_written as u64;
        shard_index += 1;
    }

    Ok(ShardResult::Sharded {
        shard_count: shard_index,
        total_tensors,
        total_bytes,
    })
}

/// Create a new GGUF builder with metadata.
fn new_builder_with_metadata(arch: &Option<String>, name: &Option<String>) -> GGUFBuilder {
    let mut builder = GGUFBuilder::new();

    if let Some(arch) = arch {
        builder =
            builder.add_metadata("general.architecture", MetadataValue::String(arch.clone()));
    }
    if let Some(name) = name {
        builder = builder.add_metadata("general.name", MetadataValue::String(name.clone()));
    }

    builder
}

/// Generate shard filename: model-00001.gguf, model-00002.gguf, etc.
fn shard_filename(parent: &Path, stem: &str, shard_index: usize) -> PathBuf {
    parent.join(format!("{}-{:05}.gguf", stem, shard_index + 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // ==================== shard_filename tests ====================

    #[test]
    fn test_shard_filename_first_shard() {
        let result = shard_filename(Path::new("/output"), "model", 0);
        assert_eq!(result, PathBuf::from("/output/model-00001.gguf"));
    }

    #[test]
    fn test_shard_filename_second_shard() {
        let result = shard_filename(Path::new("/output"), "model", 1);
        assert_eq!(result, PathBuf::from("/output/model-00002.gguf"));
    }

    #[test]
    fn test_shard_filename_large_index() {
        let result = shard_filename(Path::new("/output"), "model", 999);
        assert_eq!(result, PathBuf::from("/output/model-01000.gguf"));
    }

    #[test]
    fn test_shard_filename_custom_stem() {
        let result = shard_filename(Path::new("."), "llama-7b-q4", 0);
        assert_eq!(result, PathBuf::from("./llama-7b-q4-00001.gguf"));
    }

    #[test]
    fn test_shard_filename_nested_path() {
        let result = shard_filename(Path::new("/home/user/models/output"), "model", 5);
        assert_eq!(
            result,
            PathBuf::from("/home/user/models/output/model-00006.gguf")
        );
    }

    // ==================== ShardResult tests ====================

    #[test]
    fn test_shard_result_single() {
        let result = ShardResult::Single {
            tensor_count: 100,
            bytes_written: 1_000_000,
        };

        match result {
            ShardResult::Single {
                tensor_count,
                bytes_written,
            } => {
                assert_eq!(tensor_count, 100);
                assert_eq!(bytes_written, 1_000_000);
            }
            _ => panic!("Expected Single variant"),
        }
    }

    #[test]
    fn test_shard_result_sharded() {
        let result = ShardResult::Sharded {
            shard_count: 5,
            total_tensors: 200,
            total_bytes: 10_000_000_000,
        };

        match result {
            ShardResult::Sharded {
                shard_count,
                total_tensors,
                total_bytes,
            } => {
                assert_eq!(shard_count, 5);
                assert_eq!(total_tensors, 200);
                assert_eq!(total_bytes, 10_000_000_000);
            }
            _ => panic!("Expected Sharded variant"),
        }
    }

    // ==================== ProcessedTensor tests ====================

    #[test]
    fn test_processed_tensor_creation() {
        use transform_storage::conversions::gguf_quant::GgufQuantType;

        let tensor = ProcessedTensor {
            target_name: "blk.0.attn_q.weight".to_string(),
            shape: (4096, 4096),
            quant_data: GgufQuantizedData {
                data: vec![0u8; 1000],
                quant_type: GgufQuantType::Q4_K_M,
                shape: (4096, 4096),
            },
        };

        assert_eq!(tensor.target_name, "blk.0.attn_q.weight");
        assert_eq!(tensor.shape, (4096, 4096));
        assert_eq!(tensor.quant_data.data.len(), 1000);
    }

    // ==================== new_builder_with_metadata tests ====================

    #[test]
    fn test_new_builder_with_no_metadata() {
        let builder = new_builder_with_metadata(&None, &None);
        // Just verify it doesn't panic and returns a builder
        let _ = builder;
    }

    #[test]
    fn test_new_builder_with_arch_only() {
        let builder = new_builder_with_metadata(&Some("llama".to_string()), &None);
        let _ = builder;
    }

    #[test]
    fn test_new_builder_with_name_only() {
        let builder = new_builder_with_metadata(&None, &Some("my-model".to_string()));
        let _ = builder;
    }

    #[test]
    fn test_new_builder_with_both_metadata() {
        let builder = new_builder_with_metadata(
            &Some("llama".to_string()),
            &Some("Llama-7B".to_string()),
        );
        let _ = builder;
    }
}
