//! Sharded GGUF output - write tensors across multiple files.
//!
//! Provides streaming writer that writes tensors incrementally without
//! holding entire model in memory.

use anyhow::Result;
use gguf_rs_lib::builder::GGUFBuilder;
use gguf_rs_lib::format::MetadataValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use transform_storage::conversions::gguf_quant::GgufQuantizedData;

use super::token_metadata::TokenMetadata;
use super::util::quant_type_to_gguf_tensor_type;
use crate::config::export_config::GgufMetadataValues;

/// Quantized tensor ready for GGUF output.
pub struct ProcessedTensor {
    pub target_name: String,
    pub shape: (usize, usize),
    pub quant_data: GgufQuantizedData,
}

/// Result from the writer.
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

/// Streaming GGUF writer that writes tensors incrementally.
/// For sharded mode, writes a new shard when current one exceeds max size.
pub struct GgufStreamWriter {
    builder: GGUFBuilder,
    gguf_metadata: GgufMetadataValues,
    gguf_architecture: String,
    special_token_keys: HashMap<String, String>,
    token_meta: Option<TokenMetadata>,
    output_file: String,
    max_shard_size: Option<u64>,

    // Shard tracking
    shard_index: usize,
    current_shard_size: u64,
    tensors_in_shard: usize,

    // Totals
    total_tensors: usize,
    total_bytes: u64,
}

impl GgufStreamWriter {
    pub fn new(
        gguf_metadata: &GgufMetadataValues,
        output_file: &str,
        max_shard_size: Option<u64>,
        token_meta: Option<TokenMetadata>,
        gguf_architecture: &str,
        special_token_keys: &HashMap<String, String>,
    ) -> Self {
        let builder = new_builder_with_metadata(
            gguf_metadata,
            token_meta.as_ref(),
            gguf_architecture,
            special_token_keys,
        );

        Self {
            builder,
            gguf_metadata: gguf_metadata.clone(),
            gguf_architecture: gguf_architecture.to_string(),
            special_token_keys: special_token_keys.clone(),
            token_meta,
            output_file: output_file.to_string(),
            max_shard_size,
            shard_index: 0,
            current_shard_size: 0,
            tensors_in_shard: 0,
            total_tensors: 0,
            total_bytes: 0,
        }
    }

    /// Add a tensor to the current shard. May trigger shard flush if size exceeded.
    pub fn add_tensor(&mut self, tensor: ProcessedTensor) -> Result<()> {
        let tensor_size = tensor.quant_data.data.len() as u64;

        // Check if we need to flush current shard (sharded mode only)
        if let Some(max_size) = self.max_shard_size
            && self.tensors_in_shard > 0
            && self.current_shard_size + tensor_size > max_size
        {
            self.flush_shard()?;
        }

        // Add tensor to builder
        let (rows, cols) = tensor.shape;
        self.builder = std::mem::replace(&mut self.builder, GGUFBuilder::new())
            .add_quantized_tensor(
                tensor.target_name,
                vec![cols as u64, rows as u64],
                quant_type_to_gguf_tensor_type(&tensor.quant_data.quant_type),
                tensor.quant_data.data,
            );

        self.current_shard_size += tensor_size;
        self.tensors_in_shard += 1;
        self.total_tensors += 1;

        Ok(())
    }

    /// Flush current shard to disk and start a new one.
    fn flush_shard(&mut self) -> Result<()> {
        let shard_file = self.shard_filename();
        println!("\nWriting shard: {}", shard_file.display());

        let result = std::mem::replace(
            &mut self.builder,
            new_builder_with_metadata(
                &self.gguf_metadata,
                self.token_meta.as_ref(),
                &self.gguf_architecture,
                &self.special_token_keys,
            ),
        )
        .build_to_file(&shard_file)
        .map_err(|e| anyhow::anyhow!("Failed to write shard {}: {}", self.shard_index, e))?;

        self.total_bytes += result.total_bytes_written as u64;
        self.shard_index += 1;
        self.current_shard_size = 0;
        self.tensors_in_shard = 0;

        Ok(())
    }

    fn shard_filename(&self) -> PathBuf {
        let output_path = Path::new(&self.output_file);
        let stem = output_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let parent = output_path.parent().unwrap_or(Path::new("."));
        parent.join(format!("{}-{:05}.gguf", stem, self.shard_index + 1))
    }

    /// Finish writing. Returns the result.
    pub fn finish(mut self) -> Result<ShardResult> {
        if self.tensors_in_shard == 0 {
            // Nothing to write
            if self.max_shard_size.is_some() {
                return Ok(ShardResult::Sharded {
                    shard_count: self.shard_index,
                    total_tensors: self.total_tensors,
                    total_bytes: self.total_bytes,
                });
            } else {
                return Ok(ShardResult::Single {
                    tensor_count: 0,
                    bytes_written: 0,
                });
            }
        }

        if self.max_shard_size.is_some() {
            // Sharded mode - write final shard
            self.flush_shard()?;
            Ok(ShardResult::Sharded {
                shard_count: self.shard_index,
                total_tensors: self.total_tensors,
                total_bytes: self.total_bytes,
            })
        } else {
            // Single file mode
            println!("\nWriting: {}", self.output_file);
            let result = self
                .builder
                .build_to_file(&self.output_file)
                .map_err(|e| anyhow::anyhow!("Failed to write GGUF: {}", e))?;

            Ok(ShardResult::Single {
                tensor_count: self.total_tensors,
                bytes_written: result.total_bytes_written as u64,
            })
        }
    }
}

/// Create a new GGUF builder with pre-resolved model metadata.
fn new_builder_with_metadata(
    meta: &GgufMetadataValues,
    token_meta: Option<&TokenMetadata>,
    gguf_architecture: &str,
    special_token_keys: &HashMap<String, String>,
) -> GGUFBuilder {
    let mut builder = GGUFBuilder::new();

    let arch = gguf_architecture;

    // General metadata
    builder = builder.add_metadata(
        "general.architecture",
        MetadataValue::String(arch.to_string()),
    );
    if let Some(name) = &meta.name {
        builder = builder.add_metadata("general.name", MetadataValue::String(name.clone()));
    }

    // Architecture-specific metadata (prefixed with arch name)
    if let Some(v) = meta.vocab_size {
        builder =
            builder.add_metadata(format!("{}.vocab_size", arch), MetadataValue::U32(v as u32));
    }
    if let Some(v) = meta.hidden_size {
        builder = builder.add_metadata(
            format!("{}.embedding_length", arch),
            MetadataValue::U32(v as u32),
        );
    }
    if let Some(v) = meta.num_layers {
        builder = builder.add_metadata(
            format!("{}.block_count", arch),
            MetadataValue::U32(v as u32),
        );
    }
    if let Some(v) = meta.num_attention_heads {
        builder = builder.add_metadata(
            format!("{}.attention.head_count", arch),
            MetadataValue::U32(v as u32),
        );
    }
    if let Some(v) = meta.num_key_value_heads {
        builder = builder.add_metadata(
            format!("{}.attention.head_count_kv", arch),
            MetadataValue::U32(v as u32),
        );
    }
    if let Some(v) = meta.intermediate_size {
        builder = builder.add_metadata(
            format!("{}.feed_forward_length", arch),
            MetadataValue::U32(v as u32),
        );
    }
    if let Some(v) = meta.max_position_embeddings {
        builder = builder.add_metadata(
            format!("{}.context_length", arch),
            MetadataValue::U32(v as u32),
        );
    }
    if let Some(v) = meta.rms_norm_eps {
        builder = builder.add_metadata(
            format!("{}.attention.layer_norm_rms_epsilon", arch),
            MetadataValue::F32(v as f32),
        );
    }
    if let Some(v) = meta.rope_theta {
        builder = builder.add_metadata(
            format!("{}.rope.freq_base", arch),
            MetadataValue::F32(v as f32),
        );
    }

    // MoE metadata
    if let Some(v) = meta.num_experts {
        builder = builder.add_metadata(
            format!("{}.expert_count", arch),
            MetadataValue::U32(v as u32),
        );
    }
    if let Some(v) = meta.num_experts_per_tok {
        builder = builder.add_metadata(
            format!("{}.expert_used_count", arch),
            MetadataValue::U32(v as u32),
        );
    }

    // Tokenizer metadata with pre-resolved special token keys
    if let Some(token_meta) = token_meta
        && let Ok(token_metadata) = token_meta.to_gguf_metadata_with_keys(special_token_keys)
    {
        for (key, value) in token_metadata {
            builder = builder.add_metadata(key, value);
        }
    }

    builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Generate shard filename: model-00001.gguf, model-00002.gguf, etc.
    fn shard_filename(parent: &Path, stem: &str, shard_index: usize) -> PathBuf {
        parent.join(format!("{}-{:05}.gguf", stem, shard_index + 1))
    }

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
        let meta = GgufMetadataValues::default();
        let special_token_keys = HashMap::new();
        let builder = new_builder_with_metadata(&meta, None, "llama", &special_token_keys);
        // Just verify it doesn't panic and returns a builder
        let _ = builder;
    }

    #[test]
    fn test_new_builder_with_name_only() {
        let meta = GgufMetadataValues {
            name: Some("my-model".to_string()),
            ..Default::default()
        };
        let special_token_keys = HashMap::new();
        let builder = new_builder_with_metadata(&meta, None, "llama", &special_token_keys);
        let _ = builder;
    }

    #[test]
    fn test_new_builder_with_full_metadata() {
        let meta = GgufMetadataValues {
            name: Some("Llama-7B".to_string()),
            vocab_size: Some(32000),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            intermediate_size: Some(11008),
            max_position_embeddings: Some(4096),
            rms_norm_eps: Some(1e-5),
            rope_theta: Some(10000.0),
            num_experts: None,
            num_experts_per_tok: None,
        };
        let special_token_keys = HashMap::new();
        let builder = new_builder_with_metadata(&meta, None, "llama", &special_token_keys);
        let _ = builder;
    }

    #[test]
    fn test_new_builder_with_moe_metadata() {
        let meta = GgufMetadataValues {
            num_experts: Some(64),
            num_experts_per_tok: Some(8),
            ..Default::default()
        };
        let special_token_keys = HashMap::new();
        let builder = new_builder_with_metadata(&meta, None, "qwen2", &special_token_keys);
        let _ = builder;
    }
}
