//! Import configuration for SafeTensors/GGUF to GMAT conversion.

use serde::{Deserialize, Serialize};

/// Configuration for importing a model to GMAT format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Source format: "safetensors" or "gguf"
    pub source_format: String,

    /// Block format to use for storage
    pub block_format: String,

    /// Tensor mappings (source name -> GMAT layer name)
    pub tensor_map: Vec<TensorMapping>,

    /// Model metadata to preserve
    pub metadata: ModelMetadata,
}

/// Mapping from source tensor name to GMAT storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMapping {
    /// Source tensor name pattern (supports wildcards)
    pub source: String,

    /// Target UUID string for GMAT storage
    pub target: String,

    /// Whether to include this tensor
    pub include: bool,
}

/// Model metadata configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub architecture: Option<String>,
    pub vocab_size: Option<u64>,
    pub hidden_size: Option<u64>,
    pub num_layers: Option<u64>,
    pub num_attention_heads: Option<u64>,
    pub intermediate_size: Option<u64>,
    pub max_position_embeddings: Option<u64>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            source_format: "safetensors".to_string(),
            block_format: "B8x8".to_string(),
            tensor_map: Vec::new(),
            metadata: ModelMetadata::default(),
        }
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            architecture: None,
            vocab_size: None,
            hidden_size: None,
            num_layers: None,
            num_attention_heads: None,
            intermediate_size: None,
            max_position_embeddings: None,
        }
    }
}
