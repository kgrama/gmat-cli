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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ImportConfig tests ====================

    #[test]
    fn test_import_config_default() {
        let config = ImportConfig::default();
        assert_eq!(config.source_format, "safetensors");
        assert_eq!(config.block_format, "B8x8");
        assert!(config.tensor_map.is_empty());
    }

    #[test]
    fn test_import_config_serialization() {
        let config = ImportConfig {
            source_format: "safetensors".to_string(),
            block_format: "B16x8".to_string(),
            tensor_map: vec![TensorMapping {
                source: "layer.0.weight".to_string(),
                target: "uuid-1234".to_string(),
                include: true,
            }],
            metadata: ModelMetadata {
                architecture: Some("llama".to_string()),
                vocab_size: Some(32000),
                ..Default::default()
            },
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: ImportConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.source_format, "safetensors");
        assert_eq!(parsed.block_format, "B16x8");
        assert_eq!(parsed.tensor_map.len(), 1);
        assert_eq!(parsed.tensor_map[0].source, "layer.0.weight");
        assert_eq!(parsed.metadata.architecture, Some("llama".to_string()));
        assert_eq!(parsed.metadata.vocab_size, Some(32000));
    }

    #[test]
    fn test_import_config_deserialization() {
        let json = r#"{
            "source_format": "gguf",
            "block_format": "B8x4",
            "tensor_map": [
                {"source": "tensor1", "target": "uuid1", "include": true},
                {"source": "tensor2", "target": "uuid2", "include": false}
            ],
            "metadata": {
                "architecture": "mistral",
                "hidden_size": 4096
            }
        }"#;

        let config: ImportConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.source_format, "gguf");
        assert_eq!(config.block_format, "B8x4");
        assert_eq!(config.tensor_map.len(), 2);
        assert!(config.tensor_map[0].include);
        assert!(!config.tensor_map[1].include);
        assert_eq!(config.metadata.architecture, Some("mistral".to_string()));
        assert_eq!(config.metadata.hidden_size, Some(4096));
        assert!(config.metadata.vocab_size.is_none());
    }

    // ==================== TensorMapping tests ====================

    #[test]
    fn test_tensor_mapping_serialization() {
        let mapping = TensorMapping {
            source: "model.layer.0.weight".to_string(),
            target: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            include: true,
        };

        let json = serde_json::to_string(&mapping).unwrap();
        assert!(json.contains("model.layer.0.weight"));
        assert!(json.contains("550e8400"));

        let parsed: TensorMapping = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source, mapping.source);
        assert_eq!(parsed.target, mapping.target);
        assert_eq!(parsed.include, mapping.include);
    }

    // ==================== ModelMetadata tests ====================

    #[test]
    fn test_model_metadata_default() {
        let metadata = ModelMetadata::default();
        assert!(metadata.architecture.is_none());
        assert!(metadata.vocab_size.is_none());
        assert!(metadata.hidden_size.is_none());
        assert!(metadata.num_layers.is_none());
        assert!(metadata.num_attention_heads.is_none());
        assert!(metadata.intermediate_size.is_none());
        assert!(metadata.max_position_embeddings.is_none());
    }

    #[test]
    fn test_model_metadata_full() {
        let metadata = ModelMetadata {
            architecture: Some("llama".to_string()),
            vocab_size: Some(32000),
            hidden_size: Some(4096),
            num_layers: Some(32),
            num_attention_heads: Some(32),
            intermediate_size: Some(11008),
            max_position_embeddings: Some(4096),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let parsed: ModelMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.architecture, Some("llama".to_string()));
        assert_eq!(parsed.vocab_size, Some(32000));
        assert_eq!(parsed.hidden_size, Some(4096));
        assert_eq!(parsed.num_layers, Some(32));
        assert_eq!(parsed.num_attention_heads, Some(32));
        assert_eq!(parsed.intermediate_size, Some(11008));
        assert_eq!(parsed.max_position_embeddings, Some(4096));
    }

    #[test]
    fn test_model_metadata_partial_deserialization() {
        // Only some fields provided - others should be None
        let json = r#"{"architecture": "phi", "num_layers": 24}"#;
        let metadata: ModelMetadata = serde_json::from_str(json).unwrap();

        assert_eq!(metadata.architecture, Some("phi".to_string()));
        assert_eq!(metadata.num_layers, Some(24));
        assert!(metadata.vocab_size.is_none());
        assert!(metadata.hidden_size.is_none());
    }
}
