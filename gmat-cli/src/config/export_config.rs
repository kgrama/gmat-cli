//! Export configuration for GMAT to GGUF conversion.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for exporting a GMAT model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Target format: "gguf" or "safetensors"
    pub target_format: String,

    /// Quantization settings (optional)
    pub quantization: Option<QuantizationConfig>,

    /// Tensor export mappings
    pub tensor_map: Vec<TensorExportMapping>,

    /// Shard size in bytes (optional). If set, output will be split into multiple files.
    /// Common values: 5GB = 5_000_000_000, 8GB = 8_000_000_000
    #[serde(default)]
    pub shard_size: Option<u64>,

    /// Special token type to GGUF key mapping overrides.
    /// Maps special_type (e.g., "bos") to GGUF metadata key (e.g., "tokenizer.ggml.bos_token_id").
    /// Merges with defaults; set value to empty string to disable a mapping.
    #[serde(default)]
    pub special_token_keys: HashMap<String, String>,
}

/// Quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Default quantization type: "q4_0", "q4_1", "q8_0", etc.
    pub default_type: String,

    /// Scale optimization: "standard" or "trellis"
    #[serde(default = "default_scale_optimization")]
    pub scale_optimization: String,

    /// Trellis lambda (smoothness penalty), used when scale_optimization = "trellis"
    #[serde(default = "default_trellis_lambda")]
    pub trellis_lambda: f32,

    /// Per-tensor quantization overrides (UUID -> quant type)
    #[serde(default)]
    pub per_tensor: HashMap<String, String>,
}

fn default_scale_optimization() -> String {
    "trellis".to_string()
}

fn default_trellis_lambda() -> f32 {
    0.3
}

/// Mapping for tensor export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorExportMapping {
    /// GMAT tensor UUID string
    pub source: String,

    /// Target tensor name in export format (e.g., GGUF tensor name)
    pub target: String,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            target_format: "gguf".to_string(),
            quantization: None,
            tensor_map: Vec::new(),
            shard_size: None,
            special_token_keys: HashMap::new(),
        }
    }
}

/// Returns the default special token type to GGUF key mappings.
pub fn default_special_token_keys() -> HashMap<String, String> {
    let mut map = HashMap::new();
    map.insert("bos".to_string(), "tokenizer.ggml.bos_token_id".to_string());
    map.insert("eos".to_string(), "tokenizer.ggml.eos_token_id".to_string());
    map.insert("unk".to_string(), "tokenizer.ggml.unknown_token_id".to_string());
    map.insert("pad".to_string(), "tokenizer.ggml.padding_token_id".to_string());
    map.insert("sep".to_string(), "tokenizer.ggml.seperator_token_id".to_string()); // typo matches llama.cpp
    map.insert("cls".to_string(), "tokenizer.ggml.cls_token_id".to_string());
    map.insert("mask".to_string(), "tokenizer.ggml.mask_token_id".to_string());
    map.insert("eot".to_string(), "tokenizer.ggml.eot_token_id".to_string());
    map
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            default_type: "q8_0".to_string(),
            scale_optimization: default_scale_optimization(),
            trellis_lambda: default_trellis_lambda(),
            per_tensor: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== ExportConfig tests ====================

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.target_format, "gguf");
        assert!(config.quantization.is_none());
        assert!(config.tensor_map.is_empty());
        assert!(config.shard_size.is_none());
        assert!(config.special_token_keys.is_empty());
    }

    #[test]
    fn test_export_config_serialization() {
        let config = ExportConfig {
            target_format: "gguf".to_string(),
            quantization: Some(QuantizationConfig::default()),
            tensor_map: vec![TensorExportMapping {
                source: "uuid-1234".to_string(),
                target: "blk.0.attn_q.weight".to_string(),
            }],
            shard_size: Some(5_000_000_000),
            special_token_keys: HashMap::new(),
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: ExportConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.target_format, "gguf");
        assert!(parsed.quantization.is_some());
        assert_eq!(parsed.tensor_map.len(), 1);
        assert_eq!(parsed.shard_size, Some(5_000_000_000));
    }

    #[test]
    fn test_export_config_deserialization() {
        let json = r#"{
            "target_format": "safetensors",
            "quantization": {
                "default_type": "q4_k_m",
                "scale_optimization": "trellis",
                "trellis_lambda": 0.5,
                "per_tensor": {"uuid1": "q8_0", "uuid2": "q6_k"}
            },
            "tensor_map": [
                {"source": "uuid1", "target": "tensor1"},
                {"source": "uuid2", "target": "tensor2"}
            ],
            "shard_size": 8000000000
        }"#;

        let config: ExportConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.target_format, "safetensors");
        assert_eq!(config.tensor_map.len(), 2);
        assert_eq!(config.shard_size, Some(8_000_000_000));

        let quant = config.quantization.unwrap();
        assert_eq!(quant.default_type, "q4_k_m");
        assert_eq!(quant.scale_optimization, "trellis");
        assert!((quant.trellis_lambda - 0.5).abs() < 0.001);
        assert_eq!(quant.per_tensor.get("uuid1"), Some(&"q8_0".to_string()));
        assert_eq!(quant.per_tensor.get("uuid2"), Some(&"q6_k".to_string()));
    }

    #[test]
    fn test_export_config_without_shard_size() {
        let json = r#"{
            "target_format": "gguf",
            "tensor_map": []
        }"#;

        let config: ExportConfig = serde_json::from_str(json).unwrap();
        assert!(config.shard_size.is_none());
    }

    // ==================== QuantizationConfig tests ====================

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.default_type, "q8_0");
        assert_eq!(config.scale_optimization, "trellis");
        assert!((config.trellis_lambda - 0.3).abs() < 0.001);
        assert!(config.per_tensor.is_empty());
    }

    #[test]
    fn test_quantization_config_with_per_tensor() {
        let mut per_tensor = HashMap::new();
        per_tensor.insert("tensor1".to_string(), "q8_0".to_string());
        per_tensor.insert("tensor2".to_string(), "q4_k_m".to_string());

        let config = QuantizationConfig {
            default_type: "q4_0".to_string(),
            scale_optimization: "standard".to_string(),
            trellis_lambda: 0.0,
            per_tensor,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: QuantizationConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.per_tensor.len(), 2);
        assert_eq!(parsed.per_tensor.get("tensor1"), Some(&"q8_0".to_string()));
    }

    #[test]
    fn test_quantization_config_defaults_on_deserialize() {
        // Minimal JSON should use defaults
        let json = r#"{"default_type": "q4_0"}"#;
        let config: QuantizationConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.default_type, "q4_0");
        assert_eq!(config.scale_optimization, "trellis"); // default
        assert!((config.trellis_lambda - 0.3).abs() < 0.001); // default
        assert!(config.per_tensor.is_empty()); // default
    }

    // ==================== TensorExportMapping tests ====================

    #[test]
    fn test_tensor_export_mapping_serialization() {
        let mapping = TensorExportMapping {
            source: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            target: "blk.0.attn_q.weight".to_string(),
        };

        let json = serde_json::to_string(&mapping).unwrap();
        assert!(json.contains("550e8400"));
        assert!(json.contains("blk.0.attn_q.weight"));

        let parsed: TensorExportMapping = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source, mapping.source);
        assert_eq!(parsed.target, mapping.target);
    }

    // ==================== Default function tests ====================

    #[test]
    fn test_default_scale_optimization() {
        assert_eq!(default_scale_optimization(), "trellis");
    }

    #[test]
    fn test_default_trellis_lambda() {
        assert!((default_trellis_lambda() - 0.3).abs() < 0.001);
    }
}
