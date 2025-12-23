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
        }
    }
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
