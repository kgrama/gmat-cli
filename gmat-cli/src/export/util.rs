//! Export utility functions - quantization recommendations, type conversions.
//!
//! Note: Tensor name mapping has been moved to tensor_overrides.rs
//! which loads patterns from JSON files in export-overrides/tensor-names/

use anyhow::Result;
use transform_storage::conversions::gguf_quant::GgufQuantType;

use super::tensor_overrides::TensorNameOverrides;

/// Default importance thresholds for quantization decisions.
/// Importance is measured as octave-shift ratio (0-1 range).
/// Typical distribution: min ~0.005, avg ~0.05, max ~1.0
pub const DEFAULT_IMPORTANCE_HIGH: f32 = 0.2; // High importance: upgrade to higher quality
pub const DEFAULT_IMPORTANCE_MEDIUM: f32 = 0.1; // Medium importance: moderate quality boost

/// Importance thresholds for quantization decisions.
#[derive(Clone, Copy, Debug)]
pub struct ImportanceThresholds {
    pub high: f32,
    pub medium: f32,
}

impl Default for ImportanceThresholds {
    fn default() -> Self {
        Self {
            high: DEFAULT_IMPORTANCE_HIGH,
            medium: DEFAULT_IMPORTANCE_MEDIUM,
        }
    }
}

impl ImportanceThresholds {
    pub fn new(high: f32, medium: f32) -> Self {
        Self { high, medium }
    }
}

/// Recommend quantization type based on tensor name, importance, shape, and size.
///
/// Uses heuristics:
/// - Unaligned cols (not divisible by 32) → F16 (fallback)
/// - Small tensors (<1M params) → Q8_0
/// - Pattern-based recommendations from JSON override files
///
/// Importance thresholds (octave-shift ratio):
/// - HIGH (default 0.2): ~4× average, significant dynamic range
/// - MEDIUM (default 0.1): ~2× average, above-normal dynamic range
pub fn recommend_quant_type(
    tensor_name: &str,
    importance: f32,
    rows: usize,
    cols: usize,
    thresholds: &ImportanceThresholds,
    overrides: &TensorNameOverrides,
) -> String {
    // Unaligned columns: must use F16 (1D biases, vision tensors, etc.)
    if !cols.is_multiple_of(32) {
        return "f16".to_string();
    }

    let size = rows * cols;

    // Small tensors: use Q8_0 for minimal impact
    if size < 1_000_000 {
        return "q8_0".to_string();
    }

    // Use JSON-driven pattern matching for recommendations
    let is_high_importance = importance > thresholds.high;
    overrides.recommend_quant(tensor_name, is_high_importance).to_string()
}

/// Parse quantization type string to GgufQuantType enum.
pub fn parse_quant_type(s: &str) -> Result<GgufQuantType> {
    match s.to_lowercase().as_str() {
        "f32" => Ok(GgufQuantType::F32),
        "f16" => Ok(GgufQuantType::F16),
        "q8_0" => Ok(GgufQuantType::Q8_0),
        "q4_0" => Ok(GgufQuantType::Q4_0),
        "q4_1" => Ok(GgufQuantType::Q4_1),
        "q5_0" => Ok(GgufQuantType::Q5_0),
        "q5_1" => Ok(GgufQuantType::Q5_1),
        "q2_k" => Ok(GgufQuantType::Q2_K),
        "q3_k_s" => Ok(GgufQuantType::Q3_K_S),
        "q3_k_m" | "q3_k" => Ok(GgufQuantType::Q3_K_M),
        "q3_k_l" => Ok(GgufQuantType::Q3_K_L),
        "q4_k_s" => Ok(GgufQuantType::Q4_K_S),
        "q4_k_m" | "q4_k" => Ok(GgufQuantType::Q4_K_M),
        "q5_k_s" => Ok(GgufQuantType::Q5_K_S),
        "q5_k_m" | "q5_k" => Ok(GgufQuantType::Q5_K_M),
        "q6_k" => Ok(GgufQuantType::Q6_K),
        "iq4_xs" => Ok(GgufQuantType::IQ4_XS),
        "iq4_nl" => Ok(GgufQuantType::IQ4_NL),
        _ => Err(anyhow::anyhow!("Unknown quantization type: {}", s)),
    }
}

/// Convert GgufQuantType to gguf-rs-lib TensorType.
pub fn quant_type_to_gguf_tensor_type(
    quant_type: &GgufQuantType,
) -> gguf_rs_lib::format::GGUFTensorType {
    use gguf_rs_lib::format::GGUFTensorType;

    match quant_type {
        GgufQuantType::F32 => GGUFTensorType::F32,
        GgufQuantType::F16 => GGUFTensorType::F16,
        GgufQuantType::Q4_0 => GGUFTensorType::Q4_0,
        GgufQuantType::Q4_1 => GGUFTensorType::Q4_1,
        GgufQuantType::Q5_0 => GGUFTensorType::Q5_0,
        GgufQuantType::Q5_1 => GGUFTensorType::Q5_1,
        GgufQuantType::Q8_0 => GGUFTensorType::Q8_0,
        GgufQuantType::Q2_K => GGUFTensorType::Q2_K,
        GgufQuantType::Q3_K_S => GGUFTensorType::Q3_K,
        GgufQuantType::Q3_K_M => GGUFTensorType::Q3_K,
        GgufQuantType::Q3_K_L => GGUFTensorType::Q3_K,
        GgufQuantType::Q4_K_S => GGUFTensorType::Q4_K,
        GgufQuantType::Q4_K_M => GGUFTensorType::Q4_K,
        GgufQuantType::Q5_K_S => GGUFTensorType::Q5_K,
        GgufQuantType::Q5_K_M => GGUFTensorType::Q5_K,
        GgufQuantType::Q6_K => GGUFTensorType::Q6_K,
        GgufQuantType::IQ2_XXS => GGUFTensorType::IQ2_XXS,
        GgufQuantType::IQ2_XS => GGUFTensorType::IQ2_XS,
        GgufQuantType::IQ3_XXS => GGUFTensorType::IQ3_XXS,
        GgufQuantType::IQ1_S => GGUFTensorType::IQ1_S,
        GgufQuantType::IQ4_NL => GGUFTensorType::IQ4_NL,
        GgufQuantType::IQ3_S => GGUFTensorType::IQ3_S,
        GgufQuantType::IQ2_S => GGUFTensorType::IQ2_S,
        GgufQuantType::IQ4_XS => GGUFTensorType::IQ4_XS,
        GgufQuantType::IQ1_M => GGUFTensorType::IQ1_M,
    }
}

pub use crate::common::runtime::num_cpus;

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // ==================== recommend_quant_type tests ====================
    // Note: safetensor_to_gguf_name tests are now in tensor_overrides.rs
    // Note: Pattern-based quant recommendation tests are in tensor_overrides.rs

    fn get_overrides() -> TensorNameOverrides {
        TensorNameOverrides::from_file(Path::new("../export-overrides/models/llama.json")).unwrap()
    }

    #[test]
    fn test_unaligned_cols_gets_f16() {
        let thresholds = ImportanceThresholds::default();
        let overrides = get_overrides();
        // 1D bias tensor: cols=1 (not 32-aligned) -> f16
        assert_eq!(
            recommend_quant_type("router.bias", 0.5, 64, 1, &thresholds, &overrides),
            "f16"
        );
        // Vision tensor: cols=17 (not 32-aligned) -> f16
        assert_eq!(
            recommend_quant_type("vision.proj", 0.5, 100, 17, &thresholds, &overrides),
            "f16"
        );
    }

    #[test]
    fn test_small_tensor_gets_q8_0() {
        let thresholds = ImportanceThresholds::default();
        let overrides = get_overrides();
        // Under 1M params should get Q8_0 (500 rows * 1024 cols = 512K)
        assert_eq!(
            recommend_quant_type("any.tensor", 0.5, 500, 1024, &thresholds, &overrides),
            "q8_0"
        );
    }

    #[test]
    fn test_embed_tokens_high_importance() {
        let thresholds = ImportanceThresholds::default();
        let overrides = get_overrides();
        // Above high threshold (0.2) -> q8_0 (1000 rows * 2048 cols = ~2M)
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.25, 1000, 2048, &thresholds, &overrides),
            "q8_0"
        );
    }

    #[test]
    fn test_embed_tokens_lower_importance() {
        let thresholds = ImportanceThresholds::default();
        let overrides = get_overrides();
        // Below high threshold (0.2) -> q6_k
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.15, 1000, 2048, &thresholds, &overrides),
            "q6_k"
        );
    }

    #[test]
    fn test_lm_head_high_importance() {
        let thresholds = ImportanceThresholds::default();
        let overrides = get_overrides();
        // Above high threshold (0.2) -> q8_0
        assert_eq!(
            recommend_quant_type("lm_head.weight", 0.25, 1000, 2048, &thresholds, &overrides),
            "q8_0"
        );
    }

    #[test]
    fn test_large_tensor_uses_json_patterns() {
        let thresholds = ImportanceThresholds::default();
        let overrides = get_overrides();
        // Large tensors use JSON-driven pattern matching
        // Gate proj -> q4_k_m (from JSON)
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.mlp.gate_proj.weight",
                0.5,
                1000,
                2048,
                &thresholds,
                &overrides
            ),
            "q4_k_m"
        );
    }

    #[test]
    fn test_default_tensor_gets_q4_k_m() {
        let thresholds = ImportanceThresholds::default();
        let overrides = get_overrides();
        assert_eq!(
            recommend_quant_type("some.random.tensor", 0.5, 1000, 2048, &thresholds, &overrides),
            "q4_k_m"
        );
    }

    #[test]
    fn test_custom_thresholds() {
        // Custom thresholds: high=0.5, medium=0.3
        let thresholds = ImportanceThresholds::new(0.5, 0.3);
        let overrides = get_overrides();

        // 0.4 is below custom high (0.5), so gets q6_k not q8_0
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.4, 1000, 2048, &thresholds, &overrides),
            "q6_k"
        );

        // 0.6 is above custom high (0.5), so gets q8_0
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.6, 1000, 2048, &thresholds, &overrides),
            "q8_0"
        );
    }

    // ==================== parse_quant_type tests ====================

    #[test]
    fn test_parse_unquantized() {
        assert_eq!(parse_quant_type("f32").unwrap(), GgufQuantType::F32);
        assert_eq!(parse_quant_type("f16").unwrap(), GgufQuantType::F16);
        assert_eq!(parse_quant_type("F32").unwrap(), GgufQuantType::F32);
        assert_eq!(parse_quant_type("F16").unwrap(), GgufQuantType::F16);
    }

    #[test]
    fn test_parse_legacy_quants() {
        assert_eq!(parse_quant_type("q8_0").unwrap(), GgufQuantType::Q8_0);
        assert_eq!(parse_quant_type("q4_0").unwrap(), GgufQuantType::Q4_0);
        assert_eq!(parse_quant_type("q4_1").unwrap(), GgufQuantType::Q4_1);
        assert_eq!(parse_quant_type("q5_0").unwrap(), GgufQuantType::Q5_0);
        assert_eq!(parse_quant_type("q5_1").unwrap(), GgufQuantType::Q5_1);
    }

    #[test]
    fn test_parse_k_quants() {
        assert_eq!(parse_quant_type("q2_k").unwrap(), GgufQuantType::Q2_K);
        assert_eq!(parse_quant_type("q3_k_s").unwrap(), GgufQuantType::Q3_K_S);
        assert_eq!(parse_quant_type("q3_k_m").unwrap(), GgufQuantType::Q3_K_M);
        assert_eq!(parse_quant_type("q3_k").unwrap(), GgufQuantType::Q3_K_M); // alias
        assert_eq!(parse_quant_type("q3_k_l").unwrap(), GgufQuantType::Q3_K_L);
        assert_eq!(parse_quant_type("q4_k_s").unwrap(), GgufQuantType::Q4_K_S);
        assert_eq!(parse_quant_type("q4_k_m").unwrap(), GgufQuantType::Q4_K_M);
        assert_eq!(parse_quant_type("q4_k").unwrap(), GgufQuantType::Q4_K_M); // alias
        assert_eq!(parse_quant_type("q5_k_s").unwrap(), GgufQuantType::Q5_K_S);
        assert_eq!(parse_quant_type("q5_k_m").unwrap(), GgufQuantType::Q5_K_M);
        assert_eq!(parse_quant_type("q5_k").unwrap(), GgufQuantType::Q5_K_M); // alias
        assert_eq!(parse_quant_type("q6_k").unwrap(), GgufQuantType::Q6_K);
    }

    #[test]
    fn test_parse_i_quants() {
        assert_eq!(parse_quant_type("iq4_xs").unwrap(), GgufQuantType::IQ4_XS);
        assert_eq!(parse_quant_type("iq4_nl").unwrap(), GgufQuantType::IQ4_NL);
    }

    #[test]
    fn test_parse_case_insensitive() {
        assert_eq!(parse_quant_type("Q8_0").unwrap(), GgufQuantType::Q8_0);
        assert_eq!(parse_quant_type("Q4_K_M").unwrap(), GgufQuantType::Q4_K_M);
        assert_eq!(parse_quant_type("IQ4_XS").unwrap(), GgufQuantType::IQ4_XS);
    }

    #[test]
    fn test_parse_unknown_quant_type() {
        let result = parse_quant_type("q99_ultra");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unknown quantization type")
        );
    }

    // ==================== quant_type_to_gguf_tensor_type tests ====================

    #[test]
    fn test_quant_type_conversion_legacy() {
        use gguf_rs_lib::format::GGUFTensorType;

        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::Q4_0),
            GGUFTensorType::Q4_0
        ));
        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::Q8_0),
            GGUFTensorType::Q8_0
        ));
    }

    #[test]
    fn test_quant_type_conversion_k_quants() {
        use gguf_rs_lib::format::GGUFTensorType;

        // Q3_K variants all map to Q3_K
        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::Q3_K_S),
            GGUFTensorType::Q3_K
        ));
        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::Q3_K_M),
            GGUFTensorType::Q3_K
        ));
        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::Q3_K_L),
            GGUFTensorType::Q3_K
        ));

        // Q4_K variants
        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::Q4_K_S),
            GGUFTensorType::Q4_K
        ));
        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::Q4_K_M),
            GGUFTensorType::Q4_K
        ));
    }

    #[test]
    fn test_quant_type_conversion_i_quants() {
        use gguf_rs_lib::format::GGUFTensorType;

        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::IQ4_XS),
            GGUFTensorType::IQ4_XS
        ));
        assert!(matches!(
            quant_type_to_gguf_tensor_type(&GgufQuantType::IQ4_NL),
            GGUFTensorType::IQ4_NL
        ));
    }

    // ==================== num_cpus tests ====================

    #[test]
    fn test_num_cpus_returns_positive() {
        let cpus = num_cpus();
        assert!(cpus > 0);
    }
}
