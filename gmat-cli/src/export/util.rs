//! Export utility functions - name mapping, quantization recommendations, type conversions.

use anyhow::Result;
use transform_storage::conversions::gguf_quant::GgufQuantType;

/// Convert SafeTensor name to GGUF tensor name (llama.cpp convention).
pub fn safetensor_to_gguf_name(st_name: &str) -> String {
    // Handle token embeddings
    if st_name == "model.embed_tokens.weight" {
        return "token_embd.weight".to_string();
    }

    // Handle output layer and norm
    if st_name == "lm_head.weight" {
        return "output.weight".to_string();
    }
    if st_name == "model.norm.weight" {
        return "output_norm.weight".to_string();
    }

    // Handle layer-specific tensors
    if let Some(rest) = st_name.strip_prefix("model.layers.") {
        // Extract layer number
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let tensor_type = &rest[dot_pos + 1..];

            // Map tensor types
            let gguf_type = match tensor_type {
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                _ => tensor_type,
            };

            return format!("blk.{}.{}", layer_num, gguf_type);
        }
    }

    // Fallback: use original name if no mapping found
    st_name.to_string()
}

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

/// Recommend quantization type based on tensor name, importance, and size.
///
/// Uses heuristics:
/// - High-importance tensors (embeddings, attention output) → Q6_K
/// - Standard tensors → Q4_K_M
/// - Small tensors (<1M params) → Q8_0
///
/// Importance thresholds (octave-shift ratio):
/// - HIGH (default 0.2): ~4× average, significant dynamic range
/// - MEDIUM (default 0.1): ~2× average, above-normal dynamic range
pub fn recommend_quant_type(
    tensor_name: &str,
    importance: f32,
    size: usize,
    thresholds: &ImportanceThresholds,
) -> String {
    // Small tensors: use Q8_0 for minimal impact
    if size < 1_000_000 {
        return "q8_0".to_string();
    }

    // Check tensor type patterns
    let name_lower = tensor_name.to_lowercase();

    // Token embeddings and output layers: highest quality
    if name_lower.contains("embed_tokens")
        || name_lower.contains("lm_head")
        || name_lower.contains("model.norm")
    {
        return if importance > thresholds.high {
            "q8_0"
        } else {
            "q6_k"
        }
        .to_string();
    }

    // Attention output and value: sensitive to quantization
    if name_lower.contains("attn.o_proj") || name_lower.contains("attn.v_proj") {
        return if importance > thresholds.medium {
            "q6_k"
        } else {
            "q5_k_m"
        }
        .to_string();
    }

    // FFN down projection: output path, more sensitive
    if name_lower.contains("mlp.down_proj") {
        return if importance > thresholds.medium {
            "q6_k"
        } else {
            "q5_k_m"
        }
        .to_string();
    }

    // Attention Q/K: can tolerate more compression
    if name_lower.contains("attn.q_proj") || name_lower.contains("attn.k_proj") {
        return "q4_k_m".to_string();
    }

    // FFN gate and up: large, compressible
    if name_lower.contains("mlp.gate_proj") || name_lower.contains("mlp.up_proj") {
        return "q4_k_m".to_string();
    }

    // Default: Q4_K_M for standard tensors
    "q4_k_m".to_string()
}

/// Parse quantization type string to GgufQuantType enum.
pub fn parse_quant_type(s: &str) -> Result<GgufQuantType> {
    match s.to_lowercase().as_str() {
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

/// Get number of available CPU cores.
pub fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== safetensor_to_gguf_name tests ====================

    #[test]
    fn test_embed_tokens_mapping() {
        assert_eq!(
            safetensor_to_gguf_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
    }

    #[test]
    fn test_lm_head_mapping() {
        assert_eq!(safetensor_to_gguf_name("lm_head.weight"), "output.weight");
    }

    #[test]
    fn test_model_norm_mapping() {
        assert_eq!(
            safetensor_to_gguf_name("model.norm.weight"),
            "output_norm.weight"
        );
    }

    #[test]
    fn test_attention_projections() {
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.self_attn.q_proj.weight"),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            safetensor_to_gguf_name("model.layers.5.self_attn.k_proj.weight"),
            "blk.5.attn_k.weight"
        );
        assert_eq!(
            safetensor_to_gguf_name("model.layers.31.self_attn.v_proj.weight"),
            "blk.31.attn_v.weight"
        );
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.self_attn.o_proj.weight"),
            "blk.0.attn_output.weight"
        );
    }

    #[test]
    fn test_mlp_projections() {
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.mlp.gate_proj.weight"),
            "blk.0.ffn_gate.weight"
        );
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.mlp.up_proj.weight"),
            "blk.0.ffn_up.weight"
        );
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.mlp.down_proj.weight"),
            "blk.0.ffn_down.weight"
        );
    }

    #[test]
    fn test_layer_norms() {
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.input_layernorm.weight"),
            "blk.0.attn_norm.weight"
        );
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.post_attention_layernorm.weight"),
            "blk.0.ffn_norm.weight"
        );
    }

    #[test]
    fn test_unknown_tensor_passthrough() {
        assert_eq!(
            safetensor_to_gguf_name("some.unknown.tensor"),
            "some.unknown.tensor"
        );
    }

    #[test]
    fn test_unknown_layer_tensor_passthrough() {
        // Unknown tensor type within a layer should pass through
        assert_eq!(
            safetensor_to_gguf_name("model.layers.0.some_new_thing.weight"),
            "blk.0.some_new_thing.weight"
        );
    }

    // ==================== recommend_quant_type tests ====================

    #[test]
    fn test_small_tensor_gets_q8_0() {
        let thresholds = ImportanceThresholds::default();
        // Under 1M params should get Q8_0
        assert_eq!(
            recommend_quant_type("any.tensor", 0.5, 500_000, &thresholds),
            "q8_0"
        );
    }

    #[test]
    fn test_embed_tokens_high_importance() {
        let thresholds = ImportanceThresholds::default();
        // Above high threshold (0.2) -> q8_0
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.25, 2_000_000, &thresholds),
            "q8_0"
        );
    }

    #[test]
    fn test_embed_tokens_lower_importance() {
        let thresholds = ImportanceThresholds::default();
        // Below high threshold (0.2) -> q6_k
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.15, 2_000_000, &thresholds),
            "q6_k"
        );
    }

    #[test]
    fn test_lm_head_high_importance() {
        let thresholds = ImportanceThresholds::default();
        // Above high threshold (0.2) -> q8_0
        assert_eq!(
            recommend_quant_type("lm_head.weight", 0.25, 2_000_000, &thresholds),
            "q8_0"
        );
    }

    #[test]
    fn test_attention_output_high_importance() {
        let thresholds = ImportanceThresholds::default();
        // Above medium threshold (0.1) -> q6_k
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.self_attn.o_proj.weight",
                0.15,
                2_000_000,
                &thresholds
            ),
            "q6_k"
        );
    }

    #[test]
    fn test_attention_output_lower_importance() {
        let thresholds = ImportanceThresholds::default();
        // Below medium threshold (0.1) -> q5_k_m
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.self_attn.o_proj.weight",
                0.05,
                2_000_000,
                &thresholds
            ),
            "q5_k_m"
        );
    }

    #[test]
    fn test_attention_qk_gets_q4_k_m() {
        let thresholds = ImportanceThresholds::default();
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.self_attn.q_proj.weight",
                0.5,
                2_000_000,
                &thresholds
            ),
            "q4_k_m"
        );
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.self_attn.k_proj.weight",
                0.5,
                2_000_000,
                &thresholds
            ),
            "q4_k_m"
        );
    }

    #[test]
    fn test_mlp_gate_up_gets_q4_k_m() {
        let thresholds = ImportanceThresholds::default();
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.mlp.gate_proj.weight",
                0.5,
                2_000_000,
                &thresholds
            ),
            "q4_k_m"
        );
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.mlp.up_proj.weight",
                0.5,
                2_000_000,
                &thresholds
            ),
            "q4_k_m"
        );
    }

    #[test]
    fn test_mlp_down_high_importance() {
        let thresholds = ImportanceThresholds::default();
        // Above medium threshold (0.1) -> q6_k
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.mlp.down_proj.weight",
                0.15,
                2_000_000,
                &thresholds
            ),
            "q6_k"
        );
    }

    #[test]
    fn test_mlp_down_lower_importance() {
        let thresholds = ImportanceThresholds::default();
        // Below medium threshold (0.1) -> q5_k_m
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.mlp.down_proj.weight",
                0.05,
                2_000_000,
                &thresholds
            ),
            "q5_k_m"
        );
    }

    #[test]
    fn test_default_tensor_gets_q4_k_m() {
        let thresholds = ImportanceThresholds::default();
        assert_eq!(
            recommend_quant_type("some.random.tensor", 0.5, 2_000_000, &thresholds),
            "q4_k_m"
        );
    }

    #[test]
    fn test_custom_thresholds() {
        // Custom thresholds: high=0.5, medium=0.3
        let thresholds = ImportanceThresholds::new(0.5, 0.3);

        // 0.4 is below custom high (0.5), so gets q6_k not q8_0
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.4, 2_000_000, &thresholds),
            "q6_k"
        );

        // 0.6 is above custom high (0.5), so gets q8_0
        assert_eq!(
            recommend_quant_type("model.embed_tokens.weight", 0.6, 2_000_000, &thresholds),
            "q8_0"
        );

        // 0.2 is below custom medium (0.3), so gets q5_k_m
        assert_eq!(
            recommend_quant_type(
                "model.layers.0.self_attn.o_proj.weight",
                0.2,
                2_000_000,
                &thresholds
            ),
            "q5_k_m"
        );
    }

    // ==================== parse_quant_type tests ====================

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
