//! Tensor name override loading and pattern matching.
//! Used only during generate-config, NOT during export.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Default quantization type when no pattern matches.
pub const DEFAULT_QUANT_TYPE: &str = "q4_k_m";

/// Quantization recommendation for a tensor pattern.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantRecommendation {
    /// Quantization type when importance is high
    pub high: Option<String>,
    /// Default quantization type
    pub default: String,
}

/// JSON structure for tensor name override files.
#[derive(Debug, Deserialize)]
struct TensorOverrideFile {
    /// GGUF architecture name for `general.architecture` metadata
    #[serde(default)]
    gguf_architecture: Option<String>,
    /// Metadata key aliases for HF config parsing (canonical -> [alternatives])
    #[serde(default)]
    metadata_key_aliases: HashMap<String, Vec<String>>,
    /// Quantization recommendations by tensor name pattern (substring match)
    #[serde(default)]
    quant_recommendations: HashMap<String, QuantRecommendation>,
    /// Tensor name mapping patterns
    #[serde(default)]
    tensor_map_patterns: HashMap<String, String>,
    /// Special token type to GGUF key mappings
    #[serde(default)]
    special_token_keys: HashMap<String, String>,
}

/// Compiled pattern for efficient string-based matching.
/// Uses simple string splitting with named placeholders like {N} and {E}.
#[derive(Debug)]
struct CompiledPattern {
    /// Literal segments between placeholders: ["model.layers.", ".mlp.experts.", ".weight"]
    segments: Vec<String>,
    /// Placeholder names in order of appearance: ["N", "E"]
    placeholders: Vec<String>,
    /// Output template with named placeholders: "blk.{N}.ffn_gate.{E}.weight"
    output: String,
}

impl CompiledPattern {
    /// Compile a pattern string and output template into a CompiledPattern.
    fn compile(pattern: &str, output: &str) -> Option<Self> {
        let mut segments = Vec::new();
        let mut placeholders = Vec::new();
        let mut last_end = 0;
        let mut i = 0;

        let bytes = pattern.as_bytes();
        while i < bytes.len() {
            if bytes[i] == b'{' {
                // Find the closing brace
                if let Some(close_offset) = pattern[i..].find('}') {
                    let close = i + close_offset;
                    // Add literal segment before placeholder
                    segments.push(pattern[last_end..i].to_string());
                    // Add placeholder name (text between braces)
                    placeholders.push(pattern[i + 1..close].to_string());
                    last_end = close + 1;
                    i = last_end;
                    continue;
                }
            }
            i += 1;
        }
        // Add final literal segment
        segments.push(pattern[last_end..].to_string());

        Some(Self {
            segments,
            placeholders,
            output: output.to_string(),
        })
    }

    /// Try to match an input string against this pattern.
    /// Returns the output string with placeholders replaced, or None if no match.
    fn try_match(&self, input: &str) -> Option<String> {
        let mut remaining = input;
        let mut captures: HashMap<&str, String> = HashMap::new();

        for (i, segment) in self.segments.iter().enumerate() {
            if i > 0 {
                // Extract number before this segment
                let num_end = remaining.find(segment.as_str())?;
                let num = &remaining[..num_end];
                // Validate that it's all digits
                if !num.chars().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                captures.insert(&self.placeholders[i - 1], num.to_string());
                remaining = &remaining[num_end..];
            }
            remaining = remaining.strip_prefix(segment.as_str())?;
        }

        // Must have consumed entire input
        if !remaining.is_empty() {
            return None;
        }

        // Named replacement - order doesn't matter
        let mut result = self.output.clone();
        for (name, value) in captures {
            result = result.replace(&format!("{{{}}}", name), &value);
        }
        Some(result)
    }
}

/// Tensor name overrides with compiled patterns for efficient matching.
#[derive(Debug)]
pub struct TensorNameOverrides {
    patterns: Vec<CompiledPattern>,
    /// GGUF architecture name for `general.architecture` metadata
    gguf_architecture: String,
    /// Metadata key aliases for HF config parsing (canonical -> [alternatives])
    metadata_key_aliases: HashMap<String, Vec<String>>,
    /// Quantization recommendations by tensor name pattern (substring match, case-insensitive)
    quant_recommendations: Vec<(String, QuantRecommendation)>,
    /// Special token type to GGUF key mappings
    special_token_keys: HashMap<String, String>,
}

impl TensorNameOverrides {
    /// Load from file path.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read: {}", path.display()))?;
        Self::from_json(&content)
    }

    /// Load from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        let file: TensorOverrideFile =
            serde_json::from_str(json).context("Failed to parse model config JSON")?;

        let patterns: Vec<_> = file
            .tensor_map_patterns
            .iter()
            .filter_map(|(p, o)| CompiledPattern::compile(p, o))
            .collect();

        let quant_recommendations = file
            .quant_recommendations
            .into_iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .collect();

        Ok(Self {
            patterns,
            gguf_architecture: file.gguf_architecture.unwrap_or_else(|| "llama".into()),
            metadata_key_aliases: file.metadata_key_aliases,
            quant_recommendations,
            special_token_keys: file.special_token_keys,
        })
    }

    /// Get the GGUF architecture name.
    pub fn gguf_architecture(&self) -> &str {
        &self.gguf_architecture
    }

    /// Get metadata key aliases for a canonical key.
    /// Returns the list of alternative HF config keys to try.
    pub fn metadata_key_aliases(&self, canonical: &str) -> Option<&[String]> {
        self.metadata_key_aliases.get(canonical).map(|v| v.as_slice())
    }

    /// Get special token type to GGUF key mappings.
    pub fn special_token_keys(&self) -> &HashMap<String, String> {
        &self.special_token_keys
    }

    /// Map a tensor name using patterns. Returns None if no pattern matches.
    pub fn map_name(&self, name: &str) -> Option<String> {
        self.patterns.iter().find_map(|p| p.try_match(name))
    }

    /// Recommend a quantization type for a tensor based on patterns.
    /// Returns the recommended quant type string, or DEFAULT_QUANT_TYPE if no pattern matches.
    pub fn recommend_quant(&self, tensor_name: &str, is_high_importance: bool) -> &str {
        let name_lower = tensor_name.to_lowercase();
        for (pattern, rec) in &self.quant_recommendations {
            if name_lower.contains(pattern) {
                return if is_high_importance {
                    rec.high.as_deref().unwrap_or(&rec.default)
                } else {
                    &rec.default
                };
            }
        }
        DEFAULT_QUANT_TYPE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_matching() {
        let json = r#"{
            "tensor_map_patterns": {
                "model.layers.{N}.self_attn.q_proj.weight": "blk.{N}.attn_q.weight"
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();
        assert_eq!(
            o.map_name("model.layers.5.self_attn.q_proj.weight"),
            Some("blk.5.attn_q.weight".into())
        );
    }

    #[test]
    fn test_expert_pattern() {
        let json = r#"{
            "tensor_map_patterns": {
                "model.layers.{N}.mlp.experts.{E}.gate_proj.weight": "blk.{N}.ffn_gate.{E}.weight"
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();
        assert_eq!(
            o.map_name("model.layers.3.mlp.experts.7.gate_proj.weight"),
            Some("blk.3.ffn_gate.7.weight".into())
        );
    }

    #[test]
    fn test_reversed_placeholder_order() {
        // Test that {E} before {N} in pattern works correctly
        let json = r#"{
            "tensor_map_patterns": {
                "experts.{E}.layers.{N}.weight": "blk.{N}.expert.{E}.weight"
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();
        assert_eq!(
            o.map_name("experts.7.layers.3.weight"),
            Some("blk.3.expert.7.weight".into())
        );
    }

    #[test]
    fn test_no_match_returns_none() {
        let json = r#"{
            "tensor_map_patterns": {
                "model.layers.{N}.weight": "blk.{N}.weight"
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();
        assert_eq!(o.map_name("some.unknown.tensor"), None);
    }

    #[test]
    fn test_static_pattern() {
        let json = r#"{
            "tensor_map_patterns": {
                "model.embed_tokens.weight": "token_embd.weight",
                "lm_head.weight": "output.weight"
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();
        assert_eq!(
            o.map_name("model.embed_tokens.weight"),
            Some("token_embd.weight".into())
        );
        assert_eq!(o.map_name("lm_head.weight"), Some("output.weight".into()));
    }

    #[test]
    fn test_quant_recommendations() {
        let json = r#"{
            "quant_recommendations": {
                "embed_tokens": { "high": "q8_0", "default": "q6_k" },
                "gate_proj": { "default": "q4_k_m" }
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();

        // High importance -> q8_0
        assert_eq!(o.recommend_quant("model.embed_tokens.weight", true), "q8_0");
        // Default importance -> q6_k
        assert_eq!(
            o.recommend_quant("model.embed_tokens.weight", false),
            "q6_k"
        );

        // Gate proj (no high override) -> q4_k_m
        assert_eq!(o.recommend_quant("gate_proj.weight", false), "q4_k_m");
        assert_eq!(o.recommend_quant("gate_proj.weight", true), "q4_k_m");

        // Unknown -> DEFAULT_QUANT_TYPE
        assert_eq!(o.recommend_quant("unknown.tensor", false), DEFAULT_QUANT_TYPE);
    }

    #[test]
    fn test_gguf_architecture() {
        let json = r#"{"gguf_architecture": "phi3"}"#;
        let o = TensorNameOverrides::from_json(json).unwrap();
        assert_eq!(o.gguf_architecture(), "phi3");
    }

    #[test]
    fn test_default_architecture() {
        let json = r#"{}"#;
        let o = TensorNameOverrides::from_json(json).unwrap();
        assert_eq!(o.gguf_architecture(), "llama");
    }

    #[test]
    fn test_metadata_key_aliases() {
        let json = r#"{
            "metadata_key_aliases": {
                "hidden_size": ["hidden_size", "d_model"],
                "num_layers": ["num_hidden_layers", "n_layer"]
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();

        let aliases = o.metadata_key_aliases("hidden_size").unwrap();
        assert!(aliases.contains(&"hidden_size".to_string()));
        assert!(aliases.contains(&"d_model".to_string()));

        assert!(o.metadata_key_aliases("unknown").is_none());
    }

    #[test]
    fn test_special_token_keys() {
        let json = r#"{
            "special_token_keys": {
                "bos": "tokenizer.ggml.bos_token_id",
                "eos": "tokenizer.ggml.eos_token_id"
            }
        }"#;
        let o = TensorNameOverrides::from_json(json).unwrap();

        let keys = o.special_token_keys();
        assert_eq!(
            keys.get("bos"),
            Some(&"tokenizer.ggml.bos_token_id".to_string())
        );
        assert_eq!(
            keys.get("eos"),
            Some(&"tokenizer.ggml.eos_token_id".to_string())
        );
    }

    #[test]
    fn test_from_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");

        let json = r#"{
            "gguf_architecture": "test",
            "tensor_map_patterns": {
                "input.weight": "output.weight"
            }
        }"#;
        std::fs::File::create(&path)
            .unwrap()
            .write_all(json.as_bytes())
            .unwrap();

        let o = TensorNameOverrides::from_file(&path).unwrap();
        assert_eq!(o.gguf_architecture(), "test");
        assert_eq!(
            o.map_name("input.weight"),
            Some("output.weight".to_string())
        );
    }
}
