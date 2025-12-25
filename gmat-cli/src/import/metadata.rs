//! Metadata extraction from SafeTensors and config.json files.

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use crate::config::import_config::ModelMetadata;

/// Extract model metadata from safetensor files and/or config.json.
///
/// Tries multiple sources in order:
/// 1. config.json in the same directory (HuggingFace format)
/// 2. __metadata__ from safetensor header
///
/// Uses memory-mapped I/O to avoid loading entire files for metadata extraction.
pub fn extract_model_metadata(safetensor_files: &[PathBuf]) -> Result<ModelMetadata> {
    // Try to find config.json in the parent directory of the first safetensor file
    if let Some(first_file) = safetensor_files.first() {
        if let Some(parent) = first_file.parent() {
            let config_path = parent.join("config.json");
            if config_path.exists() {
                if let Ok(metadata) = extract_from_config_json(&config_path) {
                    println!("Loaded metadata from config.json");
                    return Ok(metadata);
                }
            }
        }
    }

    // Try to extract from safetensor __metadata__ using mmap
    for file_path in safetensor_files {
        let file = File::open(file_path)?;
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("Failed to mmap: {}", file_path.display()))?;

        let (_, st_metadata) = SafeTensors::read_metadata(&mmap)
            .with_context(|| format!("Failed to read metadata from: {}", file_path.display()))?;

        if let Some(meta_map) = st_metadata.metadata() {
            let metadata = extract_from_safetensor_metadata(meta_map);
            if metadata.architecture.is_some() || metadata.vocab_size.is_some() {
                println!("Loaded metadata from safetensor header");
                return Ok(metadata);
            }
        }
    }

    Ok(ModelMetadata::default())
}

/// Nested config keys for VLMs and composite models (checked in order).
const NESTED_TEXT_CONFIG_KEYS: &[&str] = &["text_config", "language_config", "llm_config"];

/// Extract metadata from a HuggingFace config.json file.
///
/// Handles multiple config patterns:
/// - Flat configs (standard LLMs, some TTS/ASR)
/// - Nested text configs for VLMs (text_config, language_config, llm_config)
/// - Encoder-decoder models (d_model, d_ff, decoder_layers)
fn extract_from_config_json(config_path: &Path) -> Result<ModelMetadata> {
    let content = fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    // Find nested text config if present (for VLMs)
    let text_cfg = NESTED_TEXT_CONFIG_KEYS
        .iter()
        .find_map(|key| json.get(*key))
        .filter(|v| v.is_object());

    // Helper to get string from json, checking nested config first for non-arch fields
    let get_str = |key: &str| -> Option<String> {
        // model_type should come from top-level
        if key == "model_type" {
            json.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
        } else {
            text_cfg
                .and_then(|cfg| cfg.get(key))
                .or_else(|| json.get(key))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        }
    };

    // Helper to get number, checking nested config then top-level then encoder-decoder aliases
    let get_num = |key: &str| -> Option<u64> {
        text_cfg
            .and_then(|cfg| cfg.get(key))
            .or_else(|| json.get(key))
            .and_then(|v| v.as_u64())
    };

    // Architecture from model_type or architectures array
    let architecture = get_str("model_type")
        .or_else(|| get_str("architecture"))
        .or_else(|| {
            json.get("architectures")
                .and_then(|a| a.get(0))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        });

    // hidden_size with encoder-decoder fallback (d_model)
    let hidden_size = get_num("hidden_size")
        .or_else(|| get_num("d_model"));

    // num_layers with multiple fallbacks
    let num_layers = get_num("num_hidden_layers")
        .or_else(|| get_num("num_layers"))
        .or_else(|| get_num("decoder_layers"))
        .or_else(|| get_num("n_layer"));

    // num_attention_heads with encoder-decoder fallbacks
    let num_attention_heads = get_num("num_attention_heads")
        .or_else(|| get_num("num_heads"))
        .or_else(|| get_num("decoder_attention_heads"))
        .or_else(|| get_num("n_head"));

    // intermediate_size with encoder-decoder fallbacks
    let intermediate_size = get_num("intermediate_size")
        .or_else(|| get_num("d_ff"))
        .or_else(|| get_num("decoder_ffn_dim"));

    // max_position_embeddings with fallbacks
    let max_position_embeddings = get_num("max_position_embeddings")
        .or_else(|| get_num("n_positions"))
        .or_else(|| get_num("max_target_positions"));

    Ok(ModelMetadata {
        architecture,
        vocab_size: get_num("vocab_size"),
        hidden_size,
        num_layers,
        num_attention_heads,
        intermediate_size,
        max_position_embeddings,
    })
}

/// Extract metadata from safetensor __metadata__ HashMap.
fn extract_from_safetensor_metadata(meta_map: &HashMap<String, String>) -> ModelMetadata {
    extract_metadata_fields(
        |key| meta_map.get(key).cloned(),
        |key| meta_map.get(key).and_then(|v| v.parse().ok()),
        || None,
    )
}

/// Common metadata field extraction logic.
fn extract_metadata_fields<S, N, A>(get_str: S, get_num: N, get_arch_fallback: A) -> ModelMetadata
where
    S: Fn(&str) -> Option<String>,
    N: Fn(&str) -> Option<u64>,
    A: Fn() -> Option<String>,
{
    ModelMetadata {
        architecture: get_str("model_type")
            .or_else(|| get_str("architecture"))
            .or_else(get_arch_fallback),
        vocab_size: get_num("vocab_size"),
        hidden_size: get_num("hidden_size"),
        num_layers: get_num("num_hidden_layers").or_else(|| get_num("num_layers")),
        num_attention_heads: get_num("num_attention_heads"),
        intermediate_size: get_num("intermediate_size"),
        max_position_embeddings: get_num("max_position_embeddings"),
    }
}
