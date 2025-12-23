//! Metadata extraction from SafeTensors and config.json files.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::import_config::ModelMetadata;

/// Extract model metadata from safetensor files and/or config.json.
///
/// Tries multiple sources in order:
/// 1. config.json in the same directory (HuggingFace format)
/// 2. __metadata__ from safetensor header
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

    // Try to extract from safetensor __metadata__
    for file_path in safetensor_files {
        let data = fs::read(file_path)?;
        let (_, st_metadata) = SafeTensors::read_metadata(&data)
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

/// Extract metadata from a HuggingFace config.json file.
fn extract_from_config_json(config_path: &Path) -> Result<ModelMetadata> {
    let content = fs::read_to_string(config_path)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    Ok(extract_metadata_fields(
        |key| json.get(key).and_then(|v| v.as_str()).map(|s| s.to_string()),
        |key| json.get(key).and_then(|v| v.as_u64()),
        || json.get("architectures").and_then(|a| a.get(0)).and_then(|v| v.as_str()).map(|s| s.to_string()),
    ))
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
