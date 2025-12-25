//! SafeTensor import with metadata extraction.
//!
//! Extracts model configuration and tensor metadata from SafeTensor files
//! and imports them into GraphMatrix format with preserved metadata.

use crate::formats::{metadata_keys, GmatMetadata};
use safetensors::tensor::Metadata;
use safetensors::SafeTensors;

/// Detect dtype string from SafeTensor tensor dtype
fn dtype_to_string(dtype: safetensors::Dtype) -> &'static str {
    match dtype {
        safetensors::Dtype::BF16 => "BF16",
        safetensors::Dtype::F32 => "F32",
        safetensors::Dtype::F16 => "F16",
        safetensors::Dtype::F64 => "F64",
        safetensors::Dtype::I8 => "I8",
        safetensors::Dtype::I16 => "I16",
        safetensors::Dtype::I32 => "I32",
        safetensors::Dtype::I64 => "I64",
        safetensors::Dtype::U8 => "U8",
        safetensors::Dtype::U16 => "U16",
        safetensors::Dtype::U32 => "U32",
        safetensors::Dtype::U64 => "U64",
        safetensors::Dtype::BOOL => "BOOL",
        _ => "UNKNOWN",
    }
}

/// Standard config keys to extract from SafeTensor metadata
const CONFIG_KEYS: &[&str] = &[
    "architecture",
    "vocab_size",
    "hidden_size",
    "num_layers",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "max_position_embeddings",
    "model_type",
    "torch_dtype",
];

/// Extract metadata from SafeTensor Metadata struct.
fn extract_from_st_metadata(st_metadata: &Metadata, gmat_metadata: &mut GmatMetadata) {
    if let Some(ref meta_map) = *st_metadata.metadata() {
        for &key in CONFIG_KEYS {
            if let Some(value) = meta_map.get(key) {
                // Try to parse as JSON value to preserve type, otherwise store as string
                if let Ok(json_val) = serde_json::from_str::<serde_json::Value>(value) {
                    gmat_metadata.set(key, json_val);
                } else {
                    gmat_metadata.set_str(key, value.clone());
                }
            }
        }
    }
}

/// Extract metadata from SafeTensor file.
///
/// Extracts:
/// - Model configuration from __metadata__ section (architecture, vocab_size, etc.)
/// - Original dtype from tensor inspection
/// - Source format marker
///
/// # Arguments
/// * `safetensor` - Deserialized SafeTensor object
/// * `tensor_name` - Optional specific tensor to inspect for dtype (uses first if None)
///
/// # Returns
/// GmatMetadata with extracted information
pub fn extract_metadata(safetensor: &SafeTensors, tensor_name: Option<&str>) -> GmatMetadata {
    let mut metadata = GmatMetadata::new();

    // Set source format
    metadata.set_str(metadata_keys::SOURCE_FORMAT, "safetensors");

    // Detect original dtype from tensor
    let dtype_str = if let Some(name) = tensor_name {
        safetensor
            .tensor(name)
            .map(|t| dtype_to_string(t.dtype()))
            .unwrap_or("UNKNOWN")
    } else {
        // Use first tensor if no specific name provided
        safetensor
            .names()
            .first()
            .and_then(|name| safetensor.tensor(name).ok())
            .map(|t| dtype_to_string(t.dtype()))
            .unwrap_or("UNKNOWN")
    };

    metadata.set_str(metadata_keys::ORIGINAL_DTYPE, dtype_str);

    // Add creation timestamp
    let timestamp = chrono::Utc::now().to_rfc3339();
    metadata.set_str(metadata_keys::CREATED_AT, timestamp);

    metadata
}

/// Extract metadata from SafeTensor file bytes.
///
/// Convenience wrapper that deserializes the SafeTensor data and extracts metadata.
/// Also extracts model configuration from the header metadata.
///
/// # Arguments
/// * `data` - Raw SafeTensor file bytes
/// * `tensor_name` - Optional specific tensor to inspect for dtype
///
/// # Returns
/// Result containing GmatMetadata or error string
pub fn extract_metadata_from_bytes(
    data: &[u8],
    tensor_name: Option<&str>,
) -> Result<GmatMetadata, String> {
    // First read the metadata from the header
    let (_, st_metadata) = SafeTensors::read_metadata(data)
        .map_err(|e| format!("Failed to read SafeTensor metadata: {}", e))?;

    let safetensor = SafeTensors::deserialize(data)
        .map_err(|e| format!("Failed to deserialize SafeTensor: {}", e))?;

    let mut metadata = extract_metadata(&safetensor, tensor_name);

    // Extract config from SafeTensor metadata
    extract_from_st_metadata(&st_metadata, &mut metadata);

    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(dtype_to_string(safetensors::Dtype::BF16), "BF16");
        assert_eq!(dtype_to_string(safetensors::Dtype::F32), "F32");
        assert_eq!(dtype_to_string(safetensors::Dtype::F16), "F16");
    }
}
