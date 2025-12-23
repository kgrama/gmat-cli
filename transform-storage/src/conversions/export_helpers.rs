//! Export helpers for adapting metadata to new tensor settings.
//!
//! Provides utilities to prepare metadata for export, updating format-specific
//! fields while preserving original model configuration.

use crate::formats::{GmatMetadata, metadata_keys};
use crate::blocks::BlockFormat;

/// Convert BlockFormat enum to string representation
fn block_format_to_string(format: BlockFormat) -> &'static str {
    match format {
        BlockFormat::B8x4 => "B8x4",
        BlockFormat::B8x8 => "B8x8",
        BlockFormat::B16x4 => "B16x4",
        BlockFormat::B16x8 => "B16x8",
        BlockFormat::DualRow8x4 => "DualRow8x4",
        BlockFormat::DualRow8x8 => "DualRow8x8",
        BlockFormat::DualRow16x4 => "DualRow16x4",
        BlockFormat::DualRow16x8 => "DualRow16x8",
    }
}

/// Prepare metadata for export by adapting it to current tensor settings.
///
/// This function:
/// - Preserves all original model configuration keys (architecture, vocab_size, etc.)
/// - Updates `gmat.block_format` to reflect the current block format
/// - Adds/updates `gmat.conversion_info` to track transformations
/// - Preserves original dtype, source format, and creation timestamp if present
///
/// # Arguments
/// * `original` - Original metadata from import (if available)
/// * `current_format` - Current BlockFormat being exported
/// * `conversion_notes` - Optional notes about transformations applied (e.g., "quantized to i8", "reformat from B8x4")
///
/// # Returns
/// Adapted GmatMetadata ready for export
///
/// # Example
/// ```no_run
/// use transform_storage::conversions::prepare_export_metadata;
/// use transform_storage::blocks::BlockFormat;
/// use transform_storage::GmatMetadata;
///
/// let mut original = GmatMetadata::new();
/// original.set_str("architecture", "gpt2");
/// original.set_i64("vocab_size", 50257);
///
/// let adapted = prepare_export_metadata(
///     Some(&original),
///     BlockFormat::B16x8,
///     Some("converted to DualRow format")
/// );
/// ```
pub fn prepare_export_metadata(
    original: Option<&GmatMetadata>,
    current_format: BlockFormat,
    conversion_notes: Option<&str>,
) -> GmatMetadata {
    let mut metadata = GmatMetadata::new();

    // Copy all fields from original metadata (preserves model config with types)
    if let Some(orig) = original {
        for (key, value) in orig.iter() {
            metadata.set(key.to_string(), value.clone());
        }
    }

    // Update block format to current
    metadata.set_str(
        metadata_keys::BLOCK_FORMAT,
        block_format_to_string(current_format),
    );

    // Add or append to conversion info
    if let Some(notes) = conversion_notes {
        let existing_info = metadata.get_str(metadata_keys::CONVERSION_INFO);
        let new_info = if let Some(existing) = existing_info {
            format!("{}; {}", existing, notes)
        } else {
            notes.to_string()
        };
        metadata.set_str(metadata_keys::CONVERSION_INFO, new_info);
    }

    // Update timestamp for this export
    let timestamp = chrono::Utc::now().to_rfc3339();
    metadata.set_str(metadata_keys::CREATED_AT, timestamp);

    metadata
}

/// Prepare metadata for export with dtype update.
///
/// Like `prepare_export_metadata`, but also updates the original dtype field.
/// Use this when exporting to a different dtype than the original import.
///
/// # Arguments
/// * `original` - Original metadata from import (if available)
/// * `current_format` - Current BlockFormat being exported
/// * `new_dtype` - New dtype string (e.g., "F16", "BF16", "F32")
/// * `conversion_notes` - Optional notes about transformations applied
///
/// # Returns
/// Adapted GmatMetadata with updated dtype
pub fn prepare_export_metadata_with_dtype(
    original: Option<&GmatMetadata>,
    current_format: BlockFormat,
    new_dtype: &str,
    conversion_notes: Option<&str>,
) -> GmatMetadata {
    let mut metadata = prepare_export_metadata(original, current_format, conversion_notes);
    
    // Update original dtype to reflect conversion
    metadata.set_str(metadata_keys::ORIGINAL_DTYPE, new_dtype);
    
    metadata
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_export_metadata_preserves_config() {
        let mut original = GmatMetadata::new();
        original.set_str("architecture", "gpt2");
        original.set_i64("vocab_size", 50257);
        original.set_i64("hidden_size", 768);

        let adapted = prepare_export_metadata(
            Some(&original),
            BlockFormat::B16x8,
            None,
        );

        assert_eq!(adapted.get_str("architecture"), Some("gpt2"));
        assert_eq!(adapted.get_i64("vocab_size"), Some(50257));
        assert_eq!(adapted.get_i64("hidden_size"), Some(768));
    }

    #[test]
    fn test_prepare_export_metadata_updates_format() {
        let mut original = GmatMetadata::new();
        original.set_str(metadata_keys::BLOCK_FORMAT, "B8x4");

        let adapted = prepare_export_metadata(
            Some(&original),
            BlockFormat::DualRow16x8,
            None,
        );

        assert_eq!(adapted.get_str(metadata_keys::BLOCK_FORMAT), Some("DualRow16x8"));
    }

    #[test]
    fn test_prepare_export_metadata_adds_conversion_info() {
        let original = GmatMetadata::new();

        let adapted = prepare_export_metadata(
            Some(&original),
            BlockFormat::B16x8,
            Some("quantized to i8"),
        );

        assert_eq!(adapted.get_str(metadata_keys::CONVERSION_INFO), Some("quantized to i8"));
    }

    #[test]
    fn test_prepare_export_metadata_appends_conversion_info() {
        let mut original = GmatMetadata::new();
        original.set_str(metadata_keys::CONVERSION_INFO, "imported from safetensors");

        let adapted = prepare_export_metadata(
            Some(&original),
            BlockFormat::B16x8,
            Some("reformat to DualRow"),
        );

        assert_eq!(
            adapted.get_str(metadata_keys::CONVERSION_INFO),
            Some("imported from safetensors; reformat to DualRow")
        );
    }

    #[test]
    fn test_prepare_export_metadata_with_dtype_updates_dtype() {
        let mut original = GmatMetadata::new();
        original.set_str(metadata_keys::ORIGINAL_DTYPE, "BF16");

        let adapted = prepare_export_metadata_with_dtype(
            Some(&original),
            BlockFormat::B16x8,
            "F32",
            Some("upcast to F32"),
        );

        assert_eq!(adapted.get_str(metadata_keys::ORIGINAL_DTYPE), Some("F32"));
        assert_eq!(adapted.get_str(metadata_keys::CONVERSION_INFO), Some("upcast to F32"));
    }

    #[test]
    fn test_prepare_export_metadata_no_original() {
        let adapted = prepare_export_metadata(
            None,
            BlockFormat::B8x4,
            Some("created from scratch"),
        );

        assert_eq!(adapted.get_str(metadata_keys::BLOCK_FORMAT), Some("B8x4"));
        assert_eq!(adapted.get_str(metadata_keys::CONVERSION_INFO), Some("created from scratch"));
        assert!(adapted.get_str(metadata_keys::CREATED_AT).is_some());
    }

    #[test]
    fn test_block_format_strings() {
        assert_eq!(block_format_to_string(BlockFormat::B8x4), "B8x4");
        assert_eq!(block_format_to_string(BlockFormat::B8x8), "B8x8");
        assert_eq!(block_format_to_string(BlockFormat::B16x4), "B16x4");
        assert_eq!(block_format_to_string(BlockFormat::B16x8), "B16x8");
        assert_eq!(block_format_to_string(BlockFormat::DualRow8x4), "DualRow8x4");
        assert_eq!(block_format_to_string(BlockFormat::DualRow8x8), "DualRow8x8");
        assert_eq!(block_format_to_string(BlockFormat::DualRow16x4), "DualRow16x4");
        assert_eq!(block_format_to_string(BlockFormat::DualRow16x8), "DualRow16x8");
    }
}
