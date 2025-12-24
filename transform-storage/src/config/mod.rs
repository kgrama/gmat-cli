//! Configuration types for GraphMatrix construction.

use crate::blocks::BlockFormat;

mod any_matrix;
mod dual_any_matrix;

pub use any_matrix::AnyGraphMatrix;
pub use dual_any_matrix::DualAnyGraphMatrix;

/// Configuration for GraphMatrix construction.
#[derive(Debug, Clone, Default)]
pub struct StorageConfig {
    format: BlockFormat,
    build_col_index: bool,
}

impl StorageConfig {
    /// Create a new StorageConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the block format.
    pub fn format(mut self, format: BlockFormat) -> Self {
        self.format = format;
        self
    }

    /// Enable column index building.
    pub fn with_col_index(mut self) -> Self {
        self.build_col_index = true;
        self
    }

    /// Get the block format.
    pub fn get_format(&self) -> BlockFormat {
        self.format
    }

    /// Check if column index should be built.
    pub fn should_build_col_index(&self) -> bool {
        self.build_col_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // BlockFormat tests
    // ========================================================================

    #[test]
    fn test_block_format_default() {
        let format = BlockFormat::default();
        assert_eq!(format, BlockFormat::B16x8);
    }

    #[test]
    fn test_block_format_to_u8() {
        assert_eq!(u8::from(BlockFormat::B8x4), 0);
        assert_eq!(u8::from(BlockFormat::B8x8), 1);
        assert_eq!(u8::from(BlockFormat::B16x4), 2);
        assert_eq!(u8::from(BlockFormat::B16x8), 3);
        assert_eq!(u8::from(BlockFormat::DualRow8x4), 4);
        assert_eq!(u8::from(BlockFormat::DualRow8x8), 5);
        assert_eq!(u8::from(BlockFormat::DualRow16x4), 6);
        assert_eq!(u8::from(BlockFormat::DualRow16x8), 7);
    }

    #[test]
    fn test_block_format_from_u8() {
        assert_eq!(BlockFormat::try_from(0).unwrap(), BlockFormat::B8x4);
        assert_eq!(BlockFormat::try_from(1).unwrap(), BlockFormat::B8x8);
        assert_eq!(BlockFormat::try_from(2).unwrap(), BlockFormat::B16x4);
        assert_eq!(BlockFormat::try_from(3).unwrap(), BlockFormat::B16x8);
        assert_eq!(BlockFormat::try_from(4).unwrap(), BlockFormat::DualRow8x4);
        assert_eq!(BlockFormat::try_from(5).unwrap(), BlockFormat::DualRow8x8);
        assert_eq!(BlockFormat::try_from(6).unwrap(), BlockFormat::DualRow16x4);
        assert_eq!(BlockFormat::try_from(7).unwrap(), BlockFormat::DualRow16x8);
    }

    #[test]
    fn test_block_format_from_u8_invalid() {
        assert!(BlockFormat::try_from(8).is_err());
        assert!(BlockFormat::try_from(255).is_err());
    }

    #[test]
    fn test_block_format_roundtrip() {
        for format in [BlockFormat::B16x8, BlockFormat::B16x4, BlockFormat::B8x4] {
            let byte = u8::from(format);
            let restored = BlockFormat::try_from(byte).unwrap();
            assert_eq!(format, restored);
        }
    }

    // ========================================================================
    // StorageConfig tests
    // ========================================================================

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::new();
        assert_eq!(config.get_format(), BlockFormat::B16x8);
        assert!(!config.should_build_col_index());
    }

    #[test]
    fn test_storage_config_format() {
        let config = StorageConfig::new().format(BlockFormat::B8x4);
        assert_eq!(config.get_format(), BlockFormat::B8x4);
    }

    #[test]
    fn test_storage_config_with_col_index() {
        let config = StorageConfig::new().with_col_index();
        assert!(config.should_build_col_index());
    }

    #[test]
    fn test_storage_config_chaining() {
        let config = StorageConfig::new()
            .format(BlockFormat::B16x4)
            .with_col_index();

        assert_eq!(config.get_format(), BlockFormat::B16x4);
        assert!(config.should_build_col_index());
    }
}
