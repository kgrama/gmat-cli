//! Configuration types for GraphMatrix construction.

use crate::blocks::BlockFormat;
use crate::graph_matrix::GraphMatrix;
use candle_core::{Device, Tensor, Result};

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

/// Type-erased wrapper for GraphMatrix that can hold any block format.
#[derive(Debug, Clone)]
pub enum AnyGraphMatrix {
    B16x8(GraphMatrix),
    B16x4(GraphMatrix),
    B8x8(GraphMatrix),
    B8x4(GraphMatrix),
}

impl AnyGraphMatrix {
    /// Create AnyGraphMatrix from dense f32 data with configuration.
    pub fn from_dense(data: &[f32], shape: (usize, usize), config: &StorageConfig) -> Self {
        match config.format {
            BlockFormat::B16x8 => Self::B16x8(GraphMatrix::from_dense(data, shape, BlockFormat::B16x8)),
            BlockFormat::B16x4 => Self::B16x4(GraphMatrix::from_dense(data, shape, BlockFormat::B16x4)),
            BlockFormat::B8x8 => Self::B8x8(GraphMatrix::from_dense(data, shape, BlockFormat::B8x8)),
            BlockFormat::B8x4 => Self::B8x4(GraphMatrix::from_dense(data, shape, BlockFormat::B8x4)),
            BlockFormat::DualRow8x4 | BlockFormat::DualRow8x8 | BlockFormat::DualRow16x4 | BlockFormat::DualRow16x8 => {
                panic!("AnyGraphMatrix does not support DualRow formats")
            }
        }
    }

    /// Get the block format of this matrix.
    pub fn format(&self) -> BlockFormat {
        match self {
            Self::B16x8(_) => BlockFormat::B16x8,
            Self::B16x4(_) => BlockFormat::B16x4,
            Self::B8x8(_) => BlockFormat::B8x8,
            Self::B8x4(_) => BlockFormat::B8x4,
        }
    }

    /// Get matrix dimensions (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::B16x8(m) => m.shape(),
            Self::B16x4(m) => m.shape(),
            Self::B8x8(m) => m.shape(),
            Self::B8x4(m) => m.shape(),
        }
    }

    /// Count total non-zero elements.
    pub fn nnz(&self) -> usize {
        match self {
            Self::B16x8(m) => m.nnz(),
            Self::B16x4(m) => m.nnz(),
            Self::B8x8(m) => m.nnz(),
            Self::B8x4(m) => m.nnz(),
        }
    }

    /// Calculate density as ratio of non-zero to total elements.
    /// Returns 1.0 for a fully dense matrix, 0.0 for an all-zeros matrix.
    pub fn density(&self) -> f32 {
        match self {
            Self::B16x8(m) => m.density(),
            Self::B16x4(m) => m.density(),
            Self::B8x8(m) => m.density(),
            Self::B8x4(m) => m.density(),
        }
    }

    /// Alias for density() - deprecated, use density() instead.
    #[deprecated(note = "Use density() instead - this returns nnz/total, not sparsity")]
    pub fn sparsity(&self) -> f32 {
        self.density()
    }

    /// Calculate total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::B16x8(m) => m.memory_bytes(),
            Self::B16x4(m) => m.memory_bytes(),
            Self::B8x8(m) => m.memory_bytes(),
            Self::B8x4(m) => m.memory_bytes(),
        }
    }

    // Type accessors - panic versions

    pub fn as_block16x8(&self) -> &GraphMatrix {
        match self {
            Self::B16x8(m) => m,
            _ => panic!("Expected Block16x8, got {:?}", self.format()),
        }
    }

    pub fn as_block16x4(&self) -> &GraphMatrix {
        match self {
            Self::B16x4(m) => m,
            _ => panic!("Expected Block16x4, got {:?}", self.format()),
        }
    }

    pub fn as_block8x8(&self) -> &GraphMatrix {
        match self {
            Self::B8x8(m) => m,
            _ => panic!("Expected Block8x8, got {:?}", self.format()),
        }
    }

    pub fn as_block8x4(&self) -> &GraphMatrix {
        match self {
            Self::B8x4(m) => m,
            _ => panic!("Expected Block8x4, got {:?}", self.format()),
        }
    }

    // Type accessors - Option versions

    pub fn try_as_block16x8(&self) -> Option<&GraphMatrix> {
        match self { Self::B16x8(m) => Some(m), _ => None }
    }

    pub fn try_as_block16x4(&self) -> Option<&GraphMatrix> {
        match self { Self::B16x4(m) => Some(m), _ => None }
    }

    pub fn try_as_block8x8(&self) -> Option<&GraphMatrix> {
        match self { Self::B8x8(m) => Some(m), _ => None }
    }

    pub fn try_as_block8x4(&self) -> Option<&GraphMatrix> {
        match self { Self::B8x4(m) => Some(m), _ => None }
    }

    /// Create AnyGraphMatrix from a candle Tensor with configuration.
    ///
    /// Extracts f32 data from the tensor and creates the appropriate GraphMatrix variant.
    ///
    /// # Arguments
    /// - `tensor`: 2D candle Tensor with f32 dtype
    /// - `config`: Storage configuration specifying block format
    ///
    /// # Errors
    /// Returns error if tensor extraction fails or tensor is not 2D
    pub fn from_tensor(tensor: &Tensor, config: &StorageConfig) -> Result<Self> {
        Ok(match config.format {
            BlockFormat::B16x8 => Self::B16x8(GraphMatrix::from_tensor(tensor, BlockFormat::B16x8)?),
            BlockFormat::B16x4 => Self::B16x4(GraphMatrix::from_tensor(tensor, BlockFormat::B16x4)?),
            BlockFormat::B8x8 => Self::B8x8(GraphMatrix::from_tensor(tensor, BlockFormat::B8x8)?),
            BlockFormat::B8x4 => Self::B8x4(GraphMatrix::from_tensor(tensor, BlockFormat::B8x4)?),
            BlockFormat::DualRow8x4 | BlockFormat::DualRow8x8 | BlockFormat::DualRow16x4 | BlockFormat::DualRow16x8 => {
                panic!("AnyGraphMatrix does not support DualRow formats")
            }
        })
    }

    /// Convert AnyGraphMatrix to a candle Tensor.
    ///
    /// Decodes all blocks to f32 values and creates a tensor on the specified device.
    ///
    /// # Arguments
    /// - `device`: Target device for the tensor
    ///
    /// # Errors
    /// Returns error if tensor creation fails
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        match self {
            Self::B16x8(m) => m.to_tensor(device),
            Self::B16x4(m) => m.to_tensor(device),
            Self::B8x8(m) => m.to_tensor(device),
            Self::B8x4(m) => m.to_tensor(device),
        }
    }

    /// Save AnyGraphMatrix to a file in GMAT format.
    ///
    /// # Arguments
    /// - `path`: File path to save to
    ///
    /// # Errors
    /// Returns error if file creation or writing fails
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        match self {
            Self::B16x8(m) => m.save(path),
            Self::B16x4(m) => m.save(path),
            Self::B8x8(m) => m.save(path),
            Self::B8x4(m) => m.save(path),
        }
    }

    /// Load AnyGraphMatrix from a file in GMAT format.
    ///
    /// Reads the header to determine block format and delegates to the appropriate GraphMatrix::load.
    ///
    /// # Arguments
    /// - `path`: File path to load from
    ///
    /// # Errors
    /// Returns error if file reading fails or format is invalid
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        
        let mut file = std::fs::File::open(path)?;
        
        // Read magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GMAT" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid magic: expected GMAT, got {:?}", magic),
            ));
        }
        
        // Read version
        let mut version_bytes = [0u8; 2];
        file.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", version),
            ));
        }
        
        // Read format
        let mut format_byte = [0u8; 1];
        file.read_exact(&mut format_byte)?;
        let format = BlockFormat::try_from(format_byte[0])?;
        
        // Reopen file for full load (GraphMatrix::load expects to read from start)
        drop(file);
        
        match format {
            BlockFormat::B16x8 => Ok(Self::B16x8(GraphMatrix::load(path)?)),
            BlockFormat::B16x4 => Ok(Self::B16x4(GraphMatrix::load(path)?)),
            BlockFormat::B8x8 => Ok(Self::B8x8(GraphMatrix::load(path)?)),
            BlockFormat::B8x4 => Ok(Self::B8x4(GraphMatrix::load(path)?)),
            BlockFormat::DualRow8x4 | BlockFormat::DualRow8x8 | BlockFormat::DualRow16x4 | BlockFormat::DualRow16x8 => {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "AnyGraphMatrix does not support DualRow formats",
                ))
            }
        }
    }
}

/// Type-erased wrapper for GraphMatrix with dual-row block formats.
#[derive(Debug, Clone)]
pub enum DualAnyGraphMatrix {
    DualRow8x4(GraphMatrix),
    DualRow8x8(GraphMatrix),
    DualRow16x4(GraphMatrix),
    DualRow16x8(GraphMatrix),
}

impl DualAnyGraphMatrix {
    /// Create DualAnyGraphMatrix from dense f32 data with configuration.
    pub fn from_dense(data: &[f32], shape: (usize, usize), config: &StorageConfig) -> Self {
        match config.format {
            BlockFormat::DualRow8x4 => Self::DualRow8x4(GraphMatrix::from_dense(data, shape, BlockFormat::DualRow8x4)),
            BlockFormat::DualRow8x8 => Self::DualRow8x8(GraphMatrix::from_dense(data, shape, BlockFormat::DualRow8x8)),
            BlockFormat::DualRow16x4 => Self::DualRow16x4(GraphMatrix::from_dense(data, shape, BlockFormat::DualRow16x4)),
            BlockFormat::DualRow16x8 => Self::DualRow16x8(GraphMatrix::from_dense(data, shape, BlockFormat::DualRow16x8)),
            BlockFormat::B8x4 | BlockFormat::B8x8 | BlockFormat::B16x4 | BlockFormat::B16x8 => {
                panic!("DualAnyGraphMatrix does not support single-row formats")
            }
        }
    }

    /// Get the block format of this matrix.
    pub fn format(&self) -> BlockFormat {
        match self {
            Self::DualRow8x4(_) => BlockFormat::DualRow8x4,
            Self::DualRow8x8(_) => BlockFormat::DualRow8x8,
            Self::DualRow16x4(_) => BlockFormat::DualRow16x4,
            Self::DualRow16x8(_) => BlockFormat::DualRow16x8,
        }
    }

    /// Get matrix dimensions (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::DualRow8x4(m) => m.shape(),
            Self::DualRow8x8(m) => m.shape(),
            Self::DualRow16x4(m) => m.shape(),
            Self::DualRow16x8(m) => m.shape(),
        }
    }

    /// Count total non-zero elements.
    pub fn nnz(&self) -> usize {
        match self {
            Self::DualRow8x4(m) => m.nnz(),
            Self::DualRow8x8(m) => m.nnz(),
            Self::DualRow16x4(m) => m.nnz(),
            Self::DualRow16x8(m) => m.nnz(),
        }
    }

    /// Calculate density as ratio of non-zero to total elements.
    /// Returns 1.0 for a fully dense matrix, 0.0 for an all-zeros matrix.
    pub fn density(&self) -> f32 {
        match self {
            Self::DualRow8x4(m) => m.density(),
            Self::DualRow8x8(m) => m.density(),
            Self::DualRow16x4(m) => m.density(),
            Self::DualRow16x8(m) => m.density(),
        }
    }

    /// Alias for density() - deprecated, use density() instead.
    #[deprecated(note = "Use density() instead - this returns nnz/total, not sparsity")]
    pub fn sparsity(&self) -> f32 {
        self.density()
    }

    /// Calculate total memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::DualRow8x4(m) => m.memory_bytes(),
            Self::DualRow8x8(m) => m.memory_bytes(),
            Self::DualRow16x4(m) => m.memory_bytes(),
            Self::DualRow16x8(m) => m.memory_bytes(),
        }
    }

    // Type accessors - panic versions

    pub fn as_dualrow8x4(&self) -> &GraphMatrix {
        match self {
            Self::DualRow8x4(m) => m,
            _ => panic!("Expected DualRow8x4, got {:?}", self.format()),
        }
    }

    pub fn as_dualrow8x8(&self) -> &GraphMatrix {
        match self {
            Self::DualRow8x8(m) => m,
            _ => panic!("Expected DualRow8x8, got {:?}", self.format()),
        }
    }

    pub fn as_dualrow16x4(&self) -> &GraphMatrix {
        match self {
            Self::DualRow16x4(m) => m,
            _ => panic!("Expected DualRow16x4, got {:?}", self.format()),
        }
    }

    pub fn as_dualrow16x8(&self) -> &GraphMatrix {
        match self {
            Self::DualRow16x8(m) => m,
            _ => panic!("Expected DualRow16x8, got {:?}", self.format()),
        }
    }

    // Type accessors - Option versions

    pub fn try_as_dualrow8x4(&self) -> Option<&GraphMatrix> {
        match self { Self::DualRow8x4(m) => Some(m), _ => None }
    }

    pub fn try_as_dualrow8x8(&self) -> Option<&GraphMatrix> {
        match self { Self::DualRow8x8(m) => Some(m), _ => None }
    }

    pub fn try_as_dualrow16x4(&self) -> Option<&GraphMatrix> {
        match self { Self::DualRow16x4(m) => Some(m), _ => None }
    }

    pub fn try_as_dualrow16x8(&self) -> Option<&GraphMatrix> {
        match self { Self::DualRow16x8(m) => Some(m), _ => None }
    }

    /// Create DualAnyGraphMatrix from a candle Tensor with configuration.
    ///
    /// Extracts f32 data from the tensor and creates the appropriate GraphMatrix variant.
    ///
    /// # Arguments
    /// - `tensor`: 2D candle Tensor with f32 dtype
    /// - `config`: Storage configuration specifying block format
    ///
    /// # Errors
    /// Returns error if tensor extraction fails or tensor is not 2D
    pub fn from_tensor(tensor: &Tensor, config: &StorageConfig) -> Result<Self> {
        Ok(match config.format {
            BlockFormat::DualRow8x4 => Self::DualRow8x4(GraphMatrix::from_tensor(tensor, BlockFormat::DualRow8x4)?),
            BlockFormat::DualRow8x8 => Self::DualRow8x8(GraphMatrix::from_tensor(tensor, BlockFormat::DualRow8x8)?),
            BlockFormat::DualRow16x4 => Self::DualRow16x4(GraphMatrix::from_tensor(tensor, BlockFormat::DualRow16x4)?),
            BlockFormat::DualRow16x8 => Self::DualRow16x8(GraphMatrix::from_tensor(tensor, BlockFormat::DualRow16x8)?),
            BlockFormat::B8x4 | BlockFormat::B8x8 | BlockFormat::B16x4 | BlockFormat::B16x8 => {
                panic!("DualAnyGraphMatrix does not support single-row formats")
            }
        })
    }

    /// Convert DualAnyGraphMatrix to a candle Tensor.
    ///
    /// Decodes all blocks to f32 values and creates a tensor on the specified device.
    ///
    /// # Arguments
    /// - `device`: Target device for the tensor
    ///
    /// # Errors
    /// Returns error if tensor creation fails
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        match self {
            Self::DualRow8x4(m) => m.to_tensor(device),
            Self::DualRow8x8(m) => m.to_tensor(device),
            Self::DualRow16x4(m) => m.to_tensor(device),
            Self::DualRow16x8(m) => m.to_tensor(device),
        }
    }

    /// Save DualAnyGraphMatrix to a file in GMAT format.
    ///
    /// # Arguments
    /// - `path`: File path to save to
    ///
    /// # Errors
    /// Returns error if file creation or writing fails
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        match self {
            Self::DualRow8x4(m) => m.save(path),
            Self::DualRow8x8(m) => m.save(path),
            Self::DualRow16x4(m) => m.save(path),
            Self::DualRow16x8(m) => m.save(path),
        }
    }

    /// Load DualAnyGraphMatrix from a file in GMAT format.
    ///
    /// Reads the header to determine block format and delegates to the appropriate GraphMatrix::load.
    ///
    /// # Arguments
    /// - `path`: File path to load from
    ///
    /// # Errors
    /// Returns error if file reading fails or format is invalid
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        
        let mut file = std::fs::File::open(path)?;
        
        // Read magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GMAT" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Invalid magic: expected GMAT, got {:?}", magic),
            ));
        }
        
        // Read version
        let mut version_bytes = [0u8; 2];
        file.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if version != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", version),
            ));
        }
        
        // Read format
        let mut format_byte = [0u8; 1];
        file.read_exact(&mut format_byte)?;
        let format = BlockFormat::try_from(format_byte[0])?;
        
        // Reopen file for full load (GraphMatrix::load expects to read from start)
        drop(file);
        
        match format {
            BlockFormat::DualRow8x4 => Ok(Self::DualRow8x4(GraphMatrix::load(path)?)),
            BlockFormat::DualRow8x8 => Ok(Self::DualRow8x8(GraphMatrix::load(path)?)),
            BlockFormat::DualRow16x4 => Ok(Self::DualRow16x4(GraphMatrix::load(path)?)),
            BlockFormat::DualRow16x8 => Ok(Self::DualRow16x8(GraphMatrix::load(path)?)),
            BlockFormat::B8x4 | BlockFormat::B8x8 | BlockFormat::B16x4 | BlockFormat::B16x8 => {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "DualAnyGraphMatrix does not support single-row formats",
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    /// Generate test data with narrow dynamic range suitable for block quantization.
    fn narrow_range_data(cols: usize, rows: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(cols * rows);
        for r in 0..rows {
            for c in 0..cols {
                // Values between 1.0 and 4.0 with some zeros
                let val = if (r + c) % 3 == 0 {
                    0.0
                } else {
                    1.0 + (((r * cols + c) % 10) as f32) * 0.3
                };
                data.push(val);
            }
        }
        data
    }

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

    // ========================================================================
    // AnyGraphMatrix from_dense tests
    // ========================================================================

    #[test]
    fn test_any_graph_matrix_from_dense_block16x8() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new().format(BlockFormat::B16x8);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        assert_eq!(matrix.format(), BlockFormat::B16x8);
        assert_eq!(matrix.shape(), (2, 16));
    }

    #[test]
    fn test_any_graph_matrix_from_dense_block16x4() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new().format(BlockFormat::B16x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        assert_eq!(matrix.format(), BlockFormat::B16x4);
        assert_eq!(matrix.shape(), (2, 16));
    }

    #[test]
    fn test_any_graph_matrix_from_dense_block8x4() {
        let data = vec![1.0f32; 16];
        let config = StorageConfig::new().format(BlockFormat::B8x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 8), &config);

        assert_eq!(matrix.format(), BlockFormat::B8x4);
        assert_eq!(matrix.shape(), (2, 8));
    }

    // ========================================================================
    // AnyGraphMatrix accessors tests
    // ========================================================================

    #[test]
    fn test_any_graph_matrix_nnz() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;
        data[15] = 2.0;
        data[16] = 3.0;

        let config = StorageConfig::new();
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        assert_eq!(matrix.nnz(), 3);
    }

    #[test]
    fn test_any_graph_matrix_sparsity() {
        let mut data = vec![0.0f32; 16];
        for i in 0..8 {
            data[i] = 1.0;
        }

        let config = StorageConfig::new();
        let matrix = AnyGraphMatrix::from_dense(&data, (1, 16), &config);

        assert_eq!(matrix.density(), 0.5);
    }

    #[test]
    fn test_any_graph_matrix_memory_bytes() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new();
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        assert!(matrix.memory_bytes() > 0);
    }

    // ========================================================================
    // AnyGraphMatrix type accessor tests
    // ========================================================================

    #[test]
    fn test_as_block16x8() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new().format(BlockFormat::B16x8);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        let inner = matrix.as_block16x8();
        assert_eq!(inner.shape(), (2, 16));
    }

    #[test]
    fn test_as_block16x4() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new().format(BlockFormat::B16x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        let inner = matrix.as_block16x4();
        assert_eq!(inner.shape(), (2, 16));
    }

    #[test]
    fn test_as_block8x4() {
        let data = vec![1.0f32; 16];
        let config = StorageConfig::new().format(BlockFormat::B8x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 8), &config);

        let inner = matrix.as_block8x4();
        assert_eq!(inner.shape(), (2, 8));
    }

    #[test]
    #[should_panic(expected = "Expected Block16x8")]
    fn test_as_block16x8_wrong_type() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new().format(BlockFormat::B16x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);
        let _ = matrix.as_block16x8();
    }

    #[test]
    fn test_try_as_block16x8_some() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new().format(BlockFormat::B16x8);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        assert!(matrix.try_as_block16x8().is_some());
        assert!(matrix.try_as_block16x4().is_none());
        assert!(matrix.try_as_block8x4().is_none());
    }

    #[test]
    fn test_try_as_block16x4_some() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new().format(BlockFormat::B16x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        assert!(matrix.try_as_block16x8().is_none());
        assert!(matrix.try_as_block16x4().is_some());
        assert!(matrix.try_as_block8x4().is_none());
    }

    #[test]
    fn test_try_as_block8x4_some() {
        let data = vec![1.0f32; 16];
        let config = StorageConfig::new().format(BlockFormat::B8x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 8), &config);

        assert!(matrix.try_as_block16x8().is_none());
        assert!(matrix.try_as_block16x4().is_none());
        assert!(matrix.try_as_block8x4().is_some());
    }

    // ========================================================================
    // AnyGraphMatrix tensor tests
    // ========================================================================

    #[test]
    fn test_any_graph_matrix_from_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::ones((2, 16), candle_core::DType::F32, &device).unwrap();

        let config = StorageConfig::new().format(BlockFormat::B16x8);
        let matrix = AnyGraphMatrix::from_tensor(&tensor, &config).unwrap();

        assert_eq!(matrix.shape(), (2, 16));
        assert_eq!(matrix.format(), BlockFormat::B16x8);
    }

    #[test]
    fn test_any_graph_matrix_to_tensor() {
        let data = vec![1.0f32; 32];
        let config = StorageConfig::new();
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        let device = Device::Cpu;
        let tensor = matrix.to_tensor(&device).unwrap();

        assert_eq!(tensor.dims(), &[2, 16]);
    }

    #[test]
    fn test_any_graph_matrix_tensor_roundtrip_all_formats() {
        let device = Device::Cpu;

        for format in [BlockFormat::B16x8, BlockFormat::B16x4, BlockFormat::B8x4] {
            let cols = if format == BlockFormat::B8x4 { 8 } else { 16 };
            // Use data with narrow dynamic range suitable for block quantization
            let data = narrow_range_data(cols, 2);
            let tensor = Tensor::from_vec(data, (2, cols), &device).unwrap();

            let config = StorageConfig::new().format(format);
            let matrix = AnyGraphMatrix::from_tensor(&tensor, &config).unwrap();
            let restored = matrix.to_tensor(&device).unwrap();

            assert_eq!(restored.dims(), &[2, cols]);
        }
    }

    // ========================================================================
    // AnyGraphMatrix save/load tests
    // ========================================================================

    #[test]
    fn test_any_graph_matrix_save_load_block16x8() {
        let data = narrow_range_data(16, 2);
        let config = StorageConfig::new().format(BlockFormat::B16x8);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = AnyGraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(loaded.format(), BlockFormat::B16x8);
        assert_eq!(loaded.shape(), (2, 16));
        assert_eq!(loaded.nnz(), matrix.nnz());
    }

    #[test]
    fn test_any_graph_matrix_save_load_block16x4() {
        let data = narrow_range_data(16, 2);
        let config = StorageConfig::new().format(BlockFormat::B16x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = AnyGraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(loaded.format(), BlockFormat::B16x4);
        assert_eq!(loaded.shape(), (2, 16));
    }

    #[test]
    fn test_any_graph_matrix_save_load_block8x4() {
        let data = narrow_range_data(8, 2);
        let config = StorageConfig::new().format(BlockFormat::B8x4);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 8), &config);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = AnyGraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(loaded.format(), BlockFormat::B8x4);
        assert_eq!(loaded.shape(), (2, 8));
    }

    #[test]
    fn test_any_graph_matrix_load_preserves_format() {
        // Save with each format and verify load detects correct format
        for format in [BlockFormat::B16x8, BlockFormat::B16x4, BlockFormat::B8x4] {
            let cols = if format == BlockFormat::B8x4 { 8 } else { 16 };
            let data = narrow_range_data(cols, 2);
            let config = StorageConfig::new().format(format);
            let matrix = AnyGraphMatrix::from_dense(&data, (2, cols), &config);

            let temp_file = NamedTempFile::new().unwrap();
            matrix.save(temp_file.path()).unwrap();

            let loaded = AnyGraphMatrix::load(temp_file.path()).unwrap();
            assert_eq!(loaded.format(), format, "Format should be preserved after save/load");
        }
    }

    #[test]
    fn test_any_graph_matrix_load_invalid_magic() {
        let temp_file = NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), b"XXXX").unwrap();

        let result = AnyGraphMatrix::load(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_any_graph_matrix_load_invalid_version() {
        let temp_file = NamedTempFile::new().unwrap();
        // GMAT magic + invalid version (2)
        let mut data = b"GMAT".to_vec();
        data.extend_from_slice(&2u16.to_le_bytes());
        std::fs::write(temp_file.path(), data).unwrap();

        let result = AnyGraphMatrix::load(temp_file.path());
        assert!(result.is_err());
    }

    // ========================================================================
    // DualAnyGraphMatrix tests
    // ========================================================================

    #[test]
    fn test_dual_any_graph_matrix_from_dense_dualrow8x4() {
        let data = narrow_range_data(8, 4); // 4 rows, 8 cols
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);
        assert_eq!(matrix.format(), BlockFormat::DualRow8x4);
        assert_eq!(matrix.shape(), (4, 8));
    }

    #[test]
    fn test_dual_any_graph_matrix_from_dense_dualrow16x8() {
        let data = narrow_range_data(16, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow16x8);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 16), &config);
        assert_eq!(matrix.format(), BlockFormat::DualRow16x8);
    }

    #[test]
    fn test_dual_any_graph_matrix_from_dense_dualrow8x8() {
        let data = narrow_range_data(8, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow8x8);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);
        assert_eq!(matrix.format(), BlockFormat::DualRow8x8);
        assert_eq!(matrix.shape(), (4, 8));
    }

    #[test]
    fn test_dual_any_graph_matrix_from_dense_dualrow16x4() {
        let data = narrow_range_data(16, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow16x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 16), &config);
        assert_eq!(matrix.format(), BlockFormat::DualRow16x4);
        assert_eq!(matrix.shape(), (4, 16));
    }

    #[test]
    fn test_dual_any_graph_matrix_nnz() {
        let data = narrow_range_data(8, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);
        assert!(matrix.nnz() > 0);
    }

    #[test]
    fn test_dual_any_graph_matrix_density() {
        let data = narrow_range_data(8, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);
        let density = matrix.density();
        assert!(density >= 0.0 && density <= 1.0);
    }

    #[test]
    fn test_dual_any_graph_matrix_memory_bytes() {
        let data = narrow_range_data(8, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);
        assert!(matrix.memory_bytes() > 0);
    }

    #[test]
    fn test_dual_any_graph_matrix_save_load_roundtrip() {
        let data = narrow_range_data(8, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);
        
        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();
        
        let loaded = DualAnyGraphMatrix::load(temp_file.path()).unwrap();
        assert_eq!(loaded.format(), BlockFormat::DualRow8x4);
        assert_eq!(loaded.shape(), (4, 8));
    }

    #[test]
    fn test_dual_any_graph_matrix_save_load_all_formats() {
        for format in [BlockFormat::DualRow8x4, BlockFormat::DualRow8x8, BlockFormat::DualRow16x4, BlockFormat::DualRow16x8] {
            let cols = if format == BlockFormat::DualRow8x4 || format == BlockFormat::DualRow8x8 { 8 } else { 16 };
            let data = narrow_range_data(cols, 4);
            let config = StorageConfig::new().format(format);
            let matrix = DualAnyGraphMatrix::from_dense(&data, (4, cols), &config);

            let temp_file = NamedTempFile::new().unwrap();
            matrix.save(temp_file.path()).unwrap();

            let loaded = DualAnyGraphMatrix::load(temp_file.path()).unwrap();
            assert_eq!(loaded.format(), format, "Format should be preserved after save/load");
            assert_eq!(loaded.shape(), (4, cols));
        }
    }

    #[test]
    fn test_dual_any_graph_matrix_tensor_roundtrip() {
        use candle_core::Device;
        let data = narrow_range_data(16, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow16x8);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 16), &config);
        
        let tensor = matrix.to_tensor(&Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[4, 16]);
    }

    #[test]
    fn test_dual_any_graph_matrix_from_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::ones((4, 8), candle_core::DType::F32, &device).unwrap();

        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_tensor(&tensor, &config).unwrap();

        assert_eq!(matrix.shape(), (4, 8));
        assert_eq!(matrix.format(), BlockFormat::DualRow8x4);
    }

    #[test]
    fn test_dual_any_graph_matrix_tensor_roundtrip_all_formats() {
        let device = Device::Cpu;

        for format in [BlockFormat::DualRow8x4, BlockFormat::DualRow8x8, BlockFormat::DualRow16x4, BlockFormat::DualRow16x8] {
            let cols = if format == BlockFormat::DualRow8x4 || format == BlockFormat::DualRow8x8 { 8 } else { 16 };
            let data = narrow_range_data(cols, 4);
            let tensor = Tensor::from_vec(data, (4, cols), &device).unwrap();

            let config = StorageConfig::new().format(format);
            let matrix = DualAnyGraphMatrix::from_tensor(&tensor, &config).unwrap();
            let restored = matrix.to_tensor(&device).unwrap();

            assert_eq!(restored.dims(), &[4, cols]);
        }
    }

    #[test]
    fn test_dual_any_graph_matrix_type_accessors() {
        let data = narrow_range_data(8, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);
        
        assert!(matrix.try_as_dualrow8x4().is_some());
        assert!(matrix.try_as_dualrow8x8().is_none());
        assert!(matrix.try_as_dualrow16x4().is_none());
        assert!(matrix.try_as_dualrow16x8().is_none());
    }

    #[test]
    fn test_dual_any_graph_matrix_as_dualrow8x4() {
        let data = narrow_range_data(8, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 8), &config);

        let inner = matrix.as_dualrow8x4();
        assert_eq!(inner.shape(), (4, 8));
    }

    #[test]
    fn test_dual_any_graph_matrix_as_dualrow16x8() {
        let data = narrow_range_data(16, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow16x8);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 16), &config);

        let inner = matrix.as_dualrow16x8();
        assert_eq!(inner.shape(), (4, 16));
    }

    #[test]
    #[should_panic(expected = "Expected DualRow8x4")]
    fn test_dual_any_graph_matrix_as_dualrow8x4_wrong_type() {
        let data = narrow_range_data(16, 4);
        let config = StorageConfig::new().format(BlockFormat::DualRow16x8);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (4, 16), &config);
        let _ = matrix.as_dualrow8x4();
    }

    #[test]
    #[should_panic(expected = "does not support single-row")]
    fn test_dual_any_graph_matrix_rejects_single_row_b16x8() {
        let data = narrow_range_data(16, 2);
        let config = StorageConfig::new().format(BlockFormat::B16x8);
        let _matrix = DualAnyGraphMatrix::from_dense(&data, (2, 16), &config);
    }

    #[test]
    #[should_panic(expected = "does not support single-row")]
    fn test_dual_any_graph_matrix_rejects_single_row_b8x4() {
        let data = narrow_range_data(8, 2);
        let config = StorageConfig::new().format(BlockFormat::B8x4);
        let _matrix = DualAnyGraphMatrix::from_dense(&data, (2, 8), &config);
    }

    #[test]
    fn test_dual_any_graph_matrix_odd_row_count() {
        // Test that odd row counts work (should pad with zeros)
        let data = narrow_range_data(8, 3); // 3 rows - odd!
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (3, 8), &config);
        
        assert_eq!(matrix.shape(), (3, 8));
        assert!(matrix.nnz() > 0);
    }

    #[test]
    fn test_dual_any_graph_matrix_odd_row_count_tensor_roundtrip() {
        let device = Device::Cpu;
        let data = narrow_range_data(16, 5); // 5 rows - odd!
        let tensor = Tensor::from_vec(data, (5, 16), &device).unwrap();

        let config = StorageConfig::new().format(BlockFormat::DualRow16x8);
        let matrix = DualAnyGraphMatrix::from_tensor(&tensor, &config).unwrap();
        let restored = matrix.to_tensor(&device).unwrap();

        assert_eq!(restored.dims(), &[5, 16]);
    }

    #[test]
    fn test_dual_any_graph_matrix_single_row() {
        // Single row should work (padded to 2 rows internally)
        let data = narrow_range_data(8, 1); // 1 row
        let config = StorageConfig::new().format(BlockFormat::DualRow8x4);
        let matrix = DualAnyGraphMatrix::from_dense(&data, (1, 8), &config);
        
        assert_eq!(matrix.shape(), (1, 8));
    }

    #[test]
    fn test_dual_any_graph_matrix_load_rejects_single_row_format() {
        // Save a single-row format matrix, then try to load with DualAnyGraphMatrix
        let data = narrow_range_data(16, 2);
        let config = StorageConfig::new().format(BlockFormat::B16x8);
        let matrix = AnyGraphMatrix::from_dense(&data, (2, 16), &config);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let result = DualAnyGraphMatrix::load(temp_file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("does not support single-row"));
    }
}
