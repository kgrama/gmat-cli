//! Type-erased block enum that can hold any block configuration

use half::f16;
use std::io::{Read, Write, Result};
use super::configs::{Config8x4, Config8x8, Config16x4, Config16x8};
use super::unified_block::UnifiedBlock;

/// Block format identifier for serialization/construction
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BlockFormat {
    B8x4,
    B8x8,
    B16x4,
    #[default]
    B16x8,
    DualRow8x4,
    DualRow8x8,
    DualRow16x4,
    DualRow16x8,
}

impl BlockFormat {
    /// Get the block size (number of elements per row)
    pub fn block_size(&self) -> usize {
        match self {
            Self::B8x4 | Self::B8x8 | Self::DualRow8x4 | Self::DualRow8x8 => 8,
            Self::B16x4 | Self::B16x8 | Self::DualRow16x4 | Self::DualRow16x8 => 16,
        }
    }

    /// Check if this format is dual-row
    pub fn is_dual_row(&self) -> bool {
        matches!(self, Self::DualRow8x4 | Self::DualRow8x8 | Self::DualRow16x4 | Self::DualRow16x8)
    }
}

impl From<BlockFormat> for u8 {
    fn from(format: BlockFormat) -> u8 {
        match format {
            BlockFormat::B8x4 => 0,
            BlockFormat::B8x8 => 1,
            BlockFormat::B16x4 => 2,
            BlockFormat::B16x8 => 3,
            BlockFormat::DualRow8x4 => 4,
            BlockFormat::DualRow8x8 => 5,
            BlockFormat::DualRow16x4 => 6,
            BlockFormat::DualRow16x8 => 7,
        }
    }
}

impl TryFrom<u8> for BlockFormat {
    type Error = std::io::Error;

    fn try_from(value: u8) -> std::io::Result<Self> {
        match value {
            0 => Ok(BlockFormat::B8x4),
            1 => Ok(BlockFormat::B8x8),
            2 => Ok(BlockFormat::B16x4),
            3 => Ok(BlockFormat::B16x8),
            4 => Ok(BlockFormat::DualRow8x4),
            5 => Ok(BlockFormat::DualRow8x8),
            6 => Ok(BlockFormat::DualRow16x4),
            7 => Ok(BlockFormat::DualRow16x8),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unknown block format: {}", value),
            )),
        }
    }
}

/// Type-erased block that can hold any block configuration
#[derive(Clone, Debug)]
pub enum AnyBlock {
    B8x4(UnifiedBlock<1, Config8x4>),
    B8x8(UnifiedBlock<1, Config8x8>),
    B16x4(UnifiedBlock<1, Config16x4>),
    B16x8(UnifiedBlock<1, Config16x8>),
    DualRow8x4(UnifiedBlock<2, Config8x4>),
    DualRow8x8(UnifiedBlock<2, Config8x8>),
    DualRow16x4(UnifiedBlock<2, Config16x4>),
    DualRow16x8(UnifiedBlock<2, Config16x8>),
}

/// Helper macro to reduce match arm duplication
macro_rules! any_block_dispatch {
    ($self:expr, $method:ident $(, $args:expr)*) => {
        match $self {
            AnyBlock::B8x4(b) => b.$method($($args),*),
            AnyBlock::B8x8(b) => b.$method($($args),*),
            AnyBlock::B16x4(b) => b.$method($($args),*),
            AnyBlock::B16x8(b) => b.$method($($args),*),
            AnyBlock::DualRow8x4(b) => b.$method($($args),*),
            AnyBlock::DualRow8x8(b) => b.$method($($args),*),
            AnyBlock::DualRow16x4(b) => b.$method($($args),*),
            AnyBlock::DualRow16x8(b) => b.$method($($args),*),
        }
    };
}

impl AnyBlock {
    /// Get the block size (number of elements per row)
    pub fn size(&self) -> usize {
        match self {
            Self::B8x4(_) | Self::B8x8(_) => 8,
            Self::B16x4(_) | Self::B16x8(_) => 16,
            Self::DualRow8x4(_) | Self::DualRow8x8(_) => 8,
            Self::DualRow16x4(_) | Self::DualRow16x8(_) => 16,
        }
    }

    /// Get the encoding bits (4 or 8)
    pub fn bits(&self) -> u8 {
        match self {
            Self::B8x4(_) | Self::B16x4(_) => 4,
            Self::B8x8(_) | Self::B16x8(_) => 8,
            Self::DualRow8x4(_) | Self::DualRow16x4(_) => 4,
            Self::DualRow8x8(_) | Self::DualRow16x8(_) => 8,
        }
    }

    /// Get the format of this block
    pub fn format(&self) -> BlockFormat {
        match self {
            Self::B8x4(_) => BlockFormat::B8x4,
            Self::B8x8(_) => BlockFormat::B8x8,
            Self::B16x4(_) => BlockFormat::B16x4,
            Self::B16x8(_) => BlockFormat::B16x8,
            Self::DualRow8x4(_) => BlockFormat::DualRow8x4,
            Self::DualRow8x8(_) => BlockFormat::DualRow8x8,
            Self::DualRow16x4(_) => BlockFormat::DualRow16x4,
            Self::DualRow16x8(_) => BlockFormat::DualRow16x8,
        }
    }

    /// Encode values into an AnyBlock of 8x4 type
    pub fn encode_8x4(values: &[f32]) -> Self {
        Self::B8x4(UnifiedBlock::<1, Config8x4>::encode_single(values))
    }

    /// Encode values into an AnyBlock of 8x8 type
    pub fn encode_8x8(values: &[f32]) -> Self {
        Self::B8x8(UnifiedBlock::<1, Config8x8>::encode_single(values))
    }

    /// Encode values into an AnyBlock of 16x4 type
    pub fn encode_16x4(values: &[f32]) -> Self {
        Self::B16x4(UnifiedBlock::<1, Config16x4>::encode_single(values))
    }

    /// Encode values into an AnyBlock of 16x8 type
    pub fn encode_16x8(values: &[f32]) -> Self {
        Self::B16x8(UnifiedBlock::<1, Config16x8>::encode_single(values))
    }

    /// Encode two rows into an AnyBlock of DualRow8x4 type
    pub fn encode_dualrow_8x4(row0: &[f32], row1: &[f32]) -> Self {
        Self::DualRow8x4(UnifiedBlock::<2, Config8x4>::encode_dual(row0, row1))
    }

    /// Encode two rows into an AnyBlock of DualRow8x8 type
    pub fn encode_dualrow_8x8(row0: &[f32], row1: &[f32]) -> Self {
        Self::DualRow8x8(UnifiedBlock::<2, Config8x8>::encode_dual(row0, row1))
    }

    /// Encode two rows into an AnyBlock of DualRow16x4 type
    pub fn encode_dualrow_16x4(row0: &[f32], row1: &[f32]) -> Self {
        Self::DualRow16x4(UnifiedBlock::<2, Config16x4>::encode_dual(row0, row1))
    }

    /// Encode two rows into an AnyBlock of DualRow16x8 type
    pub fn encode_dualrow_16x8(row0: &[f32], row1: &[f32]) -> Self {
        Self::DualRow16x8(UnifiedBlock::<2, Config16x8>::encode_dual(row0, row1))
    }

    /// Decode a single element
    /// Note: For DualRow variants, decodes from row 0.
    pub fn decode(&self, idx: usize) -> f32 {
        any_block_dispatch!(self, decode_element, 0, idx)
    }

    /// Decode element from specific row (for DualRow support)
    /// For single-row blocks, row must be 0
    /// For dual-row blocks, row can be 0 or 1
    pub fn decode_row(&self, row: usize, idx: usize) -> f32 {
        match self {
            Self::B8x4(_) | Self::B8x8(_) | Self::B16x4(_) | Self::B16x8(_) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                self.decode(idx)
            }
            Self::DualRow8x4(_) | Self::DualRow8x8(_) | Self::DualRow16x4(_) | Self::DualRow16x8(_) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                any_block_dispatch!(self, decode_element, row, idx)
            }
        }
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        match self {
            Self::B8x4(b) => b.row_nnz(0),
            Self::B8x8(b) => b.row_nnz(0),
            Self::B16x4(b) => b.row_nnz(0),
            Self::B16x8(b) => b.row_nnz(0),
            Self::DualRow8x4(b) => b.row_nnz(0) + b.row_nnz(1),
            Self::DualRow8x8(b) => b.row_nnz(0) + b.row_nnz(1),
            Self::DualRow16x4(b) => b.row_nnz(0) + b.row_nnz(1),
            Self::DualRow16x8(b) => b.row_nnz(0) + b.row_nnz(1),
        }
    }

    /// Check if element exists
    /// Note: For DualRow variants, checks row 0 only.
    pub fn has_element(&self, idx: usize) -> bool {
        match self {
            Self::B8x4(b) => {
                if idx >= 8 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
            Self::B8x8(b) => {
                if idx >= 8 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
            Self::B16x4(b) => {
                if idx >= 16 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
            Self::B16x8(b) => {
                if idx >= 16 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
            Self::DualRow8x4(b) => {
                if idx >= 8 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
            Self::DualRow8x8(b) => {
                if idx >= 8 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
            Self::DualRow16x4(b) => {
                if idx >= 16 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
            Self::DualRow16x8(b) => {
                if idx >= 16 { return false; }
                (b.zero_map[0] >> idx) & 1 == 1
            }
        }
    }

    /// Get sign of element
    /// Note: For DualRow variants, checks row 0 only.
    pub fn sign(&self, idx: usize) -> bool {
        match self {
            Self::B8x4(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
            Self::B8x8(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
            Self::B16x4(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
            Self::B16x8(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
            Self::DualRow8x4(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
            Self::DualRow8x8(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
            Self::DualRow16x4(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
            Self::DualRow16x8(b) => {
                (b.signs[0] >> idx) & 1 == 1
            }
        }
    }

    /// Check if block is empty
    pub fn is_empty(&self) -> bool {
        any_block_dispatch!(self, is_empty)
    }

    /// Get byte size
    pub fn byte_size(&self) -> usize {
        any_block_dispatch!(self, byte_size)
    }

    /// Get scale log
    pub fn scale_log(&self) -> f16 {
        match self {
            Self::B8x4(b) => b.scale_log,
            Self::B8x8(b) => b.scale_log,
            Self::B16x4(b) => b.scale_log,
            Self::B16x8(b) => b.scale_log,
            Self::DualRow8x4(b) => b.scale_log,
            Self::DualRow8x8(b) => b.scale_log,
            Self::DualRow16x4(b) => b.scale_log,
            Self::DualRow16x8(b) => b.scale_log,
        }
    }

    /// Write to output
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        any_block_dispatch!(self, write_to, w)
    }

    /// Read from input with known format
    pub fn read_from<R: Read>(format: BlockFormat, r: &mut R) -> Result<Self> {
        match format {
            BlockFormat::B8x4 => Ok(Self::B8x4(UnifiedBlock::<1, Config8x4>::read_from(r)?)),
            BlockFormat::B8x8 => Ok(Self::B8x8(UnifiedBlock::<1, Config8x8>::read_from(r)?)),
            BlockFormat::B16x4 => Ok(Self::B16x4(UnifiedBlock::<1, Config16x4>::read_from(r)?)),
            BlockFormat::B16x8 => Ok(Self::B16x8(UnifiedBlock::<1, Config16x8>::read_from(r)?)),
            BlockFormat::DualRow8x4 => Ok(Self::DualRow8x4(UnifiedBlock::<2, Config8x4>::read_from(r)?)),
            BlockFormat::DualRow8x8 => Ok(Self::DualRow8x8(UnifiedBlock::<2, Config8x8>::read_from(r)?)),
            BlockFormat::DualRow16x4 => Ok(Self::DualRow16x4(UnifiedBlock::<2, Config16x4>::read_from(r)?)),
            BlockFormat::DualRow16x8 => Ok(Self::DualRow16x8(UnifiedBlock::<2, Config16x8>::read_from(r)?)),
        }
    }

    /// Iterator over (index, value) pairs
    /// Note: For DualRow variants, iterates row 0 only.
    pub fn iter(&self) -> Box<dyn Iterator<Item = (usize, f32)> + '_> {
        self.row_iter(0)
    }

    /// Iterator over (index, value) pairs for a specific row
    /// For single-row blocks, row must be 0
    /// For dual-row blocks, row can be 0 or 1
    pub fn row_iter(&self, row: usize) -> Box<dyn Iterator<Item = (usize, f32)> + '_> {
        match self {
            Self::B8x4(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.row_iter(0))
            }
            Self::B8x8(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.row_iter(0))
            }
            Self::B16x4(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.row_iter(0))
            }
            Self::B16x8(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.row_iter(0))
            }
            Self::DualRow8x4(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.row_iter(row))
            }
            Self::DualRow8x8(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.row_iter(row))
            }
            Self::DualRow16x4(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.row_iter(row))
            }
            Self::DualRow16x8(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.row_iter(row))
            }
        }
    }

    /// Iterator over (index, log2_magnitude, sign) tuples for a specific row
    /// For single-row blocks, row must be 0
    /// For dual-row blocks, row can be 0 or 1
    pub fn log_row_iter(&self, row: usize) -> Box<dyn Iterator<Item = (usize, f32, u8)> + '_> {
        match self {
            Self::B8x4(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.log_row_iter(0))
            }
            Self::B8x8(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.log_row_iter(0))
            }
            Self::B16x4(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.log_row_iter(0))
            }
            Self::B16x8(b) => {
                assert_eq!(row, 0, "Single-row block only supports row 0");
                Box::new(b.log_row_iter(0))
            }
            Self::DualRow8x4(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.log_row_iter(row))
            }
            Self::DualRow8x8(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.log_row_iter(row))
            }
            Self::DualRow16x4(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.log_row_iter(row))
            }
            Self::DualRow16x8(b) => {
                assert!(row < 2, "DualRow block only supports rows 0 and 1");
                Box::new(b.log_row_iter(row))
            }
        }
    }

    /// Iterator over (index, log2_magnitude, sign) tuples
    /// Note: For DualRow variants, iterates row 0 only.
    pub fn log_iter(&self) -> Box<dyn Iterator<Item = (usize, f32, u8)> + '_> {
        self.log_row_iter(0)
    }
}
