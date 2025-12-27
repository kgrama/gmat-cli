//! Type-erased block enum that can hold any block configuration

use super::configs::{BlockConfig, Config16x4, Config16x8, Config8x4, Config8x8};
use super::unified_block::{EncodeHelper, UnifiedBlock};
use half::f16;
use std::io::{Read, Result, Write};

/// Dispatch macro for matching all AnyBlock variants with a uniform operation.
///
/// Usage:
/// ```ignore
/// match_block!(self, b => { b.some_method() })
/// ```
macro_rules! match_block {
    ($self:expr, $b:ident => $body:expr) => {
        match $self {
            AnyBlock::B8x4($b) => $body,
            AnyBlock::B8x8($b) => $body,
            AnyBlock::B16x4($b) => $body,
            AnyBlock::B16x8($b) => $body,
            AnyBlock::DualRow8x4($b) => $body,
            AnyBlock::DualRow8x8($b) => $body,
            AnyBlock::DualRow16x4($b) => $body,
            AnyBlock::DualRow16x8($b) => $body,
        }
    };
}

/// Internal trait for unified block operations across all configurations.
/// This enables generic functions to work with any UnifiedBlock variant.
pub(crate) trait UnifiedBlockOps {
    fn scale_log(&self) -> f16;
    fn decode_element(&self, row: usize, idx: usize) -> f32;
    fn row_nnz(&self, row: usize) -> usize;
    fn is_empty(&self) -> bool;
    fn byte_size(&self) -> usize;
    fn has_element_at(&self, row: usize, idx: usize) -> bool;
    fn sign_at(&self, row: usize, idx: usize) -> bool;
    fn importance_stats(&self) -> (usize, usize);
}

impl<const ROWS: usize, C: BlockConfig + EncodeHelper> UnifiedBlockOps for UnifiedBlock<ROWS, C> {
    fn scale_log(&self) -> f16 {
        self.scale_log
    }
    fn decode_element(&self, row: usize, idx: usize) -> f32 {
        UnifiedBlock::decode_element(self, row, idx)
    }
    fn row_nnz(&self, row: usize) -> usize {
        UnifiedBlock::row_nnz(self, row)
    }
    fn is_empty(&self) -> bool {
        UnifiedBlock::is_empty(self)
    }
    fn byte_size(&self) -> usize {
        UnifiedBlock::byte_size(self)
    }
    fn has_element_at(&self, row: usize, idx: usize) -> bool {
        let mask: u64 = self.zero_map[row].into();
        (mask >> idx) & 1 == 1
    }
    fn sign_at(&self, row: usize, idx: usize) -> bool {
        let mask: u64 = self.signs[row].into();
        (mask >> idx) & 1 == 1
    }
    fn importance_stats(&self) -> (usize, usize) {
        // Aggregate across all rows
        let mut total_shifted = 0usize;
        let mut total_nnz = 0usize;
        for row in 0..ROWS {
            let zero_map: u64 = self.zero_map[row].into();
            let octave_shift: u64 = self.octave_shift[row].into();
            total_nnz += zero_map.count_ones() as usize;
            total_shifted += (octave_shift & zero_map).count_ones() as usize;
        }
        (total_shifted, total_nnz)
    }
}

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
        matches!(
            self,
            Self::DualRow8x4 | Self::DualRow8x8 | Self::DualRow16x4 | Self::DualRow16x8
        )
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

impl AnyBlock {
    /// Get a reference to the inner block as a trait object
    fn inner(&self) -> &dyn UnifiedBlockOps {
        match_block!(self, b => b)
    }

    /// Get the block size (number of elements per row)
    pub fn size(&self) -> usize {
        self.format().block_size()
    }

    /// Get the encoding bits (4 or 8)
    pub fn bits(&self) -> u8 {
        match self {
            Self::B8x4(_) | Self::B16x4(_) | Self::DualRow8x4(_) | Self::DualRow16x4(_) => 4,
            Self::B8x8(_) | Self::B16x8(_) | Self::DualRow8x8(_) | Self::DualRow16x8(_) => 8,
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
        self.inner().decode_element(0, idx)
    }

    /// Decode element from specific row (for DualRow support)
    /// For single-row blocks, row must be 0
    /// For dual-row blocks, row can be 0 or 1
    pub fn decode_row(&self, row: usize, idx: usize) -> f32 {
        let max_row = if self.format().is_dual_row() { 2 } else { 1 };
        assert!(
            row < max_row,
            "Row {} out of bounds for block with {} rows",
            row,
            max_row
        );
        self.inner().decode_element(row, idx)
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        let inner = self.inner();
        if self.format().is_dual_row() {
            inner.row_nnz(0) + inner.row_nnz(1)
        } else {
            inner.row_nnz(0)
        }
    }

    /// Check if element exists
    /// Note: For DualRow variants, checks row 0 only.
    pub fn has_element(&self, idx: usize) -> bool {
        if idx >= self.size() {
            return false;
        }
        self.inner().has_element_at(0, idx)
    }

    /// Get sign of element
    /// Note: For DualRow variants, checks row 0 only.
    pub fn sign(&self, idx: usize) -> bool {
        self.inner().sign_at(0, idx)
    }

    /// Check if block is empty
    pub fn is_empty(&self) -> bool {
        self.inner().is_empty()
    }

    /// Get byte size
    pub fn byte_size(&self) -> usize {
        self.inner().byte_size()
    }

    /// Get scale log
    pub fn scale_log(&self) -> f16 {
        self.inner().scale_log()
    }

    /// Compute importance stats: (octave_shift_count, nnz)
    /// Returns ratio of elements using octave shift to total non-zero elements.
    pub fn importance_stats(&self) -> (usize, usize) {
        self.inner().importance_stats()
    }

    /// Write to output
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        match_block!(self, b => b.write_to(w))
    }

    /// Read from input with known format
    pub fn read_from<R: Read>(format: BlockFormat, r: &mut R) -> Result<Self> {
        match format {
            BlockFormat::B8x4 => Ok(Self::B8x4(UnifiedBlock::<1, Config8x4>::read_from(r)?)),
            BlockFormat::B8x8 => Ok(Self::B8x8(UnifiedBlock::<1, Config8x8>::read_from(r)?)),
            BlockFormat::B16x4 => Ok(Self::B16x4(UnifiedBlock::<1, Config16x4>::read_from(r)?)),
            BlockFormat::B16x8 => Ok(Self::B16x8(UnifiedBlock::<1, Config16x8>::read_from(r)?)),
            BlockFormat::DualRow8x4 => Ok(Self::DualRow8x4(
                UnifiedBlock::<2, Config8x4>::read_from(r)?,
            )),
            BlockFormat::DualRow8x8 => Ok(Self::DualRow8x8(
                UnifiedBlock::<2, Config8x8>::read_from(r)?,
            )),
            BlockFormat::DualRow16x4 => Ok(Self::DualRow16x4(
                UnifiedBlock::<2, Config16x4>::read_from(r)?,
            )),
            BlockFormat::DualRow16x8 => Ok(Self::DualRow16x8(
                UnifiedBlock::<2, Config16x8>::read_from(r)?,
            )),
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
        let max_row = if self.format().is_dual_row() { 2 } else { 1 };
        assert!(
            row < max_row,
            "Row {} out of bounds for block with {} rows",
            row,
            max_row
        );
        match_block!(self, b => Box::new(b.row_iter(row)))
    }

    /// Iterator over (index, log2_magnitude, sign) tuples for a specific row
    /// For single-row blocks, row must be 0
    /// For dual-row blocks, row can be 0 or 1
    pub fn log_row_iter(&self, row: usize) -> Box<dyn Iterator<Item = (usize, f32, u8)> + '_> {
        let max_row = if self.format().is_dual_row() { 2 } else { 1 };
        assert!(
            row < max_row,
            "Row {} out of bounds for block with {} rows",
            row,
            max_row
        );
        match_block!(self, b => Box::new(b.log_row_iter(row)))
    }

    /// Iterator over (index, log2_magnitude, sign) tuples
    /// Note: For DualRow variants, iterates row 0 only.
    pub fn log_iter(&self) -> Box<dyn Iterator<Item = (usize, f32, u8)> + '_> {
        self.log_row_iter(0)
    }

    /// Visit each log element without Box allocation.
    /// Closure receives (idx, log2_mag, sign) for each non-zero element.
    /// Note: For DualRow variants, iterates row 0 only.
    #[inline]
    pub fn for_each_log<F>(&self, mut f: F)
    where
        F: FnMut(usize, f32, u8),
    {
        match_block!(self, b => {
            for (idx, log2_mag, sign) in b.log_row_iter(0) {
                f(idx, log2_mag, sign);
            }
        })
    }

    /// Visit each log element for a specific row (for DualRow blocks).
    #[inline]
    pub fn for_each_log_row<F>(&self, row: usize, mut f: F)
    where
        F: FnMut(usize, f32, u8),
    {
        match_block!(self, b => {
            for (idx, log2_mag, sign) in b.log_row_iter(row) {
                f(idx, log2_mag, sign);
            }
        })
    }

    /// Get raw e1m7 magnitude data for LUT-based quantization.
    /// Returns (magnitudes, shift_map, signs, scale_log) for direct LUT indexing.
    ///
    /// For each non-zero element at index i:
    /// - lut_index = magnitudes[i] + (256 if shift_map bit i is set)
    /// - sign_bit = (signs >> i) & 1
    /// - Base offset for LUT = scale_log - d.log2()
    ///
    /// Only available for 8-bit encodings. Returns None for 4-bit encodings.
    /// For DualRow variants, returns row 0 data.
    pub fn raw_e1m7_data(&self) -> Option<(&[u8], u16, u16, f32)> {
        match self {
            // 8-bit variants - supported
            Self::B8x8(b) => {
                let shift_map: u64 = b.octave_shift[0].into();
                let signs: u64 = b.signs[0].into();
                Some((
                    &b.magnitudes[0][..8],
                    shift_map as u16,
                    signs as u16,
                    b.scale_log.to_f32(),
                ))
            }
            Self::B16x8(b) => {
                let shift_map: u64 = b.octave_shift[0].into();
                let signs: u64 = b.signs[0].into();
                Some((
                    &b.magnitudes[0][..16],
                    shift_map as u16,
                    signs as u16,
                    b.scale_log.to_f32(),
                ))
            }
            Self::DualRow8x8(b) => {
                let shift_map: u64 = b.octave_shift[0].into();
                let signs: u64 = b.signs[0].into();
                Some((
                    &b.magnitudes[0][..8],
                    shift_map as u16,
                    signs as u16,
                    b.scale_log.to_f32(),
                ))
            }
            Self::DualRow16x8(b) => {
                let shift_map: u64 = b.octave_shift[0].into();
                let signs: u64 = b.signs[0].into();
                Some((
                    &b.magnitudes[0][..16],
                    shift_map as u16,
                    signs as u16,
                    b.scale_log.to_f32(),
                ))
            }
            // 4-bit variants - not supported
            _ => None,
        }
    }

    /// Create an empty block of the given format
    pub fn new_empty(format: BlockFormat) -> Self {
        match format {
            BlockFormat::B8x4 => Self::B8x4(UnifiedBlock::new_empty()),
            BlockFormat::B8x8 => Self::B8x8(UnifiedBlock::new_empty()),
            BlockFormat::B16x4 => Self::B16x4(UnifiedBlock::new_empty()),
            BlockFormat::B16x8 => Self::B16x8(UnifiedBlock::new_empty()),
            BlockFormat::DualRow8x4 => Self::DualRow8x4(UnifiedBlock::new_empty()),
            BlockFormat::DualRow8x8 => Self::DualRow8x8(UnifiedBlock::new_empty()),
            BlockFormat::DualRow16x4 => Self::DualRow16x4(UnifiedBlock::new_empty()),
            BlockFormat::DualRow16x8 => Self::DualRow16x8(UnifiedBlock::new_empty()),
        }
    }

    /// Encode a block from log2 components (scale, zero_map, signs, offsets).
    ///
    /// This encodes directly from log-space data without f32 magnitude conversion.
    /// Offsets are relative to scale_log: actual_log2 = scale_log + offset
    pub fn encode_from_log_components(
        format: BlockFormat,
        scale_log: half::f16,
        zero_map: u16,
        signs: u16,
        offsets: &[f32],
    ) -> Self {
        use super::block::set_packed_nibble;

        if zero_map == 0 {
            return Self::new_empty(format);
        }

        // Determine encoding parameters based on format
        let (shift_threshold, shift_amount, use_4bit, size) = match format {
            BlockFormat::B8x4 | BlockFormat::DualRow8x4 => (8.0, 8.0, true, 8),
            BlockFormat::B16x4 | BlockFormat::DualRow16x4 => (8.0, 8.0, true, 16),
            BlockFormat::B8x8 | BlockFormat::DualRow8x8 => (128.0, 128.0, false, 8),
            BlockFormat::B16x8 | BlockFormat::DualRow16x8 => (128.0, 128.0, false, 16),
        };

        let mut magnitudes = [0u8; 16];
        let mut octave_shift: u16 = 0;

        for i in 0..size {
            if (zero_map >> i) & 1 == 1 {
                let offset = offsets[i];
                let (stored_offset, needs_shift) = if offset >= shift_threshold {
                    (offset - shift_amount, true)
                } else {
                    (offset, false)
                };

                if needs_shift {
                    octave_shift |= 1 << i;
                }

                let encoded = if use_4bit {
                    // E0M4 encoding
                    (stored_offset.round() as u8).min(15)
                } else {
                    // E1M7 encoding
                    stored_offset.round() as u8
                };

                if use_4bit {
                    set_packed_nibble(&mut magnitudes, i, encoded);
                } else {
                    magnitudes[i] = encoded;
                }
            }
        }

        match format {
            BlockFormat::B8x4 => Self::B8x4(UnifiedBlock::from_components(
                scale_log,
                [zero_map as u8, 0],
                [signs as u8, 0],
                [octave_shift as u8, 0],
                [magnitudes, [0; 16]],
            )),
            BlockFormat::B8x8 => Self::B8x8(UnifiedBlock::from_components(
                scale_log,
                [zero_map as u8, 0],
                [signs as u8, 0],
                [octave_shift as u8, 0],
                [magnitudes, [0; 16]],
            )),
            BlockFormat::B16x4 => Self::B16x4(UnifiedBlock::from_components(
                scale_log,
                [zero_map, 0],
                [signs, 0],
                [octave_shift, 0],
                [magnitudes, [0; 16]],
            )),
            BlockFormat::B16x8 => Self::B16x8(UnifiedBlock::from_components(
                scale_log,
                [zero_map, 0],
                [signs, 0],
                [octave_shift, 0],
                [magnitudes, [0; 16]],
            )),
            BlockFormat::DualRow8x4 => Self::DualRow8x4(UnifiedBlock::from_components(
                scale_log,
                [zero_map as u8, 0],
                [signs as u8, 0],
                [octave_shift as u8, 0],
                [magnitudes, [0; 16]],
            )),
            BlockFormat::DualRow8x8 => Self::DualRow8x8(UnifiedBlock::from_components(
                scale_log,
                [zero_map as u8, 0],
                [signs as u8, 0],
                [octave_shift as u8, 0],
                [magnitudes, [0; 16]],
            )),
            BlockFormat::DualRow16x4 => Self::DualRow16x4(UnifiedBlock::from_components(
                scale_log,
                [zero_map, 0],
                [signs, 0],
                [octave_shift, 0],
                [magnitudes, [0; 16]],
            )),
            BlockFormat::DualRow16x8 => Self::DualRow16x8(UnifiedBlock::from_components(
                scale_log,
                [zero_map, 0],
                [signs, 0],
                [octave_shift, 0],
                [magnitudes, [0; 16]],
            )),
        }
    }
}
