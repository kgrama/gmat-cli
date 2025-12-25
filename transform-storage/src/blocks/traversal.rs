//! Block traversal utilities for iterating over AnyBlock slices.
//!
//! Provides unified iteration patterns for GraphMatrix and similar structures
//! that store blocks in row-major or column-major order.

use super::{AnyBlock, BlockFormat};

/// Axis for block traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// Row-major traversal
    Row,
    /// Column-major traversal (requires column index)
    Col,
}

/// Configuration for block traversal
#[derive(Debug, Clone, Copy)]
pub struct TraversalConfig {
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
    /// Block size (elements per block row)
    pub block_size: usize,
    /// Whether blocks use dual-row format
    pub is_dual_row: bool,
}

impl TraversalConfig {
    /// Create a new traversal configuration
    pub fn new(shape: (usize, usize), block_size: usize, is_dual_row: bool) -> Self {
        Self {
            shape,
            block_size,
            is_dual_row,
        }
    }

    /// Calculate blocks per row (or per column for col-major)
    #[inline]
    pub fn blocks_per_dim(&self, axis: Axis) -> usize {
        let dim_size = match axis {
            Axis::Row => self.shape.1, // cols
            Axis::Col => self.shape.0, // rows
        };
        dim_size.div_ceil(self.block_size)
    }

    /// Calculate block offset for a given index
    #[inline]
    pub fn block_offset(&self, index: usize, axis: Axis) -> usize {
        let blocks_per_dim = self.blocks_per_dim(axis);
        if self.is_dual_row && axis == Axis::Row {
            (index / 2) * blocks_per_dim
        } else {
            index * blocks_per_dim
        }
    }

    /// For dual-row formats, get which sub-row (0 or 1) within the block
    #[inline]
    pub fn row_in_block(&self, row: usize) -> usize {
        if self.is_dual_row {
            row % 2
        } else {
            0
        }
    }

    /// Get the dimension size for bounds checking
    #[inline]
    pub fn dim_size(&self, axis: Axis) -> usize {
        match axis {
            Axis::Row => self.shape.1, // cols
            Axis::Col => self.shape.0, // rows
        }
    }
}

/// Block traversal for iterating over a slice of AnyBlocks.
///
/// Provides unified iteration patterns that handle:
/// - Block offset calculation
/// - Dual-row format handling
/// - Local-to-global index conversion
/// - Bounds checking for partial blocks
#[derive(Debug, Clone, Copy)]
pub struct BlockTraversal<'a> {
    /// The blocks to traverse
    blocks: &'a [AnyBlock],
    /// Traversal configuration
    config: TraversalConfig,
    /// Which axis we're traversing
    axis: Axis,
    /// The row or column index to traverse
    index: usize,
}

impl<'a> BlockTraversal<'a> {
    /// Create a new block traversal
    pub fn new(blocks: &'a [AnyBlock], config: TraversalConfig, axis: Axis, index: usize) -> Self {
        Self {
            blocks,
            config,
            axis,
            index,
        }
    }

    /// Create a row traversal
    pub fn row(blocks: &'a [AnyBlock], config: TraversalConfig, row: usize) -> Self {
        Self::new(blocks, config, Axis::Row, row)
    }

    /// Create a column traversal
    pub fn col(blocks: &'a [AnyBlock], config: TraversalConfig, col: usize) -> Self {
        Self::new(blocks, config, Axis::Col, col)
    }

    /// Get the block offset for this traversal
    #[inline]
    fn block_offset(&self) -> usize {
        self.config.block_offset(self.index, self.axis)
    }

    /// Get the number of blocks to iterate
    #[inline]
    fn blocks_count(&self) -> usize {
        self.config.blocks_per_dim(self.axis)
    }

    /// Get the dimension size for bounds checking
    #[inline]
    fn dim_size(&self) -> usize {
        self.config.dim_size(self.axis)
    }

    /// Get the row within block (for dual-row handling)
    #[inline]
    fn row_in_block(&self) -> usize {
        if self.axis == Axis::Row {
            self.config.row_in_block(self.index)
        } else {
            0 // Column traversal always uses row 0 of transposed blocks
        }
    }

    /// Iterate over (block_idx, block_ref, block_start_offset)
    pub fn block_iter(&self) -> impl Iterator<Item = (usize, &'a AnyBlock, usize)> {
        let block_offset = self.block_offset();
        let blocks_count = self.blocks_count();
        let block_size = self.config.block_size;
        let blocks = self.blocks;

        (0..blocks_count).map(move |block_idx| {
            let block = &blocks[block_offset + block_idx];
            let block_start = block_idx * block_size;
            (block_idx, block, block_start)
        })
    }

    /// Iterate over (global_idx, value) pairs for non-zero elements
    pub fn value_iter(&self) -> impl Iterator<Item = (usize, f32)> + 'a {
        let dim_size = self.dim_size();
        let row_in_block = self.row_in_block();
        let is_dual_row = self.config.is_dual_row;
        let axis = self.axis;
        let block_offset = self.block_offset();
        let blocks_count = self.blocks_count();
        let block_size = self.config.block_size;
        let blocks = self.blocks;

        (0..blocks_count).flat_map(move |block_idx| {
            let block = &blocks[block_offset + block_idx];
            let block_start = block_idx * block_size;

            // Use row_iter for dual-row row traversal, iter() otherwise
            let iter: Box<dyn Iterator<Item = (usize, f32)> + '_> =
                if is_dual_row && axis == Axis::Row {
                    block.row_iter(row_in_block)
                } else {
                    block.iter()
                };

            iter.filter_map(move |(local_idx, value)| {
                let global_idx = block_start + local_idx;
                if global_idx < dim_size {
                    Some((global_idx, value))
                } else {
                    None
                }
            })
        })
    }

    /// Iterate over (global_idx, log2_magnitude, sign) for non-zero elements
    pub fn log_iter(&self) -> impl Iterator<Item = (usize, f32, u8)> + 'a {
        let dim_size = self.dim_size();
        let row_in_block = self.row_in_block();
        let is_dual_row = self.config.is_dual_row;
        let axis = self.axis;
        let block_offset = self.block_offset();
        let blocks_count = self.blocks_count();
        let block_size = self.config.block_size;
        let blocks = self.blocks;

        (0..blocks_count).flat_map(move |block_idx| {
            let block = &blocks[block_offset + block_idx];
            let block_start = block_idx * block_size;

            // Use log_row_iter for dual-row row traversal, log_iter() otherwise
            let iter: Box<dyn Iterator<Item = (usize, f32, u8)> + '_> =
                if is_dual_row && axis == Axis::Row {
                    block.log_row_iter(row_in_block)
                } else {
                    block.log_iter()
                };

            iter.filter_map(move |(local_idx, log_mag, sign)| {
                let global_idx = block_start + local_idx;
                if global_idx < dim_size {
                    Some((global_idx, log_mag, sign))
                } else {
                    None
                }
            })
        })
    }

    /// Collect log2 magnitudes into a vector (useful for statistics)
    pub fn collect_log_magnitudes(&self) -> Vec<f32> {
        self.log_iter().map(|(_, log_mag, _)| log_mag).collect()
    }

    /// Iterate over ALL elements (including zeros) in dense order.
    ///
    /// Returns an iterator of (global_idx, value) pairs for every element
    /// in the row/column, including zeros. Useful for dense decoding.
    pub fn dense_iter(&self) -> impl Iterator<Item = (usize, f32)> + 'a {
        let dim_size = self.dim_size();
        let row_in_block = self.row_in_block();
        let block_offset = self.block_offset();
        let blocks_count = self.blocks_count();
        let block_size = self.config.block_size;
        let blocks = self.blocks;

        (0..blocks_count).flat_map(move |block_idx| {
            let block = &blocks[block_offset + block_idx];
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(dim_size);
            let local_count = block_end - block_start;

            (0..local_count).map(move |local_idx| {
                let global_idx = block_start + local_idx;
                let value = block.decode_row(row_in_block, local_idx);
                (global_idx, value)
            })
        })
    }

    /// Decode all elements into a pre-allocated buffer.
    ///
    /// This is more efficient than dense_iter() when you need all values
    /// in a contiguous buffer, as it avoids iterator overhead.
    ///
    /// # Arguments
    /// - `buffer`: Pre-allocated buffer of size >= dim_size
    ///
    /// # Panics
    /// Panics if buffer is smaller than dim_size
    pub fn decode_to_buffer(&self, buffer: &mut [f32]) {
        let dim_size = self.dim_size();
        assert!(
            buffer.len() >= dim_size,
            "Buffer size {} is smaller than dimension size {}",
            buffer.len(),
            dim_size
        );

        let row_in_block = self.row_in_block();
        let block_offset = self.block_offset();
        let blocks_count = self.blocks_count();
        let block_size = self.config.block_size;

        for block_idx in 0..blocks_count {
            let block = &self.blocks[block_offset + block_idx];
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(dim_size);

            for local_idx in 0..(block_end - block_start) {
                buffer[block_start + local_idx] = block.decode_row(row_in_block, local_idx);
            }
        }
    }
}

/// Helper trait for types that can create BlockTraversal instances
pub trait BlockTraversable {
    /// Get the block slice for traversal
    fn blocks(&self) -> &[AnyBlock];

    /// Get the traversal configuration
    fn traversal_config(&self) -> TraversalConfig;

    /// Create a row traversal
    fn row_traversal(&self, row: usize) -> BlockTraversal<'_> {
        BlockTraversal::row(self.blocks(), self.traversal_config(), row)
    }

    /// Create a column traversal (requires column index to be built)
    fn col_traversal(&self, col: usize) -> BlockTraversal<'_> {
        BlockTraversal::col(self.blocks(), self.traversal_config(), col)
    }
}

/// Transpose a tile of row blocks into column blocks.
///
/// Takes `block_size` row blocks from consecutive rows (all at the same block_idx)
/// and outputs `block_size` column blocks. Works entirely in log2 space without
/// converting to f32.
///
/// # Arguments
/// - `row_blocks`: References to `block_size` row blocks from consecutive rows
/// - `row_in_block`: For dual-row formats, which sub-row (0 or 1) to read from each block
/// - `out`: Output buffer for column blocks (must be len >= block_size)
/// - `scratch`: Scratch buffer for log2 data (must be len >= block_size * block_size)
pub fn transpose_tile(
    row_blocks: &[&AnyBlock],
    row_in_block: &[usize],
    out: &mut [AnyBlock],
    scratch: &mut [(f32, u8)],
) {
    assert!(!row_blocks.is_empty());
    let format = row_blocks[0].format();
    let block_size = format.block_size();
    assert_eq!(row_blocks.len(), block_size);
    assert_eq!(row_in_block.len(), block_size);
    assert!(out.len() >= block_size);
    assert!(scratch.len() >= block_size * block_size);

    // scratch is laid out as [block_size][block_size] in column-major order
    // scratch[col * block_size + row] = (log2_mag, sign) for element at (row, col)

    // Initialize scratch to zeros (NEG_INFINITY means zero)
    for s in scratch.iter_mut().take(block_size * block_size) {
        *s = (f32::NEG_INFINITY, 0);
    }

    // Gather log2 data from each row block
    for (row_idx, (block, &sub_row)) in row_blocks.iter().zip(row_in_block.iter()).enumerate() {
        for (local_col, log2_mag, sign) in block.log_row_iter(sub_row) {
            if local_col < block_size {
                scratch[local_col * block_size + row_idx] = (log2_mag, sign);
            }
        }
    }

    // Encode each column as a block
    for (col, out_block) in out.iter_mut().enumerate().take(block_size) {
        let col_start = col * block_size;
        *out_block = encode_from_log2(&scratch[col_start..col_start + block_size], format);
    }
}

/// Encode a block directly from log2 magnitude and sign data.
///
/// Avoids f32 conversion by working directly with log2 values:
/// - Finds min log2 as scale_log
/// - Computes offsets as (element_log2 - scale_log)
/// - Encodes offsets with octave shift as needed
fn encode_from_log2(elements: &[(f32, u8)], format: BlockFormat) -> AnyBlock {
    use half::f16;

    // Find min scale (min log2 magnitude among non-zero elements)
    let min_log2 = elements
        .iter()
        .filter(|(log2, _)| *log2 != f32::NEG_INFINITY)
        .map(|(log2, _)| *log2)
        .fold(f32::INFINITY, f32::min);

    if min_log2 == f32::INFINITY {
        // All zeros
        return AnyBlock::new_empty(format);
    }

    let scale_log = f16::from_f32(min_log2);
    let scale_f32 = scale_log.to_f32();

    // Build zero_map, signs, offsets
    let mut zero_map: u16 = 0;
    let mut signs: u16 = 0;
    let mut offsets = [0.0f32; 16];

    for (i, (log2, sign)) in elements.iter().enumerate() {
        if *log2 != f32::NEG_INFINITY {
            zero_map |= 1 << i;
            if *sign == 1 {
                signs |= 1 << i;
            }
            offsets[i] = *log2 - scale_f32;
        }
    }

    AnyBlock::encode_from_log_components(format, scale_log, zero_map, signs, &offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traversal_config() {
        let config = TraversalConfig::new((100, 200), 16, false);
        assert_eq!(config.blocks_per_dim(Axis::Row), 13); // ceil(200/16)
        assert_eq!(config.blocks_per_dim(Axis::Col), 7); // ceil(100/16)
        assert_eq!(config.block_offset(5, Axis::Row), 5 * 13);
        assert_eq!(config.row_in_block(5), 0); // not dual-row
    }

    #[test]
    fn test_dual_row_config() {
        let config = TraversalConfig::new((100, 200), 16, true);
        assert_eq!(config.row_in_block(0), 0);
        assert_eq!(config.row_in_block(1), 1);
        assert_eq!(config.row_in_block(2), 0);
        assert_eq!(config.row_in_block(3), 1);
        // Block offset should be halved for dual-row
        assert_eq!(config.block_offset(0, Axis::Row), 0);
        assert_eq!(config.block_offset(1, Axis::Row), 0); // same block pair
        assert_eq!(config.block_offset(2, Axis::Row), 13); // next block pair
    }
}
