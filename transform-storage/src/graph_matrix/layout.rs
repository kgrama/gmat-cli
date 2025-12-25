//! Block layout helpers for GraphMatrix.

use crate::blocks::BlockFormat;

/// Static helper: calculate blocks per row given cols and block_size.
#[inline]
pub(crate) fn calc_blocks_per_row(cols: usize, block_size: usize) -> usize {
    cols.div_ceil(block_size)
}

/// Calculate expected block count for given shape and format.
pub(crate) fn expected_block_count(rows: usize, cols: usize, format: &BlockFormat) -> usize {
    let block_size = format.block_size();
    let blocks_per_row = calc_blocks_per_row(cols, block_size);
    if format.is_dual_row() {
        let row_pairs = rows.div_ceil(2);
        row_pairs * blocks_per_row
    } else {
        rows * blocks_per_row
    }
}
