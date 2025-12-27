//! Utility helpers for GraphMatrix: block layout and statistics.

use crate::blocks::BlockFormat;

// ============================================================================
// Block layout helpers
// ============================================================================

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

// ============================================================================
// Statistics helpers
// ============================================================================

/// Compute median of a float slice (returns 0.0 for empty).
pub(crate) fn compute_median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}
