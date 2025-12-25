//! Encoding helpers for GraphMatrix.

use crate::blocks::{AnyBlock, BlockFormat};
use rayon::prelude::*;

/// Encode a single-row block using the given format.
pub(crate) fn encode_single_row_block(format: BlockFormat, data: &[f32]) -> AnyBlock {
    match format {
        BlockFormat::B8x4 => AnyBlock::encode_8x4(data),
        BlockFormat::B8x8 => AnyBlock::encode_8x8(data),
        BlockFormat::B16x4 => AnyBlock::encode_16x4(data),
        BlockFormat::B16x8 => AnyBlock::encode_16x8(data),
        _ => unreachable!("encode_single_row_block called with dual-row format"),
    }
}

/// Encode a dual-row block using the given format.
pub(crate) fn encode_dual_row_block(format: BlockFormat, row0: &[f32], row1: &[f32]) -> AnyBlock {
    match format {
        BlockFormat::DualRow8x4 => AnyBlock::encode_dualrow_8x4(row0, row1),
        BlockFormat::DualRow8x8 => AnyBlock::encode_dualrow_8x8(row0, row1),
        BlockFormat::DualRow16x4 => AnyBlock::encode_dualrow_16x4(row0, row1),
        BlockFormat::DualRow16x8 => AnyBlock::encode_dualrow_16x8(row0, row1),
        _ => unreachable!("encode_dual_row_block called with single-row format"),
    }
}

/// Copy a row segment into a buffer, zero-padding only the tail if needed.
#[inline]
pub(crate) fn copy_row_segment(
    data: &[f32],
    row: usize,
    cols: usize,
    block_col_start: usize,
    block_len: usize,
    buffer: &mut [f32],
) {
    if row * cols < data.len() {
        let data_start = row * cols + block_col_start;
        let data_end = data_start + block_len;
        buffer[..block_len].copy_from_slice(&data[data_start..data_end]);
        // Only zero-pad the tail if block_len < buffer.len()
        if block_len < buffer.len() {
            buffer[block_len..].fill(0.0);
        }
    } else {
        buffer.fill(0.0);
    }
}

/// Encode dual-row format blocks in parallel.
///
/// Processes pairs of rows simultaneously, creating blocks that contain data from two rows.
pub(crate) fn encode_dual_row_blocks(
    data: &[f32],
    rows: usize,
    cols: usize,
    format: BlockFormat,
    block_size: usize,
    blocks_per_row: usize,
) -> Vec<AnyBlock> {
    let row_pairs = rows.div_ceil(2);
    (0..row_pairs)
        .into_par_iter()
        .flat_map(|row_pair| {
            let row0 = row_pair * 2;
            let row1 = row0 + 1;

            (0..blocks_per_row)
                .map(move |block_idx| {
                    let block_col_start = block_idx * block_size;
                    let block_len = (block_col_start + block_size).min(cols) - block_col_start;

                    let mut buf0 = [0.0f32; 16];
                    let mut buf1 = [0.0f32; 16];

                    copy_row_segment(
                        data,
                        row0,
                        cols,
                        block_col_start,
                        block_len,
                        &mut buf0[..block_size],
                    );
                    if row1 < rows {
                        copy_row_segment(
                            data,
                            row1,
                            cols,
                            block_col_start,
                            block_len,
                            &mut buf1[..block_size],
                        );
                    } else {
                        buf1[..block_size].fill(0.0);
                    }

                    encode_dual_row_block(format, &buf0[..block_size], &buf1[..block_size])
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Encode single-row format blocks in parallel.
///
/// Processes one row at a time, creating independent blocks for each row.
pub(crate) fn encode_single_row_blocks(
    data: &[f32],
    rows: usize,
    cols: usize,
    format: BlockFormat,
    block_size: usize,
    blocks_per_row: usize,
) -> Vec<AnyBlock> {
    (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            (0..blocks_per_row)
                .map(move |block_idx| {
                    let block_col_start = block_idx * block_size;
                    let block_len = (block_col_start + block_size).min(cols) - block_col_start;

                    let mut buf = [0.0f32; 16];
                    copy_row_segment(
                        data,
                        row,
                        cols,
                        block_col_start,
                        block_len,
                        &mut buf[..block_size],
                    );
                    encode_single_row_block(format, &buf[..block_size])
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
