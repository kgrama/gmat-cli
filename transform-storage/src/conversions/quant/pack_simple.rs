//! Simple packed quantization (2×i4 → u8, i8 → u8).

use crate::graph_matrix::GraphMatrix;
use candle_core::{Device, Tensor, Result};

use super::pack_none::quantize_unpacked;
use super::types::{PackFormat, QuantDType, QuantParams, QuantizedTensors};

/// Quantize with simple packing (no grouping).
pub fn quantize_simple_packed(
    matrix: &GraphMatrix,
    dtype: QuantDType,
    clip_percentile: f32,
    log2_center: f32,
    log2_range: f32,
    nnz: usize,
    device: &Device,
) -> Result<QuantizedTensors> {
    let (rows, cols) = matrix.shape();

    // Compute effective max from log2_center + half range
    let half_range = log2_range / 2.0;
    let effective_max_log = if clip_percentile > 0.0 {
        log2_center + half_range * (1.0 - clip_percentile)
    } else {
        log2_center + half_range
    };
    let effective_max_abs = f32::exp2(effective_max_log);
    let scale = effective_max_abs / dtype.int_max() as f32;

    let weights = match dtype {
        QuantDType::I4 => {
            // Pack 2 i4 values per u8: low nibble = even idx, high nibble = odd idx
            let packed_cols = cols.div_ceil(2);
            let mut data = vec![0u8; rows * packed_cols];

            for row in 0..rows {
                let mut row_vals = vec![0i8; cols];
                for (col, value) in matrix.row_iter(row) {
                    let q = (value / scale).round().clamp(-8.0, 7.0) as i8;
                    row_vals[col] = q;
                }

                for col_pair in 0..packed_cols {
                    let idx0 = col_pair * 2;
                    let idx1 = idx0 + 1;
                    let v0 = (row_vals.get(idx0).copied().unwrap_or(0) & 0x0F) as u8;
                    let v1 = (row_vals.get(idx1).copied().unwrap_or(0) & 0x0F) as u8;
                    data[row * packed_cols + col_pair] = v0 | (v1 << 4);
                }
            }
            Tensor::from_vec(data, (rows, packed_cols), device)?
        }
        QuantDType::I8 => {
            // i8 stored as u8 (reinterpret)
            let mut data = vec![0u8; rows * cols];
            for row in 0..rows {
                for (col, value) in matrix.row_iter(row) {
                    let q = (value / scale).round().clamp(-128.0, 127.0) as i8;
                    data[row * cols + col] = q as u8;
                }
            }
            Tensor::from_vec(data, (rows, cols), device)?
        }
        _ => {
            // Fall back to unpacked for other types
            return quantize_unpacked(matrix, dtype, clip_percentile, log2_center, log2_range, nnz, device);
        }
    };

    let params = QuantParams {
        dtype,
        pack_format: PackFormat::Packed,
        group_size: 0,
        scale: Some(scale),
        log2_center,
        log2_range,
        nnz,
    };

    Ok(QuantizedTensors {
        weights,
        scales: None,
        zeros: None,
        g_idx: None,
        redundancy_mask: None,
        params,
    })
}
