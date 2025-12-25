//! Unpacked quantization (one value per element).

use crate::graph_matrix::GraphMatrix;
use candle_core::{Device, Result, Tensor};
use half::f16;

use super::float_quant::quantize_float_bits;
use super::types::{PackFormat, QuantDType, QuantParams, QuantizedTensors};

/// Quantize without packing - one value per tensor element.
pub fn quantize_unpacked(
    matrix: &GraphMatrix,
    dtype: QuantDType,
    clip_percentile: f32,
    log2_center: f32,
    log2_range: f32,
    nnz: usize,
    device: &Device,
) -> Result<QuantizedTensors> {
    let (rows, cols) = matrix.shape();

    // Log-aware scale selection:
    // Instead of using max_abs (sensitive to outliers), use the log distribution
    // to set scale based on geometric median + half_range (covers ~50% of values above median)
    //
    // For symmetric signed quant: scale = effective_max / int_max
    // where effective_max is based on log2_center + effective_half_range
    let half_range = log2_range / 2.0;
    let effective_half_range = if clip_percentile > 0.0 {
        half_range * (1.0 - clip_percentile)
    } else {
        half_range
    };

    // effective_max from log domain - this clips outliers naturally
    let effective_max_log = log2_center + effective_half_range;
    let effective_max_abs = f32::exp2(effective_max_log);

    let scale = if dtype.is_integer() {
        effective_max_abs / dtype.int_max() as f32
    } else {
        1.0
    };

    let weights = match dtype {
        QuantDType::F16 => {
            let mut data = vec![f16::ZERO; rows * cols];
            for row in 0..rows {
                for (col, value) in matrix.row_iter(row) {
                    data[row * cols + col] = f16::from_f32(value);
                }
            }
            Tensor::from_vec(data, (rows, cols), device)?
        }
        QuantDType::I4 | QuantDType::I6 | QuantDType::I8 | QuantDType::I16 => {
            let min_int = dtype.int_min() as f32;
            let max_int = dtype.int_max() as f32;
            let mut data = vec![0i64; rows * cols];
            for row in 0..rows {
                for (col, value) in matrix.row_iter(row) {
                    // Standard symmetric quantization: q = round(value / scale)
                    let q = (value / scale).round().clamp(min_int, max_int) as i64;
                    data[row * cols + col] = q;
                }
            }
            Tensor::from_vec(data, (rows, cols), device)?
        }
        QuantDType::F4 | QuantDType::F6 | QuantDType::F8 => {
            let mut data = vec![0.0f32; rows * cols];
            for row in 0..rows {
                for (col, value) in matrix.row_iter(row) {
                    data[row * cols + col] = quantize_float_bits(value, dtype);
                }
            }
            Tensor::from_vec(data, (rows, cols), device)?
        }
    };

    let params = QuantParams {
        dtype,
        pack_format: PackFormat::None,
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
