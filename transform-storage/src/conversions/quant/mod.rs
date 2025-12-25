//! Unified quantization module for GraphMatrix.
//!
//! Supports integer (i4-i16) and float (f4-f16) quantization with:
//! - Log-aware scale computation for improved precision
//! - Activation-aware quantization for importance-weighted scales
//! - AWQ/GPTQ-compatible bit-packed tensor output
//!
//! All outputs are Candle tensors with proper packing for inference.

mod float_quant;
mod pack_awq;
mod pack_gptq;
mod pack_none;
mod pack_simple;
mod pack_trellis;
mod types;

#[cfg(test)]
mod tests;

use crate::graph_matrix::GraphMatrix;
use candle_core::{DType, Device, Result, Tensor};

pub use types::{
    ActivationStats, PackFormat, QuantDType, QuantParams, QuantizedTensors, StaticSaliency,
    TrellisConfig,
};

/// Quantize a GraphMatrix with specified format and packing.
///
/// # Arguments
/// - `matrix`: The GraphMatrix to quantize
/// - `dtype`: Target quantization type (I4-I16, F4-F16)
/// - `pack_format`: Bit-packing format (None, Awq, Gptq, Packed)
/// - `clip_percentile`: Fraction of outliers to clip (0.0 = none)
/// - `device`: Target device for output tensors
///
/// # Returns
/// - `QuantizedTensors` containing packed weights and optional scales/zeros
pub fn quantize(
    matrix: &GraphMatrix,
    dtype: QuantDType,
    pack_format: PackFormat,
    clip_percentile: f32,
    device: &Device,
) -> Result<QuantizedTensors> {
    quantize_with_center(matrix, dtype, pack_format, clip_percentile, None, device)
}

/// Quantize a GraphMatrix with an optional pre-computed log2 center.
///
/// # Arguments
/// - `matrix`: The GraphMatrix to quantize
/// - `dtype`: Target quantization type (I4-I16, F4-F16)
/// - `pack_format`: Bit-packing format (None, Awq, Gptq, Packed)
/// - `clip_percentile`: Fraction of outliers to clip (0.0 = none)
/// - `log2_center`: Optional pre-computed log2 center (e.g., from `matrix.log2_stats()`)
///   If None, computed from the matrix
/// - `device`: Target device for output tensors
///
/// # Returns
/// - `QuantizedTensors` containing packed weights and optional scales/zeros
pub fn quantize_with_center(
    matrix: &GraphMatrix,
    dtype: QuantDType,
    pack_format: PackFormat,
    clip_percentile: f32,
    log2_center: Option<f32>,
    device: &Device,
) -> Result<QuantizedTensors> {
    let (rows, cols) = matrix.shape();

    // Use matrix's log2_stats if no center provided
    let (computed_center, log2_range, nnz) = matrix.log2_stats();

    // Handle empty matrix
    if nnz == 0 {
        return create_empty_output(rows, cols, dtype, pack_format, device);
    }

    // Use provided center or computed
    let center = log2_center.unwrap_or(computed_center);

    match pack_format {
        PackFormat::None => pack_none::quantize_unpacked(
            matrix,
            dtype,
            clip_percentile,
            center,
            log2_range,
            nnz,
            device,
        ),
        PackFormat::Packed => pack_simple::quantize_simple_packed(
            matrix,
            dtype,
            clip_percentile,
            center,
            log2_range,
            nnz,
            device,
        ),
        PackFormat::Awq => pack_awq::quantize_awq(
            matrix,
            dtype,
            clip_percentile,
            center,
            log2_range,
            nnz,
            None,
            device,
        ),
        PackFormat::Gptq => pack_gptq::quantize_gptq(
            matrix,
            dtype,
            clip_percentile,
            center,
            log2_range,
            nnz,
            None,
            device,
        ),
        PackFormat::TrellisSingle => pack_trellis::quantize_trellis_single(
            matrix,
            dtype,
            clip_percentile,
            center,
            log2_range,
            nnz,
            None,
            device,
        ),
        PackFormat::TrellisDual => pack_trellis::quantize_trellis_dual(
            matrix,
            dtype,
            clip_percentile,
            center,
            log2_range,
            nnz,
            None,
            device,
            TrellisConfig::default(),
        ),
    }
}

/// Quantize a GraphMatrix with activation-aware importance weighting.
///
/// This implements AWQ-style activation-aware quantization where weights are
/// scaled based on the magnitude of activations they multiply. Channels with
/// larger activations get tighter quantization (smaller scales) to preserve
/// precision where it matters most.
///
/// # Arguments
/// - `matrix`: The GraphMatrix to quantize (shape: out_features × in_features)
/// - `activation_stats`: Per-channel activation statistics (length: in_features)
/// - `dtype`: Target quantization type (I4-I16, F4-F16)
/// - `pack_format`: Bit-packing format (None, Awq, Gptq, Packed)
/// - `clip_percentile`: Fraction of outliers to clip (0.0 = none)
/// - `device`: Target device for output tensors
///
/// # Returns
/// - `QuantizedTensors` with activation-aware scales
///
/// # Example
/// ```ignore
/// // Collect activation stats from calibration
/// let act_stats = ActivationStats::from_scales(channel_means);
///
/// // Quantize with activation awareness
/// let quantized = quantize_with_activations(
///     &weights,
///     &act_stats,
///     QuantDType::I4,
///     PackFormat::Awq,
///     0.01,  // 1% outlier clipping
///     &Device::Cpu,
/// )?;
/// ```
pub fn quantize_with_activations(
    matrix: &GraphMatrix,
    activation_stats: &ActivationStats,
    dtype: QuantDType,
    pack_format: PackFormat,
    clip_percentile: f32,
    device: &Device,
) -> Result<QuantizedTensors> {
    let (rows, cols) = matrix.shape();

    // Validate activation stats match matrix dimensions
    if activation_stats.num_channels() != cols {
        return Err(candle_core::Error::Msg(format!(
            "Activation stats channels ({}) != matrix cols ({})",
            activation_stats.num_channels(),
            cols
        )));
    }

    let (log2_center, log2_range, nnz) = matrix.log2_stats();

    if nnz == 0 {
        return create_empty_output(rows, cols, dtype, pack_format, device);
    }

    match pack_format {
        PackFormat::None => {
            // For unpacked, use standard quantization (activation awareness less critical)
            pack_none::quantize_unpacked(
                matrix,
                dtype,
                clip_percentile,
                log2_center,
                log2_range,
                nnz,
                device,
            )
        }
        PackFormat::Packed => pack_simple::quantize_simple_packed(
            matrix,
            dtype,
            clip_percentile,
            log2_center,
            log2_range,
            nnz,
            device,
        ),
        PackFormat::Awq => pack_awq::quantize_awq(
            matrix,
            dtype,
            clip_percentile,
            log2_center,
            log2_range,
            nnz,
            Some(activation_stats),
            device,
        ),
        PackFormat::Gptq => pack_gptq::quantize_gptq(
            matrix,
            dtype,
            clip_percentile,
            log2_center,
            log2_range,
            nnz,
            Some(activation_stats),
            device,
        ),
        PackFormat::TrellisSingle => pack_trellis::quantize_trellis_single(
            matrix,
            dtype,
            clip_percentile,
            log2_center,
            log2_range,
            nnz,
            Some(activation_stats),
            device,
        ),
        PackFormat::TrellisDual => pack_trellis::quantize_trellis_dual(
            matrix,
            dtype,
            clip_percentile,
            log2_center,
            log2_range,
            nnz,
            Some(activation_stats),
            device,
            TrellisConfig::default(),
        ),
    }
}

/// Quantize a GraphMatrix using static saliency derived from block scales.
///
/// This provides activation-aware quantization without calibration data.
/// Saliency is computed from weight statistics (block scales) which approximate
/// the diagonal of the Hessian matrix.
///
/// # Arguments
/// - `matrix`: The GraphMatrix to quantize (shape: out_features × in_features)
/// - `saliency`: Static saliency scores per input channel (from block scales)
/// - `dtype`: Target quantization type (I4-I16, F4-F16)
/// - `pack_format`: Bit-packing format (None, Awq, Gptq, Packed)
/// - `clip_percentile`: Fraction of outliers to clip (0.0 = none)
/// - `device`: Target device for output tensors
///
/// # Returns
/// - `QuantizedTensors` with saliency-aware scales
///
/// # Example
/// ```ignore
/// // Extract column scales from the matrix's blocks
/// let col_scales = matrix.column_block_scales();
///
/// // Optionally chain with upstream layer (e.g., embedding)
/// let saliency = StaticSaliency::from_chained_scales(&embed_scales, &col_scales);
///
/// // Quantize with static saliency
/// let quantized = quantize_with_saliency(
///     &weights,
///     &saliency,
///     QuantDType::I4,
///     PackFormat::Awq,
///     0.01,
///     &Device::Cpu,
/// )?;
/// ```
pub fn quantize_with_saliency(
    matrix: &GraphMatrix,
    saliency: &StaticSaliency,
    dtype: QuantDType,
    pack_format: PackFormat,
    clip_percentile: f32,
    device: &Device,
) -> Result<QuantizedTensors> {
    // Convert saliency to activation stats and delegate
    let act_stats = saliency.to_activation_stats();
    quantize_with_activations(
        matrix,
        &act_stats,
        dtype,
        pack_format,
        clip_percentile,
        device,
    )
}

fn create_empty_output(
    rows: usize,
    cols: usize,
    dtype: QuantDType,
    pack_format: PackFormat,
    device: &Device,
) -> Result<QuantizedTensors> {
    let pack_factor = pack_format.pack_factor(dtype.bits());
    let packed_cols = cols.div_ceil(pack_factor);

    let weights = match dtype {
        QuantDType::F16 => Tensor::zeros((rows, cols), DType::F16, device)?,
        QuantDType::I8 | QuantDType::F8 => Tensor::zeros((rows, cols), DType::U8, device)?,
        _ => match pack_format {
            PackFormat::None => Tensor::zeros((rows, cols), DType::I64, device)?,
            PackFormat::Packed if dtype.bits() == 4 => {
                Tensor::zeros((rows, packed_cols), DType::U8, device)?
            }
            PackFormat::Awq | PackFormat::Gptq => {
                Tensor::zeros((rows, packed_cols), DType::U32, device)?
            }
            _ => Tensor::zeros((rows, cols), DType::I64, device)?,
        },
    };

    let params = QuantParams {
        dtype,
        pack_format,
        group_size: pack_format.default_group_size(),
        scale: Some(1.0),
        log2_center: 0.0,
        log2_range: 0.0,
        nnz: 0,
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
