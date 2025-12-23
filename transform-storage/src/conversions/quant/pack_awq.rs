//! AWQ format quantization (Activation-aware Weight Quantization).
//!
//! Implements the AWQ tensor format compatible with AutoAWQ/vLLM:
//! - qweight: (K/8, N) int32 - 8 x 4-bit weights packed per int32, interleaved
//! - scales: (K/group_size, N) f16 - per-group scales
//! - qzeros: (K/group_size/8, N) int32 - packed zero points (stored as value-1)
//!
//! Supports activation-aware quantization when ActivationStats are provided.

use crate::graph_matrix::GraphMatrix;
use candle_core::{Device, Tensor, Result};
use half::f16;

use super::pack_simple::quantize_simple_packed;
use super::types::{ActivationStats, PackFormat, QuantDType, QuantParams, QuantizedTensors};

/// Compute per-group scale using log-aware statistics and optional activation weighting.
///
/// When activation_stats is provided, weights in channels with higher activations
/// get tighter scales (more precision) based on importance = |weight| × |activation|.
fn compute_group_scale(
    values: &[(usize, f32)],  // (channel_idx, weight_value)
    clip_percentile: f32,
    activation_stats: Option<&ActivationStats>,
) -> f32 {
    if values.is_empty() {
        return 1.0;
    }

    if let Some(act_stats) = activation_stats {
        // Activation-aware: use importance-weighted scale
        // importance = |weight| × |activation|
        // We want tighter scale for high-importance weights

        let mut max_importance = 0.0f32;
        let mut log2_importance_sum = 0.0f32;
        let mut log2_imp_min = f32::INFINITY;
        let mut log2_imp_max = f32::NEG_INFINITY;
        let mut count = 0usize;

        for &(channel, w) in values {
            if w != 0.0 {
                let importance = act_stats.importance(channel, w.abs());
                max_importance = max_importance.max(importance);

                let log2_imp = importance.log2();
                log2_importance_sum += log2_imp;
                log2_imp_min = log2_imp_min.min(log2_imp);
                log2_imp_max = log2_imp_max.max(log2_imp);
                count += 1;
            }
        }

        if count == 0 {
            return 1.0;
        }

        // Use importance-weighted effective max
        let effective_max = if clip_percentile > 0.0 && count > 1 {
            // Log-domain clipping on importance
            let log2_center = log2_importance_sum / count as f32;
            let half_range = (log2_imp_max - log2_imp_min) / 2.0;
            let effective_half_range = half_range * (1.0 - clip_percentile);
            f32::exp2(log2_center + effective_half_range)
        } else {
            max_importance
        };

        // Scale based on importance: effective_max_importance / 7.0
        // But we need to convert back to weight scale
        // importance = weight × activation, so weight = importance / activation
        // For the group, use the mean activation scale
        let mean_act: f32 = values.iter()
            .map(|&(ch, _)| act_stats.scale(ch))
            .sum::<f32>() / values.len() as f32;

        let effective_weight_max = if mean_act > 0.0 {
            effective_max / mean_act
        } else {
            effective_max
        };

        if effective_weight_max > 0.0 { effective_weight_max / 7.0 } else { 1.0 }
    } else {
        // Standard weight-only scale computation
        let mut max_abs = 0.0f32;

        if clip_percentile > 0.0 {
            let mut log2_sum = 0.0f32;
            let mut log2_min = f32::INFINITY;
            let mut log2_max = f32::NEG_INFINITY;
            let mut count = 0usize;

            for &(_, v) in values {
                if v != 0.0 {
                    let log2_abs = v.abs().log2();
                    log2_sum += log2_abs;
                    log2_min = log2_min.min(log2_abs);
                    log2_max = log2_max.max(log2_abs);
                    count += 1;
                }
            }

            if count > 0 {
                let log2_center = log2_sum / count as f32;
                let half_range = (log2_max - log2_min) / 2.0;
                let effective_half_range = half_range * (1.0 - clip_percentile);
                max_abs = f32::exp2(log2_center + effective_half_range);
            }
        } else {
            for &(_, v) in values {
                max_abs = max_abs.max(v.abs());
            }
        }

        if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 }
    }
}

/// Quantize to AWQ format.
///
/// AWQ tensor layout (compatible with AutoAWQ GEMM kernels):
/// - qweight: (in_features/8, out_features) int32
/// - scales: (in_features/group_size, out_features) f16
/// - qzeros: (in_features/group_size/8, out_features) int32
///
/// Note: GraphMatrix is (rows=out_features, cols=in_features), so we transpose.
pub fn quantize_awq(
    matrix: &GraphMatrix,
    dtype: QuantDType,
    clip_percentile: f32,
    log2_center: f32,
    log2_range: f32,
    nnz: usize,
    activation_stats: Option<&ActivationStats>,
    device: &Device,
) -> Result<QuantizedTensors> {
    if dtype != QuantDType::I4 {
        return quantize_simple_packed(matrix, dtype, clip_percentile, log2_center, log2_range, nnz, device);
    }

    let (rows, cols) = matrix.shape();
    let out_features = rows;
    let in_features = cols;
    let group_size = 128;
    let num_groups = (in_features + group_size - 1) / group_size;
    let packed_in = in_features / 8;

    // Collect all values into dense form for group processing
    let mut dense = vec![0.0f32; out_features * in_features];
    for row in 0..out_features {
        for (col, value) in matrix.row_iter(row) {
            dense[row * in_features + col] = value;
        }
    }

    // Compute per-group scales (with activation awareness if provided)
    let mut scales_data = vec![0.0f32; out_features * num_groups];

    for out_idx in 0..out_features {
        for g in 0..num_groups {
            let start = g * group_size;
            let end = ((g + 1) * group_size).min(in_features);
            let group_vals: Vec<(usize, f32)> = (start..end)
                .map(|in_idx| (in_idx, dense[out_idx * in_features + in_idx]))
                .collect();
            scales_data[out_idx * num_groups + g] = compute_group_scale(&group_vals, clip_percentile, activation_stats);
        }
    }

    // Quantize weights
    let mut qweight_data = vec![0u32; packed_in * out_features];

    for out_idx in 0..out_features {
        for pack_idx in 0..packed_in {
            let mut packed: u32 = 0;
            for i in 0..8 {
                let in_idx = pack_idx * 8 + i;
                if in_idx < in_features {
                    let group = in_idx / group_size;
                    let scale = scales_data[out_idx * num_groups + group];
                    let w = dense[out_idx * in_features + in_idx];
                    let q_signed = (w / scale).round().clamp(-8.0, 7.0) as i8;
                    let q_unsigned = (q_signed + 8) as u32 & 0x0F;
                    packed |= q_unsigned << (i * 4);
                }
            }
            qweight_data[pack_idx * out_features + out_idx] = packed;
        }
    }

    // Pack zeros
    let zeros_packed_groups = (num_groups + 7) / 8;
    let mut qzeros_data = vec![0u32; zeros_packed_groups * out_features];

    for out_idx in 0..out_features {
        for pack_idx in 0..zeros_packed_groups {
            let mut packed: u32 = 0;
            for i in 0..8 {
                let g = pack_idx * 8 + i;
                if g < num_groups {
                    let zero_minus_one: u32 = 7;
                    packed |= zero_minus_one << (i * 4);
                }
            }
            qzeros_data[pack_idx * out_features + out_idx] = packed;
        }
    }

    // Convert scales to f16
    let mut scales_f16 = vec![f16::ZERO; num_groups * out_features];
    for out_idx in 0..out_features {
        for g in 0..num_groups {
            scales_f16[g * out_features + out_idx] = f16::from_f32(scales_data[out_idx * num_groups + g]);
        }
    }

    let weights = Tensor::from_vec(qweight_data, (packed_in, out_features), device)?;
    let scales = Tensor::from_vec(scales_f16, (num_groups, out_features), device)?;
    let zeros = Tensor::from_vec(qzeros_data, (zeros_packed_groups, out_features), device)?;

    let params = QuantParams {
        dtype,
        pack_format: PackFormat::Awq,
        group_size,
        scale: None,
        log2_center,
        log2_range,
        nnz,
    };

    Ok(QuantizedTensors {
        weights,
        scales: Some(scales),
        zeros: Some(zeros),
        g_idx: None,
        redundancy_mask: None,
        params,
    })
}
