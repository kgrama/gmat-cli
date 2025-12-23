//! GPTQ format quantization.
//!
//! Implements the GPTQ tensor format compatible with AutoGPTQ/Transformers:
//! - qweight: (in_features/8, out_features) int32 - packed weights
//! - scales: (num_groups, out_features) f16 - per-group scales
//! - qzeros: (num_groups/8, out_features) int32 - packed zero points
//! - g_idx: (in_features,) int32 - group index per input feature
//!
//! Supports activation-aware quantization when ActivationStats are provided.

use crate::graph_matrix::GraphMatrix;
use candle_core::{Device, Tensor, Result};
use half::f16;

use super::pack_simple::quantize_simple_packed;
use super::types::{ActivationStats, PackFormat, QuantDType, QuantParams, QuantizedTensors};

/// Compute per-group min/max with optional activation-aware importance weighting.
fn compute_group_minmax(
    values: &[(usize, f32)],  // (channel_idx, weight_value)
    clip_percentile: f32,
    activation_stats: Option<&ActivationStats>,
) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mut linear_min = f32::INFINITY;
    let mut linear_max = f32::NEG_INFINITY;

    for &(_, v) in values {
        linear_min = linear_min.min(v);
        linear_max = linear_max.max(v);
    }

    if linear_min == f32::INFINITY {
        return (0.0, 0.0);
    }

    // If activation stats provided, use importance-weighted clipping
    if let Some(act_stats) = activation_stats {
        if clip_percentile > 0.0 {
            // Compute importance for each weight
            let mut log2_importance_sum = 0.0f32;
            let mut log2_imp_min = f32::INFINITY;
            let mut log2_imp_max = f32::NEG_INFINITY;
            let mut count = 0usize;

            for &(channel, w) in values {
                if w != 0.0 {
                    let importance = act_stats.importance(channel, w.abs());
                    let log2_imp = importance.log2();
                    log2_importance_sum += log2_imp;
                    log2_imp_min = log2_imp_min.min(log2_imp);
                    log2_imp_max = log2_imp_max.max(log2_imp);
                    count += 1;
                }
            }

            if count > 0 {
                // Clip in importance space
                let log2_center = log2_importance_sum / count as f32;
                let half_range = (log2_imp_max - log2_imp_min) / 2.0;
                let effective_half_range = half_range * (1.0 - clip_percentile);
                let effective_max_importance = f32::exp2(log2_center + effective_half_range);

                // Convert back to weight space using mean activation
                let mean_act: f32 = values.iter()
                    .map(|&(ch, _)| act_stats.scale(ch))
                    .sum::<f32>() / values.len() as f32;

                let effective_max_weight = if mean_act > 0.0 {
                    effective_max_importance / mean_act
                } else {
                    effective_max_importance
                };

                // Clip linear min/max to effective range
                let eff_min = linear_min.max(-effective_max_weight);
                let eff_max = linear_max.min(effective_max_weight);

                return (eff_min, eff_max);
            }
        }
        // No clipping requested, use linear min/max
        return (linear_min, linear_max);
    }

    // Standard weight-only clipping
    if clip_percentile <= 0.0 {
        return (linear_min, linear_max);
    }

    // Log-aware clipping
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

    if count == 0 {
        return (linear_min, linear_max);
    }

    let log2_center = log2_sum / count as f32;
    let half_range = (log2_max - log2_min) / 2.0;
    let effective_half_range = half_range * (1.0 - clip_percentile);
    let effective_max_mag = f32::exp2(log2_center + effective_half_range);

    let eff_min = linear_min.max(-effective_max_mag);
    let eff_max = linear_max.min(effective_max_mag);

    (eff_min, eff_max)
}

/// Quantize to GPTQ format.
///
/// GPTQ tensor layout (compatible with AutoGPTQ):
/// - qweight: (in_features/8, out_features) int32
/// - scales: (num_groups, out_features) f16
/// - qzeros: (num_groups/8, out_features) int32
/// - g_idx: (in_features,) int32
///
/// Note: GraphMatrix is (rows=out_features, cols=in_features).
pub fn quantize_gptq(
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

    // Collect all values into dense form
    let mut dense = vec![0.0f32; out_features * in_features];
    for row in 0..out_features {
        for (col, value) in matrix.row_iter(row) {
            dense[row * in_features + col] = value;
        }
    }

    // Compute per-group scales and zeros (with activation awareness if provided)
    let mut scales_data = vec![0.0f32; out_features * num_groups];
    let mut zeros_data = vec![0u8; out_features * num_groups];

    for out_idx in 0..out_features {
        for g in 0..num_groups {
            let start = g * group_size;
            let end = ((g + 1) * group_size).min(in_features);
            let group_vals: Vec<(usize, f32)> = (start..end)
                .map(|in_idx| (in_idx, dense[out_idx * in_features + in_idx]))
                .collect();

            let (min_val, max_val) = compute_group_minmax(&group_vals, clip_percentile, activation_stats);

            if min_val == 0.0 && max_val == 0.0 || max_val == min_val {
                scales_data[out_idx * num_groups + g] = 1.0;
                zeros_data[out_idx * num_groups + g] = 0;
            } else {
                let scale = (max_val - min_val) / 15.0;
                scales_data[out_idx * num_groups + g] = scale;
                let zero = ((-min_val / scale).round().clamp(0.0, 15.0)) as u8;
                zeros_data[out_idx * num_groups + g] = zero;
            }
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
                    let zero = zeros_data[out_idx * num_groups + group] as f32;
                    let w = dense[out_idx * in_features + in_idx];
                    let q = ((w / scale) + zero).round().clamp(0.0, 15.0) as u32;
                    packed |= (q & 0x0F) << (i * 4);
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
                    let z = zeros_data[out_idx * num_groups + g] as u32 & 0x0F;
                    packed |= z << (i * 4);
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

    // g_idx
    let g_idx_data: Vec<u32> = (0..in_features).map(|i| (i / group_size) as u32).collect();

    let weights = Tensor::from_vec(qweight_data, (packed_in, out_features), device)?;
    let scales = Tensor::from_vec(scales_f16, (num_groups, out_features), device)?;
    let zeros = Tensor::from_vec(qzeros_data, (zeros_packed_groups, out_features), device)?;
    let g_idx = Tensor::from_vec(g_idx_data, (in_features,), device)?;

    let params = QuantParams {
        dtype,
        pack_format: PackFormat::Gptq,
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
        g_idx: Some(g_idx),
        redundancy_mask: None,
        params,
    })
}
