//! Trellis quantization with dynamic programming optimization.
//!
//! Implements two quantization modes:
//! - TrellisSingle: Single-view DP optimization with smoothness penalty
//! - TrellisDual: Dual-view joint optimization with shared scale and inter-row coupling

use crate::graph_matrix::GraphMatrix;
use candle_core::{Device, Result, Tensor};
use half::f16;
use std::collections::HashMap;

use super::pack_simple::quantize_simple_packed;
use super::types::{ActivationStats, PackFormat, QuantDType, QuantParams, QuantizedTensors, TrellisConfig};

const LAMBDA: f32 = 0.3; // Smoothness penalty
const GAMMA: f32 = 0.2;  // Dual-row coupling
const TOP_K: usize = 8;  // Candidate states

/// Compute quantization error: (true_offset - decoded_offset)²
fn compute_quant_error(true_offset: f32, state: u8, num_states: usize) -> f32 {
    let decoded = (state as f32) / ((num_states - 1) as f32);
    (true_offset - decoded).powi(2)
}

/// Compute transition cost: lambda * |prev_state - curr_state|
fn compute_transition_cost(prev_state: u8, curr_state: u8, lambda: f32) -> f32 {
    lambda * (prev_state as i16 - curr_state as i16).abs() as f32
}

/// Get top K states with smallest quantization error
fn get_top_k_states(offset: f32, k: usize, num_states: usize) -> Vec<u8> {
    let mut errors: Vec<(u8, f32)> = (0..num_states as u8)
        .map(|s| (s, compute_quant_error(offset, s, num_states)))
        .collect();
    errors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    errors.into_iter().take(k).map(|(s, _)| s).collect()
}

/// Single-view trellis: DP over log offset states with smoothness penalty
fn trellis_single(log_offsets: &[f32], lambda: f32, num_states: usize) -> Vec<u8> {
    let size = log_offsets.len();
    if size == 0 {
        return Vec::new();
    }

    let mut cost = vec![vec![f32::INFINITY; num_states]; size];
    let mut parent = vec![vec![0u8; num_states]; size];

    for s in 0..num_states {
        cost[0][s] = compute_quant_error(log_offsets[0], s as u8, num_states);
    }

    for i in 1..size {
        for curr_s in 0..num_states {
            for prev_s in 0..num_states {
                let candidate_cost = cost[i - 1][prev_s]
                    + compute_quant_error(log_offsets[i], curr_s as u8, num_states)
                    + compute_transition_cost(prev_s as u8, curr_s as u8, lambda);
                if candidate_cost < cost[i][curr_s] {
                    cost[i][curr_s] = candidate_cost;
                    parent[i][curr_s] = prev_s as u8;
                }
            }
        }
    }

    let mut path = vec![0u8; size];
    path[size - 1] = cost[size - 1]
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u8)
        .unwrap_or(0);

    for i in (0..size - 1).rev() {
        path[i] = parent[i + 1][path[i + 1] as usize];
    }

    path
}

/// Dual-view trellis: Joint 2D state space with shared scale and inter-row coupling
fn trellis_dual(
    offsets0: &[f32],
    offsets1: &[f32],
    lambda: f32,
    gamma: f32,
    k: usize,
    num_states: usize,
) -> (Vec<u8>, Vec<u8>) {
    let size = offsets0.len();
    if size == 0 {
        return (Vec::new(), Vec::new());
    }

    let candidates0: Vec<Vec<u8>> = offsets0
        .iter()
        .map(|&off| get_top_k_states(off, k, num_states))
        .collect();
    let candidates1: Vec<Vec<u8>> = offsets1
        .iter()
        .map(|&off| get_top_k_states(off, k, num_states))
        .collect();

    let mut cost: Vec<HashMap<(u8, u8), f32>> = vec![HashMap::new(); size];
    let mut parent: Vec<HashMap<(u8, u8), (u8, u8)>> = vec![HashMap::new(); size];

    for &s0 in &candidates0[0] {
        for &s1 in &candidates1[0] {
            let err0 = compute_quant_error(offsets0[0], s0, num_states);
            let err1 = compute_quant_error(offsets1[0], s1, num_states);
            let coupling = gamma * (s0 as i16 - s1 as i16).abs() as f32;
            cost[0].insert((s0, s1), err0 + err1 + coupling);
        }
    }

    for i in 1..size {
        for &curr_s0 in &candidates0[i] {
            for &curr_s1 in &candidates1[i] {
                let mut best_cost = f32::INFINITY;
                let mut best_parent = (0u8, 0u8);

                for &prev_s0 in &candidates0[i - 1] {
                    for &prev_s1 in &candidates1[i - 1] {
                        if let Some(&prev_cost) = cost[i - 1].get(&(prev_s0, prev_s1)) {
                            let err0 = compute_quant_error(offsets0[i], curr_s0, num_states);
                            let err1 = compute_quant_error(offsets1[i], curr_s1, num_states);
                            let trans0 = compute_transition_cost(prev_s0, curr_s0, lambda);
                            let trans1 = compute_transition_cost(prev_s1, curr_s1, lambda);
                            let coupling = gamma * (curr_s0 as i16 - curr_s1 as i16).abs() as f32;

                            let candidate = prev_cost + err0 + err1 + trans0 + trans1 + coupling;

                            if candidate < best_cost {
                                best_cost = candidate;
                                best_parent = (prev_s0, prev_s1);
                            }
                        }
                    }
                }

                if best_cost < f32::INFINITY {
                    cost[i].insert((curr_s0, curr_s1), best_cost);
                    parent[i].insert((curr_s0, curr_s1), best_parent);
                }
            }
        }
    }

    let mut path0 = vec![0u8; size];
    let mut path1 = vec![0u8; size];

    let (final_s0, final_s1) = cost[size - 1]
        .iter()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|((s0, s1), _)| (*s0, *s1))
        .unwrap_or((0, 0));

    path0[size - 1] = final_s0;
    path1[size - 1] = final_s1;

    for i in (0..size - 1).rev() {
        if let Some(&(prev_s0, prev_s1)) = parent[i + 1].get(&(path0[i + 1], path1[i + 1])) {
            path0[i] = prev_s0;
            path1[i] = prev_s1;
        }
    }

    (path0, path1)
}

/// Pack nibbles into u32 (8 nibbles per u32)
fn pack_nibbles(values: &[u8], start: usize, end: usize) -> u32 {
    let mut packed = 0u32;
    for i in start..end {
        if i < values.len() {
            packed |= (values[i] as u32 & 0x0F) << ((i - start) * 4);
        }
    }
    packed
}

/// Normalize values to [0, 1] range for trellis input
fn normalize_block(values: &[f32], scale: f32, num_states: usize) -> Vec<f32> {
    values
        .iter()
        .map(|&v| ((v / scale).abs().clamp(0.0, 1.0) * (num_states - 1) as f32) / (num_states - 1) as f32)
        .collect()
}

/// Quantize to TrellisSingle format with single-view DP optimization
pub fn quantize_trellis_single(
    matrix: &GraphMatrix,
    dtype: QuantDType,
    clip_percentile: f32,
    log2_center: f32,
    log2_range: f32,
    nnz: usize,
    _activation_stats: Option<&ActivationStats>,
    device: &Device,
) -> Result<QuantizedTensors> {
    if dtype != QuantDType::I4 {
        return quantize_simple_packed(matrix, dtype, clip_percentile, log2_center, log2_range, nnz, device);
    }

    let (rows, cols) = matrix.shape();
    let block_size = 16;
    let num_blocks = (cols + block_size - 1) / block_size;
    let num_states = 16;

    let mut dense = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for (col, value) in matrix.row_iter(row) {
            dense[row * cols + col] = value;
        }
    }

    let mut qweight_data = vec![0u32; rows * num_blocks * 2];
    let mut scales_data = vec![f16::ZERO; rows * num_blocks];

    for row in 0..rows {
        for blk in 0..num_blocks {
            let start = blk * block_size;
            let end = (start + block_size).min(cols);
            let block_vals: Vec<f32> = dense[row * cols + start..row * cols + end].to_vec();

            let max_abs = block_vals.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 15.0 } else { 1.0 };
            scales_data[row * num_blocks + blk] = f16::from_f32(scale);

            let log_offsets = normalize_block(&block_vals, scale, num_states);
            let path = trellis_single(&log_offsets, LAMBDA, num_states);

            qweight_data[row * num_blocks * 2 + blk * 2] = pack_nibbles(&path, 0, 8);
            qweight_data[row * num_blocks * 2 + blk * 2 + 1] = pack_nibbles(&path, 8, 16);
        }
    }

    let weights = Tensor::from_vec(qweight_data, (rows, num_blocks * 2), device)?;
    let scales = Tensor::from_vec(scales_data, (rows, num_blocks), device)?;

    let params = QuantParams {
        dtype,
        pack_format: PackFormat::TrellisSingle,
        group_size: 0,
        scale: None,
        log2_center,
        log2_range,
        nnz,
    };

    Ok(QuantizedTensors {
        weights,
        scales: Some(scales),
        zeros: None,
        g_idx: None,
        redundancy_mask: None,
        params,
    })
}

/// Quantize to TrellisDual format with dual-view joint optimization
pub fn quantize_trellis_dual(
    matrix: &GraphMatrix,
    dtype: QuantDType,
    clip_percentile: f32,
    log2_center: f32,
    log2_range: f32,
    nnz: usize,
    _activation_stats: Option<&ActivationStats>,
    device: &Device,
    config: TrellisConfig,
) -> Result<QuantizedTensors> {
    if dtype != QuantDType::I4 {
        return quantize_simple_packed(matrix, dtype, clip_percentile, log2_center, log2_range, nnz, device);
    }

    let (rows, cols) = matrix.shape();
    let block_size = 16;
    let num_blocks = (cols + block_size - 1) / block_size;
    let num_states = 16;
    let row_pairs = (rows + 1) / 2;

    let mut dense = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for (col, value) in matrix.row_iter(row) {
            dense[row * cols + col] = value;
        }
    }

    let mut qweight_data = vec![0u32; row_pairs * 2 * num_blocks * 2];
    let mut scales_data = vec![f16::ZERO; row_pairs * num_blocks];

    for pair_idx in 0..row_pairs {
        let row0 = pair_idx * 2;
        let row1 = (pair_idx * 2 + 1).min(rows - 1);

        for blk in 0..num_blocks {
            let start = blk * block_size;
            let end = (start + block_size).min(cols);

            let vals0: Vec<f32> = dense[row0 * cols + start..row0 * cols + end].to_vec();
            let vals1: Vec<f32> = dense[row1 * cols + start..row1 * cols + end].to_vec();

            let max0 = vals0.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            let max1 = vals1.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
            let scale0 = if max0 > 0.0 { max0 / 15.0 } else { 1.0 };
            let scale1 = if max1 > 0.0 { max1 / 15.0 } else { 1.0 };

            let shared_scale = scale0.min(scale1).max(1e-6);
            scales_data[pair_idx * num_blocks + blk] = f16::from_f32(shared_scale);

            let offsets0 = normalize_block(&vals0, shared_scale, num_states);
            let offsets1 = normalize_block(&vals1, shared_scale, num_states);

            let (path0, path1) = trellis_dual(&offsets0, &offsets1, LAMBDA, GAMMA, TOP_K, num_states);

            for (row_offset, path) in [(0, &path0), (1, &path1)] {
                let base_idx = pair_idx * 2 * num_blocks * 2 + row_offset * num_blocks * 2 + blk * 2;
                qweight_data[base_idx] = pack_nibbles(path, 0, 8);
                qweight_data[base_idx + 1] = pack_nibbles(path, 8, 16);
            }
        }
    }

    // Compute redundancy mask if enabled (before moving qweight_data)
    let redundancy_mask = if config.emit_redundancy_mask {
        let mut mask_data = vec![0u8; row_pairs * num_blocks];
        
        for pair_idx in 0..row_pairs {
            for blk in 0..num_blocks {
                let start = blk * block_size;
                let end = (start + block_size).min(cols);
                let block_len = end - start;
                
                // Extract paths for this block from qweight_data
                let base_idx0 = pair_idx * 2 * num_blocks * 2 + blk * 2;
                let base_idx1 = pair_idx * 2 * num_blocks * 2 + num_blocks * 2 + blk * 2;
                
                let packed0_0 = qweight_data[base_idx0];
                let packed0_1 = qweight_data[base_idx0 + 1];
                let packed1_0 = qweight_data[base_idx1];
                let packed1_1 = qweight_data[base_idx1 + 1];
                
                // Unpack nibbles and compute mean absolute difference
                let mut total_diff = 0u32;
                for i in 0..8.min(block_len) {
                    let s0 = ((packed0_0 >> (i * 4)) & 0x0F) as u8;
                    let s1 = ((packed1_0 >> (i * 4)) & 0x0F) as u8;
                    total_diff += (s0 as i16 - s1 as i16).abs() as u32;
                }
                for i in 0..(block_len.saturating_sub(8)) {
                    let s0 = ((packed0_1 >> (i * 4)) & 0x0F) as u8;
                    let s1 = ((packed1_1 >> (i * 4)) & 0x0F) as u8;
                    total_diff += (s0 as i16 - s1 as i16).abs() as u32;
                }
                
                let mean_diff = (total_diff as f32) / (block_len as f32);
                if mean_diff <= (config.redundancy_threshold as f32) {
                    mask_data[pair_idx * num_blocks + blk] = 1;
                }
            }
        }
        
        Some(Tensor::from_vec(mask_data, (row_pairs, num_blocks), device)?)
    } else {
        None
    };

    let weights = Tensor::from_vec(qweight_data, (row_pairs * 2, num_blocks * 2), device)?;
    let scales = Tensor::from_vec(scales_data, (row_pairs, num_blocks), device)?;

    let params = QuantParams {
        dtype,
        pack_format: PackFormat::TrellisDual,
        group_size: 0,
        scale: None,
        log2_center,
        log2_range,
        nnz,
    };

    Ok(QuantizedTensors {
        weights,
        scales: Some(scales),
        zeros: None,
        g_idx: None,
        redundancy_mask,
        params,
    })
}
