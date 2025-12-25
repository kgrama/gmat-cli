//! Quantization dispatch and tensor-level operations

use half::f16;

use crate::blocks::AnyBlock;
use crate::graph_matrix::GraphMatrix;

use super::iquant::{
    compute_iq4nl_scale, compute_iq4xs_scales, encode_iq4nl_block, encode_iq4xs_block,
    IQ4_NL_CONFIG, IQ4_XS_CONFIG,
};
use super::kquant::{
    compute_q2k_scales, compute_q3k_scales, compute_q6k_scales, compute_scales_standard,
    encode_q2k_block, encode_q3k_block, encode_q4k_block, encode_q5k_block, encode_q6k_block,
    Q2_K_CONFIG, Q3_K_CONFIG, Q4_K_CONFIG, Q5_K_CONFIG, Q6_K_CONFIG,
};
use super::legacy;
use super::trellis::optimize_group_scales_trellis;
use super::types::*;
use super::utils::{
    compute_superblock_scales, quantize_legacy_blocks, quantize_superblocks,
    select_quant_with_fallback,
};

/// Main entry point - quantize GMAT matrix to GGUF format
pub fn quantize_to_gguf(
    matrix: &GraphMatrix,
    quant_type: GgufQuantType,
    scale_opt: ScaleOptimization,
    _activation_stats: Option<&ActivationStats>,
) -> Result<GgufQuantizedData, String> {
    let (rows, cols) = matrix.shape();
    let actual_type = select_quant_with_fallback(cols, quant_type);

    let all_blocks: Vec<&AnyBlock> = matrix.block_iter().collect();
    let gmat_block_size = all_blocks.first().map(|b| b.size()).unwrap_or(8);

    let data = match actual_type {
        // Legacy formats (32-element blocks)
        GgufQuantType::Q8_0 => quantize_legacy_blocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            34,
            legacy::blocks_to_q8_0,
        ),
        GgufQuantType::Q4_0 => quantize_legacy_blocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            18,
            legacy::blocks_to_q4_0,
        ),
        GgufQuantType::Q4_1 => quantize_legacy_blocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            20,
            legacy::blocks_to_q4_1,
        ),
        GgufQuantType::Q5_0 => quantize_legacy_blocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            22,
            legacy::blocks_to_q5_0,
        ),
        GgufQuantType::Q5_1 => quantize_legacy_blocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            24,
            legacy::blocks_to_q5_1,
        ),

        // K-quants with scale optimization
        GgufQuantType::Q4_K_S | GgufQuantType::Q4_K_M => quantize_kquant_with_opt(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            scale_opt,
            Q4_K_CONFIG.block_bytes,
            encode_q4k_superblock,
        ),
        GgufQuantType::Q5_K_S | GgufQuantType::Q5_K_M => quantize_kquant_with_opt(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            scale_opt,
            Q5_K_CONFIG.block_bytes,
            encode_q5k_superblock,
        ),
        GgufQuantType::Q6_K => quantize_superblocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            Q6_K_CONFIG.block_bytes,
            |out, ctx| {
                let (d, _) = compute_superblock_scales(ctx.block_refs);
                let scales = compute_q6k_scales(ctx.block_refs, d, ctx.gmat_block_size);
                encode_q6k_block(out, d, &scales, ctx.block_refs, ctx.gmat_block_size);
            },
        ),
        GgufQuantType::Q2_K => quantize_superblocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            Q2_K_CONFIG.block_bytes,
            |out, ctx| {
                let (d, dmin) = compute_superblock_scales(ctx.block_refs);
                let scales = compute_q2k_scales(ctx.block_refs, d, ctx.gmat_block_size);
                encode_q2k_block(out, d, dmin, &scales, ctx.block_refs, ctx.gmat_block_size);
            },
        ),
        GgufQuantType::Q3_K_S | GgufQuantType::Q3_K_M | GgufQuantType::Q3_K_L => {
            quantize_superblocks(
                &all_blocks,
                rows,
                cols,
                gmat_block_size,
                Q3_K_CONFIG.block_bytes,
                |out, ctx| {
                    let (d, _) = compute_superblock_scales(ctx.block_refs);
                    let scales = compute_q3k_scales(ctx.block_refs, d, ctx.gmat_block_size);
                    encode_q3k_block(out, d, &scales, ctx.block_refs, ctx.gmat_block_size);
                },
            )
        }

        // I-quants
        GgufQuantType::IQ4_XS => quantize_superblocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            IQ4_XS_CONFIG.block_bytes,
            |out, ctx| {
                let (d, _) = compute_superblock_scales(ctx.block_refs);
                let scales = compute_iq4xs_scales(ctx.block_refs, d, ctx.gmat_block_size);
                encode_iq4xs_block(out, d, &scales, ctx.block_refs, ctx.gmat_block_size);
            },
        ),
        GgufQuantType::IQ4_NL => quantize_superblocks(
            &all_blocks,
            rows,
            cols,
            gmat_block_size,
            IQ4_NL_CONFIG.block_bytes,
            |out, ctx| {
                let d = compute_iq4nl_scale(ctx.block_refs);
                encode_iq4nl_block(out, d, ctx.block_refs, ctx.gmat_block_size);
            },
        ),

        _ => return Err(format!("Unimplemented: {:?}", actual_type)),
    };

    Ok(GgufQuantizedData {
        data,
        quant_type: actual_type,
        shape: (rows, cols),
    })
}

/// Helper for K-quants that support scale optimization (Q4_K, Q5_K).
#[inline]
fn quantize_kquant_with_opt<F>(
    all_blocks: &[&AnyBlock],
    rows: usize,
    cols: usize,
    gmat_block_size: usize,
    scale_opt: ScaleOptimization,
    bytes_per_sb: usize,
    encode_fn: F,
) -> Vec<u8>
where
    F: Fn(&mut [u8], f16, f16, &[u8; 8], &[u8; 8], &[&AnyBlock], usize) + Sync,
{
    quantize_superblocks(
        all_blocks,
        rows,
        cols,
        gmat_block_size,
        bytes_per_sb,
        |out, ctx| {
            let (d, dmin) = compute_superblock_scales(ctx.block_refs);
            let scales = match scale_opt {
                ScaleOptimization::Standard => {
                    compute_scales_standard(ctx.block_refs, d, dmin, ctx.gmat_block_size)
                }
                ScaleOptimization::Trellis { lambda } => optimize_group_scales_trellis(
                    ctx.block_refs,
                    d,
                    dmin,
                    None,
                    lambda,
                    ctx.gmat_block_size,
                ),
                ScaleOptimization::TrellisDual { .. } => {
                    unimplemented!("TrellisDual requires paired rows")
                }
            };
            let mins = scales;
            encode_fn(
                out,
                d,
                dmin,
                &scales,
                &mins,
                ctx.block_refs,
                ctx.gmat_block_size,
            );
        },
    )
}

#[inline]
fn encode_q4k_superblock(
    out: &mut [u8],
    d: f16,
    dmin: f16,
    scales: &[u8; 8],
    mins: &[u8; 8],
    blocks: &[&AnyBlock],
    bs: usize,
) {
    encode_q4k_block(out, d, dmin, scales, mins, blocks, bs);
}

#[inline]
fn encode_q5k_superblock(
    out: &mut [u8],
    d: f16,
    dmin: f16,
    scales: &[u8; 8],
    mins: &[u8; 8],
    blocks: &[&AnyBlock],
    bs: usize,
) {
    encode_q5k_block(out, d, dmin, scales, mins, blocks, bs);
}

/// Compute tensor importance from block statistics.
/// Uses ratio of octave-shifted elements to total non-zero elements.
/// Higher ratio indicates larger dynamic range / more important values.
pub fn compute_tensor_importance(matrix: &GraphMatrix) -> f32 {
    let mut total_shifted = 0usize;
    let mut total_nnz = 0usize;

    for blk in matrix.block_iter() {
        let (shifted, nnz) = blk.importance_stats();
        total_shifted += shifted;
        total_nnz += nnz;
    }

    if total_nnz == 0 {
        return 0.0;
    }

    total_shifted as f32 / total_nnz as f32
}
