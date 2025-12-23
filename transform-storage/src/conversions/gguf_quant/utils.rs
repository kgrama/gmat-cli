//! Utility functions for GGUF quantization

use half::f16;

use crate::blocks::AnyBlock;
use super::types::GgufQuantType;

//=============================================================================
// Fast Math Utilities
//=============================================================================

/// Fast exp2 approximation using polynomial interpolation.
/// ~5x faster than libm exp2, <0.1% relative error.
/// Uses integer bit manipulation for 2^floor(x) and polynomial for fractional part.
#[inline]
pub fn fast_exp2(x: f32) -> f32 {
    let i = x.floor() as i32;
    let f = x - i as f32;
    // Polynomial approximation for 2^f where f in [0, 1)
    // Coefficients: ln(2), ln(2)^2/2!, ln(2)^3/3!
    let poly = 1.0 + f * (0.693147 + f * (0.240226 + f * 0.055504));
    // 2^i via IEEE 754 bit manipulation
    f32::from_bits(((i + 127) as u32) << 23) * poly
}

//=============================================================================
// Log-Scale Statistics
//=============================================================================

/// Statistics collected from blocks in log2 domain.
/// Avoids exp2 conversion during data gathering.
#[derive(Debug, Clone)]
pub struct LogGroupStats {
    /// Sum of log2_mag values (for geometric mean computation)
    pub log2_sum: f64,
    /// Maximum log2_mag for positive values
    pub log2_max: f32,
    /// Total element count
    pub count: usize,
    /// Count of negative values
    pub neg_count: usize,
    /// Maximum log2_mag for negative values
    pub neg_log2_max: f32,
}

impl Default for LogGroupStats {
    fn default() -> Self {
        Self {
            log2_sum: 0.0,
            log2_max: f32::NEG_INFINITY,
            count: 0,
            neg_count: 0,
            neg_log2_max: f32::NEG_INFINITY,
        }
    }
}

/// Collect log2 statistics from GMAT blocks without exp2 conversion.
/// This enables efficient scale computation in log domain.
#[inline]
pub fn log2_group_stats(gmat_blocks: &[&AnyBlock]) -> LogGroupStats {
    let mut stats = LogGroupStats::default();
    
    for blk in gmat_blocks {
        for (_, log2_mag, sign) in blk.log_iter() {
            stats.log2_sum += log2_mag as f64;
            stats.count += 1;
            
            if sign == 1 {
                // Negative value
                stats.neg_count += 1;
                stats.neg_log2_max = stats.neg_log2_max.max(log2_mag);
            } else {
                // Positive value
                stats.log2_max = stats.log2_max.max(log2_mag);
            }
        }
    }
    
    stats
}

/// Compute scale factor from log2_max using single exp2 call.
/// Formula: d = exp2(log2_max - log2(q_max)) = exp2(log2_max) / q_max
#[inline]
pub fn compute_scale_from_log2_max(log2_max: f32, q_max: u8) -> f16 {
    if log2_max.is_finite() && log2_max > f32::NEG_INFINITY {
        let scale = f32::exp2(log2_max) / (q_max as f32);
        f16::from_f32(scale)
    } else {
        f16::from_f32(1.0)
    }
}

//=============================================================================
// Super-block Scale Computation
//=============================================================================

/// Compute super-block scales (d, dmin) from GMAT blocks.
/// These are the f16 values stored in the GGUF block header.
/// Uses log-domain computation with single exp2 call per scale.
#[inline]
pub fn compute_superblock_scales(gmat_blocks: &[&AnyBlock]) -> (f16, f16) {
    let stats = log2_group_stats(gmat_blocks);
    
    // Compute d from maximum positive value: d = exp2(log2_max) / 63
    let d = compute_scale_from_log2_max(stats.log2_max, 63);
    
    // Compute dmin from maximum negative value: dmin = exp2(neg_log2_max) / 63
    let dmin = compute_scale_from_log2_max(stats.neg_log2_max, 63);
    
    (d, dmin)
}

//=============================================================================
// Format Selection
//=============================================================================

/// Determine best quant type given tensor dimensions and target.
/// Falls back to legacy format if K-quant alignment not met.
pub fn select_quant_with_fallback(cols: usize, target: GgufQuantType) -> GgufQuantType {
    use GgufQuantType::*;
    match target {
        // K-quants and I-quants require 256 alignment
        Q2_K | Q3_K_S | Q3_K_M | Q3_K_L | Q4_K_S | Q4_K_M | Q5_K_S | Q5_K_M | Q6_K | IQ1_S
        | IQ1_M | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ4_XS | IQ4_NL => {
            if cols % 256 == 0 {
                target
            } else if cols % 32 == 0 {
                Q8_0 // Fall back to legacy format
            } else {
                panic!("tensor cols must be multiple of 32")
            }
        }
        // Legacy formats require 32 alignment
        Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 => {
            if cols % 32 == 0 {
                target
            } else {
                panic!("tensor cols must be multiple of 32")
            }
        }
    }
}

//=============================================================================
// Block Size Helpers
//=============================================================================

/// Calculate how many GMAT blocks make up one K-quant group (32 elements)
#[inline]
pub fn gmat_blocks_per_group(gmat_block_size: usize) -> usize {
    32 / gmat_block_size
}

//=============================================================================
// Nibble and Bit Packing Utilities
//=============================================================================

/// Pack a 4-bit value into output at element index (nibble-packed format).
/// Even indices go in low nibble, odd indices in high nibble.
#[inline(always)]
pub fn pack_nibble(out: &mut [u8], offset: usize, elem_idx: usize, q: u8) {
    let byte_idx = offset + (elem_idx >> 1);
    let shift = (elem_idx & 1) << 2;  // 0 or 4
    let mask = 0x0F << shift;
    out[byte_idx] = (out[byte_idx] & !mask) | ((q & 0x0F) << shift);
}

/// Set high bit for 5-bit formats (bit 4 stored separately).
#[inline(always)]
pub fn set_high_bit_5(out: &mut [u8], high_offset: usize, elem_idx: usize, q: u8) {
    let high_bit = (q >> 4) & 1;
    out[high_offset + (elem_idx >> 3)] |= high_bit << (elem_idx & 7);
}

/// Set high 2 bits for 6-bit formats (bits 4-5 stored separately).
#[inline(always)]
pub fn set_high_bits_6(out: &mut [u8], high_offset: usize, elem_idx: usize, q: u8) {
    let high_bits = (q >> 4) & 0x03;
    let byte_idx = high_offset + (elem_idx >> 2);
    let bit_offset = (elem_idx & 3) << 1;
    out[byte_idx] |= high_bits << bit_offset;
}

/// Pack 2-bit value at element index.
#[inline(always)]
pub fn pack_2bit(out: &mut [u8], offset: usize, elem_idx: usize, q: u8) {
    let byte_idx = offset + (elem_idx >> 2);
    let shift = (elem_idx & 3) << 1;
    out[byte_idx] |= (q & 0x03) << shift;
}

//=============================================================================
// Log2-Domain Quantization
//=============================================================================

/// Binary search quantization in log2 domain. No per-element exp2 calls.
/// Thresholds must be sorted ascending. Returns bucket index [0, N).
#[inline]
pub fn log2_quantize<const N: usize>(log2_mag: f32, thresholds: &[f32; N]) -> u8 {
    let mut lo = 0usize;
    let mut hi = N;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if log2_mag > thresholds[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.min(N - 1) as u8
}

/// Build log2-domain thresholds for asymmetric quantization.
/// threshold[q] = log2((q + 0.5) * scale) = log2(q + 0.5) + log2_scale
#[inline]
pub fn build_log2_thresholds<const N: usize>(log2_scale: f32) -> [f32; N] {
    let mut thresholds = [f32::INFINITY; N];
    let mut q = 0usize;
    while q < N - 1 {
        thresholds[q] = log2_scale + (q as f32 + 0.5).log2();
        q += 1;
    }
    thresholds
}

//=============================================================================
// Additional Group Statistics
//=============================================================================

/// Compute max log2 magnitude across blocks. Returns NEG_INFINITY if empty.
#[inline]
pub fn log2_group_max(blocks: &[&AnyBlock]) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for blk in blocks {
        for (_, log2_mag, _) in blk.log_iter() {
            max = if log2_mag > max { log2_mag } else { max };
        }
    }
    max
}

/// Compute max absolute value in linear domain. Uses single exp2 call.
#[inline]
pub fn group_max_abs(blocks: &[&AnyBlock]) -> f32 {
    let log2_max = log2_group_max(blocks);
    if log2_max.is_finite() {
        fast_exp2(log2_max)
    } else {
        0.0
    }
}

//=============================================================================
// Generic Superblock Iteration
//=============================================================================

/// Context passed to superblock encoder functions.
pub struct SuperblockCtx<'a> {
    pub block_refs: &'a [&'a AnyBlock],
    pub gmat_block_size: usize,
}

/// Generic superblock quantization loop. Eliminates 7x code duplication.
#[inline]
pub fn quantize_superblocks<F>(
    all_blocks: &[&AnyBlock],
    rows: usize,
    cols: usize,
    gmat_block_size: usize,
    bytes_per_sb: usize,
    mut encode_sb: F,
) -> Vec<u8>
where
    F: FnMut(&mut [u8], SuperblockCtx<'_>),
{
    let gmat_blocks_per_row = cols / gmat_block_size;
    let gmat_blocks_per_sb = 256 / gmat_block_size;
    let superblocks_per_row = cols / 256;
    let total_bytes = rows * superblocks_per_row * bytes_per_sb;

    let mut output = vec![0u8; total_bytes];

    for row in 0..rows {
        let row_start = row * gmat_blocks_per_row;
        let out_row_start = row * superblocks_per_row * bytes_per_sb;

        for sb in 0..superblocks_per_row {
            let gmat_start = row_start + sb * gmat_blocks_per_sb;
            let gmat_end = gmat_start + gmat_blocks_per_sb;

            let out_offset = out_row_start + sb * bytes_per_sb;
            let out_slice = &mut output[out_offset..out_offset + bytes_per_sb];

            let ctx = SuperblockCtx {
                block_refs: &all_blocks[gmat_start..gmat_end],
                gmat_block_size,
            };

            encode_sb(out_slice, ctx);
        }
    }

    output
}

/// Legacy (32-element) block quantization loop.
#[inline]
pub fn quantize_legacy_blocks<F>(
    all_blocks: &[&AnyBlock],
    rows: usize,
    cols: usize,
    gmat_block_size: usize,
    bytes_per_block: usize,
    mut encode_fn: F,
) -> Vec<u8>
where
    F: FnMut(&[&AnyBlock]) -> Vec<u8>,
{
    let gmat_blocks_per_row = cols / gmat_block_size;
    let gmat_blocks_per_gguf = 32 / gmat_block_size;
    let gguf_blocks_per_row = cols / 32;

    let mut output = vec![0u8; rows * gguf_blocks_per_row * bytes_per_block];

    for row in 0..rows {
        let row_start = row * gmat_blocks_per_row;
        let out_row_start = row * gguf_blocks_per_row * bytes_per_block;

        for blk in 0..gguf_blocks_per_row {
            let gmat_start = row_start + blk * gmat_blocks_per_gguf;
            let gmat_end = gmat_start + gmat_blocks_per_gguf;

            let block_refs: Vec<&AnyBlock> = all_blocks[gmat_start..gmat_end].to_vec();
            let encoded = encode_fn(&block_refs);

            let out_offset = out_row_start + blk * bytes_per_block;
            output[out_offset..out_offset + encoded.len()].copy_from_slice(&encoded);
        }
    }

    output
}

//=============================================================================
// Generic Scale Computation
//=============================================================================

/// Compute per-group scales with custom scale function (unsigned).
#[inline]
pub fn compute_group_scales_u8<const N: usize, F>(
    gmat_blocks: &[&AnyBlock],
    d: f16,
    gmat_block_size: usize,
    elements_per_group: usize,
    scale_fn: F,
) -> [u8; N]
where
    F: Fn(f32, f32) -> u8,
{
    let mut scales = [0u8; N];
    let blocks_per_group = elements_per_group / gmat_block_size;
    let d_f32 = f32::from(d);

    for g in 0..N {
        let start = g * blocks_per_group;
        let end = (start + blocks_per_group).min(gmat_blocks.len());
        let log2_max = log2_group_max(&gmat_blocks[start..end]);

        if d_f32 > 1e-10 && log2_max.is_finite() {
            scales[g] = scale_fn(log2_max, d_f32);
        }
    }

    scales
}

/// Compute per-group scales with custom scale function (signed).
#[inline]
pub fn compute_group_scales_i8<const N: usize, F>(
    gmat_blocks: &[&AnyBlock],
    d: f16,
    gmat_block_size: usize,
    elements_per_group: usize,
    scale_fn: F,
) -> [i8; N]
where
    F: Fn(f32, f32) -> i8,
{
    let mut scales = [0i8; N];
    let blocks_per_group = elements_per_group / gmat_block_size;
    let d_f32 = f32::from(d);

    for g in 0..N {
        let start = g * blocks_per_group;
        let end = (start + blocks_per_group).min(gmat_blocks.len());
        let log2_max = log2_group_max(&gmat_blocks[start..end]);

        if d_f32 > 1e-10 && log2_max.is_finite() {
            scales[g] = scale_fn(log2_max, d_f32);
        }
    }

    scales
}
