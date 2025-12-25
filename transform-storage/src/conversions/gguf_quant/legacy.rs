//! Legacy GGUF quantization formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
//!
//! These formats use 32-element blocks with a single scale value.
//! No trellis optimization needed - straightforward quantization.

use half::f16;

use crate::blocks::AnyBlock;
use super::utils::{log2_group_stats, compute_scale_from_log2_max, log2_quantize, build_log2_thresholds, log2_group_max};

//=============================================================================
// Configuration
//=============================================================================

/// Legacy format configuration - const for zero-cost abstraction
#[derive(Clone, Copy)]
pub struct LegacyConfig {
    pub block_bytes: usize,
    pub quant_bits: u8,
    pub has_min: bool,
    pub header_bytes: usize,
    pub high_bits_bytes: usize,
    /// Quantization range max (7 for 4-bit sym, 15 for 4-bit asym, etc.)
    pub q_max: i8,
    /// Offset for symmetric formats (8 for 4-bit, 16 for 5-bit)
    pub q_offset: i8,
}

pub const Q8_0_CONFIG: LegacyConfig = LegacyConfig {
    block_bytes: 34,
    quant_bits: 8,
    has_min: false,
    header_bytes: 2,
    high_bits_bytes: 0,
    q_max: 127,
    q_offset: 0,
};

pub const Q4_0_CONFIG: LegacyConfig = LegacyConfig {
    block_bytes: 18,
    quant_bits: 4,
    has_min: false,
    header_bytes: 2,
    high_bits_bytes: 0,
    q_max: 7,
    q_offset: 8,
};

pub const Q4_1_CONFIG: LegacyConfig = LegacyConfig {
    block_bytes: 20,
    quant_bits: 4,
    has_min: true,
    header_bytes: 4,
    high_bits_bytes: 0,
    q_max: 15,
    q_offset: 0,
};

pub const Q5_0_CONFIG: LegacyConfig = LegacyConfig {
    block_bytes: 22,
    quant_bits: 5,
    has_min: false,
    header_bytes: 2,
    high_bits_bytes: 4,
    q_max: 15,
    q_offset: 16,
};

pub const Q5_1_CONFIG: LegacyConfig = LegacyConfig {
    block_bytes: 24,
    quant_bits: 5,
    has_min: true,
    header_bytes: 4,
    high_bits_bytes: 4,
    q_max: 31,
    q_offset: 0,
};

//=============================================================================
// Log-Domain Statistics (zero-copy, cache-efficient)
//=============================================================================


//=============================================================================
// Log-Domain LUT-Based Quantization for Asymmetric Formats
//=============================================================================



//=============================================================================
// Nibble Packing (shared, inlined)
//=============================================================================

#[inline(always)]
#[allow(dead_code)] // Used in tests
fn pack_nibble_pair(q0: u8, q1: u8) -> u8 {
    (q0 & 0x0F) | ((q1 & 0x0F) << 4)
}

#[inline(always)]
fn set_high_bit(out: &mut [u8], high_offset: usize, idx: usize, q: u8) {
    let high_bit = (q >> 4) & 1;
    out[high_offset + idx / 8] |= high_bit << (idx % 8);
}

#[inline(always)]
fn set_low_nibble(out: &mut [u8], low_offset: usize, idx: usize, q: u8) {
    let low_nibble = q & 0x0F;
    if idx.is_multiple_of(2) {
        out[low_offset + idx / 2] = low_nibble;
    } else {
        out[low_offset + idx / 2] |= low_nibble << 4;
    }
}

//=============================================================================
// Generic Encoder
//=============================================================================

/// Encode a legacy block using config-driven logic.
/// Uses log-domain statistics with single exp2 per scale computation.
#[inline]
pub fn encode_legacy_block(
    config: &LegacyConfig,
    gmat_blocks: &[&AnyBlock],
    out: &mut [u8],
) {
    debug_assert!(out.len() >= config.block_bytes);

    let (d, min_val, inv_scale) = if config.has_min {
        // Asymmetric: compute min/max from log2 stats (2 exp2 calls)
        let stats = log2_group_stats(gmat_blocks);
        
        let min_val = if stats.neg_count > 0 {
            -f32::exp2(stats.neg_log2_max)
        } else {
            0.0
        };
        
        let max_val = if stats.count > stats.neg_count {
            f32::exp2(stats.log2_max)
        } else {
            0.0
        };
        
        let range = max_val - min_val;
        let scale = range / config.q_max as f32;
        let inv = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
        (f16::from_f32(scale), min_val, inv)
    } else {
        // Symmetric: use log2_max with single exp2 call
        let log2_max = log2_group_max(gmat_blocks);
        let d = compute_scale_from_log2_max(log2_max, config.q_max as u8);
        let scale = d.to_f32();
        let inv = if scale > 1e-10 { 1.0 / scale } else { 0.0 };
        (d, 0.0, inv)
    };

    // Write header
    out[0..2].copy_from_slice(&d.to_le_bytes());
    if config.has_min {
        out[2..4].copy_from_slice(&f16::from_f32(min_val).to_le_bytes());
    }

    let block_size = gmat_blocks.first().map(|b| b.size()).unwrap_or(8);

    // Quantize based on bit width (per-element exp2 for GGUF compliance)
    match config.quant_bits {
        8 => encode_8bit(gmat_blocks, block_size, inv_scale, &mut out[2..]),
        4 if config.high_bits_bytes == 0 => {
            encode_4bit(gmat_blocks, block_size, inv_scale, min_val, config.q_offset, &mut out[config.header_bytes..])
        }
        5 => {
            let high_offset = config.header_bytes;
            let low_offset = high_offset + config.high_bits_bytes;
            encode_5bit(gmat_blocks, block_size, inv_scale, min_val, config.q_offset, out, high_offset, low_offset)
        }
        _ => unreachable!(),
    }
}

/// Encode 8-bit quantization using log-domain formula.
/// Uses fast_exp2(log2_mag - log2_scale) for efficient quantization.
#[inline]
fn encode_8bit(gmat_blocks: &[&AnyBlock], block_size: usize, inv_scale: f32, out: &mut [u8]) {
    let log2_scale = inv_scale.log2();
    for (blk_idx, blk) in gmat_blocks.iter().enumerate() {
        let base = blk_idx * block_size;
        for (idx, log2_mag, sign) in blk.log_iter() {
            let global_idx = base + idx;
            if global_idx < 32 {
                let q_mag = super::utils::fast_exp2(log2_mag - log2_scale).round().clamp(0.0, 127.0);
                let q = if sign == 1 { -(q_mag as i8) } else { q_mag as i8 };
                out[global_idx] = q as u8;
            }
        }
    }
}

/// Encode 4-bit quantization using log-domain formula.
/// Symmetric (Q4_0): uses fast_exp2(log2_mag - log2_scale) - single exp2 per element
/// Asymmetric (Q4_1): uses log2-domain LUT - no per-element exp2
#[inline]
fn encode_4bit(
    gmat_blocks: &[&AnyBlock],
    block_size: usize,
    inv_scale: f32,
    _min_val: f32,
    offset: i8,
    out: &mut [u8],
) {
    // Initialize output
    for byte in out.iter_mut() {
        *byte = 0;
    }

    // Compute log2_scale once for the block
    let log2_scale = inv_scale.log2();

    if offset != 0 {
        // Symmetric (Q4_0): use fast_exp2 with log2 formula
        // q = round(exp2(log2_mag - log2_scale)) * sign + offset
        for (blk_idx, blk) in gmat_blocks.iter().enumerate() {
            let base = blk_idx * block_size;
            for (idx, log2_mag, sign) in blk.log_iter() {
                let global_idx = base + idx;
                if global_idx < 32 {
                    let q_mag = super::utils::fast_exp2(log2_mag - log2_scale).round().clamp(0.0, 7.0);
                    let q_signed = if sign == 1 { -(q_mag as i8) } else { q_mag as i8 };
                    let q = (q_signed + offset) as u8;

                    // Pack nibble
                    let byte_idx = global_idx / 2;
                    if global_idx.is_multiple_of(2) {
                        out[byte_idx] = (out[byte_idx] & 0xF0) | (q & 0x0F);
                    } else {
                        out[byte_idx] = (out[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
                    }
                }
            }
        }
    } else {
        // Asymmetric (Q4_1): use log2-domain LUT - no per-element exp2
        // Build log2 thresholds once for the block (O(16) work)
        let log2_thresholds = build_log2_thresholds::<16>(-log2_scale);

        for (blk_idx, blk) in gmat_blocks.iter().enumerate() {
            let base = blk_idx * block_size;
            for (idx, log2_mag, sign) in blk.log_iter() {
                let global_idx = base + idx;
                if global_idx < 32 {
                    // Log2-domain quantization: no exp2 per element
                    // Negative values map to 0
                    let q = if sign == 1 {
                        0u8
                    } else {
                        log2_quantize::<16>(log2_mag, &log2_thresholds)
                    };

                    // Pack nibble
                    let byte_idx = global_idx / 2;
                    if global_idx.is_multiple_of(2) {
                        out[byte_idx] = (out[byte_idx] & 0xF0) | (q & 0x0F);
                    } else {
                        out[byte_idx] = (out[byte_idx] & 0x0F) | ((q & 0x0F) << 4);
                    }
                }
            }
        }
    }
}

/// Encode 5-bit quantization using log-domain formula.
/// Symmetric (Q5_0): uses fast_exp2(log2_mag - log2_scale) - single exp2 per element
/// Asymmetric (Q5_1): uses log2-domain LUT - no per-element exp2
#[inline]
fn encode_5bit(
    gmat_blocks: &[&AnyBlock],
    block_size: usize,
    inv_scale: f32,
    _min_val: f32,
    offset: i8,
    out: &mut [u8],
    high_offset: usize,
    low_offset: usize,
) {
    // Clear high bits section
    for i in 0..4 {
        out[high_offset + i] = 0;
    }

    // Clear low nibbles section
    for i in 0..16 {
        out[low_offset + i] = 0;
    }

    // Compute log2_scale once for the block
    let log2_scale = inv_scale.log2();

    if offset != 0 {
        // Symmetric (Q5_0): use fast_exp2 with log2 formula
        // q = round(exp2(log2_mag - log2_scale)) * sign + offset
        for (blk_idx, blk) in gmat_blocks.iter().enumerate() {
            let base = blk_idx * block_size;
            for (idx, log2_mag, sign) in blk.log_iter() {
                let global_idx = base + idx;
                if global_idx < 32 {
                    let q_mag = super::utils::fast_exp2(log2_mag - log2_scale).round().clamp(0.0, 15.0);
                    let q_signed = if sign == 1 { -(q_mag as i8) } else { q_mag as i8 };
                    let q = (q_signed + offset) as u8;

                    set_high_bit(out, high_offset, global_idx, q);
                    set_low_nibble(out, low_offset, global_idx, q);
                }
            }
        }
    } else {
        // Asymmetric (Q5_1): use log2-domain LUT - no per-element exp2
        // Build log2 thresholds once for the block (O(32) work)
        let log2_thresholds = build_log2_thresholds::<32>(-log2_scale);

        for (blk_idx, blk) in gmat_blocks.iter().enumerate() {
            let base = blk_idx * block_size;
            for (idx, log2_mag, sign) in blk.log_iter() {
                let global_idx = base + idx;
                if global_idx < 32 {
                    // Log2-domain quantization: no exp2 per element
                    // Negative values map to 0
                    let q = if sign == 1 {
                        0u8
                    } else {
                        log2_quantize::<32>(log2_mag, &log2_thresholds)
                    };

                    set_high_bit(out, high_offset, global_idx, q);
                    set_low_nibble(out, low_offset, global_idx, q);
                }
            }
        }
    }
}

//=============================================================================
// Public API (thin wrappers for compatibility)
//=============================================================================

/// Q8_0: f16 scale + 32 x int8 quants
pub fn blocks_to_q8_0(gmat_blocks: &[&AnyBlock]) -> Vec<u8> {
    let mut out = vec![0u8; Q8_0_CONFIG.block_bytes];
    encode_legacy_block(&Q8_0_CONFIG, gmat_blocks, &mut out);
    out
}

/// Q4_0: f16 scale + 16 bytes nibble-packed (symmetric)
pub fn blocks_to_q4_0(gmat_blocks: &[&AnyBlock]) -> Vec<u8> {
    let mut out = vec![0u8; Q4_0_CONFIG.block_bytes];
    encode_legacy_block(&Q4_0_CONFIG, gmat_blocks, &mut out);
    out
}

/// Q4_1: f16 scale + f16 min + 16 bytes nibble-packed (asymmetric)
pub fn blocks_to_q4_1(gmat_blocks: &[&AnyBlock]) -> Vec<u8> {
    let mut out = vec![0u8; Q4_1_CONFIG.block_bytes];
    encode_legacy_block(&Q4_1_CONFIG, gmat_blocks, &mut out);
    out
}

/// Q5_0: f16 scale + 4B high bits + 16B low nibbles (symmetric)
pub fn blocks_to_q5_0(gmat_blocks: &[&AnyBlock]) -> Vec<u8> {
    let mut out = vec![0u8; Q5_0_CONFIG.block_bytes];
    encode_legacy_block(&Q5_0_CONFIG, gmat_blocks, &mut out);
    out
}

/// Q5_1: f16 scale + f16 min + 4B high bits + 16B low nibbles (asymmetric)
pub fn blocks_to_q5_1(gmat_blocks: &[&AnyBlock]) -> Vec<u8> {
    let mut out = vec![0u8; Q5_1_CONFIG.block_bytes];
    encode_legacy_block(&Q5_1_CONFIG, gmat_blocks, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_sizes() {
        assert_eq!(Q8_0_CONFIG.block_bytes, 34);
        assert_eq!(Q4_0_CONFIG.block_bytes, 18);
        assert_eq!(Q4_1_CONFIG.block_bytes, 20);
        assert_eq!(Q5_0_CONFIG.block_bytes, 22);
        assert_eq!(Q5_1_CONFIG.block_bytes, 24);
    }

    #[test]
    fn test_pack_nibble_pair() {
        assert_eq!(pack_nibble_pair(0x05, 0x0A), 0xA5);
        assert_eq!(pack_nibble_pair(0x0F, 0x0F), 0xFF);
        assert_eq!(pack_nibble_pair(0x00, 0x00), 0x00);
    }
}
