//! K-Quant scale computation and packing

use half::f16;
use crate::blocks::AnyBlock;
use super::super::utils::{fast_exp2, compute_group_scales_u8, compute_group_scales_i8};

//=============================================================================
// Q4_K Scale Packing (12 bytes for 8 groups)
//=============================================================================

/// Decode Q4_K packed scales from 12 bytes
#[inline]
#[allow(dead_code)]
pub fn decode_q4k_scales(packed: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for i in 0..4 {
        scales[i] = packed[i] & 0x3F;
        mins[i] = packed[4 + i] & 0x3F;
    }
    for i in 0..4 {
        scales[4 + i] = (packed[8 + i] & 0x0F) | ((packed[i] >> 2) & 0x30);
        mins[4 + i] = (packed[8 + i] >> 4) | ((packed[4 + i] >> 2) & 0x30);
    }

    (scales, mins)
}

/// Encode Q4_K scales into 12 packed bytes
#[inline]
pub fn encode_q4k_scales(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
    let mut packed = [0u8; 12];

    for i in 0..4 {
        packed[i] = (scales[i] & 0x3F) | ((scales[4 + i] & 0x30) << 2);
        packed[4 + i] = (mins[i] & 0x3F) | ((mins[4 + i] & 0x30) << 2);
    }
    for i in 0..4 {
        packed[8 + i] = (scales[4 + i] & 0x0F) | ((mins[4 + i] & 0x0F) << 4);
    }

    packed
}

//=============================================================================
// Scale Computation (using generic helpers)
//=============================================================================

/// Q4_K/Q5_K standard scales (8 groups, 6-bit unsigned)
#[inline]
pub fn compute_scales_standard(
    blocks: &[&AnyBlock],
    d: f16,
    _dmin: f16,
    gmat_block_size: usize,
) -> [u8; 8] {
    compute_group_scales_u8::<8, _>(blocks, d, gmat_block_size, 32, |log2_max, d_f32| {
        let max_abs = fast_exp2(log2_max);
        ((max_abs / d_f32).round() as u8).min(63)
    })
}

/// Q6_K scales (16 groups, signed 8-bit)
#[inline]
pub fn compute_q6k_scales(
    gmat_blocks: &[&AnyBlock],
    d: f16,
    gmat_block_size: usize,
) -> [i8; 16] {
    compute_group_scales_i8::<16, _>(gmat_blocks, d, gmat_block_size, 16, |log2_max, d_f32| {
        let max_abs = fast_exp2(log2_max);
        let scale = (max_abs / (31.0 * d_f32)).round().clamp(-128.0, 127.0) as i8;
        scale.max(1)
    })
}

/// Q2_K scales (16 groups, 4-bit unsigned)
#[inline]
pub fn compute_q2k_scales(
    gmat_blocks: &[&AnyBlock],
    d: f16,
    gmat_block_size: usize,
) -> [u8; 16] {
    let log2_d = f32::from(d).log2();
    compute_group_scales_u8::<16, _>(gmat_blocks, d, gmat_block_size, 16, |log2_max, _d_f32| {
        let scale = fast_exp2(log2_max - log2_d);
        (scale.round() as u8).min(15)
    })
}

/// Q3_K scales (16 groups, signed)
#[inline]
pub fn compute_q3k_scales(
    gmat_blocks: &[&AnyBlock],
    d: f16,
    gmat_block_size: usize,
) -> [i8; 16] {
    compute_group_scales_i8::<16, _>(gmat_blocks, d, gmat_block_size, 16, |log2_max, d_f32| {
        let max_abs = fast_exp2(log2_max);
        let scale = (max_abs / (3.0 * d_f32)).round().clamp(-128.0, 127.0) as i8;
        scale.max(1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_scale_roundtrip() {
        let scales = [1, 15, 31, 63, 7, 22, 44, 55];
        let mins = [0, 8, 16, 32, 3, 19, 38, 60];

        let packed = encode_q4k_scales(&scales, &mins);
        let (dec_scales, dec_mins) = decode_q4k_scales(&packed);

        assert_eq!(scales, dec_scales);
        assert_eq!(mins, dec_mins);
    }

    #[test]
    fn test_q4k_scale_edge_cases() {
        let scales = [0u8; 8];
        let mins = [0u8; 8];
        let packed = encode_q4k_scales(&scales, &mins);
        let (dec_scales, dec_mins) = decode_q4k_scales(&packed);
        assert_eq!(scales, dec_scales);
        assert_eq!(mins, dec_mins);

        let scales = [63u8; 8];
        let mins = [63u8; 8];
        let packed = encode_q4k_scales(&scales, &mins);
        let (dec_scales, dec_mins) = decode_q4k_scales(&packed);
        assert_eq!(scales, dec_scales);
        assert_eq!(mins, dec_mins);
    }
}
