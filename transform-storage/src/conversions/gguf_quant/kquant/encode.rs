//! K-Quant block encoding

use super::super::compute;
use super::super::utils::{fast_exp2, iter_group_elements, pack_nibble, set_high_bit_5, set_high_bits_6};
use super::config::*;
use super::scales::encode_q4k_scales;
use crate::blocks::AnyBlock;
use half::f16;

//=============================================================================
// LUT-Based Log-Domain Quantization
//=============================================================================

#[inline]
fn build_log2_positive_lut<const N: usize>(log2_ds: f32) -> [f32; N] {
    let mut thresholds = [f32::INFINITY; N];
    for (q, threshold) in thresholds.iter_mut().enumerate().take(N - 1) {
        *threshold = log2_ds + (q as f32 + 0.5).log2();
    }
    thresholds
}

#[inline]
fn log2_quantize<const N: usize>(log2_mag: f32, log2_thresholds: &[f32; N]) -> u8 {
    let mut lo = 0;
    let mut hi = N;
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if log2_mag > log2_thresholds[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.min(N - 1) as u8
}

//=============================================================================
// Q2_K Log-Domain Helpers
//=============================================================================

#[inline]
fn build_q2k_log2_thresholds(log2_d: f32, log2_scale: f32) -> [f32; 3] {
    let log2_ds = log2_d + log2_scale;
    [
        log2_ds + 0.5f32.log2(),
        log2_ds + 1.5f32.log2(),
        log2_ds + 2.5f32.log2(),
    ]
}

//=============================================================================
// Generic K-Quant Block Encoder
//=============================================================================

/// Encode a K-quant super-block using config-driven logic
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn encode_kquant_block(
    config: &KQuantConfig,
    output: &mut [u8],
    d: f16,
    dmin: Option<f16>,
    scales: &[u8],
    mins: Option<&[u8]>,
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    debug_assert!(output.len() >= config.block_bytes);
    debug_assert!(scales.len() >= config.num_groups);

    let blocks_per_grp = config.elements_per_group / gmat_block_size;
    let d_f32 = f32::from(d);

    if config.high_bits_offset.is_some() {
        output[..config.block_bytes].fill(0);
    }

    // Encode header
    match config.quant_bits {
        4 | 5 if config.has_min => {
            output[0..2].copy_from_slice(&d.to_le_bytes());
            output[2..4].copy_from_slice(&dmin.unwrap().to_le_bytes());
            let scales_arr: [u8; 8] = scales[..8].try_into().unwrap();
            let mins_arr: [u8; 8] = mins.unwrap()[..8].try_into().unwrap();
            output[4..16].copy_from_slice(&encode_q4k_scales(&scales_arr, &mins_arr));
        }
        6 => {
            for (i, &s) in scales.iter().take(16).enumerate() {
                output[192 + i] = s;
            }
            output[208..210].copy_from_slice(&d.to_le_bytes());
        }
        _ => unreachable!(),
    }

    let log2_d = d_f32.log2();
    let log2_dmin = dmin.map(|dm| f32::from(dm).log2());

    for (g, &scale) in scales.iter().enumerate().take(config.num_groups) {
        let log2_scale = (scale as f32).max(1.0).log2();
        let log2_ds = log2_d + log2_scale;

        match config.quant_bits {
            4 if config.has_min => {
                let log2_thresholds = build_log2_positive_lut::<16>(log2_ds);
                let _log2_min_scale =
                    log2_dmin.unwrap() + (mins.unwrap()[g] as f32).max(1.0).log2();

                for (elem_idx, log2_mag, sign) in iter_group_elements(
                    gmat_blocks, g, blocks_per_grp, gmat_block_size, config.elements_per_group
                ) {
                    let q = if sign == 1 {
                        0u8
                    } else {
                        log2_quantize::<16>(log2_mag, &log2_thresholds)
                    };
                    pack_nibble(output, config.quants_offset, elem_idx, q);
                }
            }
            5 if config.has_min => {
                let log2_thresholds = build_log2_positive_lut::<32>(log2_ds);

                for (elem_idx, log2_mag, sign) in iter_group_elements(
                    gmat_blocks, g, blocks_per_grp, gmat_block_size, config.elements_per_group
                ) {
                    let q = if sign == 1 {
                        0u8
                    } else {
                        log2_quantize::<32>(log2_mag, &log2_thresholds)
                    };
                    set_high_bit_5(output, config.high_bits_offset.unwrap(), elem_idx, q);
                    pack_nibble(output, config.quants_offset, elem_idx, q & 0x0F);
                }
            }
            6 => {
                for (elem_idx, log2_mag, sign) in iter_group_elements(
                    gmat_blocks, g, blocks_per_grp, gmat_block_size, config.elements_per_group
                ) {
                    let q = compute::compute_q6k(log2_mag, sign, log2_ds);
                    pack_nibble(output, config.quants_offset, elem_idx, q & 0x0F);
                    set_high_bits_6(output, config.high_bits_offset.unwrap(), elem_idx, q);
                }
            }
            _ => {
                let scale = d_f32 * (scales[g] as f32);
                let _inv_scale = if scale > 1e-10 { 1.0 / scale } else { 0.0 };

                for (elem_idx, log2_mag, sign) in iter_group_elements(
                    gmat_blocks, g, blocks_per_grp, gmat_block_size, config.elements_per_group
                ) {
                    let q_mag = fast_exp2(log2_mag - log2_ds)
                        .round()
                        .clamp(0.0, config.q_max as f32);
                    let q = if sign == 1 { 0 } else { q_mag as u8 };
                    pack_nibble(output, config.quants_offset, elem_idx, q);
                }
            }
        }
    }
}

//=============================================================================
// Format-Specific Wrappers
//=============================================================================

pub fn encode_q4k_block(
    output: &mut [u8],
    d: f16,
    dmin: f16,
    scales: &[u8; 8],
    mins: &[u8; 8],
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    encode_kquant_block(
        &Q4_K_CONFIG,
        output,
        d,
        Some(dmin),
        scales,
        Some(mins),
        gmat_blocks,
        gmat_block_size,
    );
}

pub fn encode_q5k_block(
    output: &mut [u8],
    d: f16,
    dmin: f16,
    scales: &[u8; 8],
    mins: &[u8; 8],
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    encode_kquant_block(
        &Q5_K_CONFIG,
        output,
        d,
        Some(dmin),
        scales,
        Some(mins),
        gmat_blocks,
        gmat_block_size,
    );
}

pub fn encode_q6k_block(
    output: &mut [u8],
    d: f16,
    scales: &[i8; 16],
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    let scales_u8: [u8; 16] = scales.map(|s| s as u8);
    encode_kquant_block(
        &Q6_K_CONFIG,
        output,
        d,
        None,
        &scales_u8,
        None,
        gmat_blocks,
        gmat_block_size,
    );
}

pub fn encode_q2k_block(
    output: &mut [u8],
    d: f16,
    dmin: f16,
    scales: &[u8; 16],
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    debug_assert!(output.len() >= Q2_K_CONFIG.block_bytes);
    output[..Q2_K_CONFIG.block_bytes].fill(0);

    output[0..2].copy_from_slice(&d.to_le_bytes());
    output[2..4].copy_from_slice(&dmin.to_le_bytes());

    for (g, &scale) in scales.iter().enumerate().take(16) {
        output[4 + g] = scale & 0x0F;
    }

    let log2_d = f32::from(d).log2();
    let blocks_per_group = 16 / gmat_block_size;

    for (g, &scale) in scales.iter().enumerate().take(16) {
        let log2_scale = (scale as f32).max(1.0).log2();
        let thresholds = build_q2k_log2_thresholds(log2_d, log2_scale);

        for (elem_idx, log2_mag, _sign) in iter_group_elements(
            gmat_blocks, g, blocks_per_group, gmat_block_size, 16
        ) {
            let q = compute::compute_q2k(log2_mag, &thresholds);
            let byte_idx = 20 + elem_idx / 4;
            let shift = (elem_idx % 4) * 2;
            output[byte_idx] |= (q & 0x03) << shift;
        }
    }
}

pub fn encode_q3k_block(
    output: &mut [u8],
    d: f16,
    scales: &[i8; 16],
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    debug_assert!(output.len() >= Q3_K_CONFIG.block_bytes);
    output[..Q3_K_CONFIG.block_bytes].fill(0);

    output[0..2].copy_from_slice(&d.to_le_bytes());

    for (i, &s) in scales.iter().enumerate().take(12) {
        output[98 + i] = s as u8;
    }

    let log2_d = f32::from(d).log2();
    let blocks_per_group = 16 / gmat_block_size;

    for (g, &scale) in scales.iter().enumerate().take(16) {
        let scale_abs = scale.abs() as f32;
        let log2_scale = if scale_abs > 0.0 {
            log2_d + scale_abs.log2()
        } else {
            f32::NEG_INFINITY
        };

        for (elem_idx, log2_mag, sign) in iter_group_elements(
            gmat_blocks, g, blocks_per_group, gmat_block_size, 16
        ) {
            let q_mag = compute::compute_q3k_mag(log2_mag, log2_scale);

            let quant_byte_idx = 34 + elem_idx / 4;
            let quant_shift = (elem_idx % 4) * 2;
            output[quant_byte_idx] |= (q_mag & 0x03) << quant_shift;

            if sign == 1 {
                let hmask_byte_idx = 2 + elem_idx / 8;
                let hmask_bit = elem_idx % 8;
                output[hmask_byte_idx] |= 1 << hmask_bit;
            }
        }
    }
}

//=============================================================================
// Dequantization (for validation)
//=============================================================================

#[inline]
#[allow(dead_code)]
pub fn dequantize_q4k_group(d: f16, dmin: f16, sc: u8, m: u8, quants: &[u8; 16]) -> [f32; 32] {
    let scale = f32::from(d) * (sc as f32);
    let min = f32::from(dmin) * (m as f32);

    let mut out = [0f32; 32];
    for i in 0..16 {
        out[i * 2] = scale * ((quants[i] & 0x0F) as f32) - min;
        out[i * 2 + 1] = scale * ((quants[i] >> 4) as f32) - min;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_dequantize() {
        let d = f16::from_f32(0.1);
        let dmin = f16::from_f32(0.05);
        let sc = 32u8;
        let m = 16u8;
        let quants = [0x21u8; 16];

        let out = dequantize_q4k_group(d, dmin, sc, m, &quants);

        let expected_low = 3.2 * 1.0 - 0.8;
        let expected_high = 3.2 * 2.0 - 0.8;

        assert!((out[0] - expected_low).abs() < 0.01);
        assert!((out[1] - expected_high).abs() < 0.01);
    }

    #[test]
    fn test_config_sizes() {
        assert_eq!(Q4_K_CONFIG.block_bytes, 144);
        assert_eq!(Q5_K_CONFIG.block_bytes, 176);
        assert_eq!(Q6_K_CONFIG.block_bytes, 210);
    }
}
