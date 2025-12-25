//! I-Quant formats (IQ1 through IQ4)
//!
//! I-quants use learned importance matrices and non-linear lookup tables
//! for higher quality than K-quants at the same size.
//!
//! Key differences from K-quants:
//! - Non-linear lookup tables instead of linear d * sc * q
//! - Importance weighting is built into the format design
//! - Scale encoding differs per format

use half::f16;

use super::utils::{group_max_abs, pack_nibble};
use crate::blocks::AnyBlock;

//=============================================================================
// Configuration
//=============================================================================

/// I-Quant format configuration - const for zero-cost abstraction
#[derive(Clone, Copy)]
pub struct IQuantConfig {
    pub block_bytes: usize,
    pub num_groups: usize,
    pub elements_per_group: usize,
    pub has_group_scales: bool,
    pub quants_offset: usize,
}

pub const IQ4_XS_CONFIG: IQuantConfig = IQuantConfig {
    block_bytes: 136,
    num_groups: 8,
    elements_per_group: 32,
    has_group_scales: true,
    quants_offset: 8,
};

pub const IQ4_NL_CONFIG: IQuantConfig = IQuantConfig {
    block_bytes: 130,
    num_groups: 1,
    elements_per_group: 256,
    has_group_scales: false,
    quants_offset: 2,
};

//=============================================================================
// Lookup Tables
//=============================================================================

/// IQ4_NL lookup table (from llama.cpp)
/// Non-linear 4-bit quantization levels - sorted for binary search
pub const IQ4_NL_QUANTS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Log2 thresholds for IQ4_NL binary search.
/// Midpoints between adjacent IQ4_NL_QUANTS values in log2 domain.
/// Thresholds[i] = log2((|QUANTS[i]| + |QUANTS[i+1]|) / 2)
pub const IQ4_NL_LOG2_THRESHOLDS: [f32; 15] = [
    6.8522818, // log2((127 + 104) / 2) = log2(115.5)
    6.5466433, // log2((104 + 83) / 2) = log2(93.5)
    6.2094533, // log2((83 + 65) / 2) = log2(74.0)
    5.8329900, // log2((65 + 49) / 2) = log2(57.0)
    5.3923174, // log2((49 + 35) / 2) = log2(42.0)
    4.8329900, // log2((35 + 22) / 2) = log2(28.5)
    4.0,       // log2((22 + 10) / 2) = log2(16.0)
    2.4594316, // log2((10 + 1) / 2) = log2(5.5)
    2.8073549, // log2((1 + 13) / 2) = log2(7.0)
    4.2479275, // log2((13 + 25) / 2) = log2(19.0)
    4.9772799, // log2((25 + 38) / 2) = log2(31.5)
    5.5076081, // log2((38 + 53) / 2) = log2(45.5)
    5.9307373, // log2((53 + 69) / 2) = log2(61.0)
    6.3038369, // log2((69 + 89) / 2) = log2(79.0)
    6.6582115, // log2((89 + 113) / 2) = log2(101.0)
];

/// Find the IQ4_NL index that best represents the target value
/// Uses binary search for O(log n) performance
#[inline]
#[allow(dead_code)] // Used in tests
fn find_nearest_iq4nl(target: f32) -> u8 {
    // Binary search for insertion point
    let mut lo = 0usize;
    let mut hi = 16usize;

    while lo < hi {
        let mid = (lo + hi) / 2;
        if (IQ4_NL_QUANTS[mid] as f32) < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    // Check which neighbor is closer
    if lo == 0 {
        0
    } else if lo >= 16 {
        15
    } else {
        let dist_lo = (target - IQ4_NL_QUANTS[lo - 1] as f32).abs();
        let dist_hi = (target - IQ4_NL_QUANTS[lo] as f32).abs();
        // When equidistant, prefer the less extreme (higher index) value
        if dist_lo < dist_hi {
            (lo - 1) as u8
        } else {
            lo as u8
        }
    }
}

/// Find the IQ4_NL index using log2 domain binary search
/// Avoids exp2 calls by working directly in log2 space
#[inline]
fn find_nearest_iq4nl_log2(log2_mag: f32, log2_scale: f32) -> u8 {
    let normalized = log2_mag - log2_scale;

    // Binary search in log2 thresholds
    let mut lo = 0;
    let mut hi = 15;

    while lo < hi {
        let mid = (lo + hi) / 2;
        if normalized > IQ4_NL_LOG2_THRESHOLDS[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    lo as u8
}

/// Fast IQ4_NL index lookup using unrolled decision tree.
/// Replaces binary search with fixed 4-comparison decision tree.
///
/// # Performance
/// - Guaranteed 4 comparisons (vs ~3.9 average for binary search)
/// - Better branch prediction due to fixed structure
/// - No loop overhead
///
/// # Arguments
/// - `log2_mag`: log2 of absolute value
/// - `log2_scale`: log2 of scale factor
///
/// # Returns
/// Index (0-15) into IQ4_NL_QUANTS lookup table
#[inline(always)]
fn find_iq4nl_index_fast(log2_mag: f32, log2_scale: f32) -> u8 {
    let normalized = log2_mag - log2_scale;

    // Balanced decision tree: 16 outcomes, 4 levels, 4 comparisons max
    // Thresholds from IQ4_NL_LOG2_THRESHOLDS array
    if normalized > 2.4594316 {
        // THRESHOLDS[7] - root split
        // Indices 0-7 (higher magnitudes)
        if normalized > 5.8329900 {
            // THRESHOLDS[3]
            // Indices 0-3
            if normalized > 6.5466433 {
                // THRESHOLDS[1]
                // Indices 0-1
                if normalized > 6.8522818 {
                    0
                } else {
                    1
                } // THRESHOLDS[0]
            } else {
                // Indices 2-3
                if normalized > 6.2094533 {
                    2
                } else {
                    3
                } // THRESHOLDS[2]
            }
        } else {
            // Indices 4-7
            if normalized > 4.8329900 {
                // THRESHOLDS[5]
                // Indices 4-5
                if normalized > 5.3923174 {
                    4
                } else {
                    5
                } // THRESHOLDS[4]
            } else {
                // Indices 6-7
                if normalized > 4.0 {
                    6
                } else {
                    7
                } // THRESHOLDS[6]
            }
        }
    } else {
        // Indices 8-15 (lower magnitudes)
        if normalized > 5.5076081 {
            // THRESHOLDS[11]
            // Indices 8-11
            if normalized > 4.2479275 {
                // THRESHOLDS[9]
                // Indices 8-9
                if normalized > 2.8073549 {
                    8
                } else {
                    9
                } // THRESHOLDS[8]
            } else {
                // Indices 10-11
                if normalized > 4.9772799 {
                    10
                } else {
                    11
                } // THRESHOLDS[10]
            }
        } else {
            // Indices 12-15
            if normalized > 6.3038369 {
                // THRESHOLDS[13]
                // Indices 12-13
                if normalized > 5.9307373 {
                    12
                } else {
                    13
                } // THRESHOLDS[12]
            } else {
                // Indices 14-15
                if normalized > 6.6582115 {
                    14
                } else {
                    15
                } // THRESHOLDS[14]
            }
        }
    }
}

/// Dequantize a single IQ4_NL value
#[inline]
#[allow(dead_code)] // For validation/debugging
pub fn dequantize_iq4_nl(d: f16, q: u8) -> f32 {
    f32::from(d) * (IQ4_NL_QUANTS[q as usize] as f32)
}

//=============================================================================
// E1M7 to IQ4_NL LUT Cache
//=============================================================================

/// Cached LUT for e1m7 encoded values to IQ4_NL indices.
/// 512 entries: [0..255] non-shifted, [256..511] shifted by +2.0 octaves.
pub type Iq4nlE1m7Lut = [u8; 512];

/// Build LUT mapping e1m7 encoded bytes to IQ4_NL indices.
///
/// # Arguments
/// - `base_offset`: block.scale_log - d.log2() (shared across block)
///
/// The LUT maps raw e1m7 byte values directly to IQ4_NL indices,
/// avoiding per-element log2 arithmetic.
#[inline]
pub fn build_iq4nl_lut_e1m7(base_offset: f32) -> Iq4nlE1m7Lut {
    let mut lut = [0u8; 512];

    for raw in 0..256u16 {
        // Decode e1m7: e = raw >> 7, m = (raw & 0x7F) / 128.0, offset = e + m
        let e = (raw >> 7) as f32;
        let m = (raw & 0x7F) as f32 / 128.0;
        let log2_offset = e + m;

        // Non-shifted entry
        let normalized = base_offset + log2_offset;
        lut[raw as usize] = find_iq4nl_index_fast(normalized, 0.0);

        // Shifted entry (+2.0 octaves from shift_map bit)
        let normalized_shifted = base_offset + log2_offset + 2.0;
        lut[256 + raw as usize] = find_iq4nl_index_fast(normalized_shifted, 0.0);
    }

    lut
}

//=============================================================================
// IQ4_XS Scale Encoding
//=============================================================================

/// Encode IQ4_XS scales: 8 groups × 6-bit scales
/// Layout: scales_h (u16) + scales_l (4 bytes)
#[inline]
pub fn encode_iq4xs_scales(scales: &[u8; 8]) -> (u16, [u8; 4]) {
    let mut scales_h: u16 = 0;
    let mut scales_l = [0u8; 4];

    for (i, &s) in scales.iter().enumerate() {
        // High 2 bits into scales_h
        scales_h |= ((s >> 4) as u16 & 0x3) << (i * 2);
        // Low 4 bits into scales_l (nibble-packed)
        if i % 2 == 0 {
            scales_l[i / 2] = s & 0x0F;
        } else {
            scales_l[i / 2] |= (s & 0x0F) << 4;
        }
    }

    (scales_h, scales_l)
}

/// Decode IQ4_XS scales from packed format
#[inline]
#[allow(dead_code)] // Used in tests
pub fn decode_iq4xs_scales(scales_h: u16, scales_l: &[u8; 4]) -> [u8; 8] {
    let mut scales = [0u8; 8];

    for i in 0..8 {
        let high = ((scales_h >> (i * 2)) & 0x3) as u8;
        let low = if i % 2 == 0 {
            scales_l[i / 2] & 0x0F
        } else {
            scales_l[i / 2] >> 4
        };
        scales[i] = (high << 4) | low;
    }

    scales
}

//=============================================================================
// Shared Utilities
//=============================================================================

//=============================================================================
// Generic I-Quant Block Encoder
//=============================================================================

/// Encode an I-quant super-block using config-driven logic
#[inline]
pub fn encode_iquant_block(
    config: &IQuantConfig,
    output: &mut [u8],
    d: f16,
    scales: Option<&[u8; 8]>,
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    debug_assert!(output.len() >= config.block_bytes);

    let blocks_per_grp = config.elements_per_group / gmat_block_size;
    let d_f32 = f32::from(d);

    // Write header
    output[0..2].copy_from_slice(&d.to_le_bytes());

    if config.has_group_scales {
        // IQ4_XS: encode scales
        let scales = scales.unwrap();
        let (scales_h, scales_l) = encode_iq4xs_scales(scales);
        output[2..4].copy_from_slice(&scales_h.to_le_bytes());
        output[4..8].copy_from_slice(&scales_l);
    }

    // Clear quants section
    output[config.quants_offset..config.block_bytes].fill(0);

    // LUT cache for IQ4_NL optimization (8-bit GMAT blocks only)
    let mut cached_base_offset: Option<f32> = None;
    let mut cached_lut: Option<Iq4nlE1m7Lut> = None;

    // Quantize each group
    for g in 0..config.num_groups {
        let effective_scale = if config.has_group_scales {
            d_f32 * (scales.unwrap()[g] as f32)
        } else {
            d_f32
        };

        // Compute log2_scale once per group for log-domain quantization
        let log2_scale = if effective_scale.abs() > 1e-10 {
            effective_scale.log2()
        } else {
            0.0
        };

        let group_start = g * blocks_per_grp;
        let group_end = (group_start + blocks_per_grp).min(gmat_blocks.len());

        for (blk_idx, blk) in gmat_blocks[group_start..group_end].iter().enumerate() {
            // Fast path: IQ4_NL with 8-bit GMAT blocks using cached LUT
            if !config.has_group_scales {
                if let Some((mags, shift_map, _signs, scale_log)) = blk.raw_e1m7_data() {
                    // Compute base offset for this block
                    let base_offset = if d_f32.abs() > 1e-10 {
                        scale_log - d_f32.log2()
                    } else {
                        scale_log
                    };

                    // Build/cache LUT if base_offset changed
                    let lut = if cached_base_offset != Some(base_offset) {
                        let new_lut = build_iq4nl_lut_e1m7(base_offset);
                        cached_base_offset = Some(base_offset);
                        cached_lut = Some(new_lut);
                        cached_lut.as_ref().unwrap()
                    } else {
                        cached_lut.as_ref().unwrap()
                    };

                    // Direct LUT lookup for all elements
                    for idx in 0..gmat_block_size {
                        let lut_idx =
                            mags[idx] as usize + if (shift_map >> idx) & 1 != 0 { 256 } else { 0 };
                        let q = lut[lut_idx];
                        let elem_idx =
                            g * config.elements_per_group + blk_idx * gmat_block_size + idx;
                        pack_nibble(output, config.quants_offset, elem_idx, q);
                    }
                    continue; // Skip fallback path
                }
            }

            // Fallback path: use log_iter for 4-bit blocks or when raw data unavailable
            for (idx, log2_mag, sign) in blk.log_iter() {
                // Use log2 domain binary search - no exp2 calls in inner loop
                let q = if sign == 1 {
                    // Negative values map to indices 0-7
                    find_nearest_iq4nl_log2(log2_mag, log2_scale)
                } else {
                    // Positive values map to indices 8-15
                    find_nearest_iq4nl_log2(log2_mag, log2_scale)
                };

                let elem_idx = g * config.elements_per_group + blk_idx * gmat_block_size + idx;
                pack_nibble(output, config.quants_offset, elem_idx, q);
            }
        }
    }
}

//=============================================================================
// Format-Specific Wrappers
//=============================================================================

/// Encode one IQ4_XS super-block (256 elements)
pub fn encode_iq4xs_block(
    output: &mut [u8],
    d: f16,
    scales: &[u8; 8],
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    encode_iquant_block(
        &IQ4_XS_CONFIG,
        output,
        d,
        Some(scales),
        gmat_blocks,
        gmat_block_size,
    );
}

/// Encode one IQ4_NL super-block (256 elements)
pub fn encode_iq4nl_block(
    output: &mut [u8],
    d: f16,
    gmat_blocks: &[&AnyBlock],
    gmat_block_size: usize,
) {
    encode_iquant_block(
        &IQ4_NL_CONFIG,
        output,
        d,
        None,
        gmat_blocks,
        gmat_block_size,
    );
}

//=============================================================================
// Scale Computation
//=============================================================================

/// Compute IQ4_XS scales from GMAT blocks
#[inline]
pub fn compute_iq4xs_scales(gmat_blocks: &[&AnyBlock], d: f16, gmat_block_size: usize) -> [u8; 8] {
    // IQ4_NL range is [-127, 113], effective range ~240
    let iq4_range = (IQ4_NL_QUANTS[15] - IQ4_NL_QUANTS[0]) as f32;

    super::utils::compute_group_scales_u8::<8, _>(
        gmat_blocks,
        d,
        gmat_block_size,
        32,
        |log2_max, d_f32| {
            let max_abs = super::utils::fast_exp2(log2_max);
            let scale = (max_abs / (d_f32 * iq4_range / 2.0))
                .round()
                .clamp(0.0, 63.0) as u8;
            scale.max(1)
        },
    )
}

/// Compute optimal super-scale d for IQ4_NL
#[inline]
pub fn compute_iq4nl_scale(gmat_blocks: &[&AnyBlock]) -> f16 {
    let max_abs = group_max_abs(gmat_blocks);
    // Scale so max value maps to ~113 (max of lookup table)
    let d = max_abs / 113.0;
    f16::from_f32(d.max(1e-10))
}

//=============================================================================
// Trellis Optimization (placeholder)
//=============================================================================

/// Trellis optimization for IQ4_XS
#[allow(dead_code)] // Not yet implemented
pub fn optimize_iq4_scales_trellis(
    _gmat_blocks: &[&AnyBlock],
    _d: f16,
    _activation_weights: Option<&[f32; 256]>,
    _lambda: f32,
) -> [u8; 8] {
    todo!("Implement trellis for IQ4_XS")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iq4xs_scale_roundtrip() {
        let scales = [0, 15, 31, 63, 7, 22, 44, 55];

        let (scales_h, scales_l) = encode_iq4xs_scales(&scales);
        let decoded = decode_iq4xs_scales(scales_h, &scales_l);

        assert_eq!(scales, decoded);
    }

    #[test]
    fn test_iq4xs_scale_edge_cases() {
        // All zeros
        let scales = [0u8; 8];
        let (scales_h, scales_l) = encode_iq4xs_scales(&scales);
        let decoded = decode_iq4xs_scales(scales_h, &scales_l);
        assert_eq!(scales, decoded);

        // All max (63)
        let scales = [63u8; 8];
        let (scales_h, scales_l) = encode_iq4xs_scales(&scales);
        let decoded = decode_iq4xs_scales(scales_h, &scales_l);
        assert_eq!(scales, decoded);
    }

    #[test]
    fn test_iq4nl_lookup_coverage() {
        let min = *IQ4_NL_QUANTS.iter().min().unwrap();
        let max = *IQ4_NL_QUANTS.iter().max().unwrap();

        assert!(min < -100, "Min should be < -100, got {}", min);
        assert!(max > 100, "Max should be > 100, got {}", max);

        // Check sorted
        for i in 1..16 {
            assert!(
                IQ4_NL_QUANTS[i] >= IQ4_NL_QUANTS[i - 1],
                "Lookup table should be sorted"
            );
        }
    }

    #[test]
    fn test_find_nearest_iq4nl() {
        // Test exact matches
        assert_eq!(find_nearest_iq4nl(-127.0), 0);
        assert_eq!(find_nearest_iq4nl(113.0), 15);
        assert_eq!(find_nearest_iq4nl(1.0), 8);

        // Test midpoints (should round to closer)
        assert_eq!(find_nearest_iq4nl(-115.5), 1); // between -127 and -104
        assert_eq!(find_nearest_iq4nl(0.0), 8); // closest to 1
    }

    #[test]
    fn test_config_sizes() {
        assert_eq!(IQ4_XS_CONFIG.block_bytes, 136);
        assert_eq!(IQ4_NL_CONFIG.block_bytes, 130);
    }
}
