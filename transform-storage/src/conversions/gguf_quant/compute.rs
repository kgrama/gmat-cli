//! Inline compute functions for quantization formats.
//!
//! Each function handles the inner-loop computation for one format.
//! These are designed to be used with shared block iterators to reduce
//! code duplication across quantization encoders.
//!
//! All functions use `#[inline(always)]` to ensure zero overhead when
//! composed with iterators.

use super::utils::fast_exp2;

/// 8-bit symmetric quantization (Q8_0).
///
/// Computes quantized value from log2 magnitude and sign.
/// Output range: -127 to 127 as u8.
#[inline(always)]
pub fn compute_q8_0(log2_mag: f32, sign: u8, log2_scale: f32) -> u8 {
    let q_mag = fast_exp2(log2_mag - log2_scale).round().clamp(0.0, 127.0);
    let q_signed = if sign == 1 { -(q_mag as i8) } else { q_mag as i8 };
    q_signed as u8
}

/// 4-bit symmetric quantization (Q4_0).
///
/// Computes quantized value with offset for signed representation.
/// Output range: 0 to 15 (centered at offset).
#[inline(always)]
pub fn compute_q4_0(log2_mag: f32, sign: u8, log2_scale: f32, offset: i8) -> u8 {
    let q_mag = fast_exp2(log2_mag - log2_scale).round().clamp(0.0, 7.0);
    let q_signed = if sign == 1 { -(q_mag as i8) } else { q_mag as i8 };
    (q_signed + offset) as u8
}

/// 4-bit asymmetric quantization (Q4_1) - LUT-based.
///
/// Uses log2-domain thresholds for bucket assignment.
/// Negative values map to 0, positive values use binary search.
///
/// # Type Parameters
/// - `N`: Number of thresholds (bucket count - 1)
#[inline(always)]
pub fn compute_q4_1<const N: usize>(log2_mag: f32, sign: u8, thresholds: &[f32; N]) -> u8 {
    if sign == 1 {
        0
    } else {
        super::utils::log2_quantize::<N>(log2_mag, thresholds)
    }
}

/// 5-bit symmetric quantization (Q5_0).
///
/// Computes quantized value with offset for signed representation.
/// Output range: 0 to 31 (centered at offset).
#[inline(always)]
pub fn compute_q5_0(log2_mag: f32, sign: u8, log2_scale: f32, offset: i8) -> u8 {
    let q_mag = fast_exp2(log2_mag - log2_scale).round().clamp(0.0, 15.0);
    let q_signed = if sign == 1 { -(q_mag as i8) } else { q_mag as i8 };
    (q_signed + offset) as u8
}

/// 5-bit asymmetric quantization (Q5_1) - LUT-based.
///
/// Uses log2-domain thresholds for bucket assignment.
/// Negative values map to 0, positive values use binary search.
///
/// # Type Parameters
/// - `N`: Number of thresholds (bucket count - 1)
#[inline(always)]
pub fn compute_q5_1<const N: usize>(log2_mag: f32, sign: u8, thresholds: &[f32; N]) -> u8 {
    if sign == 1 {
        0
    } else {
        super::utils::log2_quantize::<N>(log2_mag, thresholds)
    }
}

/// 6-bit K-quant (Q6_K).
///
/// Computes 6-bit quantized value with 32 offset for signed representation.
/// Output range: 0 to 63 (centered at 32).
#[inline(always)]
pub fn compute_q6k(log2_mag: f32, sign: u8, log2_ds: f32) -> u8 {
    let q_mag = fast_exp2(log2_mag - log2_ds).round().clamp(0.0, 31.0);
    let q_signed = if sign == 1 { -(q_mag as i8) } else { q_mag as i8 };
    (q_signed + 32) as u8
}

/// 2-bit K-quant (Q2_K).
///
/// Uses 3 thresholds to assign values to buckets 0-3.
/// Thresholds must be sorted ascending.
#[inline(always)]
pub fn compute_q2k(log2_mag: f32, thresholds: &[f32; 3]) -> u8 {
    if log2_mag < thresholds[0] {
        0
    } else if log2_mag < thresholds[1] {
        1
    } else if log2_mag < thresholds[2] {
        2
    } else {
        3
    }
}

/// 3-bit K-quant magnitude (Q3_K).
///
/// Computes magnitude component only (sign stored separately in hmask).
/// Output range: 0 to 3.
#[inline(always)]
pub fn compute_q3k_mag(log2_mag: f32, log2_scale: f32) -> u8 {
    fast_exp2(log2_mag - log2_scale).round().clamp(0.0, 3.0) as u8
}

/// IQ4_NL lookup.
///
/// Finds nearest IQ4_NL quantization level using log2-domain thresholds.
/// Delegates to iquant module's lookup function.
#[inline(always)]
#[allow(dead_code)]
pub fn compute_iq4nl(log2_mag: f32, log2_scale: f32) -> u8 {
    super::iquant::find_nearest_iq4nl_log2(log2_mag, log2_scale)
}

/// Trellis error computation.
///
/// Computes weighted quantization error in log2 domain for trellis optimization.
/// Used to evaluate candidate scale factors.
///
/// # Arguments
/// - `log2_mag`: log2 of absolute value
/// - `log2_d`: log2 of superblock scale
/// - `log2_scale_factor`: log2 of group scale relative to superblock
/// - `weight`: importance weight for this element
///
/// # Returns
/// Weighted absolute error between ideal and clamped quantized values.
#[inline(always)]
#[allow(dead_code)]
pub fn compute_trellis_error(
    log2_mag: f32,
    log2_d: f32,
    log2_scale_factor: f32,
    weight: f32,
) -> f32 {
    let log2_q_ideal = log2_mag - log2_d - log2_scale_factor;
    let log2_q = log2_q_ideal.clamp(0.0, 3.91);
    weight * (log2_q_ideal - log2_q).abs()
}
