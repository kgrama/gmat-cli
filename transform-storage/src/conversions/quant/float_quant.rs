//! Float quantization helpers.

use super::types::QuantDType;

/// Quantize a float value to reduced precision (F4/F6/F8).
pub fn quantize_float_bits(value: f32, dtype: QuantDType) -> f32 {
    if value == 0.0 || !value.is_finite() {
        return value;
    }

    let (exp_bits, mantissa_bits) = match dtype {
        QuantDType::F4 => (2, 1),
        QuantDType::F6 => (3, 2),
        QuantDType::F8 => (4, 3),
        _ => return value,
    };

    let sign = value.signum();
    let abs_val = value.abs();

    let exp_bias = (1 << (exp_bits - 1)) - 1;
    let exp_max = (1 << exp_bits) - 1;

    let log2_val = abs_val.log2();
    let exp_unbiased = log2_val.floor() as i32;
    let exp_clamped = exp_unbiased.clamp(-exp_bias, exp_max - exp_bias - 1);

    let mantissa_full = abs_val / f32::exp2(exp_clamped as f32) - 1.0;
    let mantissa_steps = (1 << mantissa_bits) as f32;
    let mantissa_quantized = (mantissa_full * mantissa_steps).round() / mantissa_steps;

    let magnitude = (1.0 + mantissa_quantized) * f32::exp2(exp_clamped as f32);
    sign * magnitude
}
