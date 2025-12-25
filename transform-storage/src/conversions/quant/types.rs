//! Quantization types and parameters.

use candle_core::Tensor;

/// Configuration for trellis quantization.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrellisConfig {
    /// Emit redundancy mask during dual trellis quantization
    pub emit_redundancy_mask: bool,
    /// Maximum octave difference to consider states redundant (default 1)
    pub redundancy_threshold: u8,
}

impl TrellisConfig {
    /// Create default config with mask disabled
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            emit_redundancy_mask: false,
            redundancy_threshold: 1,
        }
    }

    /// Enable mask emission with custom threshold
    #[allow(dead_code)]
    pub fn with_mask(threshold: u8) -> Self {
        Self {
            emit_redundancy_mask: true,
            redundancy_threshold: threshold,
        }
    }
}

/// Target dtype for quantization output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDType {
    /// 4-bit signed integer [-8, 7]
    I4,
    /// 6-bit signed integer [-32, 31]
    I6,
    /// 8-bit signed integer [-128, 127]
    I8,
    /// 16-bit signed integer [-32768, 32767]
    I16,
    /// 4-bit float (1 sign, 2 exp, 1 mantissa)
    F4,
    /// 6-bit float (1 sign, 3 exp, 2 mantissa)
    F6,
    /// 8-bit float (1 sign, 4 exp, 3 mantissa)
    F8,
    /// 16-bit float (IEEE 754 half)
    F16,
}

impl QuantDType {
    /// Number of bits for this type
    pub fn bits(&self) -> u8 {
        match self {
            Self::I4 | Self::F4 => 4,
            Self::I6 | Self::F6 => 6,
            Self::I8 | Self::F8 => 8,
            Self::I16 | Self::F16 => 16,
        }
    }

    /// Whether this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::I4 | Self::I6 | Self::I8 | Self::I16)
    }

    /// Max representable integer value (for integer types)
    pub fn int_max(&self) -> i32 {
        match self {
            Self::I4 => 7,
            Self::I6 => 31,
            Self::I8 => 127,
            Self::I16 => 32767,
            _ => 0,
        }
    }

    /// Min representable integer value (for integer types)
    pub fn int_min(&self) -> i32 {
        match self {
            Self::I4 => -8,
            Self::I6 => -32,
            Self::I8 => -128,
            Self::I16 => -32768,
            _ => 0,
        }
    }
}

/// Bit-packing format for quantized tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PackFormat {
    /// No packing - one value per element (uses I64/F32)
    #[default]
    None,
    /// AWQ-style: 8×i4 → i32, with interleaving for fast dequant
    /// Group size 128, scales as f16
    Awq,
    /// GPTQ-style: 8×i4 → i32, sequential packing
    /// Group size 128, scales as f16, includes g_idx
    Gptq,
    /// Simple packing: 2×i4 → u8, 4×i8 → u32, etc.
    /// No grouping, single global scale
    Packed,
    /// Trellis-optimized quantization for single-view (single row)
    /// Uses dynamic programming to minimize quantization error with smoothness constraints
    TrellisSingle,
    /// Trellis-optimized quantization for dual-view (paired rows)
    /// Joint optimization with shared scale and inter-row coupling
    TrellisDual,
}

impl PackFormat {
    /// Default group size for this format
    pub fn default_group_size(&self) -> usize {
        match self {
            Self::Awq | Self::Gptq => 128,
            Self::None | Self::Packed | Self::TrellisSingle | Self::TrellisDual => 0, // No grouping
        }
    }

    /// Values packed per storage element for given bit width
    pub fn pack_factor(&self, bits: u8) -> usize {
        match self {
            Self::None => 1,
            Self::Awq | Self::Gptq => 32 / bits as usize, // 8 for i4, 4 for i8
            Self::Packed => match bits {
                4 => 2,  // 2×i4 → u8
                8 => 1,  // 1×i8 → u8
                16 => 1, // 1×i16 → needs i64 or 2×u8
                _ => 1,
            },
            Self::TrellisSingle | Self::TrellisDual => 32 / bits as usize, // Uses block encoding: 8 for i4, 4 for i8
        }
    }
}

/// Parameters returned from quantization, used for dequantization.
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Target dtype used
    pub dtype: QuantDType,
    /// Packing format used
    pub pack_format: PackFormat,
    /// Group size (0 = no grouping, single scale)
    pub group_size: usize,
    /// Global scale (for non-grouped) or None (for grouped)
    pub scale: Option<f32>,
    /// Log2 of geometric center
    pub log2_center: f32,
    /// Log2 range (max - min)
    pub log2_range: f32,
    /// Number of non-zero values
    pub nnz: usize,
}

impl QuantParams {
    /// Dequantize a single integer value (for non-grouped)
    pub fn dequantize_int(&self, quantized: i64) -> f32 {
        self.scale.map(|s| quantized as f32 * s).unwrap_or(0.0)
    }
}

/// Per-channel activation statistics for activation-aware quantization.
///
/// These statistics capture how activations behave per input channel,
/// allowing importance-weighted quantization similar to AWQ.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Per-channel activation scales (mean or max absolute value).
    /// Length should equal in_features (number of input channels).
    /// Higher values indicate more important channels.
    pub channel_scales: Vec<f32>,

    /// Optional: log2 of channel scales for direct log-domain computation.
    /// If None, computed from channel_scales.
    pub log2_scales: Option<Vec<f32>>,
}

impl ActivationStats {
    /// Create from raw f32 activation scales (e.g., mean |activation| per channel).
    pub fn from_scales(scales: Vec<f32>) -> Self {
        Self {
            channel_scales: scales,
            log2_scales: None,
        }
    }

    /// Create from log2 activation scales directly.
    pub fn from_log2_scales(log2_scales: Vec<f32>) -> Self {
        let channel_scales = log2_scales.iter().map(|&l| f32::exp2(l)).collect();
        Self {
            channel_scales,
            log2_scales: Some(log2_scales),
        }
    }

    /// Get log2 scale for a channel (computes lazily if not stored).
    pub fn log2_scale(&self, channel: usize) -> f32 {
        if let Some(ref log2) = self.log2_scales {
            log2.get(channel).copied().unwrap_or(0.0)
        } else {
            let s = self.channel_scales.get(channel).copied().unwrap_or(1.0);
            if s > 0.0 {
                s.log2()
            } else {
                f32::NEG_INFINITY
            }
        }
    }

    /// Get linear scale for a channel.
    pub fn scale(&self, channel: usize) -> f32 {
        self.channel_scales.get(channel).copied().unwrap_or(1.0)
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.channel_scales.len()
    }

    /// Compute importance for a weight given its channel.
    /// importance = |weight| * |activation| = exp2(log2_weight + log2_activation)
    pub fn importance(&self, channel: usize, weight_abs: f32) -> f32 {
        weight_abs * self.scale(channel)
    }

    /// Compute log2 importance for a weight given its channel.
    pub fn log2_importance(&self, channel: usize, log2_weight_abs: f32) -> f32 {
        log2_weight_abs + self.log2_scale(channel)
    }
}

/// Static saliency derived from block scales - no calibration needed.
///
/// Approximates activation-aware importance using only weight statistics:
/// - Block scales from embedding matrix indicate which hidden dims are active
/// - Block scales from weight matrix indicate which dims influence output
/// - Combined: `saliency[h] ≈ embed_scale[h] × weight_scale[h]`
///
/// This provides ~80-90% of the benefit of full activation calibration
/// for LLMs where weight distributions dominate.
#[derive(Debug, Clone)]
pub struct StaticSaliency {
    /// Per-channel saliency scores (higher = more important).
    /// Length equals in_features of the weight matrix.
    pub channel_saliency: Vec<f32>,
}

impl StaticSaliency {
    /// Create from a single layer's column block scales.
    ///
    /// For a weight matrix W with shape (out, in), this uses the max
    /// block scale per input column as a proxy for importance.
    pub fn from_column_scales(col_scales: Vec<f32>) -> Self {
        Self {
            channel_saliency: col_scales,
        }
    }

    /// Create by combining upstream and current layer block scales.
    ///
    /// `upstream_scales`: Per-channel scales from previous layer output (e.g., embedding)
    /// `weight_col_scales`: Per-input-column scales from current weight matrix
    ///
    /// Saliency = upstream × weight (in log domain: add, then exp2)
    pub fn from_chained_scales(upstream_scales: &[f32], weight_col_scales: &[f32]) -> Self {
        assert_eq!(
            upstream_scales.len(),
            weight_col_scales.len(),
            "Scale vectors must have same length"
        );

        let channel_saliency: Vec<f32> = upstream_scales
            .iter()
            .zip(weight_col_scales.iter())
            .map(|(&up, &w)| up * w)
            .collect();

        Self { channel_saliency }
    }

    /// Create by combining log2 scales (more efficient for log-domain data).
    ///
    /// `upstream_log2`: log2 of upstream scales
    /// `weight_log2`: log2 of weight column scales
    pub fn from_chained_log2(upstream_log2: &[f32], weight_log2: &[f32]) -> Self {
        assert_eq!(
            upstream_log2.len(),
            weight_log2.len(),
            "Scale vectors must have same length"
        );

        let channel_saliency: Vec<f32> = upstream_log2
            .iter()
            .zip(weight_log2.iter())
            .map(|(&up, &w)| f32::exp2(up + w))
            .collect();

        Self { channel_saliency }
    }

    /// Convert to ActivationStats for use with existing quantization functions.
    pub fn to_activation_stats(&self) -> ActivationStats {
        ActivationStats::from_scales(self.channel_saliency.clone())
    }

    /// Get saliency for a channel.
    pub fn saliency(&self, channel: usize) -> f32 {
        self.channel_saliency.get(channel).copied().unwrap_or(1.0)
    }

    /// Number of channels.
    pub fn num_channels(&self) -> usize {
        self.channel_saliency.len()
    }

    /// Compute importance for a weight given its channel.
    /// Same interface as ActivationStats for drop-in use.
    pub fn importance(&self, channel: usize, weight_abs: f32) -> f32 {
        weight_abs * self.saliency(channel)
    }
}

/// Output from quantization - contains all tensors needed for inference.
#[derive(Debug)]
pub struct QuantizedTensors {
    /// Packed weight tensor
    /// - AWQ/GPTQ i4: shape [rows, cols/8], dtype U32 (8 values per element)
    /// - Packed i4: shape [rows, cols/2], dtype U8 (2 values per element)
    /// - Packed i8: shape [rows, cols], dtype U8
    /// - F16: shape [rows, cols], dtype F16
    pub weights: Tensor,

    /// Per-group scales (for AWQ/GPTQ)
    /// Shape: [rows, cols/group_size], dtype F16
    pub scales: Option<Tensor>,

    /// Per-group zeros (for GPTQ)
    /// Shape: [rows, cols/group_size], dtype U32 (packed)
    pub zeros: Option<Tensor>,

    /// Group indices (for GPTQ with non-sequential groups)
    /// Shape: [cols], dtype U32
    pub g_idx: Option<Tensor>,

    /// Redundancy mask (for dual trellis with config.emit_redundancy_mask)
    /// Shape: [row_pairs, num_blocks], dtype U8 (0 or 1)
    /// Bit is 1 when dual-row states are within threshold (redundant)
    pub redundancy_mask: Option<Tensor>,

    /// Quantization parameters
    pub params: QuantParams,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trellis_config() {
        let default = TrellisConfig::new();
        assert!(!default.emit_redundancy_mask);

        let with_mask = TrellisConfig::with_mask(2);
        assert!(with_mask.emit_redundancy_mask);
        assert_eq!(with_mask.redundancy_threshold, 2);
    }
}
