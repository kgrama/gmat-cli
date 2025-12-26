//! GGUF quantization types and enums

/// GGUF quantization formats
///
/// Names follow llama.cpp conventions (e.g., Q4_K_M) for compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgufQuantType {
    // Unquantized (any alignment)
    F32,
    F16,
    // Legacy (block size = 32)
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    // K-Quants (super-block = 256)
    Q2_K,
    Q3_K_S,
    Q3_K_M,
    Q3_K_L,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    // I-Quants (super-block = 256)
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_XS,
    IQ4_NL,
}

/// Scale optimization strategy
#[derive(Debug, Clone, Copy)]
pub enum ScaleOptimization {
    /// Standard: group scale = max(abs(elements))
    Standard,
    /// Trellis: DP-optimized scales with smoothness penalty
    Trellis { lambda: f32 },
    /// TrellisDual: joint optimization for paired rows (Q/K in attention)
    TrellisDual { lambda: f32, gamma: f32 },
}

/// Quantized output data
pub struct GgufQuantizedData {
    pub data: Vec<u8>,
    pub quant_type: GgufQuantType,
    pub shape: (usize, usize),
}

/// Activation statistics for importance-weighted quantization
///
/// Used to weight quantization errors by importance during calibration.
/// Collected by running representative inputs through the model.
pub struct ActivationStats {
    /// Per-element mean absolute activation values
    /// Shape: same as weight tensor being quantized
    pub mean_abs: Vec<f32>,

    /// Per-element activation variance (optional, for advanced weighting)
    pub variance: Option<Vec<f32>>,

    /// Per-row importance scores (aggregated from element stats)
    /// Useful for row-wise quantization decisions
    pub row_importance: Vec<f32>,

    /// Number of calibration samples used to compute these stats
    pub sample_count: usize,
}

/// Static saliency for importance weighting without calibration data.
/// Derives importance from weight statistics alone, providing ~80-90%
/// benefit of full activation calibration for LLMs.
#[derive(Debug, Clone)]
pub struct StaticSaliency {
    /// Per-column log2 importance: log2(embed_scale) + log2(weight_scale)
    pub log2_importance: Vec<f32>,
}

impl StaticSaliency {
    /// Create from per-column maximum scales (single layer).
    pub fn from_column_scales(column_max_scales: &[f32]) -> Self {
        Self {
            log2_importance: column_max_scales
                .iter()
                .map(|&s| s.abs().max(f32::MIN_POSITIVE).log2())
                .collect(),
        }
    }

    /// Combine upstream layer log2 scales with current layer scales.
    /// log2_saliency = log2_upstream + log2_current (multiplication in linear domain)
    pub fn from_chained_log2(upstream_log2: &[f32], current_log2: &[f32]) -> Self {
        assert_eq!(
            upstream_log2.len(),
            current_log2.len(),
            "upstream and current must have same dimension"
        );
        Self {
            log2_importance: upstream_log2
                .iter()
                .zip(current_log2.iter())
                .map(|(&u, &c)| u + c)
                .collect(),
        }
    }

    /// Convert to ActivationStats for drop-in compatibility with AWQ/GGUF pipelines.
    /// Uses minimal fields required for importance weighting.
    pub fn to_activation_stats(&self) -> ActivationStats {
        let row_importance: Vec<f32> = self
            .log2_importance
            .iter()
            .map(|&log2_s| super::utils::fast_exp2(log2_s))
            .collect();
        ActivationStats {
            mean_abs: vec![1.0; self.log2_importance.len()],
            variance: None,
            row_importance,
            sample_count: 0,
        }
    }
}

impl ActivationStats {
    /// Create new empty stats for a given shape
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            mean_abs: vec![0.0; rows * cols],
            variance: None,
            row_importance: vec![0.0; rows],
            sample_count: 0,
        }
    }

    /// Get per-element weights for a super-block (256 elements)
    ///
    /// Returns importance weights normalized to sum to 256 (uniform = 1.0 each)
    pub fn get_superblock_weights(&self, row: usize, superblock: usize, cols: usize) -> [f32; 256] {
        let mut weights = [1.0f32; 256];
        let start = row * cols + superblock * 256;

        if start + 256 <= self.mean_abs.len() {
            let slice = &self.mean_abs[start..start + 256];
            let sum: f32 = slice.iter().sum();

            if sum > 1e-10 {
                let scale = 256.0 / sum;
                for (i, &v) in slice.iter().enumerate() {
                    weights[i] = v * scale;
                }
            }
        }

        weights
    }
}

impl GgufQuantType {
    /// Block size in elements
    pub fn block_size(&self) -> usize {
        use GgufQuantType::*;
        match self {
            F32 | F16 => 1, // No blocking for unquantized
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 => 32,
            _ => 256,
        }
    }

    /// Block size in bytes
    pub fn block_bytes(&self) -> usize {
        use GgufQuantType::*;
        match self {
            F32 => 4, // 4 bytes per element
            F16 => 2, // 2 bytes per element
            Q4_0 => 2 + 16,                           // f16 scale + 16 nibbles
            Q4_1 => 2 + 2 + 16,                       // f16 scale + f16 min + 16 nibbles
            Q5_0 => 2 + 4 + 16,                       // f16 scale + 4B high + 16B low
            Q5_1 => 2 + 2 + 4 + 16,                   // f16 scale + f16 min + 4B high + 16B low
            Q8_0 => 2 + 32,                           // f16 scale + 32 int8
            Q4_K_S | Q4_K_M => 2 + 2 + 12 + 128,      // d + dmin + scales + quants
            Q5_K_S | Q5_K_M => 2 + 2 + 12 + 128 + 32, // + high bits
            Q6_K => 128 + 64 + 16 + 2,                // ql + qh + scales + d = 210

            // Q2_K and Q3_K formats
            Q2_K => 2 + 2 + 16 + 64,    // d + dmin + scales + quants = 84
            Q3_K_S => 2 + 32 + 64,      // d + hmask + quants = 98 (no scales packed separately)
            Q3_K_M => 2 + 32 + 64 + 12, // d + hmask + quants + scales = 110
            Q3_K_L => 2 + 32 + 64 + 12, // same as Q3_K_M = 110

            // I-Quants (256-element super-blocks)
            IQ1_S => 2 + 2 + 32,          // d + sumd + grids = 36
            IQ1_M => 2 + 2 + 2 + 32 + 8,  // d + sumd + extra + grids + scales = 46
            IQ2_XXS => 2 + 2 + 32,        // d + extra + grids = 36
            IQ2_XS => 2 + 2 + 64 + 2,     // d + extra + grids + scales_h = 70
            IQ2_S => 2 + 64 + 2 + 1,      // d + grids + scales_h + extra = 69
            IQ3_XXS => 2 + 3 * 32 + 32,   // d + grids + signs = 130 (approx, varies)
            IQ3_S => 2 + 64 + 32 + 4 + 2, // d + grids + signs + scales + extra = 104
            IQ4_XS => 2 + 2 + 4 + 128,    // d + scales_h + scales_l + quants = 136
            IQ4_NL => 2 + 128,            // d + quants = 130 (uses lookup table)
        }
    }

    /// Check if cols are aligned for this format
    pub fn is_aligned(&self, cols: usize) -> bool {
        cols.is_multiple_of(self.block_size())
    }

    /// Returns true if this is a K-quant format
    pub fn is_kquant(&self) -> bool {
        use GgufQuantType::*;
        matches!(
            self,
            Q2_K | Q3_K_S | Q3_K_M | Q3_K_L | Q4_K_S | Q4_K_M | Q5_K_S | Q5_K_M | Q6_K
        )
    }

    /// Returns true if this is an I-quant format
    pub fn is_iquant(&self) -> bool {
        use GgufQuantType::*;
        matches!(
            self,
            IQ1_S | IQ1_M | IQ2_XXS | IQ2_XS | IQ2_S | IQ3_XXS | IQ3_S | IQ4_XS | IQ4_NL
        )
    }
}
