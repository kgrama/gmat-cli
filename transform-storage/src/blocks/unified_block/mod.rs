//! Unified block implementation supporting 1 or 2 rows with const generic ROWS

mod encode_helper;
mod io;
mod iteration;

use super::configs::{BlockConfig, Config16x4, Config16x8, Config8x4, Config8x8};
use super::traits::{ElementMask, EncodingStrategy};
use crate::blocks::block::{get_packed_nibble, set_packed_nibble, Block, EncodedBlock, EMPTY_SCALE};
use half::f16;
use std::io::{Read, Result, Write};
use std::marker::PhantomData;

pub use encode_helper::EncodeHelper;

/// Unified block supporting 1 or 2 rows with const generic ROWS
#[derive(Debug, Clone, Copy)]
pub struct UnifiedBlock<const ROWS: usize, C: BlockConfig> {
    pub scale_log: f16,
    pub zero_map: [C::Mask; 2], // Fixed size 2, use [0] for single row
    pub signs: [C::Mask; 2],
    pub octave_shift: [C::Mask; 2],
    pub magnitudes: [[u8; 16]; 2], // Fixed size 2
    _phantom: PhantomData<C>,
}

// ============================================================================
// UnifiedBlock implementation (methods for all ROWS)
// ============================================================================

impl<const ROWS: usize, C: BlockConfig + EncodeHelper> UnifiedBlock<ROWS, C> {
    /// Create an empty block
    pub fn new_empty() -> Self {
        Self {
            scale_log: EMPTY_SCALE,
            zero_map: [C::Mask::default(); 2],
            signs: [C::Mask::default(); 2],
            octave_shift: [C::Mask::default(); 2],
            magnitudes: [[0; 16]; 2],
            _phantom: PhantomData,
        }
    }

    /// Create a block from raw components (for transpose operations)
    pub fn from_components(
        scale_log: f16,
        zero_map: [C::Mask; 2],
        signs: [C::Mask; 2],
        octave_shift: [C::Mask; 2],
        magnitudes: [[u8; 16]; 2],
    ) -> Self {
        Self {
            scale_log,
            zero_map,
            signs,
            octave_shift,
            magnitudes,
            _phantom: PhantomData,
        }
    }

    /// Encode single row (for ROWS=1)
    pub fn encode_single(values: &[f32]) -> Self {
        let enc = C::do_encode(values);

        if enc.zero_map == 0 {
            return Self::new_empty();
        }

        let mut magnitudes = [0u8; 16];
        let mut octave_shift = C::Mask::default();

        let shift_threshold = C::Encoding::SHIFT_THRESHOLD;
        let shift_amount = C::Encoding::SHIFT_AMOUNT;

        for i in 0..C::SIZE {
            if (enc.zero_map >> i) & 1 == 1 {
                let offset = enc.log_offsets[i];
                let (stored_offset, needs_shift) = if offset >= shift_threshold {
                    (offset - shift_amount, true)
                } else {
                    (offset, false)
                };

                if needs_shift {
                    octave_shift |= C::Mask::from(1u8) << i;
                }

                if C::Encoding::BITS == 4 {
                    set_packed_nibble(&mut magnitudes, i, C::Encoding::encode(stored_offset));
                } else {
                    magnitudes[i] = C::Encoding::encode(stored_offset);
                }
            }
        }

        Self {
            scale_log: enc.scale_log,
            zero_map: [C::Mask::from_u64(enc.zero_map), C::Mask::default()],
            signs: [C::Mask::from_u64(enc.signs), C::Mask::default()],
            octave_shift: [octave_shift, C::Mask::default()],
            magnitudes: [magnitudes, [0; 16]],
            _phantom: PhantomData,
        }
    }

    /// Encode two rows (for ROWS=2)
    pub fn encode_dual(row0: &[f32], row1: &[f32]) -> Self {
        let enc0 = C::do_encode(row0);
        let enc1 = C::do_encode(row1);

        // If both rows are empty, return empty block
        if enc0.zero_map == 0 && enc1.zero_map == 0 {
            return Self::new_empty();
        }

        // Compute shared scale_log as minimum of both rows
        let shared_scale_log = if enc0.zero_map == 0 {
            enc1.scale_log
        } else if enc1.zero_map == 0 {
            enc0.scale_log
        } else {
            f16::from_f32(enc0.scale_log.to_f32().min(enc1.scale_log.to_f32()))
        };
        let shared_scale = shared_scale_log.to_f32();

        // Encode row 0
        let (zero_map0, signs0, octave_shift0, magnitudes0) = Self::encode_row(&enc0, shared_scale);

        // Encode row 1
        let (zero_map1, signs1, octave_shift1, magnitudes1) = Self::encode_row(&enc1, shared_scale);

        Self {
            scale_log: shared_scale_log,
            zero_map: [zero_map0, zero_map1],
            signs: [signs0, signs1],
            octave_shift: [octave_shift0, octave_shift1],
            magnitudes: [magnitudes0, magnitudes1],
            _phantom: PhantomData,
        }
    }

    /// Helper to encode a single row given a shared scale
    fn encode_row(
        enc: &EncodedBlock<16>,
        shared_scale: f32,
    ) -> (C::Mask, C::Mask, C::Mask, [u8; 16]) {
        let mut magnitudes = [0u8; 16];
        let mut octave_shift = C::Mask::default();

        if enc.zero_map == 0 {
            return (
                C::Mask::default(),
                C::Mask::default(),
                C::Mask::default(),
                magnitudes,
            );
        }

        let shift_threshold = C::Encoding::SHIFT_THRESHOLD;
        let shift_amount = C::Encoding::SHIFT_AMOUNT;

        for i in 0..C::SIZE {
            if (enc.zero_map >> i) & 1 == 1 {
                // Recompute offset relative to shared scale
                let offset = enc.log_offsets[i] + (enc.scale_log.to_f32() - shared_scale);

                let (stored_offset, needs_shift) = if offset >= shift_threshold {
                    (offset - shift_amount, true)
                } else {
                    (offset, false)
                };

                if needs_shift {
                    octave_shift |= C::Mask::from(1u8) << i;
                }

                if C::Encoding::BITS == 4 {
                    set_packed_nibble(&mut magnitudes, i, C::Encoding::encode(stored_offset));
                } else {
                    magnitudes[i] = C::Encoding::encode(stored_offset);
                }
            }
        }

        (
            C::Mask::from_u64(enc.zero_map),
            C::Mask::from_u64(enc.signs),
            octave_shift,
            magnitudes,
        )
    }

    /// Decode a single element from a specific row
    pub fn decode_element(&self, row: usize, idx: usize) -> f32 {
        if self.is_empty() || row >= ROWS || idx >= C::SIZE {
            return 0.0;
        }

        let zero_map = self.zero_map[row];
        if (zero_map >> idx) & C::Mask::from(1u8) == C::Mask::default() {
            return 0.0;
        }

        let base_offset = if C::Encoding::BITS == 4 {
            C::Encoding::decode(get_packed_nibble(&self.magnitudes[row], idx))
        } else {
            C::Encoding::decode(self.magnitudes[row][idx])
        };

        let is_shifted = (self.octave_shift[row] >> idx) & C::Mask::from(1u8) == C::Mask::from(1u8);
        let actual_offset = if is_shifted {
            base_offset + C::Encoding::SHIFT_AMOUNT
        } else {
            base_offset
        };

        let sign = if (self.signs[row] >> idx) & C::Mask::from(1u8) == C::Mask::from(1u8) {
            -1.0
        } else {
            1.0
        };

        sign * f32::exp2(self.scale_log.to_f32() + actual_offset)
    }

    /// Count non-zero elements in a specific row
    pub fn row_nnz(&self, row: usize) -> usize {
        if row >= ROWS {
            return 0;
        }
        let mask_u64: u64 = self.zero_map[row].into();
        mask_u64.count_ones() as usize
    }

    /// Check if all rows are empty
    pub fn is_empty(&self) -> bool {
        (0..ROWS).all(|i| self.zero_map[i] == C::Mask::default())
    }

    /// Compute byte size using formula
    pub fn byte_size(&self) -> usize {
        if self.is_empty() {
            ROWS * C::MASK_SIZE
        } else {
            ROWS * (3 * C::MASK_SIZE + C::MAG_ARRAY_SIZE) + 2
        }
    }
}

// ============================================================================
// Block trait implementation (only for ROWS=1)
// ============================================================================

impl<C: BlockConfig + EncodeHelper> Block for UnifiedBlock<1, C> {
    const SIZE: usize = C::SIZE;

    fn encode(values: &[f32]) -> Self {
        Self::encode_single(values)
    }

    fn decode(&self, idx: usize) -> f32 {
        self.decode_element(0, idx)
    }

    fn nnz(&self) -> usize {
        self.row_nnz(0)
    }

    fn has_element(&self, idx: usize) -> bool {
        (self.zero_map[0] >> idx) & C::Mask::from(1u8) == C::Mask::from(1u8)
    }

    fn magnitude_at(&self, idx: usize) -> u8 {
        if C::Encoding::BITS == 4 {
            get_packed_nibble(&self.magnitudes[0], idx)
        } else {
            self.magnitudes[0][idx]
        }
    }

    fn sign(&self, idx: usize) -> bool {
        (self.signs[0] >> idx) & C::Mask::from(1u8) == C::Mask::from(1u8)
    }

    fn is_empty(&self) -> bool {
        self.zero_map[0] == C::Mask::default()
    }

    fn byte_size(&self) -> usize {
        self.byte_size()
    }

    fn iter(&self) -> impl Iterator<Item = (usize, f32)> {
        self.row_iter(0)
    }

    fn log_iter(&self) -> impl Iterator<Item = (usize, f32, u8)> {
        let magnitudes = self.magnitudes[0];
        let octave_shift = self.octave_shift[0];
        let signs = self.signs[0];
        let scale_log = self.scale_log.to_f32();
        let zero_map = self.zero_map[0];
        let is_empty = self.is_empty();
        let bits = C::Encoding::BITS;
        let shift_amount = C::Encoding::SHIFT_AMOUNT;

        (0..C::SIZE).filter_map(move |i| {
            if is_empty || (zero_map >> i) & C::Mask::from(1u8) == C::Mask::default() {
                None
            } else {
                let base_offset = if bits == 4 {
                    C::Encoding::decode(get_packed_nibble(&magnitudes, i))
                } else {
                    C::Encoding::decode(magnitudes[i])
                };

                let is_shifted = (octave_shift >> i) & C::Mask::from(1u8) == C::Mask::from(1u8);
                let actual_offset = if is_shifted {
                    base_offset + shift_amount
                } else {
                    base_offset
                };

                let log2_magnitude = scale_log + actual_offset;
                let sign = if (signs >> i) & C::Mask::from(1u8) == C::Mask::from(1u8) {
                    1u8
                } else {
                    0u8
                };
                Some((i, log2_magnitude, sign))
            }
        })
    }

    fn scale_log(&self) -> f16 {
        self.scale_log
    }

    fn importance_stats(&self) -> (usize, usize) {
        let zero_map: u64 = self.zero_map[0].into();
        let octave_shift: u64 = self.octave_shift[0].into();
        let nnz = zero_map.count_ones() as usize;
        let shifted = (octave_shift & zero_map).count_ones() as usize;
        (shifted, nnz)
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        UnifiedBlock::write_to(self, w)
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        UnifiedBlock::read_from(r)
    }
}

// ============================================================================
// Type aliases for backwards compatibility and convenience
// ============================================================================

// Single-row blocks (backwards compatible)
pub type Block8x4 = UnifiedBlock<1, Config8x4>;
pub type Block8x8 = UnifiedBlock<1, Config8x8>;
pub type Block16x4 = UnifiedBlock<1, Config16x4>;
pub type Block16x8 = UnifiedBlock<1, Config16x8>;

// Dual-row blocks
pub type DualRowBlock8x4 = UnifiedBlock<2, Config8x4>;
pub type DualRowBlock8x8 = UnifiedBlock<2, Config8x8>;
pub type DualRowBlock16x4 = UnifiedBlock<2, Config16x4>;
pub type DualRowBlock16x8 = UnifiedBlock<2, Config16x8>;
