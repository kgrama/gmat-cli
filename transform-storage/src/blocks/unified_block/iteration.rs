//! Iterator implementations for UnifiedBlock

use super::super::configs::BlockConfig;
use super::super::traits::EncodingStrategy;
use super::{EncodeHelper, UnifiedBlock};
use crate::block::get_packed_nibble;

impl<const ROWS: usize, C: BlockConfig + EncodeHelper> UnifiedBlock<ROWS, C> {
    /// Iterator over non-zero elements in a specific row
    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        let magnitudes = if row < ROWS {
            self.magnitudes[row]
        } else {
            [0u8; 16]
        };
        let octave_shift = if row < ROWS {
            self.octave_shift[row]
        } else {
            C::Mask::default()
        };
        let signs = if row < ROWS {
            self.signs[row]
        } else {
            C::Mask::default()
        };
        let zero_map = if row < ROWS {
            self.zero_map[row]
        } else {
            C::Mask::default()
        };
        let scale_log = self.scale_log.to_f32();
        let is_empty = self.is_empty();
        let bits = C::Encoding::BITS;
        let shift_amount = C::Encoding::SHIFT_AMOUNT;
        let valid_row = row < ROWS;

        (0..C::SIZE).filter_map(move |i| {
            if !valid_row || is_empty || (zero_map >> i) & C::Mask::from(1u8) == C::Mask::default()
            {
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

                let sign = if (signs >> i) & C::Mask::from(1u8) == C::Mask::from(1u8) {
                    -1.0
                } else {
                    1.0
                };
                let value = sign * f32::exp2(scale_log + actual_offset);
                Some((i, value))
            }
        })
    }

    /// Iterator over log-space values for non-zero elements in a specific row
    /// Returns (index, log2_magnitude, sign) where sign is 1 for negative, 0 for positive
    pub fn log_row_iter(&self, row: usize) -> impl Iterator<Item = (usize, f32, u8)> + '_ {
        let magnitudes = if row < ROWS {
            self.magnitudes[row]
        } else {
            [0u8; 16]
        };
        let octave_shift = if row < ROWS {
            self.octave_shift[row]
        } else {
            C::Mask::default()
        };
        let signs = if row < ROWS {
            self.signs[row]
        } else {
            C::Mask::default()
        };
        let zero_map = if row < ROWS {
            self.zero_map[row]
        } else {
            C::Mask::default()
        };
        let scale_log = self.scale_log.to_f32();
        let is_empty = self.is_empty();
        let bits = C::Encoding::BITS;
        let shift_amount = C::Encoding::SHIFT_AMOUNT;
        let valid_row = row < ROWS;

        (0..C::SIZE).filter_map(move |i| {
            if !valid_row || is_empty || (zero_map >> i) & C::Mask::from(1u8) == C::Mask::default()
            {
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
}
