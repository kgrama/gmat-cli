//! Block trait definitions for sparse coefficient storage.

use half::f16;

/// Empty block scale_log sentinel
pub const EMPTY_SCALE: f16 = f16::ZERO;

/// Core trait for block storage types.
pub trait Block: Sized + Clone {
    /// Number of elements per block (16 or 8)
    const SIZE: usize;

    /// Encode f32 values into a block. Zeros in input become zeros in output.
    fn encode(values: &[f32]) -> Self;

    /// Decode a single element to f32
    fn decode(&self, idx: usize) -> f32;

    /// Count of non-zero elements
    fn nnz(&self) -> usize;

    /// Check if element at index exists (non-zero)
    fn has_element(&self, idx: usize) -> bool;

    /// Get raw magnitude byte at index (for matmul)
    fn magnitude_at(&self, idx: usize) -> u8;

    /// Get sign at index (true if negative)
    fn sign(&self, idx: usize) -> bool;

    /// Check if block is empty (scale_log == 0.0)
    fn is_empty(&self) -> bool;

    /// Byte size when serialized (2 if empty, full size otherwise)
    fn byte_size(&self) -> usize;

    /// Iterator over non-zero elements as (index, value) pairs
    fn iter(&self) -> impl Iterator<Item = (usize, f32)>;

    /// Iterator over non-zero elements as (index, log2_magnitude, sign) tuples.
    /// Returns raw log2 values for log-domain computation (no exp2 conversion).
    /// sign: 0 = positive, 1 = negative
    fn log_iter(&self) -> impl Iterator<Item = (usize, f32, u8)>;

    /// Get the scale_log value (for log-domain ops)
    fn scale_log(&self) -> f16;

    /// Compute importance as ratio of octave-shifted elements to total non-zero elements.
    /// Returns (octave_shift_count, nnz) for efficient aggregate computation.
    /// Higher ratio indicates larger dynamic range / more important values.
    fn importance_stats(&self) -> (usize, usize);

    /// Decode entire block to Vec<f32>
    fn decode_all(&self) -> Vec<f32> {
        (0..Self::SIZE).map(|i| self.decode(i)).collect()
    }

    /// Decode entire block into a provided slice (no allocation).
    /// Slice must have length >= Self::SIZE.
    #[inline]
    fn decode_into(&self, out: &mut [f32]) {
        debug_assert!(out.len() >= Self::SIZE);
        for i in 0..Self::SIZE {
            out[i] = self.decode(i);
        }
    }

    /// Write block to binary stream
    fn write_to<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()>;

    /// Read block from binary stream
    fn read_from<R: std::io::Read>(r: &mut R) -> std::io::Result<Self>;
}

// ============================================================================
// Generic encoding/decoding
// ============================================================================

/// Result of encoding values - intermediate representation before block construction
pub struct EncodedBlock<const N: usize> {
    pub scale_log: f16,
    pub zero_map: u64,
    pub signs: u64,
    pub log_offsets: [f32; N],
}

/// Encode f32 values into intermediate representation
pub fn encode_values<const N: usize>(values: &[f32]) -> EncodedBlock<N> {
    let mut zero_map = 0u64;
    let mut signs = 0u64;
    let mut log_vals = [0.0f32; N];

    values.iter().take(N).enumerate().for_each(|(i, &v)| {
        if v != 0.0 {
            zero_map |= 1 << i;
            if v < 0.0 {
                signs |= 1 << i;
            }
            log_vals[i] = v.abs().log2();
        }
    });

    if zero_map == 0 {
        return EncodedBlock {
            scale_log: EMPTY_SCALE,
            zero_map: 0,
            signs: 0,
            log_offsets: [0.0; N],
        };
    }

    // Use min as scale so all offsets are >= 0 (O(n) instead of O(n log n) median)
    let scale = (0..N)
        .filter(|&i| (zero_map >> i) & 1 == 1)
        .map(|i| log_vals[i])
        .fold(f32::INFINITY, f32::min);

    // Compute offsets in place
    let mut log_offsets = log_vals;
    (0..N).filter(|&i| (zero_map >> i) & 1 == 1).for_each(|i| {
        log_offsets[i] -= scale;
    });

    EncodedBlock {
        scale_log: f16::from_f32(scale),
        zero_map,
        signs,
        log_offsets,
    }
}

/// Encode a magnitude offset to e1m7 format (8-bit)
/// Range: [0, 2) with 128 steps per octave, 256 total values
/// Format: 1-bit exponent (0-1), 7-bit mantissa
#[inline]
pub fn encode_e1m7(offset: f32) -> u8 {
    let offset_clamped = offset.clamp(0.0, 2.0 - 1.0 / 128.0);
    let e = (offset_clamped as u8).min(1);
    let base = e as f32;
    let m = ((offset_clamped - base) * 128.0) as u8;
    (e << 7) | (m & 0x7F)
}

/// Decode e1m7 magnitude to log2 offset
/// Range: [0, 2)
#[inline]
pub fn decode_e1m7(raw: u8) -> f32 {
    let e = (raw >> 7) as f32;
    let m = (raw & 0x7F) as f32 / 128.0;
    e + m
}

/// Encode a magnitude offset to e0m4 format (4-bit)
#[inline]
pub fn encode_e0m4(offset: f32) -> u8 {
    let offset_clamped = offset.clamp(0.0, 1.0 - 1.0 / 16.0);
    (offset_clamped * 16.0) as u8
}

/// Decode e0m4 magnitude to log2 offset
#[inline]
pub fn decode_e0m4(nibble: u8) -> f32 {
    nibble as f32 / 16.0
}

/// Get nibble from packed e0m4 byte array
#[inline]
pub fn get_packed_nibble(magnitudes: &[u8], idx: usize) -> u8 {
    let byte_idx = idx / 2;
    if idx.is_multiple_of(2) {
        magnitudes[byte_idx] & 0x0F
    } else {
        magnitudes[byte_idx] >> 4
    }
}

/// Set nibble in packed e0m4 byte array
#[inline]
pub fn set_packed_nibble(magnitudes: &mut [u8], idx: usize, nibble: u8) {
    let byte_idx = idx / 2;
    if idx.is_multiple_of(2) {
        magnitudes[byte_idx] |= nibble & 0x0F;
    } else {
        magnitudes[byte_idx] |= (nibble & 0x0F) << 4;
    }
}

/// Generic block iterator - returns (index, decoded_value) for non-zero elements
#[inline]
pub fn block_iter<const N: usize, F>(
    zero_map: u64,
    scale_log: f32,
    signs: u64,
    is_empty: bool,
    decode_magnitude: F,
) -> impl Iterator<Item = (usize, f32)>
where
    F: Fn(usize) -> f32 + 'static,
{
    (0..N).filter_map(move |i| {
        if is_empty || (zero_map >> i) & 1 == 0 {
            None
        } else {
            let sign = if (signs >> i) & 1 == 1 { -1.0 } else { 1.0 };
            Some((i, sign * f32::exp2(scale_log + decode_magnitude(i))))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // e1m7 encoding/decoding tests
    // ========================================================================

    #[test]
    fn test_e1m7_roundtrip_zero() {
        // Offset of 0.0 should encode and decode back to approximately 0.0
        let encoded = encode_e1m7(0.0);
        let decoded = decode_e1m7(encoded);
        assert!(
            (decoded - 0.0).abs() < 0.01,
            "Expected ~0.0, got {}",
            decoded
        );
    }

    #[test]
    fn test_e1m7_roundtrip_positive() {
        // Test offsets within native range [0, 2)
        for offset in [0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9] {
            let encoded = encode_e1m7(offset);
            let decoded = decode_e1m7(encoded);
            assert!(
                (decoded - offset).abs() < 0.01,
                "Offset {} encoded to {} decoded to {}",
                offset,
                encoded,
                decoded
            );
        }
    }

    #[test]
    fn test_e1m7_clamping() {
        // Values outside native range [0, 2) should be clamped
        let encoded_low = encode_e1m7(-0.5);
        let decoded_low = decode_e1m7(encoded_low);
        assert_eq!(decoded_low, 0.0, "Should clamp to 0.0, got {}", decoded_low);

        let encoded_high = encode_e1m7(3.0);
        let decoded_high = decode_e1m7(encoded_high);
        assert!(
            decoded_high < 2.0,
            "Should clamp to < 2.0, got {}",
            decoded_high
        );
    }

    // ========================================================================
    // e0m4 encoding/decoding tests
    // ========================================================================

    #[test]
    fn test_e0m4_roundtrip_zero() {
        let encoded = encode_e0m4(0.0);
        assert_eq!(encoded, 0);
        let decoded = decode_e0m4(encoded);
        assert_eq!(decoded, 0.0);
    }

    #[test]
    fn test_e0m4_roundtrip_values() {
        // Test values within range
        for i in 0..16 {
            let offset = i as f32 / 16.0;
            let encoded = encode_e0m4(offset);
            let decoded = decode_e0m4(encoded);
            assert!(
                (decoded - offset).abs() < 0.07,
                "Offset {} encoded to {} decoded to {}",
                offset,
                encoded,
                decoded
            );
        }
    }

    #[test]
    fn test_e0m4_clamping() {
        // Negative values should clamp to 0
        let encoded_neg = encode_e0m4(-0.5);
        assert_eq!(encoded_neg, 0);

        // Values >= 1.0 should clamp to max (15/16)
        let encoded_high = encode_e0m4(1.5);
        assert_eq!(encoded_high, 15);
    }

    // ========================================================================
    // Nibble packing tests
    // ========================================================================

    #[test]
    fn test_nibble_packing_low() {
        let mut magnitudes = [0u8; 4];
        set_packed_nibble(&mut magnitudes, 0, 0xA);
        assert_eq!(get_packed_nibble(&magnitudes, 0), 0xA);
        assert_eq!(magnitudes[0], 0x0A);
    }

    #[test]
    fn test_nibble_packing_high() {
        let mut magnitudes = [0u8; 4];
        set_packed_nibble(&mut magnitudes, 1, 0xB);
        assert_eq!(get_packed_nibble(&magnitudes, 1), 0xB);
        assert_eq!(magnitudes[0], 0xB0);
    }

    #[test]
    fn test_nibble_packing_both() {
        let mut magnitudes = [0u8; 4];
        set_packed_nibble(&mut magnitudes, 0, 0x5);
        set_packed_nibble(&mut magnitudes, 1, 0xC);
        assert_eq!(magnitudes[0], 0xC5);
        assert_eq!(get_packed_nibble(&magnitudes, 0), 0x5);
        assert_eq!(get_packed_nibble(&magnitudes, 1), 0xC);
    }

    #[test]
    fn test_nibble_packing_all_indices() {
        let mut magnitudes = [0u8; 4];
        for i in 0..8 {
            set_packed_nibble(&mut magnitudes, i, (i as u8) % 16);
        }
        for i in 0..8 {
            assert_eq!(
                get_packed_nibble(&magnitudes, i),
                (i as u8) % 16,
                "Index {} failed",
                i
            );
        }
    }

    // ========================================================================
    // encode_values tests
    // ========================================================================

    #[test]
    fn test_encode_values_all_zeros() {
        let values = [0.0f32; 8];
        let encoded = encode_values::<8>(&values);
        assert_eq!(encoded.zero_map, 0);
        assert_eq!(encoded.scale_log, EMPTY_SCALE);
    }

    #[test]
    fn test_encode_values_single_nonzero() {
        let mut values = [0.0f32; 8];
        values[3] = 4.0;
        let encoded = encode_values::<8>(&values);
        assert_eq!(encoded.zero_map, 0b00001000); // bit 3 set
        assert_eq!(encoded.signs, 0); // positive
    }

    #[test]
    fn test_encode_values_negative_signs() {
        let mut values = [0.0f32; 8];
        values[0] = -1.0;
        values[2] = 2.0;
        values[4] = -3.0;
        let encoded = encode_values::<8>(&values);
        assert_eq!(encoded.zero_map, 0b00010101); // bits 0, 2, 4 set
        assert_eq!(encoded.signs, 0b00010001); // bits 0 and 4 are negative
    }

    #[test]
    fn test_encode_values_16_elements() {
        let mut values = [0.0f32; 16];
        values[0] = 1.0;
        values[15] = 2.0;
        let encoded = encode_values::<16>(&values);
        assert_eq!(encoded.zero_map, 0b1000000000000001); // bits 0 and 15 set
    }

    // ========================================================================
    // block_iter tests
    // ========================================================================

    #[test]
    fn test_block_iter_empty() {
        let items: Vec<_> = block_iter::<8, _>(0, 0.0, 0, true, |_| 0.0).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_block_iter_single_element() {
        let zero_map = 0b00000100; // bit 2 set
        let items: Vec<_> = block_iter::<8, _>(zero_map, 0.0, 0, false, |_| 0.0).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, 2); // index 2
        assert!((items[0].1 - 1.0).abs() < 0.001); // exp2(0 + 0) = 1.0
    }

    #[test]
    fn test_block_iter_with_sign() {
        let zero_map = 0b00000010; // bit 1 set
        let signs = 0b00000010; // bit 1 negative
        let items: Vec<_> = block_iter::<8, _>(zero_map, 0.0, signs, false, |_| 0.0).collect();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, 1);
        assert!((items[0].1 - (-1.0)).abs() < 0.001); // negative 1.0
    }
}
