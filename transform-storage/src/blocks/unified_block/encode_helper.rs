//! EncodeHelper trait for dispatching encoding based on block size

use super::super::configs::{Config16x4, Config16x8, Config8x4, Config8x8};
use crate::block::{encode_values, EncodedBlock};

// ============================================================================
// Helper trait for SIZE dispatch (8 vs 16 elements)
// ============================================================================

mod private {
    use super::*;
    pub trait Sealed {}
    impl Sealed for Config8x4 {}
    impl Sealed for Config8x8 {}
    impl Sealed for Config16x4 {}
    impl Sealed for Config16x8 {}
}

/// Helper trait for SIZE dispatch (8 vs 16 elements)
/// This trait is sealed and only implemented for Config8x4, Config8x8, Config16x4, Config16x8
pub trait EncodeHelper: private::Sealed {
    fn do_encode(values: &[f32]) -> EncodedBlock<16>;
}

impl EncodeHelper for Config8x4 {
    fn do_encode(values: &[f32]) -> EncodedBlock<16> {
        let enc = encode_values::<8>(values);
        let mut offsets = [0.0f32; 16];
        offsets[..8].copy_from_slice(&enc.log_offsets);
        EncodedBlock {
            scale_log: enc.scale_log,
            zero_map: enc.zero_map,
            signs: enc.signs,
            log_offsets: offsets,
        }
    }
}

impl EncodeHelper for Config8x8 {
    fn do_encode(values: &[f32]) -> EncodedBlock<16> {
        let enc = encode_values::<8>(values);
        let mut offsets = [0.0f32; 16];
        offsets[..8].copy_from_slice(&enc.log_offsets);
        EncodedBlock {
            scale_log: enc.scale_log,
            zero_map: enc.zero_map,
            signs: enc.signs,
            log_offsets: offsets,
        }
    }
}

impl EncodeHelper for Config16x4 {
    fn do_encode(values: &[f32]) -> EncodedBlock<16> {
        encode_values::<16>(values)
    }
}

impl EncodeHelper for Config16x8 {
    fn do_encode(values: &[f32]) -> EncodedBlock<16> {
        encode_values::<16>(values)
    }
}
