//! Core traits for block encoding and element masking

use super::block::{decode_e0m4, decode_e1m7, encode_e0m4, encode_e1m7};
use std::io::{Read, Result, Write};

/// Trait for element count dependent types (u8 for 8 elements, u16 for 16 elements)
pub trait ElementMask:
    Copy
    + Default
    + From<u8>
    + Into<u64>
    + std::ops::Shr<usize, Output = Self>
    + std::ops::Shl<usize, Output = Self>
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitOrAssign
    + PartialEq
{
    fn from_u64(v: u64) -> Self;
    fn write_le<W: Write>(&self, w: &mut W) -> Result<()>;
    fn read_le<R: Read>(r: &mut R) -> Result<Self>;
}

impl ElementMask for u8 {
    fn from_u64(v: u64) -> Self {
        v as u8
    }
    fn write_le<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&[*self])
    }
    fn read_le<R: Read>(r: &mut R) -> Result<Self> {
        let mut buf = [0u8; 1];
        r.read_exact(&mut buf)?;
        Ok(buf[0])
    }
}

impl ElementMask for u16 {
    fn from_u64(v: u64) -> Self {
        v as u16
    }
    fn write_le<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.to_le_bytes())
    }
    fn read_le<R: Read>(r: &mut R) -> Result<Self> {
        let mut buf = [0u8; 2];
        r.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }
}

/// Trait for encoding strategy (4-bit e0m4 vs 8-bit e1m7)
pub trait EncodingStrategy: Copy + Default {
    /// Size of magnitude array for N elements
    const MAG_SIZE_8: usize; // magnitude array size for 8 elements
    const MAG_SIZE_16: usize; // magnitude array size for 16 elements

    /// Bits per magnitude
    const BITS: u8;

    /// Octave shift threshold and amount
    /// When offset >= SHIFT_THRESHOLD, subtract SHIFT_AMOUNT and set shift bit
    /// On decode, if shift bit set, add SHIFT_AMOUNT
    const SHIFT_THRESHOLD: f32;
    const SHIFT_AMOUNT: f32;

    /// Encode a log offset to stored format
    fn encode(offset: f32) -> u8;

    /// Decode stored format to log offset
    fn decode(raw: u8) -> f32;
}

/// 4-bit encoding
#[derive(Copy, Clone, Default)]
pub struct E0M4;

impl EncodingStrategy for E0M4 {
    const MAG_SIZE_8: usize = 4; // 8 elements / 2 per byte
    const MAG_SIZE_16: usize = 8; // 16 elements / 2 per byte
    const BITS: u8 = 4;
    const SHIFT_THRESHOLD: f32 = 1.0; // e0m4 range is [0, 1)
    const SHIFT_AMOUNT: f32 = 1.0; // shift by 1 octave

    fn encode(offset: f32) -> u8 {
        encode_e0m4(offset)
    }
    fn decode(raw: u8) -> f32 {
        decode_e0m4(raw)
    }
}

/// 8-bit encoding
#[derive(Copy, Clone, Default)]
pub struct E1M7;

impl EncodingStrategy for E1M7 {
    const MAG_SIZE_8: usize = 8; // 8 elements × 1 byte
    const MAG_SIZE_16: usize = 16; // 16 elements × 1 byte
    const BITS: u8 = 8;
    const SHIFT_THRESHOLD: f32 = 2.0; // e1m7 range is [0, 2)
    const SHIFT_AMOUNT: f32 = 2.0; // shift by 2 octaves

    fn encode(offset: f32) -> u8 {
        encode_e1m7(offset)
    }
    fn decode(raw: u8) -> f32 {
        decode_e1m7(raw)
    }
}
