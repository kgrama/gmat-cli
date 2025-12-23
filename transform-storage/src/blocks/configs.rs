//! Block configuration types

use super::traits::{ElementMask, EncodingStrategy, E0M4, E1M7};

/// Block configuration trait combining element count and encoding
pub trait BlockConfig: Copy + Default {
    type Mask: ElementMask;
    type Encoding: EncodingStrategy;

    const SIZE: usize;
    const MAG_ARRAY_SIZE: usize;
    const MASK_SIZE: usize;
    const BYTE_SIZE_EMPTY: usize;
    const BYTE_SIZE_NON_EMPTY: usize;
}

/// 8 elements, 4-bit encoding
#[derive(Debug, Copy, Clone, Default)]
pub struct Config8x4;

impl BlockConfig for Config8x4 {
    type Mask = u8;
    type Encoding = E0M4;
    const SIZE: usize = 8;
    const MAG_ARRAY_SIZE: usize = 4;
    const MASK_SIZE: usize = 1;
    const BYTE_SIZE_EMPTY: usize = 1;      // zero_map only
    const BYTE_SIZE_NON_EMPTY: usize = 9;  // 1+2+1+1+4
}

/// 8 elements, 8-bit encoding
#[derive(Debug, Copy, Clone, Default)]
pub struct Config8x8;

impl BlockConfig for Config8x8 {
    type Mask = u8;
    type Encoding = E1M7;
    const SIZE: usize = 8;
    const MAG_ARRAY_SIZE: usize = 8;
    const MASK_SIZE: usize = 1;
    const BYTE_SIZE_EMPTY: usize = 1;       // zero_map only
    const BYTE_SIZE_NON_EMPTY: usize = 13;  // 1+2+1+1+8
}

/// 16 elements, 4-bit encoding
#[derive(Debug, Copy, Clone, Default)]
pub struct Config16x4;

impl BlockConfig for Config16x4 {
    type Mask = u16;
    type Encoding = E0M4;
    const SIZE: usize = 16;
    const MAG_ARRAY_SIZE: usize = 8;
    const MASK_SIZE: usize = 2;
    const BYTE_SIZE_EMPTY: usize = 2;       // zero_map only
    const BYTE_SIZE_NON_EMPTY: usize = 16;  // 2+2+2+2+8
}

/// 16 elements, 8-bit encoding
#[derive(Debug, Copy, Clone, Default)]
pub struct Config16x8;

impl BlockConfig for Config16x8 {
    type Mask = u16;
    type Encoding = E1M7;
    const SIZE: usize = 16;
    const MAG_ARRAY_SIZE: usize = 16;
    const MASK_SIZE: usize = 2;
    const BYTE_SIZE_EMPTY: usize = 2;       // zero_map only
    const BYTE_SIZE_NON_EMPTY: usize = 24;  // 2+2+2+2+16
}
