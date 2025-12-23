//! K-Quant format configuration constants

/// K-Quant format configuration - const for zero-cost abstraction
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct KQuantConfig {
    pub block_bytes: usize,
    pub num_groups: usize,
    pub elements_per_group: usize,
    pub quant_bits: u8,
    pub has_min: bool,
    pub quants_offset: usize,
    pub high_bits_offset: Option<usize>,
    pub high_bits_bytes: usize,
    pub q_max: u8,
}

pub const Q4_K_CONFIG: KQuantConfig = KQuantConfig {
    block_bytes: 144,
    num_groups: 8,
    elements_per_group: 32,
    quant_bits: 4,
    has_min: true,
    quants_offset: 16,
    high_bits_offset: None,
    high_bits_bytes: 0,
    q_max: 15,
};

pub const Q5_K_CONFIG: KQuantConfig = KQuantConfig {
    block_bytes: 176,
    num_groups: 8,
    elements_per_group: 32,
    quant_bits: 5,
    has_min: true,
    quants_offset: 48,
    high_bits_offset: Some(16),
    high_bits_bytes: 32,
    q_max: 31,
};

pub const Q6_K_CONFIG: KQuantConfig = KQuantConfig {
    block_bytes: 210,
    num_groups: 16,
    elements_per_group: 16,
    quant_bits: 6,
    has_min: false,
    quants_offset: 0,
    high_bits_offset: Some(128),
    high_bits_bytes: 64,
    q_max: 63,
};

pub const Q2_K_CONFIG: KQuantConfig = KQuantConfig {
    block_bytes: 84,
    num_groups: 16,
    elements_per_group: 16,
    quant_bits: 2,
    has_min: true,
    quants_offset: 20,
    high_bits_offset: None,
    high_bits_bytes: 0,
    q_max: 3,
};

pub const Q3_K_CONFIG: KQuantConfig = KQuantConfig {
    block_bytes: 110,
    num_groups: 16,
    elements_per_group: 16,
    quant_bits: 3,
    has_min: false,
    quants_offset: 34,
    high_bits_offset: Some(2),
    high_bits_bytes: 32,
    q_max: 7,
};
