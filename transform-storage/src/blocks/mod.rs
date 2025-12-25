//! Block types for transform storage

mod traits;
mod configs;
mod unified_block;
mod any_block;
mod traversal;

// Re-export traits
pub use traits::{ElementMask, EncodingStrategy, E0M4, E1M7};

// Re-export configs
pub use configs::{BlockConfig, Config8x4, Config8x8, Config16x4, Config16x8};

// Re-export unified block and type aliases
pub use unified_block::{
    UnifiedBlock,
    EncodeHelper,
    // Single-row type aliases (backwards compatible)
    Block8x4, Block8x8, Block16x4, Block16x8,
    // Dual-row type aliases
    DualRowBlock8x4, DualRowBlock8x8, DualRowBlock16x4, DualRowBlock16x8,
};

// Re-export any_block types
pub use any_block::{BlockFormat, AnyBlock};

// Re-export traversal types
pub use traversal::{Axis, TraversalConfig, BlockTraversal, BlockTraversable, transpose_tile};

// Keep backwards compatibility - re-export GenericBlock as alias
pub use unified_block::UnifiedBlock as GenericBlock;

// Legacy DualRowBlock re-export for compatibility
pub use unified_block::DualRowBlock8x4 as DualRowBlock;
