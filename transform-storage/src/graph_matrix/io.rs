//! I/O operations for GraphMatrix.

use std::io::Read;
use std::path::Path;

use crate::blocks::{AnyBlock, BlockFormat};
use crate::formats::GmatHeader;

use super::utils;
use super::GraphMatrix;

impl GraphMatrix {
    /// Save GraphMatrix to a file in GMAT format.
    ///
    /// Uses GmatHeader for format specification.
    ///
    /// # Arguments
    /// - `path`: File path to save to
    ///
    /// # Errors
    /// Returns error if file creation or writing fails
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Convert format to u8
        let format_byte: u8 = self.format.into();

        // Write header with borrowed metadata (avoids clone)
        let (rows, cols) = self.shape;
        GmatHeader::write_header_to(
            &mut file,
            format_byte,
            rows as u64,
            cols as u64,
            self.metadata.as_ref(),
        )?;

        // Write all blocks
        for block in &self.row_blocks {
            block.write_to(&mut file)?;
        }

        Ok(())
    }

    /// Load GraphMatrix from a file in GMAT format.
    ///
    /// Uses GmatHeader for format specification.
    ///
    /// # Arguments
    /// - `path`: File path to load from
    ///
    /// # Errors
    /// Returns error if:
    /// - File reading fails
    /// - Header is invalid
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        Self::load_from_reader(&mut file)
    }

    /// Load GraphMatrix from a reader in GMAT format.
    ///
    /// Uses GmatHeader for format specification. The reader should be positioned
    /// at the start of the GMAT data.
    ///
    /// # Arguments
    /// - `reader`: Reader positioned at start of GMAT data
    ///
    /// # Errors
    /// Returns error if:
    /// - Reading fails
    /// - Header is invalid
    pub fn load_from_reader<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let header = GmatHeader::read_from(reader)?;
        let format = BlockFormat::try_from(header.format)?;

        let (rows, cols) = header.shape();
        let expected_blocks = utils::expected_block_count(rows, cols, &format);

        let mut row_blocks = Vec::with_capacity(expected_blocks);
        for _ in 0..expected_blocks {
            row_blocks.push(AnyBlock::read_from(format, reader)?);
        }

        // Take ownership of metadata instead of cloning
        let GmatHeader { metadata, .. } = header;

        Ok(Self {
            row_blocks,
            col_blocks: None,
            shape: (rows, cols),
            format,
            metadata,
        })
    }
}
