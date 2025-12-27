//! GraphMatrix - Block-based sparse matrix storage.

use crate::blocks::{transpose_tile, AnyBlock, BlockFormat, BlockTraversal, TraversalConfig};
use crate::formats::GmatMetadata;
use candle_core::{Result, Tensor};

mod encoding;
mod io;
mod utils;

#[cfg(test)]
mod tests;

/// Block-based sparse matrix with optional column index.
///
/// Stores blocks in row-major order with optional column-major index for fast column access.
/// Each block covers block_size consecutive elements within a row.
#[derive(Debug, Clone)]
pub struct GraphMatrix {
    /// Row-major block storage
    row_blocks: Vec<AnyBlock>,
    /// Optional column-major block storage (built on-demand)
    col_blocks: Option<Vec<AnyBlock>>,
    /// Matrix shape (rows, cols)
    shape: (usize, usize),
    /// Block format
    format: BlockFormat,
    /// Optional metadata (model config, conversion info, etc.)
    metadata: Option<GmatMetadata>,
}

impl GraphMatrix {
    // ========================================================================
    // Block layout helpers
    // ========================================================================

    /// Calculate number of blocks per row (or per column for col-major).
    #[inline]
    pub(crate) fn blocks_per_row(&self) -> usize {
        utils::calc_blocks_per_row(self.shape.1, self.format.block_size())
    }

    /// Calculate the block offset for a given row in row_blocks.
    #[inline]
    pub(crate) fn row_block_offset(&self, row: usize) -> usize {
        let blocks_per_row = self.blocks_per_row();
        if self.format.is_dual_row() {
            (row / 2) * blocks_per_row
        } else {
            row * blocks_per_row
        }
    }

    /// For dual-row formats, get which sub-row (0 or 1) within the block.
    #[inline]
    fn row_in_block(&self, row: usize) -> usize {
        if self.format.is_dual_row() {
            row % 2
        } else {
            0
        }
    }

    /// Get traversal configuration for this matrix.
    #[inline]
    pub(crate) fn traversal_config(&self) -> TraversalConfig {
        TraversalConfig::new(
            self.shape,
            self.format.block_size(),
            self.format.is_dual_row(),
        )
    }

    /// Create a row traversal for the given row index.
    #[inline]
    fn row_traversal(&self, row: usize) -> BlockTraversal<'_> {
        BlockTraversal::row(&self.row_blocks, self.traversal_config(), row)
    }

    /// Create a column traversal for the given column index.
    /// Requires column index to be built.
    #[inline]
    fn col_traversal(&self, col: usize) -> BlockTraversal<'_> {
        let col_blocks = self
            .col_blocks
            .as_ref()
            .expect("Column index not built. Call build_col_index() first.");
        // For column traversal, we need a transposed config
        let col_config = TraversalConfig::new(
            (self.shape.1, self.shape.0), // swap rows/cols for column blocks
            self.format.block_size(),
            false, // column blocks are not dual-row
        );
        BlockTraversal::col(col_blocks, col_config, col)
    }

    // ========================================================================
    // Constructors
    // ========================================================================

    /// Create GraphMatrix from pre-encoded blocks with validation.
    ///
    /// # Arguments
    /// - `row_blocks`: Pre-encoded blocks in row-major order
    /// - `shape`: Matrix dimensions (rows, cols)
    /// - `format`: Block format
    ///
    /// # Panics
    /// Panics if block count doesn't match expected
    pub fn from_blocks(
        row_blocks: Vec<AnyBlock>,
        shape: (usize, usize),
        format: BlockFormat,
    ) -> Self {
        let (rows, cols) = shape;
        let expected_blocks = utils::expected_block_count(rows, cols, &format);

        assert_eq!(
            row_blocks.len(),
            expected_blocks,
            "Block count mismatch: expected {} blocks for {}x{} matrix with format {:?}, got {}",
            expected_blocks,
            rows,
            cols,
            format,
            row_blocks.len()
        );

        Self {
            row_blocks,
            col_blocks: None,
            shape,
            format,
            metadata: None,
        }
    }

    /// Create GraphMatrix from dense f32 data.
    ///
    /// Chunks data into blocks, encoding each block according to format.
    /// Partial blocks at row/column ends are padded with zeros.
    /// Uses parallel encoding via rayon for better performance on large matrices.
    ///
    /// # Arguments
    /// - `data`: Dense row-major f32 data (length must be rows * cols)
    /// - `shape`: Matrix dimensions (rows, cols)
    /// - `format`: Block format (determines encoding and dual-row behavior)
    ///
    /// # Panics
    /// Panics if data length doesn't match rows * cols
    pub fn from_dense(data: &[f32], shape: (usize, usize), format: BlockFormat) -> Self {
        let (rows, cols) = shape;
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} doesn't match shape {}x{} = {}",
            data.len(),
            rows,
            cols,
            rows * cols
        );

        let block_size = format.block_size();
        let blocks_per_row = cols.div_ceil(block_size);

        let row_blocks = if format.is_dual_row() {
            encoding::encode_dual_row_blocks(data, rows, cols, format, block_size, blocks_per_row)
        } else {
            encoding::encode_single_row_blocks(data, rows, cols, format, block_size, blocks_per_row)
        };

        Self {
            row_blocks,
            col_blocks: None,
            shape,
            format,
            metadata: None,
        }
    }

    /// Get matrix dimensions (rows, cols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get block format.
    #[inline]
    #[allow(dead_code)] // Used by config/ modules
    pub(crate) fn format(&self) -> BlockFormat {
        self.format
    }

    /// Get reference to row blocks.
    #[inline]
    pub(crate) fn row_blocks(&self) -> &[AnyBlock] {
        &self.row_blocks
    }

    /// Count total non-zero elements.
    pub fn nnz(&self) -> usize {
        self.row_blocks.iter().map(|block| block.nnz()).sum()
    }

    /// Calculate density as ratio of non-zero to total elements.
    /// Returns 1.0 for a fully dense matrix, 0.0 for an all-zeros matrix.
    pub fn density(&self) -> f32 {
        let (rows, cols) = self.shape;
        let total_elements = rows * cols;
        if total_elements == 0 {
            return 0.0;
        }
        self.nnz() as f32 / total_elements as f32
    }

    /// Alias for density() - deprecated, use density() instead.
    #[deprecated(note = "Use density() instead - this returns nnz/total, not sparsity")]
    pub fn sparsity(&self) -> f32 {
        self.density()
    }

    /// Calculate total memory usage in bytes.
    ///
    /// Includes:
    /// - Sum of all block byte sizes (empty blocks = 2 bytes, non-empty = full size)
    /// - Vec overhead for row_blocks allocation
    /// - Vec overhead for col_blocks if present
    pub fn memory_bytes(&self) -> usize {
        let blocks_size: usize = self.row_blocks.iter().map(|b| b.byte_size()).sum();

        // Vec overhead: 3 * size_of::<usize>() for ptr/len/cap
        let vec_overhead = std::mem::size_of::<Vec<AnyBlock>>();

        let col_blocks_size = if let Some(ref col_blocks) = self.col_blocks {
            let col_blocks_data: usize = col_blocks.iter().map(|b| b.byte_size()).sum();
            col_blocks_data + vec_overhead
        } else {
            0
        };

        blocks_size + vec_overhead + col_blocks_size
    }

    /// Create GraphMatrix from a candle Tensor.
    ///
    /// Extracts f32 data from the tensor and delegates to from_dense.
    ///
    /// # Arguments
    /// - `tensor`: 2D candle Tensor with f32 dtype
    /// - `format`: Block format
    ///
    /// # Errors
    /// Returns error if:
    /// - Tensor is not 2D
    /// - Tensor extraction fails
    pub fn from_tensor(tensor: &Tensor, format: BlockFormat) -> Result<Self> {
        let dims = tensor.dims();
        if dims.len() != 2 {
            candle_core::bail!("Expected 2D tensor, got {}D", dims.len());
        }

        let shape = (dims[0], dims[1]);
        let data = tensor.flatten_all()?.to_vec1::<f32>()?;

        Ok(Self::from_dense(&data, shape, format))
    }

    // ========================================================================
    // Log-domain statistics
    // ========================================================================

    /// Compute log2 statistics for the entire matrix.
    /// Returns (log2_center, log2_range, nnz) where:
    /// - log2_center: geometric median of magnitudes (median of log2 values)
    /// - log2_range: max_log2 - min_log2
    /// - nnz: number of non-zero elements
    pub fn log2_stats(&self) -> (f32, f32, usize) {
        // Pre-allocate based on total nnz estimate
        let estimated_nnz = self.nnz();
        let mut log2_vals = Vec::with_capacity(estimated_nnz);
        let mut min_log = f32::INFINITY;
        let mut max_log = f32::NEG_INFINITY;

        for block in self.row_blocks.iter() {
            for (_, log_mag, _) in block.log_iter() {
                log2_vals.push(log_mag);
                min_log = min_log.min(log_mag);
                max_log = max_log.max(log_mag);
            }
        }

        if log2_vals.is_empty() {
            return (0.0, 0.0, 0);
        }

        let log2_center = utils::compute_median(&mut log2_vals);
        let log2_range = max_log - min_log;
        (log2_center, log2_range, log2_vals.len())
    }

    /// Compute log2 center (geometric median) for a specific row.
    pub fn row_log2_center(&self, row: usize) -> f32 {
        assert!(row < self.shape.0, "Row {} out of bounds", row);
        let mut log2_vals = self.row_traversal(row).collect_log_magnitudes();
        utils::compute_median(&mut log2_vals)
    }

    /// Compute log2 center (geometric median) for a specific column.
    /// Requires column index to be built.
    pub fn col_log2_center(&self, col: usize) -> f32 {
        assert!(col < self.shape.1, "Column {} out of bounds", col);
        let mut log2_vals = self.col_traversal(col).collect_log_magnitudes();
        utils::compute_median(&mut log2_vals)
    }

    /// Iterate over non-zero elements in a specific row.
    ///
    /// Returns an iterator of (column_index, value) pairs for the given row.
    /// Handles partial blocks at row ends correctly.
    ///
    /// # Arguments
    /// - `row`: Row index to iterate over
    ///
    /// # Panics
    /// Panics if row >= self.shape.0
    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        assert!(
            row < self.shape.0,
            "Row index {} out of bounds for matrix with {} rows",
            row,
            self.shape.0
        );
        self.row_traversal(row).value_iter()
    }

    /// Iterate over all blocks in row-major order.
    ///
    /// Returns an iterator over references to all blocks in the matrix.
    pub fn block_iter(&self) -> impl Iterator<Item = &AnyBlock> {
        self.row_blocks.iter()
    }

    /// Iterate over non-zero elements in a specific column.
    ///
    /// Returns an iterator of (row_index, value) pairs for the given column.
    /// Handles partial blocks at column ends correctly.
    ///
    /// # Arguments
    /// - `col`: Column index to iterate over
    ///
    /// # Panics
    /// - Panics if col >= self.shape.1
    /// - Panics if column index has not been built (call build_col_index first)
    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
        assert!(
            col < self.shape.1,
            "Column index {} out of bounds for matrix with {} columns",
            col,
            self.shape.1
        );
        self.col_traversal(col).value_iter()
    }

    /// Helper: decode a row from blocks into a buffer.
    pub(crate) fn decode_row_to_buffer(&self, row: usize, buffer: &mut [f32]) {
        self.row_traversal(row).decode_to_buffer(buffer);
    }

    /// Build column index for fast column access.
    ///
    /// Transposes row_blocks into col_blocks by encoding columns directly
    /// without allocating a full matrix buffer. Enables efficient column iteration via col_iter().
    #[allow(clippy::needless_range_loop)]
    pub fn build_col_index(&mut self) {
        let (rows, cols) = self.shape;
        if rows == 0 || cols == 0 {
            self.col_blocks = Some(Vec::new());
            return;
        }

        let block_size = self.format.block_size();
        let blocks_per_row = utils::calc_blocks_per_row(cols, block_size);
        let blocks_per_col = utils::calc_blocks_per_row(rows, block_size);

        // Pre-allocate output: cols columns, each with blocks_per_col blocks
        let mut col_blocks = Vec::with_capacity(cols * blocks_per_col);

        // Reusable buffers for tile processing (fixed-size arrays, max block_size=16)
        let empty_block = AnyBlock::new_empty(self.format);
        let mut tile_block_refs: [&AnyBlock; 16] = [&empty_block; 16];
        let mut row_in_block_buf: [usize; 16] = [0; 16];
        let mut transposed_buf: [AnyBlock; 16] = std::array::from_fn(|_| AnyBlock::new_empty(self.format));
        let mut scratch: [(f32, u8); 256] = [(f32::NEG_INFINITY, 0u8); 256];

        // Process in tiles of block_size rows × block_size cols
        for row_tile in 0..blocks_per_col {
            let row_start = row_tile * block_size;

            for col_tile in 0..blocks_per_row {
                // Gather block_size row block references for this tile
                for i in 0..block_size {
                    let row = row_start + i;
                    if row < rows {
                        let block_offset = self.row_block_offset(row);
                        tile_block_refs[i] = &self.row_blocks[block_offset + col_tile];
                        row_in_block_buf[i] = self.row_in_block(row);
                    } else {
                        // Pad with empty blocks for partial tiles
                        tile_block_refs[i] = &empty_block;
                        row_in_block_buf[i] = 0;
                    }
                }

                // Transpose tile: block_size row blocks → block_size column blocks
                transpose_tile(
                    &tile_block_refs[..block_size],
                    &row_in_block_buf[..block_size],
                    &mut transposed_buf[..block_size],
                    &mut scratch[..block_size * block_size],
                );

                // Store transposed blocks in column-major order
                let col_start = col_tile * block_size;
                for local_col in 0..block_size {
                    let col = col_start + local_col;
                    if col < cols {
                        col_blocks.push((col, row_tile, transposed_buf[local_col].clone()));
                    }
                }
            }
        }

        // Reorder to column-major: col_blocks[col * blocks_per_col + block_idx]
        col_blocks.sort_by_key(|(col, row_tile, _)| (*col, *row_tile));
        self.col_blocks = Some(col_blocks.into_iter().map(|(_, _, b)| b).collect());
    }

    /// Drop the column index to free memory.
    ///
    /// Sets col_blocks to None, removing the column-major storage.
    /// Column iteration will no longer work until build_col_index is called again.
    pub fn drop_col_index(&mut self) {
        self.col_blocks = None;
    }

    /// Check if column index is built.
    ///
    /// Returns true if col_blocks exists, false otherwise.
    pub fn has_col_index(&self) -> bool {
        self.col_blocks.is_some()
    }

    /// Get reference to metadata.
    pub fn metadata(&self) -> Option<&GmatMetadata> {
        self.metadata.as_ref()
    }

    /// Get mutable reference to metadata.
    pub fn metadata_mut(&mut self) -> Option<&mut GmatMetadata> {
        self.metadata.as_mut()
    }

    /// Set metadata.
    pub fn set_metadata(&mut self, metadata: GmatMetadata) {
        self.metadata = Some(metadata);
    }

    /// Builder pattern: add metadata and return self.
    pub fn with_metadata(mut self, metadata: GmatMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

}
