//! GraphMatrix - Block-based sparse matrix storage.

use crate::blocks::{AnyBlock, BlockFormat};
use candle_core::{Tensor, Result};
use crate::formats::GmatMetadata;
use rayon::prelude::*;

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
        Self::calc_blocks_per_row(self.shape.1, self.format.block_size())
    }

    /// Static helper: calculate blocks per row given cols and block_size.
    #[inline]
    fn calc_blocks_per_row(cols: usize, block_size: usize) -> usize {
        (cols + block_size - 1) / block_size
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
        if self.format.is_dual_row() { row % 2 } else { 0 }
    }

    /// Calculate expected block count for given shape and format.
    fn expected_block_count(rows: usize, cols: usize, format: &BlockFormat) -> usize {
        let block_size = format.block_size();
        let blocks_per_row = Self::calc_blocks_per_row(cols, block_size);
        if format.is_dual_row() {
            let row_pairs = (rows + 1) / 2;
            row_pairs * blocks_per_row
        } else {
            rows * blocks_per_row
        }
    }

    // ========================================================================
    // Encoding helpers
    // ========================================================================

    /// Encode a single-row block using the given format.
    fn encode_single_row_block(format: BlockFormat, data: &[f32]) -> AnyBlock {
        match format {
            BlockFormat::B8x4 => AnyBlock::encode_8x4(data),
            BlockFormat::B8x8 => AnyBlock::encode_8x8(data),
            BlockFormat::B16x4 => AnyBlock::encode_16x4(data),
            BlockFormat::B16x8 => AnyBlock::encode_16x8(data),
            _ => unreachable!("encode_single_row_block called with dual-row format"),
        }
    }

    /// Encode a dual-row block using the given format.
    fn encode_dual_row_block(format: BlockFormat, row0: &[f32], row1: &[f32]) -> AnyBlock {
        match format {
            BlockFormat::DualRow8x4 => AnyBlock::encode_dualrow_8x4(row0, row1),
            BlockFormat::DualRow8x8 => AnyBlock::encode_dualrow_8x8(row0, row1),
            BlockFormat::DualRow16x4 => AnyBlock::encode_dualrow_16x4(row0, row1),
            BlockFormat::DualRow16x8 => AnyBlock::encode_dualrow_16x8(row0, row1),
            _ => unreachable!("encode_dual_row_block called with single-row format"),
        }
    }

    /// Copy a row segment into a buffer, zero-padding only the tail if needed.
    #[inline]
    fn copy_row_segment(
        data: &[f32],
        row: usize,
        cols: usize,
        block_col_start: usize,
        block_len: usize,
        buffer: &mut [f32],
    ) {
        if row * cols < data.len() {
            let data_start = row * cols + block_col_start;
            let data_end = data_start + block_len;
            buffer[..block_len].copy_from_slice(&data[data_start..data_end]);
            // Only zero-pad the tail if block_len < buffer.len()
            if block_len < buffer.len() {
                buffer[block_len..].fill(0.0);
            }
        } else {
            buffer.fill(0.0);
        }
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
    pub fn from_blocks(row_blocks: Vec<AnyBlock>, shape: (usize, usize), format: BlockFormat) -> Self {
        let (rows, cols) = shape;
        let expected_blocks = Self::expected_block_count(rows, cols, &format);

        assert_eq!(
            row_blocks.len(),
            expected_blocks,
            "Block count mismatch: expected {} blocks for {}x{} matrix with format {:?}, got {}",
            expected_blocks, rows, cols, format, row_blocks.len()
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
            data.len(), rows, cols, rows * cols
        );

        let block_size = format.block_size();
        let blocks_per_row = Self::calc_blocks_per_row(cols, block_size);

        let row_blocks = if format.is_dual_row() {
            let row_pairs = (rows + 1) / 2;
            (0..row_pairs)
                .into_par_iter()
                .flat_map(|row_pair| {
                    let row0 = row_pair * 2;
                    let row1 = row0 + 1;

                    (0..blocks_per_row).map(move |block_idx| {
                        let block_col_start = block_idx * block_size;
                        let block_len = (block_col_start + block_size).min(cols) - block_col_start;

                        let mut buf0 = [0.0f32; 16];
                        let mut buf1 = [0.0f32; 16];

                        Self::copy_row_segment(data, row0, cols, block_col_start, block_len, &mut buf0[..block_size]);
                        if row1 < rows {
                            Self::copy_row_segment(data, row1, cols, block_col_start, block_len, &mut buf1[..block_size]);
                        } else {
                            buf1[..block_size].fill(0.0);
                        }

                        Self::encode_dual_row_block(format, &buf0[..block_size], &buf1[..block_size])
                    }).collect::<Vec<_>>()
                })
                .collect()
        } else {
            (0..rows)
                .into_par_iter()
                .flat_map(|row| {
                    (0..blocks_per_row).map(move |block_idx| {
                        let block_col_start = block_idx * block_size;
                        let block_len = (block_col_start + block_size).min(cols) - block_col_start;

                        let mut buf = [0.0f32; 16];
                        Self::copy_row_segment(data, row, cols, block_col_start, block_len, &mut buf[..block_size]);
                        Self::encode_single_row_block(format, &buf[..block_size])
                    }).collect::<Vec<_>>()
                })
                .collect()
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

        let log2_center = Self::compute_median(&mut log2_vals);
        let log2_range = max_log - min_log;
        (log2_center, log2_range, log2_vals.len())
    }

    /// Compute log2 center (geometric median) for a specific row.
    pub fn row_log2_center(&self, row: usize) -> f32 {
        let rows = self.shape.0;
        assert!(row < rows);

        let blocks_per_row = self.blocks_per_row();
        let block_offset = self.row_block_offset(row);
        let row_in_block = self.row_in_block(row);
        // Pre-allocate for estimated nnz per row (cols / sparsity estimate, capped at cols)
        let cols = self.shape.1;
        let mut log2_vals = Vec::with_capacity(cols);

        for block_idx in 0..blocks_per_row {
            let block = &self.row_blocks[block_offset + block_idx];
            for (_, log_mag, _) in block.log_row_iter(row_in_block) {
                log2_vals.push(log_mag);
            }
        }

        Self::compute_median(&mut log2_vals)
    }

    /// Compute median of a float slice (returns 0.0 for empty).
    fn compute_median(values: &mut [f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    /// Compute log2 center (geometric median) for a specific column.
    /// Requires column index to be built.
    pub fn col_log2_center(&self, col: usize) -> f32 {
        let (rows, cols) = self.shape;
        assert!(col < cols);

        let col_blocks = self.col_blocks.as_ref()
            .expect("Column index not built. Call build_col_index() first.");

        let block_size = self.format.block_size();
        let blocks_per_col = Self::calc_blocks_per_row(rows, block_size);
        let col_offset = col * blocks_per_col;
        // Pre-allocate for estimated nnz per column (capped at rows)
        let mut log2_vals = Vec::with_capacity(rows);

        for block_idx in 0..blocks_per_col {
            let block = &col_blocks[col_offset + block_idx];
            for (_, log_mag, _) in block.log_iter() {
                log2_vals.push(log_mag);
            }
        }

        Self::compute_median(&mut log2_vals)
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
    pub fn row_iter(&self, row: usize) -> Box<dyn Iterator<Item = (usize, f32)> + '_> {
        let (rows, cols) = self.shape;
        assert!(row < rows, "Row index {} out of bounds for matrix with {} rows", row, rows);
        
        let block_size = self.format.block_size();
        let blocks_per_row = (cols + block_size - 1) / block_size;
        
        if self.format.is_dual_row() {
            // For dual-row: find which block pair this row belongs to
            let row_pair = row / 2;
            let row_in_block = row % 2;
            let block_offset = row_pair * blocks_per_row;
            
            Box::new((0..blocks_per_row).flat_map(move |block_idx| {
                let block = &self.row_blocks[block_offset + block_idx];
                let block_col_start = block_idx * block_size;
                
                block.row_iter(row_in_block).filter_map(move |(local_idx, value)| {
                    let col = block_col_start + local_idx;
                    if col < cols {
                        Some((col, value))
                    } else {
                        None
                    }
                })
            }))
        } else {
            // For single-row: standard indexing
            let row_offset = row * blocks_per_row;
            
            Box::new((0..blocks_per_row).flat_map(move |block_idx| {
                let block = &self.row_blocks[row_offset + block_idx];
                let block_col_start = block_idx * block_size;
                
                block.iter().filter_map(move |(local_idx, value)| {
                    let col = block_col_start + local_idx;
                    if col < cols {
                        Some((col, value))
                    } else {
                        None
                    }
                })
            }))
        }
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
    pub fn col_iter(&self, col: usize) -> Box<dyn Iterator<Item = (usize, f32)> + '_> {
        let (rows, cols) = self.shape;
        assert!(col < cols, "Column index {} out of bounds for matrix with {} columns", col, cols);

        let col_blocks = self.col_blocks.as_ref()
            .expect("Column index not built. Call build_col_index() first.");

        let block_size = self.format.block_size();
        let blocks_per_col = Self::calc_blocks_per_row(rows, block_size);
        let col_offset = col * blocks_per_col;
        
        Box::new((0..blocks_per_col)
            .flat_map(move |block_idx| {
                let block = &col_blocks[col_offset + block_idx];
                let block_row_start = block_idx * block_size;
                
                block.iter().filter_map(move |(local_idx, value)| {
                    let row = block_row_start + local_idx;
                    if row < rows {
                        Some((row, value))
                    } else {
                        None
                    }
                })
            }))
    }

    /// Helper: decode a row from blocks into a buffer.
    pub(crate) fn decode_row_to_buffer(&self, row: usize, buffer: &mut [f32]) {
        let cols = self.shape.1;
        let block_size = self.format.block_size();
        let blocks_per_row = self.blocks_per_row();
        let block_offset = self.row_block_offset(row);
        let block_row = self.row_in_block(row);

        for block_idx in 0..blocks_per_row {
            let block = &self.row_blocks[block_offset + block_idx];
            let block_start = block_idx * block_size;
            let block_end = (block_start + block_size).min(cols);

            for local_idx in 0..(block_end - block_start) {
                buffer[block_start + local_idx] = block.decode_row(block_row, local_idx);
            }
        }
    }

    /// Helper: encode a slice into a block using the matrix's format.
    fn encode_block(&self, data: &[f32]) -> AnyBlock {
        match self.format {
            BlockFormat::B8x4 | BlockFormat::DualRow8x4 => AnyBlock::encode_8x4(data),
            BlockFormat::B8x8 | BlockFormat::DualRow8x8 => AnyBlock::encode_8x8(data),
            BlockFormat::B16x4 | BlockFormat::DualRow16x4 => AnyBlock::encode_16x4(data),
            BlockFormat::B16x8 | BlockFormat::DualRow16x8 => AnyBlock::encode_16x8(data),
        }
    }

    /// Build column index for fast column access.
    ///
    /// Transposes row_blocks into col_blocks by encoding columns directly
    /// without allocating a full matrix buffer. Enables efficient column iteration via col_iter().
    pub fn build_col_index(&mut self) {
        let (rows, cols) = self.shape;
        if rows == 0 || cols == 0 {
            self.col_blocks = Some(Vec::new());
            return;
        }

        let block_size = self.format.block_size();
        let blocks_per_col = Self::calc_blocks_per_row(rows, block_size);

        // Pre-allocate output blocks
        let mut encoded_col_blocks = Vec::with_capacity(cols * blocks_per_col);

        // Reusable buffer for one column (much smaller than rows*cols)
        let mut col_buffer = vec![0.0f32; rows];
        let mut block_data = [0.0f32; 16];

        // For each column, gather values from all rows and encode blocks
        for col in 0..cols {
            // Gather column values by iterating through row blocks
            col_buffer.fill(0.0);

            for row in 0..rows {
                let block_offset = self.row_block_offset(row);
                let block_idx = col / block_size;
                let local_idx = col % block_size;
                let block_row = self.row_in_block(row);

                let block = &self.row_blocks[block_offset + block_idx];
                col_buffer[row] = block.decode_row(block_row, local_idx);
            }

            // Encode this column into blocks
            for block_idx in 0..blocks_per_col {
                let block_start = block_idx * block_size;
                let block_end = (block_start + block_size).min(rows);
                let block_len = block_end - block_start;

                block_data[..block_size].fill(0.0);
                block_data[..block_len].copy_from_slice(&col_buffer[block_start..block_end]);

                encoded_col_blocks.push(self.encode_block(&block_data[..block_size]));
            }
        }

        self.col_blocks = Some(encoded_col_blocks);
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

    /// Save GraphMatrix to a file in GMAT format.
    ///
    /// Uses GmatHeader for format specification.
    ///
    /// # Arguments
    /// - `path`: File path to save to
    ///
    /// # Errors
    /// Returns error if file creation or writing fails
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use crate::formats::GmatHeader;

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
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
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
    pub fn load_from_reader<R: std::io::Read>(reader: &mut R) -> std::io::Result<Self> {
        use crate::formats::GmatHeader;

        let header = GmatHeader::read_from(reader)?;
        let format = BlockFormat::try_from(header.format)?;

        let (rows, cols) = header.shape();
        let expected_blocks = Self::expected_block_count(rows, cols, &format);

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
