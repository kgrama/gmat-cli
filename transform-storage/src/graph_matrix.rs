//! GraphMatrix - Block-based sparse matrix storage.

use crate::blocks::{AnyBlock, BlockFormat};
use candle_core::{Device, Tensor, Result};
use crate::formats::GmatMetadata;

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
    fn blocks_per_row(&self) -> usize {
        Self::calc_blocks_per_row(self.shape.1, self.format.block_size())
    }

    /// Static helper: calculate blocks per row given cols and block_size.
    #[inline]
    fn calc_blocks_per_row(cols: usize, block_size: usize) -> usize {
        (cols + block_size - 1) / block_size
    }

    /// Calculate the block offset for a given row in row_blocks.
    #[inline]
    fn row_block_offset(&self, row: usize) -> usize {
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

    /// Copy a row segment into a buffer, zero-padding as needed.
    #[inline]
    fn copy_row_segment(
        data: &[f32],
        row: usize,
        cols: usize,
        block_col_start: usize,
        block_len: usize,
        buffer: &mut [f32],
    ) {
        buffer.fill(0.0);
        if row * cols < data.len() {
            let data_start = row * cols + block_col_start;
            let data_end = data_start + block_len;
            buffer[..block_len].copy_from_slice(&data[data_start..data_end]);
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
        let expected_blocks = Self::expected_block_count(rows, cols, &format);
        let mut row_blocks = Vec::with_capacity(expected_blocks);

        // Stack-allocated buffers reused across all blocks (max block size is 16)
        let mut buf0 = [0.0f32; 16];
        let mut buf1 = [0.0f32; 16];

        if format.is_dual_row() {
            let row_pairs = (rows + 1) / 2;
            for row_pair in 0..row_pairs {
                let row0 = row_pair * 2;
                let row1 = row0 + 1;

                for block_idx in 0..blocks_per_row {
                    let block_col_start = block_idx * block_size;
                    let block_len = (block_col_start + block_size).min(cols) - block_col_start;

                    Self::copy_row_segment(data, row0, cols, block_col_start, block_len, &mut buf0[..block_size]);
                    if row1 < rows {
                        Self::copy_row_segment(data, row1, cols, block_col_start, block_len, &mut buf1[..block_size]);
                    } else {
                        buf1[..block_size].fill(0.0);
                    }

                    row_blocks.push(Self::encode_dual_row_block(format, &buf0[..block_size], &buf1[..block_size]));
                }
            }
        } else {
            for row in 0..rows {
                for block_idx in 0..blocks_per_row {
                    let block_col_start = block_idx * block_size;
                    let block_len = (block_col_start + block_size).min(cols) - block_col_start;

                    Self::copy_row_segment(data, row, cols, block_col_start, block_len, &mut buf0[..block_size]);
                    row_blocks.push(Self::encode_single_row_block(format, &buf0[..block_size]));
                }
            }
        }

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

    /// Convert GraphMatrix to a candle Tensor.
    ///
    /// Decodes all blocks to f32 values in row-major order and creates a tensor.
    ///
    /// # Arguments
    /// - `device`: Target device for the tensor
    ///
    /// # Errors
    /// Returns error if tensor creation fails
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let (rows, cols) = self.shape;
        let mut data = Vec::with_capacity(rows * cols);
        let mut row_buffer = vec![0.0f32; cols];

        for row in 0..rows {
            row_buffer.fill(0.0);
            self.decode_row_to_buffer(row, &mut row_buffer);
            data.extend_from_slice(&row_buffer);
        }

        Tensor::from_vec(data, (rows, cols), device)
    }

    /// Convert GraphMatrix to f16 tensor.
    pub fn to_tensor_f16(&self, device: &Device) -> Result<Tensor> {
        let tensor = self.to_tensor(device)?;
        tensor.to_dtype(candle_core::DType::F16)
    }

    /// Export to CSR format.
    pub fn to_csr(&self) -> crate::conversions::CsrMatrix {
        let (rows, _cols) = self.shape;
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = Vec::with_capacity(rows + 1);
        row_ptr.push(0);
        for row in 0..rows {
            for (col, value) in self.row_iter(row) {
                values.push(value);
                col_indices.push(col);
            }
            row_ptr.push(values.len());
        }
        crate::conversions::CsrMatrix::new(values, col_indices, row_ptr, self.shape)
    }

    /// Export to COO format.
    pub fn to_coo(&self) -> crate::conversions::CooMatrix {
        let (rows, _cols) = self.shape;
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        for row in 0..rows {
            for (col, value) in self.row_iter(row) {
                row_indices.push(row);
                col_indices.push(col);
                values.push(value);
            }
        }
        crate::conversions::CooMatrix::new(values, row_indices, col_indices, self.shape)
    }

    /// Helper: iterate over log-sparse entries, calling visitor for each (row, col, log_mag, sign).
    fn for_each_log_entry<F>(&self, mut visitor: F)
    where
        F: FnMut(usize, usize, f32, u8),
    {
        let (rows, cols) = self.shape;
        let block_size = self.format.block_size();
        let blocks_per_row = self.blocks_per_row();

        for row in 0..rows {
            let row_offset = self.row_block_offset(row);

            for block_idx in 0..blocks_per_row {
                let block = &self.row_blocks[row_offset + block_idx];
                let block_col_start = block_idx * block_size;
                for (local_idx, log_mag, sign) in block.log_iter() {
                    let col = block_col_start + local_idx;
                    if col < cols {
                        visitor(row, col, log_mag, sign);
                    }
                }
            }
        }
    }

    /// Export to log-sparse COO format.
    pub fn to_log_sparse(&self) -> crate::conversions::LogSparseMatrix {
        let mut log2_values = Vec::new();
        let mut signs = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();

        self.for_each_log_entry(|row, col, log_mag, sign| {
            row_indices.push(row);
            col_indices.push(col);
            log2_values.push(log_mag);
            signs.push(sign);
        });

        crate::conversions::LogSparseMatrix::new(log2_values, signs, row_indices, col_indices, self.shape)
    }

    /// Export to log-sparse CSR format.
    pub fn to_log_sparse_csr(&self) -> crate::conversions::LogSparseCsrMatrix {
        let (rows, _) = self.shape;
        let mut log2_values = Vec::new();
        let mut signs = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = vec![0usize; rows + 1];
        let mut current_row = 0;

        self.for_each_log_entry(|row, col, log_mag, sign| {
            // Fill in row_ptr for any skipped rows
            while current_row < row {
                current_row += 1;
                row_ptr[current_row] = log2_values.len();
            }
            col_indices.push(col);
            log2_values.push(log_mag);
            signs.push(sign);
        });

        // Fill remaining row_ptr entries
        for i in (current_row + 1)..=rows {
            row_ptr[i] = log2_values.len();
        }

        crate::conversions::LogSparseCsrMatrix::new(log2_values, signs, col_indices, row_ptr, self.shape)
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
        let mut log2_vals = Vec::new();
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
        let mut log2_vals = Vec::new();

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
        let mut log2_vals = Vec::new();

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
    fn decode_row_to_buffer(&self, row: usize, buffer: &mut [f32]) {
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
    /// Transposes row_blocks into col_blocks by decoding to column-major format
    /// and re-encoding. Enables efficient column iteration via col_iter().
    pub fn build_col_index(&mut self) {
        let (rows, cols) = self.shape;
        if rows == 0 || cols == 0 {
            self.col_blocks = Some(Vec::new());
            return;
        }

        let block_size = self.format.block_size();
        let blocks_per_col = Self::calc_blocks_per_row(rows, block_size);

        // Decode to column-major buffer
        let mut col_major = vec![0.0f32; rows * cols];
        let mut row_buffer = vec![0.0f32; cols];

        for row in 0..rows {
            row_buffer.fill(0.0);
            self.decode_row_to_buffer(row, &mut row_buffer);
            for col in 0..cols {
                col_major[col * rows + row] = row_buffer[col];
            }
        }

        // Encode columns into blocks
        let mut encoded_col_blocks = Vec::with_capacity(cols * blocks_per_col);
        let mut block_data = [0.0f32; 16];

        for col in 0..cols {
            let col_start = col * rows;

            for block_idx in 0..blocks_per_col {
                let block_start = block_idx * block_size;
                let block_end = (block_start + block_size).min(rows);

                block_data[..block_size].fill(0.0);
                block_data[..(block_end - block_start)]
                    .copy_from_slice(&col_major[col_start + block_start..col_start + block_end]);

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
        
        let (rows, cols) = self.shape;
        let header = if let Some(ref metadata) = self.metadata {
            GmatHeader::new_with_metadata(format_byte, rows as u64, cols as u64, metadata.clone())
        } else {
            GmatHeader::new(format_byte, rows as u64, cols as u64)
        };
        header.write_to(&mut file)?;
        
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
        use crate::formats::GmatHeader;

        let mut file = std::fs::File::open(path)?;

        let header = GmatHeader::read_from(&mut file)?;
        let format = BlockFormat::try_from(header.format)?;

        let (rows, cols) = header.shape();
        let expected_blocks = Self::expected_block_count(rows, cols, &format);

        let mut row_blocks = Vec::with_capacity(expected_blocks);
        for _ in 0..expected_blocks {
            row_blocks.push(AnyBlock::read_from(format, &mut file)?);
        }

        Ok(Self {
            row_blocks,
            col_blocks: None,
            shape: (rows, cols),
            format,
            metadata: header.metadata().cloned(),
        })
    }
}
