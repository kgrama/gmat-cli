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
        let block_size = format.block_size();
        
        let expected_blocks = if format.is_dual_row() {
            // For dual-row: each block covers 2 rows
            let rows_in_blocks = (rows + 1) / 2; // ceil division
            let blocks_per_row_pair = (cols + block_size - 1) / block_size;
            rows_in_blocks * blocks_per_row_pair
        } else {
            // For single-row: each block covers 1 row
            let blocks_per_row = (cols + block_size - 1) / block_size;
            rows * blocks_per_row
        };
        
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
        let mut row_blocks = Vec::new();

        // Stack-allocated buffer reused across all blocks (max block size is 16)
        let mut block_data = [0.0f32; 16];
        let mut block_data2 = [0.0f32; 16];

        if format.is_dual_row() {
            // Process 2 rows at a time
            let blocks_per_row = (cols + block_size - 1) / block_size;
            let row_pairs = (rows + 1) / 2; // ceil division
            
            for row_pair in 0..row_pairs {
                let row0 = row_pair * 2;
                let row1 = row0 + 1;
                
                for block_idx in 0..blocks_per_row {
                    let block_col_start = block_idx * block_size;
                    let block_col_end = (block_col_start + block_size).min(cols);
                    let block_len = block_col_end - block_col_start;
                    
                    // Prepare row 0 data
                    block_data[..block_size].fill(0.0);
                    if row0 < rows {
                        let data_start = row0 * cols + block_col_start;
                        let data_end = row0 * cols + block_col_end;
                        block_data[..block_len].copy_from_slice(&data[data_start..data_end]);
                    }
                    
                    // Prepare row 1 data (pad with zeros if no row1)
                    block_data2[..block_size].fill(0.0);
                    if row1 < rows {
                        let data_start = row1 * cols + block_col_start;
                        let data_end = row1 * cols + block_col_end;
                        block_data2[..block_len].copy_from_slice(&data[data_start..data_end]);
                    }
                    
                    // Encode dual-row block
                    let block = match format {
                        BlockFormat::DualRow8x4 => AnyBlock::encode_dualrow_8x4(&block_data[..8], &block_data2[..8]),
                        BlockFormat::DualRow8x8 => AnyBlock::encode_dualrow_8x8(&block_data[..8], &block_data2[..8]),
                        BlockFormat::DualRow16x4 => AnyBlock::encode_dualrow_16x4(&block_data[..16], &block_data2[..16]),
                        BlockFormat::DualRow16x8 => AnyBlock::encode_dualrow_16x8(&block_data[..16], &block_data2[..16]),
                        _ => unreachable!("is_dual_row() returned true but format is not dual-row"),
                    };
                    row_blocks.push(block);
                }
            }
        } else {
            // Single-row format: process 1 row at a time
            let blocks_per_row = (cols + block_size - 1) / block_size;
            
            for row in 0..rows {
                let row_start = row * cols;
                
                for block_idx in 0..blocks_per_row {
                    let block_start = row_start + block_idx * block_size;
                    let block_end = (block_start + block_size).min(row_start + cols);
                    let block_len = block_end - block_start;
                    
                    // Zero the buffer and copy data
                    block_data[..block_size].fill(0.0);
                    block_data[..block_len].copy_from_slice(&data[block_start..block_end]);
                    
                    // Encode single-row block
                    let block = match format {
                        BlockFormat::B8x4 => AnyBlock::encode_8x4(&block_data[..8]),
                        BlockFormat::B8x8 => AnyBlock::encode_8x8(&block_data[..8]),
                        BlockFormat::B16x4 => AnyBlock::encode_16x4(&block_data[..16]),
                        BlockFormat::B16x8 => AnyBlock::encode_16x8(&block_data[..16]),
                        _ => unreachable!("is_dual_row() returned false but format is dual-row"),
                    };
                    row_blocks.push(block);
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
        
        let block_size = self.format.block_size();
        
        if self.format.is_dual_row() {
            // Dual-row: each block contains 2 rows
            let blocks_per_row = (cols + block_size - 1) / block_size;
            let row_pairs = (rows + 1) / 2;
            
            for row_pair in 0..row_pairs {
                let row0 = row_pair * 2;
                let row1 = row0 + 1;
                let block_offset = row_pair * blocks_per_row;
                
                // Decode row 0
                if row0 < rows {
                    for block_idx in 0..blocks_per_row {
                        let block = &self.row_blocks[block_offset + block_idx];
                        let block_start_col = block_idx * block_size;
                        let block_end_col = (block_start_col + block_size).min(cols);
                        
                        for col_idx in 0..(block_end_col - block_start_col) {
                            data.push(block.decode_row(0, col_idx));
                        }
                    }
                }
                
                // Decode row 1
                if row1 < rows {
                    for block_idx in 0..blocks_per_row {
                        let block = &self.row_blocks[block_offset + block_idx];
                        let block_start_col = block_idx * block_size;
                        let block_end_col = (block_start_col + block_size).min(cols);
                        
                        for col_idx in 0..(block_end_col - block_start_col) {
                            data.push(block.decode_row(1, col_idx));
                        }
                    }
                }
            }
        } else {
            // Single-row: each block contains 1 row
            let blocks_per_row = (cols + block_size - 1) / block_size;
            
            for row in 0..rows {
                let row_offset = row * blocks_per_row;
                
                for block_idx in 0..blocks_per_row {
                    let block = &self.row_blocks[row_offset + block_idx];
                    let block_start_col = block_idx * block_size;
                    let block_end_col = (block_start_col + block_size).min(cols);
                    
                    for col_idx in 0..(block_end_col - block_start_col) {
                        data.push(block.decode(col_idx));
                    }
                }
            }
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

    /// Export to log-sparse COO format.
    /// Note: Only works for single-row formats (not DualRow).
    pub fn to_log_sparse(&self) -> crate::conversions::LogSparseMatrix {
        let (rows, cols) = self.shape;
        let block_size = self.format.block_size();
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let mut log2_values = Vec::new();
        let mut signs = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        
        for row in 0..rows {
            let row_offset = if self.format.is_dual_row() {
                (row / 2) * blocks_per_row
            } else {
                row * blocks_per_row
            };
            
            for block_idx in 0..blocks_per_row {
                let block = &self.row_blocks[row_offset + block_idx];
                let block_col_start = block_idx * block_size;
                for (local_idx, log_mag, sign) in block.log_iter() {
                    let col = block_col_start + local_idx;
                    if col < cols {
                        row_indices.push(row);
                        col_indices.push(col);
                        log2_values.push(log_mag);
                        signs.push(sign);
                    }
                }
            }
        }
        crate::conversions::LogSparseMatrix::new(log2_values, signs, row_indices, col_indices, self.shape)
    }

    /// Export to log-sparse CSR format.
    /// Note: Only works for single-row formats (not DualRow).
    pub fn to_log_sparse_csr(&self) -> crate::conversions::LogSparseCsrMatrix {
        let (rows, cols) = self.shape;
        let block_size = self.format.block_size();
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let mut log2_values = Vec::new();
        let mut signs = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = Vec::with_capacity(rows + 1);
        row_ptr.push(0);
        
        for row in 0..rows {
            let row_offset = if self.format.is_dual_row() {
                (row / 2) * blocks_per_row
            } else {
                row * blocks_per_row
            };
            
            for block_idx in 0..blocks_per_row {
                let block = &self.row_blocks[row_offset + block_idx];
                let block_col_start = block_idx * block_size;
                for (local_idx, log_mag, sign) in block.log_iter() {
                    let col = block_col_start + local_idx;
                    if col < cols {
                        col_indices.push(col);
                        log2_values.push(log_mag);
                        signs.push(sign);
                    }
                }
            }
            row_ptr.push(log2_values.len());
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

        // Compute median (geometric median in log space)
        log2_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = log2_vals.len() / 2;
        let log2_center = if log2_vals.len() % 2 == 0 {
            (log2_vals[mid - 1] + log2_vals[mid]) / 2.0
        } else {
            log2_vals[mid]
        };

        let log2_range = max_log - min_log;
        (log2_center, log2_range, log2_vals.len())
    }

    /// Compute log2 center (geometric median) for a specific row.
    pub fn row_log2_center(&self, row: usize) -> f32 {
        let (rows, cols) = self.shape;
        assert!(row < rows);

        let block_size = self.format.block_size();
        let blocks_per_row = (cols + block_size - 1) / block_size;
        let mut log2_vals = Vec::new();

        if self.format.is_dual_row() {
            let row_pair = row / 2;
            let row_in_block = row % 2;
            let block_offset = row_pair * blocks_per_row;

            for block_idx in 0..blocks_per_row {
                let block = &self.row_blocks[block_offset + block_idx];
                for (_, log_mag, _) in block.log_row_iter(row_in_block) {
                    log2_vals.push(log_mag);
                }
            }
        } else {
            let row_offset = row * blocks_per_row;
            for block_idx in 0..blocks_per_row {
                let block = &self.row_blocks[row_offset + block_idx];
                for (_, log_mag, _) in block.log_iter() {
                    log2_vals.push(log_mag);
                }
            }
        }

        if log2_vals.is_empty() {
            return 0.0;
        }

        log2_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = log2_vals.len() / 2;
        if log2_vals.len() % 2 == 0 {
            (log2_vals[mid - 1] + log2_vals[mid]) / 2.0
        } else {
            log2_vals[mid]
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
        let blocks_per_col = (rows + block_size - 1) / block_size;
        let col_offset = col * blocks_per_col;
        let mut log2_vals = Vec::new();

        for block_idx in 0..blocks_per_col {
            let block = &col_blocks[col_offset + block_idx];
            for (_, log_mag, _) in block.log_iter() {
                log2_vals.push(log_mag);
            }
        }

        if log2_vals.is_empty() {
            return 0.0;
        }

        log2_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = log2_vals.len() / 2;
        if log2_vals.len() % 2 == 0 {
            (log2_vals[mid - 1] + log2_vals[mid]) / 2.0
        } else {
            log2_vals[mid]
        }
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
        let blocks_per_col = (rows + block_size - 1) / block_size;
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

    /// Build column index for fast column access.
    ///
    /// Transposes row_blocks into col_blocks by:
    /// 1. Decoding all row blocks into a flat column-major buffer
    /// 2. Encoding each column into blocks of block_size elements
    /// 3. Storing result in col_blocks field
    ///
    /// This enables efficient column iteration via col_iter().
    pub fn build_col_index(&mut self) {
        let (rows, cols) = self.shape;
        if rows == 0 || cols == 0 {
            self.col_blocks = Some(Vec::new());
            return;
        }

        let block_size = self.format.block_size();
        let blocks_per_col = (rows + block_size - 1) / block_size;

        // Single flat buffer for column-major data
        let mut col_major = vec![0.0f32; rows * cols];

        // Decode all blocks into column-major buffer
        if self.format.is_dual_row() {
            let blocks_per_row = (cols + block_size - 1) / block_size;
            let row_pairs = (rows + 1) / 2;
            
            for row_pair in 0..row_pairs {
                let row0 = row_pair * 2;
                let row1 = row0 + 1;
                let block_offset = row_pair * blocks_per_row;
                
                // Decode row 0
                if row0 < rows {
                    for block_idx in 0..blocks_per_row {
                        let block = &self.row_blocks[block_offset + block_idx];
                        let block_start_col = block_idx * block_size;
                        let block_end_col = (block_start_col + block_size).min(cols);
                        
                        for local_idx in 0..(block_end_col - block_start_col) {
                            let col = block_start_col + local_idx;
                            col_major[col * rows + row0] = block.decode_row(0, local_idx);
                        }
                    }
                }
                
                // Decode row 1
                if row1 < rows {
                    for block_idx in 0..blocks_per_row {
                        let block = &self.row_blocks[block_offset + block_idx];
                        let block_start_col = block_idx * block_size;
                        let block_end_col = (block_start_col + block_size).min(cols);
                        
                        for local_idx in 0..(block_end_col - block_start_col) {
                            let col = block_start_col + local_idx;
                            col_major[col * rows + row1] = block.decode_row(1, local_idx);
                        }
                    }
                }
            }
        } else {
            let blocks_per_row = (cols + block_size - 1) / block_size;
            
            for row in 0..rows {
                let row_offset = row * blocks_per_row;
                
                for block_idx in 0..blocks_per_row {
                    let block = &self.row_blocks[row_offset + block_idx];
                    let block_start_col = block_idx * block_size;
                    let block_end_col = (block_start_col + block_size).min(cols);
                    
                    for local_idx in 0..(block_end_col - block_start_col) {
                        let col = block_start_col + local_idx;
                        col_major[col * rows + row] = block.decode(local_idx);
                    }
                }
            }
        }

        // Encode columns into blocks (always single-row blocks for column storage)
        let mut encoded_col_blocks = Vec::with_capacity(cols * blocks_per_col);
        let mut block_data = [0.0f32; 16];

        for col in 0..cols {
            let col_start = col * rows;

            for block_idx in 0..blocks_per_col {
                let block_start = block_idx * block_size;
                let block_end = (block_start + block_size).min(rows);
                let block_len = block_end - block_start;

                block_data[..block_size].fill(0.0);
                block_data[..block_len].copy_from_slice(&col_major[col_start + block_start..col_start + block_end]);

                // Use same format as row blocks for column blocks
                let block = match self.format {
                    BlockFormat::B8x4 | BlockFormat::DualRow8x4 => AnyBlock::encode_8x4(&block_data[..block_size]),
                    BlockFormat::B8x8 | BlockFormat::DualRow8x8 => AnyBlock::encode_8x8(&block_data[..block_size]),
                    BlockFormat::B16x4 | BlockFormat::DualRow16x4 => AnyBlock::encode_16x4(&block_data[..block_size]),
                    BlockFormat::B16x8 | BlockFormat::DualRow16x8 => AnyBlock::encode_16x8(&block_data[..block_size]),
                };
                encoded_col_blocks.push(block);
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
        let block_size = format.block_size();
        
        let expected_blocks = if format.is_dual_row() {
            let rows_in_blocks = (rows + 1) / 2;
            let blocks_per_row_pair = (cols + block_size - 1) / block_size;
            rows_in_blocks * blocks_per_row_pair
        } else {
            let blocks_per_row = (cols + block_size - 1) / block_size;
            rows * blocks_per_row
        };
        
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    // ========================================================================
    // from_dense tests
    // ========================================================================

    #[test]
    fn test_from_dense_basic() {
        let data = vec![
            1.0, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 4.0,
        ];
        let matrix = GraphMatrix::from_dense(&data, (2, 4), BlockFormat::B16x8);

        assert_eq!(matrix.shape(), (2, 4));
        assert_eq!(matrix.nnz(), 4);
    }

    #[test]
    fn test_from_dense_all_zeros() {
        let data = vec![0.0f32; 64];
        let matrix = GraphMatrix::from_dense(&data, (4, 16), BlockFormat::B16x8);

        assert_eq!(matrix.shape(), (4, 16));
        assert_eq!(matrix.nnz(), 0);
        assert_eq!(matrix.density(), 0.0);
    }

    #[test]
    fn test_from_dense_all_nonzero() {
        let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        assert_eq!(matrix.shape(), (2, 16));
        assert_eq!(matrix.nnz(), 32);
        assert_eq!(matrix.density(), 1.0);
    }

    #[test]
    fn test_from_dense_partial_blocks() {
        // 20 columns = 2 full blocks (16) + partial block (4)
        let data = vec![1.0f32; 20];
        let matrix = GraphMatrix::from_dense(&data, (1, 20), BlockFormat::B16x8);

        assert_eq!(matrix.shape(), (1, 20));
        assert_eq!(matrix.nnz(), 20);
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_from_dense_mismatched_length() {
        let data = vec![1.0f32; 10];
        let _matrix = GraphMatrix::from_dense(&data, (2, 8), BlockFormat::B16x8);
    }

    // ========================================================================
    // from_blocks tests
    // ========================================================================

    #[test]
    fn test_from_blocks_valid() {
        let blocks = vec![
            AnyBlock::encode_16x8(&[1.0; 16]),
            AnyBlock::encode_16x8(&[2.0; 16]),
        ];
        let matrix = GraphMatrix::from_blocks(blocks, (2, 16), BlockFormat::B16x8);

        assert_eq!(matrix.shape(), (2, 16));
    }

    #[test]
    #[should_panic(expected = "Block count mismatch")]
    fn test_from_blocks_wrong_count() {
        let blocks = vec![AnyBlock::encode_16x8(&[1.0; 16])];
        let _matrix = GraphMatrix::from_blocks(blocks, (2, 16), BlockFormat::B16x8); // expects 2 blocks
    }

    // ========================================================================
    // nnz and density tests
    // ========================================================================

    #[test]
    fn test_nnz_sparse() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;
        data[15] = 2.0;
        data[16] = 3.0;
        data[31] = 4.0;

        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);
        assert_eq!(matrix.nnz(), 4);
    }

    #[test]
    fn test_density_half() {
        let mut data = vec![0.0f32; 16];
        for i in 0..8 {
            data[i] = (i + 1) as f32;
        }

        let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);
        assert_eq!(matrix.density(), 0.5);
    }

    #[test]
    fn test_density_empty_matrix() {
        let matrix = GraphMatrix::from_dense(&[], (0, 0), BlockFormat::B16x8);
        assert_eq!(matrix.density(), 0.0);
    }

    // ========================================================================
    // row_iter tests
    // ========================================================================

    #[test]
    fn test_row_iter_basic() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;   // row 0, col 0
        data[5] = 2.0;   // row 0, col 5
        data[16] = 3.0;  // row 1, col 0
        data[20] = 4.0;  // row 1, col 4

        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        // Check row 0
        let row0: Vec<(usize, f32)> = matrix.row_iter(0).collect();
        assert_eq!(row0.len(), 2);
        assert_eq!(row0[0].0, 0); // column index
        assert_eq!(row0[1].0, 5); // column index

        // Check row 1
        let row1: Vec<(usize, f32)> = matrix.row_iter(1).collect();
        assert_eq!(row1.len(), 2);
        assert_eq!(row1[0].0, 0);
        assert_eq!(row1[1].0, 4);
    }

    #[test]
    fn test_row_iter_empty_row() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0; // only row 0 has data

        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        let row1: Vec<_> = matrix.row_iter(1).collect();
        assert!(row1.is_empty());
    }

    #[test]
    fn test_row_iter_partial_block() {
        // 20 columns = 2 blocks, but second block only has 4 valid elements
        let mut data = vec![0.0f32; 20];
        data[18] = 5.0; // col 18 in partial block

        let matrix = GraphMatrix::from_dense(&data, (1, 20), BlockFormat::B16x8);

        let row: Vec<_> = matrix.row_iter(0).collect();
        assert_eq!(row.len(), 1);
        assert_eq!(row[0].0, 18);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_row_iter_out_of_bounds() {
        let data = vec![1.0f32; 16];
        let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);
        let _: Vec<_> = matrix.row_iter(1).collect(); // row 1 doesn't exist
    }

    // ========================================================================
    // block_iter tests
    // ========================================================================

    #[test]
    fn test_block_iter() {
        let data = vec![1.0f32; 48]; // 3 rows x 16 cols = 3 blocks
        let matrix = GraphMatrix::from_dense(&data, (3, 16), BlockFormat::B16x8);

        let blocks: Vec<_> = matrix.block_iter().collect();
        assert_eq!(blocks.len(), 3);
    }

    // ========================================================================
    // col_iter and col_index tests
    // ========================================================================

    #[test]
    fn test_col_index_not_built() {
        let data = vec![1.0f32; 16];
        let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);

        assert!(!matrix.has_col_index());
    }

    #[test]
    fn test_build_col_index() {
        let data = vec![1.0f32; 32];
        let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        assert!(!matrix.has_col_index());
        matrix.build_col_index();
        assert!(matrix.has_col_index());
    }

    #[test]
    fn test_col_iter_basic() {
        let mut data = vec![0.0f32; 32];
        data[0] = 1.0;   // row 0, col 0
        data[16] = 2.0;  // row 1, col 0
        data[5] = 3.0;   // row 0, col 5

        let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);
        matrix.build_col_index();

        // Check col 0
        let col0: Vec<(usize, f32)> = matrix.col_iter(0).collect();
        assert_eq!(col0.len(), 2);

        // Check col 5
        let col5: Vec<(usize, f32)> = matrix.col_iter(5).collect();
        assert_eq!(col5.len(), 1);
        assert_eq!(col5[0].0, 0); // row index
    }

    #[test]
    fn test_drop_col_index() {
        let data = vec![1.0f32; 32];
        let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        matrix.build_col_index();
        assert!(matrix.has_col_index());

        matrix.drop_col_index();
        assert!(!matrix.has_col_index());
    }

    #[test]
    #[should_panic(expected = "Column index not built")]
    fn test_col_iter_without_index() {
        let data = vec![1.0f32; 16];
        let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);
        let _: Vec<_> = matrix.col_iter(0).collect();
    }

    // ========================================================================
    // memory_bytes tests
    // ========================================================================

    #[test]
    fn test_memory_bytes_empty_blocks() {
        let data = vec![0.0f32; 32];
        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        // 2 empty blocks = 2 * 2 bytes = 4 bytes + vec overhead
        let mem = matrix.memory_bytes();
        assert!(mem >= 4);
    }

    #[test]
    fn test_memory_bytes_nonempty_blocks() {
        // Use varied values to avoid scale_log = 0 (which equals EMPTY_SCALE)
        let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        // 2 non-empty Block16x8 = 2 * 22 bytes = 44 bytes + vec overhead
        let mem = matrix.memory_bytes();
        assert!(mem >= 44, "Expected >= 44 bytes, got {}", mem);
    }

    #[test]
    fn test_memory_bytes_with_col_index() {
        let data = vec![1.0f32; 32];
        let mut matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        let mem_before = matrix.memory_bytes();
        matrix.build_col_index();
        let mem_after = matrix.memory_bytes();

        assert!(mem_after > mem_before);
    }

    // ========================================================================
    // save/load tests
    // ========================================================================

    #[test]
    fn test_save_load_roundtrip_block16x8() {
        let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = GraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(matrix.shape(), loaded.shape());
        assert_eq!(matrix.nnz(), loaded.nnz());
    }

    #[test]
    fn test_save_load_roundtrip_block16x4() {
        let data: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x4);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = GraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(matrix.shape(), loaded.shape());
        assert_eq!(matrix.nnz(), loaded.nnz());
    }

    #[test]
    fn test_save_load_roundtrip_block8x4() {
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let matrix = GraphMatrix::from_dense(&data, (2, 8), BlockFormat::B8x4);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = GraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(matrix.shape(), loaded.shape());
        assert_eq!(matrix.nnz(), loaded.nnz());
    }

    #[test]
    fn test_save_load_empty_blocks() {
        let data = vec![0.0f32; 32];
        let matrix = GraphMatrix::from_dense(&data, (2, 16), BlockFormat::B16x8);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = GraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(loaded.nnz(), 0);
    }

    #[test]
    fn test_save_load_sparse_data() {
        let mut data = vec![0.0f32; 64];
        data[0] = 1.0;
        data[15] = 2.0;
        data[32] = 3.0;
        data[63] = 4.0;

        let matrix = GraphMatrix::from_dense(&data, (4, 16), BlockFormat::B16x8);

        let temp_file = NamedTempFile::new().unwrap();
        matrix.save(temp_file.path()).unwrap();

        let loaded = GraphMatrix::load(temp_file.path()).unwrap();

        assert_eq!(matrix.nnz(), loaded.nnz());
        assert_eq!(loaded.nnz(), 4);
    }

    // ========================================================================
    // Tensor conversion tests
    // ========================================================================

    #[test]
    fn test_tensor_roundtrip() {
        let device = Device::Cpu;
        // e1m7 has 4 octave range, can handle simple linear values
        let original_data: Vec<f32> = (1..=32).map(|x| x as f32).collect();

        let original = Tensor::from_vec(original_data.clone(), (4, 8), &device).unwrap();

        let matrix = GraphMatrix::from_tensor(&original, BlockFormat::B8x8).unwrap();
        let restored = matrix.to_tensor(&device).unwrap();

        assert_eq!(restored.dims(), &[4, 8]);

        // Values should be approximately equal (quantization error)
        let restored_data: Vec<f32> = restored.flatten_all().unwrap().to_vec1().unwrap();
        for (orig, rest) in original_data.iter().zip(restored_data.iter()) {
            if *orig == 0.0 {
                assert_eq!(*rest, 0.0);
            } else {
                let ratio = rest / orig;
                // e1m7 has ~4 octave range, expect ratio within 0.9-1.1
                assert!(
                    (0.9..1.1).contains(&ratio),
                    "orig={}, restored={}", orig, rest
                );
            }
        }
    }

    #[test]
    fn test_from_tensor_shape() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((4, 20), candle_core::DType::F32, &device).unwrap();

        let matrix = GraphMatrix::from_tensor(&tensor, BlockFormat::B16x8).unwrap();
        assert_eq!(matrix.shape(), (4, 20));
    }

    // ========================================================================
    // Different block type tests
    // ========================================================================

    #[test]
    fn test_block8x4_matrix() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let matrix = GraphMatrix::from_dense(&data, (3, 8), BlockFormat::B8x4);

        assert_eq!(matrix.shape(), (3, 8));
        assert_eq!(matrix.nnz(), 24);
    }

    #[test]
    fn test_block16x4_matrix() {
        let data: Vec<f32> = (1..=48).map(|x| x as f32).collect();
        let matrix = GraphMatrix::from_dense(&data, (3, 16), BlockFormat::B16x4);

        assert_eq!(matrix.shape(), (3, 16));
        assert_eq!(matrix.nnz(), 48);
    }

    // ========================================================================
    // Log-sparse reconstruction error tests
    // ========================================================================

    /// Compute relative reconstruction error: |original - reconstructed| / |original|
    fn relative_error(original: f32, reconstructed: f32) -> f32 {
        if original == 0.0 {
            if reconstructed == 0.0 { 0.0 } else { f32::INFINITY }
        } else {
            (original - reconstructed).abs() / original.abs()
        }
    }

    #[test]
    fn test_log_sparse_reconstruction_error_block8x8() {
        // Use two Block8x8 blocks for 16 values - each block handles narrower range
        // Values 2-16 span too many octaves for single block's offset range from median
        let data: [f32; 16] = [
            2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0,
            10.0, -11.0, 12.0, -14.0, 16.0, 0.0, 0.0, 0.0,
        ];

        // Split into two matrices with Block8x8
        let matrix1 = GraphMatrix::from_dense(&data[0..8], (1, 8), BlockFormat::B8x8);
        let matrix2 = GraphMatrix::from_dense(&data[8..16], (1, 8), BlockFormat::B8x8);

        let log_sparse1 = matrix1.to_log_sparse();
        let log_sparse2 = matrix2.to_log_sparse();
        let linear1 = log_sparse1.to_linear();
        let linear2 = log_sparse2.to_linear();

        let mut max_error: f32 = 0.0;
        let mut sum_error: f32 = 0.0;
        let mut count = 0;

        // Check first block
        for (&orig, &recon) in data[0..8].iter().zip(linear1.values.iter()) {
            if orig != 0.0 {
                assert_eq!(orig.signum(), recon.signum(),
                    "Sign mismatch: original={}, reconstructed={}", orig, recon);
                let err = relative_error(orig, recon);
                max_error = max_error.max(err);
                sum_error += err;
                count += 1;
            }
        }

        // Check second block
        for (&orig, &recon) in data[8..16].iter().zip(linear2.values.iter()) {
            if orig != 0.0 {
                assert_eq!(orig.signum(), recon.signum(),
                    "Sign mismatch: original={}, reconstructed={}", orig, recon);
                let err = relative_error(orig, recon);
                max_error = max_error.max(err);
                sum_error += err;
                count += 1;
            }
        }

        let avg_error = sum_error / count as f32;

        // e1m7 (8-bit) should have < 2% average error for values within each block's range
        println!("Block8x8 (e1m7): avg_error={:.4}%, max_error={:.4}%",
            avg_error * 100.0, max_error * 100.0);
        assert!(avg_error < 0.03, "Average error {:.4}% exceeds 3%", avg_error * 100.0);
        assert!(max_error < 0.08, "Max error {:.4}% exceeds 8%", max_error * 100.0);
    }

    #[test]
    fn test_log_sparse_reconstruction_error_block16x4() {
        // e0m4 encodes offsets in [0.0, 1.0) from median, so ~1 octave range
        // Using values from 8.0 to 12.0 (factor 1.5 = 0.58 octaves)
        let data: [f32; 16] = [
            8.0, -8.5, 9.0, -9.5, 10.0, -10.5, 11.0, -11.5,
            12.0, -8.2, 9.2, -10.2, 11.2, 0.0, 0.0, 0.0,
        ];

        let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x4);
        let log_sparse = matrix.to_log_sparse();
        let linear = log_sparse.to_linear();

        let mut max_error: f32 = 0.0;
        let mut sum_error: f32 = 0.0;
        let mut count = 0;

        for (&orig, &recon) in data.iter().zip(linear.values.iter()) {
            if orig != 0.0 {
                assert_eq!(orig.signum(), recon.signum(),
                    "Sign mismatch: original={}, reconstructed={}", orig, recon);

                let err = relative_error(orig, recon);
                max_error = max_error.max(err);
                sum_error += err;
                count += 1;
            }
        }

        let avg_error = sum_error / count as f32;

        // e0m4 (4-bit) has lower precision, expect < 10% average error
        println!("Block16x4 (e0m4): avg_error={:.4}%, max_error={:.4}%",
            avg_error * 100.0, max_error * 100.0);
        assert!(avg_error < 0.15, "Average error {:.4}% exceeds 15%", avg_error * 100.0);
        assert!(max_error < 0.30, "Max error {:.4}% exceeds 30%", max_error * 100.0);
    }

    #[test]
    fn test_log_sparse_reconstruction_error_block8x4() {
        // e0m4 encodes offsets in [0.0, 1.0) from median, so ~1 octave range
        // Using values from 8.0 to 11.0 (factor 1.375 = 0.46 octaves)
        let data: [f32; 8] = [8.0, -8.5, 9.0, -9.5, 10.0, -10.5, 11.0, -8.2];

        let matrix = GraphMatrix::from_dense(&data, (1, 8), BlockFormat::B8x4);
        let log_sparse = matrix.to_log_sparse();
        let linear = log_sparse.to_linear();

        let mut max_error: f32 = 0.0;
        let mut sum_error: f32 = 0.0;
        let mut count = 0;

        for (&orig, &recon) in data.iter().zip(linear.values.iter()) {
            if orig != 0.0 {
                assert_eq!(orig.signum(), recon.signum(),
                    "Sign mismatch: original={}, reconstructed={}", orig, recon);

                let err = relative_error(orig, recon);
                max_error = max_error.max(err);
                sum_error += err;
                count += 1;
            }
        }

        let avg_error = sum_error / count as f32;

        // e0m4 (4-bit) has lower precision
        println!("Block8x4 (e0m4): avg_error={:.4}%, max_error={:.4}%",
            avg_error * 100.0, max_error * 100.0);
        assert!(avg_error < 0.15, "Average error {:.4}% exceeds 15%", avg_error * 100.0);
        assert!(max_error < 0.30, "Max error {:.4}% exceeds 30%", max_error * 100.0);
    }

    #[test]
    fn test_log_sparse_csr_matches_coo() {
        // Verify CSR and COO produce same results
        let data: [f32; 16] = [
            4.0, -5.0, 6.0, -7.0, 8.0, -9.0, 10.0, -11.0,
            12.0, -13.0, 14.0, -15.0, 16.0, 0.0, 0.0, 0.0,
        ];

        let matrix = GraphMatrix::from_dense(&data, (1, 16), BlockFormat::B16x8);

        let log_coo = matrix.to_log_sparse();
        let log_csr = matrix.to_log_sparse_csr();

        let linear_coo = log_coo.to_linear();
        let linear_csr = log_csr.to_linear();

        assert_eq!(linear_coo.values.len(), linear_csr.values.len());
        for (coo_val, csr_val) in linear_coo.values.iter().zip(linear_csr.values.iter()) {
            assert!((coo_val - csr_val).abs() < 1e-6,
                "COO={} vs CSR={}", coo_val, csr_val);
        }
    }
}
