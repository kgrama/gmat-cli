//! Tensor and sparse format export methods for GraphMatrix.

use candle_core::{Device, Tensor, Result};
use crate::graph_matrix::GraphMatrix;

impl GraphMatrix {
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
        let (rows, cols) = self.shape();
        let mut data = Vec::with_capacity(rows * cols);
        // Buffer is zero-initialized and decode_row_to_buffer overwrites all values
        let mut row_buffer = vec![0.0f32; cols];

        for row in 0..rows {
            // Note: decode_row_to_buffer sets all values, so no fill(0.0) needed
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
        let (rows, _cols) = self.shape();
        // Pre-allocate based on nnz
        let nnz = self.nnz();
        let mut values = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut row_ptr = Vec::with_capacity(rows + 1);
        row_ptr.push(0);
        for row in 0..rows {
            for (col, value) in self.row_iter(row) {
                values.push(value);
                col_indices.push(col);
            }
            row_ptr.push(values.len());
        }
        crate::conversions::CsrMatrix::new(values, col_indices, row_ptr, self.shape())
    }

    /// Export to COO format.
    pub fn to_coo(&self) -> crate::conversions::CooMatrix {
        let (rows, _cols) = self.shape();
        // Pre-allocate based on nnz
        let nnz = self.nnz();
        let mut values = Vec::with_capacity(nnz);
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        for row in 0..rows {
            for (col, value) in self.row_iter(row) {
                row_indices.push(row);
                col_indices.push(col);
                values.push(value);
            }
        }
        crate::conversions::CooMatrix::new(values, row_indices, col_indices, self.shape())
    }

    /// Helper: iterate over log-sparse entries, calling visitor for each (row, col, log_mag, sign).
    fn for_each_log_entry<F>(&self, mut visitor: F)
    where
        F: FnMut(usize, usize, f32, u8),
    {
        use crate::blocks::BlockTraversal;
        
        let (rows, _) = self.shape();
        let config = self.traversal_config();
        
        for row in 0..rows {
            let traversal = BlockTraversal::row(self.row_blocks(), config, row);
            for (col, log_mag, sign) in traversal.log_iter() {
                visitor(row, col, log_mag, sign);
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

        crate::conversions::LogSparseMatrix::new(log2_values, signs, row_indices, col_indices, self.shape())
    }

    /// Export to log-sparse CSR format.
    pub fn to_log_sparse_csr(&self) -> crate::conversions::LogSparseCsrMatrix {
        let (rows, _) = self.shape();
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

        crate::conversions::LogSparseCsrMatrix::new(log2_values, signs, col_indices, row_ptr, self.shape())
    }
}
