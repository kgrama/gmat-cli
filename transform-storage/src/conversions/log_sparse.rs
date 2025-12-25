//! Log-domain sparse matrix formats.
//!
//! These formats store values in log2 domain to enable log-domain arithmetic
//! without converting back to linear scale. Useful for log-space inference
//! and numerical stability.

use super::sparse::{CooMatrix, CsrMatrix};

/// Log-domain COO sparse matrix.
///
/// Stores sparse matrix in coordinate format with values in log2 domain.
/// Each non-zero value is represented as: value = sign * 2^log2_value
///
/// # Sign Convention
/// - 0 = positive
/// - 1 = negative
///
/// # Example
/// Linear value -4.0 is stored as:
/// - log2_value: 2.0 (because log2(4.0) = 2.0)
/// - sign: 1 (negative)
#[derive(Debug, Clone, PartialEq)]
pub struct LogSparseMatrix {
    /// Log2 of absolute magnitude for each non-zero value
    pub log2_values: Vec<f32>,
    /// Sign bit for each value (0 = positive, 1 = negative)
    pub signs: Vec<u8>,
    /// Row index for each value
    pub row_indices: Vec<usize>,
    /// Column index for each value
    pub col_indices: Vec<usize>,
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
}

impl LogSparseMatrix {
    /// Create a new log-domain sparse matrix.
    ///
    /// # Panics
    /// Panics if the input vectors have different lengths.
    pub fn new(
        log2_values: Vec<f32>,
        signs: Vec<u8>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(
            log2_values.len(),
            signs.len(),
            "log2_values and signs must have same length"
        );
        assert_eq!(
            log2_values.len(),
            row_indices.len(),
            "log2_values and row_indices must have same length"
        );
        assert_eq!(
            log2_values.len(),
            col_indices.len(),
            "log2_values and col_indices must have same length"
        );

        Self {
            log2_values,
            signs,
            row_indices,
            col_indices,
            shape,
        }
    }

    /// Convert to linear (standard f32) COO format.
    ///
    /// Reconstructs linear values as: value = sign_multiplier * 2^log2_value
    /// where sign_multiplier = 1.0 for sign=0, -1.0 for sign=1.
    pub fn to_linear(&self) -> CooMatrix {
        let values: Vec<f32> = self
            .log2_values
            .iter()
            .zip(self.signs.iter())
            .map(|(&log_val, &sign)| {
                let magnitude = log_val.exp2(); // 2^log_val
                if sign == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();

        CooMatrix::new(
            values,
            self.row_indices.clone(),
            self.col_indices.clone(),
            self.shape,
        )
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.log2_values.len()
    }
}

/// Log-domain CSR sparse matrix.
///
/// Stores sparse matrix in compressed sparse row format with values in log2 domain.
/// Each non-zero value is represented as: value = sign * 2^log2_value
///
/// # Sign Convention
/// - 0 = positive
/// - 1 = negative
///
/// # Example Structure
/// For a 3x4 matrix in log domain:
/// ```text
/// [2.0  0.0  -4.0  0.0]  (linear values)
/// [0.0  0.0  0.0   8.0]
/// [1.0  0.5  0.0   0.0]
/// ```
///
/// Log CSR representation:
/// - log2_values: [1.0, 2.0, 3.0, 0.0, -1.0]
/// - signs: [0, 1, 0, 0, 0]
/// - col_indices: [0, 2, 3, 0, 1]
/// - row_ptr: [0, 2, 3, 5]
#[derive(Debug, Clone, PartialEq)]
pub struct LogSparseCsrMatrix {
    /// Log2 of absolute magnitude for each non-zero value in row-major order
    pub log2_values: Vec<f32>,
    /// Sign bit for each value (0 = positive, 1 = negative)
    pub signs: Vec<u8>,
    /// Column index for each value
    pub col_indices: Vec<usize>,
    /// Pointers to start of each row (length = num_rows + 1)
    pub row_ptr: Vec<usize>,
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
}

impl LogSparseCsrMatrix {
    /// Create a new log-domain CSR matrix.
    ///
    /// # Panics
    /// Panics if invariants are violated:
    /// - log2_values.len() != signs.len() != col_indices.len()
    /// - row_ptr.len() != num_rows + 1
    /// - row_ptr is not monotonically increasing
    pub fn new(
        log2_values: Vec<f32>,
        signs: Vec<u8>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(
            log2_values.len(),
            signs.len(),
            "log2_values and signs must have same length"
        );
        assert_eq!(
            log2_values.len(),
            col_indices.len(),
            "log2_values and col_indices must have same length"
        );
        assert_eq!(
            row_ptr.len(),
            shape.0 + 1,
            "row_ptr must have length num_rows + 1"
        );
        assert_eq!(*row_ptr.first().unwrap(), 0, "row_ptr must start at 0");
        assert_eq!(
            *row_ptr.last().unwrap(),
            log2_values.len(),
            "row_ptr must end at values.len()"
        );

        Self {
            log2_values,
            signs,
            col_indices,
            row_ptr,
            shape,
        }
    }

    /// Convert to linear (standard f32) CSR format.
    ///
    /// Reconstructs linear values as: value = sign_multiplier * 2^log2_value
    /// where sign_multiplier = 1.0 for sign=0, -1.0 for sign=1.
    pub fn to_linear(&self) -> CsrMatrix {
        let values: Vec<f32> = self
            .log2_values
            .iter()
            .zip(self.signs.iter())
            .map(|(&log_val, &sign)| {
                let magnitude = log_val.exp2(); // 2^log_val
                if sign == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            })
            .collect();

        CsrMatrix::new(
            values,
            self.col_indices.clone(),
            self.row_ptr.clone(),
            self.shape,
        )
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.log2_values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_sparse_matrix_creation() {
        let log_sparse = LogSparseMatrix::new(
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![0, 2, 1],
            (2, 3),
        );
        assert_eq!(log_sparse.nnz(), 3);
        assert_eq!(log_sparse.shape, (2, 3));
    }

    #[test]
    fn test_log_sparse_to_linear() {
        // log2_values: [1.0, 2.0, 0.0] -> linear magnitudes: [2.0, 4.0, 1.0]
        // signs: [0, 1, 0] -> [+, -, +]
        // Expected linear values: [2.0, -4.0, 1.0]
        let log_sparse = LogSparseMatrix::new(
            vec![1.0, 2.0, 0.0],
            vec![0, 1, 0],
            vec![0, 0, 1],
            vec![0, 2, 1],
            (2, 3),
        );

        let coo = log_sparse.to_linear();

        assert_eq!(coo.values.len(), 3);
        assert!((coo.values[0] - 2.0).abs() < 1e-6);
        assert!((coo.values[1] - (-4.0)).abs() < 1e-6);
        assert!((coo.values[2] - 1.0).abs() < 1e-6);
        assert_eq!(coo.row_indices, vec![0, 0, 1]);
        assert_eq!(coo.col_indices, vec![0, 2, 1]);
    }

    #[test]
    fn test_log_sparse_csr_creation() {
        let log_sparse_csr = LogSparseCsrMatrix::new(
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 0],
            vec![0, 2, 1],
            vec![0, 2, 2, 3],
            (3, 3),
        );
        assert_eq!(log_sparse_csr.nnz(), 3);
        assert_eq!(log_sparse_csr.shape, (3, 3));
    }

    #[test]
    fn test_log_sparse_csr_to_linear() {
        // log2_values: [1.0, 2.0, 0.0] -> linear magnitudes: [2.0, 4.0, 1.0]
        // signs: [0, 1, 0] -> [+, -, +]
        // Expected linear values: [2.0, -4.0, 1.0]
        let log_sparse_csr = LogSparseCsrMatrix::new(
            vec![1.0, 2.0, 0.0],
            vec![0, 1, 0],
            vec![0, 2, 1],
            vec![0, 2, 2, 3],
            (3, 3),
        );

        let csr = log_sparse_csr.to_linear();

        assert_eq!(csr.values.len(), 3);
        assert!((csr.values[0] - 2.0).abs() < 1e-6);
        assert!((csr.values[1] - (-4.0)).abs() < 1e-6);
        assert!((csr.values[2] - 1.0).abs() < 1e-6);
        assert_eq!(csr.col_indices, vec![0, 2, 1]);
        assert_eq!(csr.row_ptr, vec![0, 2, 2, 3]);
    }

    #[test]
    fn test_negative_log_values() {
        // log2_values: [-1.0] -> magnitude: 2^(-1) = 0.5
        // signs: [1] -> negative
        // Expected: -0.5
        let log_sparse = LogSparseMatrix::new(vec![-1.0], vec![1], vec![0], vec![0], (1, 1));

        let coo = log_sparse.to_linear();
        assert!((coo.values[0] - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_empty_log_sparse() {
        let log_sparse = LogSparseMatrix::new(vec![], vec![], vec![], vec![], (3, 3));

        let coo = log_sparse.to_linear();
        assert_eq!(coo.nnz(), 0);
    }
}
