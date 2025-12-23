//! Sparse matrix formats (CSR and COO).
//!
//! Provides standard sparse matrix representations for inference backends
//! that expect sparse tensors.

/// Compressed Sparse Row (CSR) matrix format.
///
/// Efficient for row-based operations and matrix-vector multiplication.
/// Memory layout: values and col_indices are parallel arrays, row_ptr provides
/// offsets into these arrays for each row.
///
/// # Example Structure
/// For a 3x4 matrix:
/// ```text
/// [1.0  0.0  2.0  0.0]
/// [0.0  0.0  0.0  3.0]
/// [4.0  5.0  0.0  0.0]
/// ```
///
/// CSR representation:
/// - values: [1.0, 2.0, 3.0, 4.0, 5.0]
/// - col_indices: [0, 2, 3, 0, 1]
/// - row_ptr: [0, 2, 3, 5]
/// - shape: (3, 4)
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    /// Non-zero values in row-major order
    pub values: Vec<f32>,
    /// Column index for each value
    pub col_indices: Vec<usize>,
    /// Pointers to start of each row in values array (length = num_rows + 1)
    pub row_ptr: Vec<usize>,
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
}

impl CsrMatrix {
    /// Create a new CSR matrix.
    ///
    /// # Panics
    /// Panics if invariants are violated:
    /// - values.len() != col_indices.len()
    /// - row_ptr.len() != num_rows + 1
    /// - row_ptr is not monotonically increasing
    pub fn new(
        values: Vec<f32>,
        col_indices: Vec<usize>,
        row_ptr: Vec<usize>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(values.len(), col_indices.len(), "values and col_indices must have same length");
        assert_eq!(row_ptr.len(), shape.0 + 1, "row_ptr must have length num_rows + 1");
        assert_eq!(*row_ptr.first().unwrap(), 0, "row_ptr must start at 0");
        assert_eq!(*row_ptr.last().unwrap(), values.len(), "row_ptr must end at values.len()");
        
        Self {
            values,
            col_indices,
            row_ptr,
            shape,
        }
    }

    /// Convert from COO format to CSR format.
    pub fn from_coo(coo: CooMatrix) -> Self {
        let (num_rows, num_cols) = coo.shape;
        let nnz = coo.values.len();

        if nnz == 0 {
            return Self {
                values: Vec::new(),
                col_indices: Vec::new(),
                row_ptr: vec![0; num_rows + 1],
                shape: (num_rows, num_cols),
            };
        }

        // Create triplets and sort by (row, col)
        let mut triplets: Vec<(usize, usize, f32)> = coo.row_indices
            .iter()
            .zip(coo.col_indices.iter())
            .zip(coo.values.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();
        
        triplets.sort_by_key(|&(r, c, _)| (r, c));

        // Build CSR arrays
        let mut values = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut row_ptr = vec![0; num_rows + 1];

        for (r, c, v) in triplets {
            values.push(v);
            col_indices.push(c);
            row_ptr[r + 1] += 1;
        }

        // Convert counts to cumulative offsets
        for i in 0..num_rows {
            row_ptr[i + 1] += row_ptr[i];
        }

        Self {
            values,
            col_indices,
            row_ptr,
            shape: (num_rows, num_cols),
        }
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// Coordinate (COO) matrix format.
///
/// Simple triplet format (row, col, value) for each non-zero element.
/// Easy to construct incrementally but less efficient for computation.
///
/// # Example Structure
/// For a 3x4 matrix:
/// ```text
/// [1.0  0.0  2.0  0.0]
/// [0.0  0.0  0.0  3.0]
/// [4.0  5.0  0.0  0.0]
/// ```
///
/// COO representation:
/// - values: [1.0, 2.0, 3.0, 4.0, 5.0]
/// - row_indices: [0, 0, 1, 2, 2]
/// - col_indices: [0, 2, 3, 0, 1]
/// - shape: (3, 4)
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix {
    /// Non-zero values
    pub values: Vec<f32>,
    /// Row index for each value
    pub row_indices: Vec<usize>,
    /// Column index for each value
    pub col_indices: Vec<usize>,
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
}

impl CooMatrix {
    /// Create a new COO matrix.
    ///
    /// # Panics
    /// Panics if values, row_indices, and col_indices have different lengths.
    pub fn new(
        values: Vec<f32>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        shape: (usize, usize),
    ) -> Self {
        assert_eq!(values.len(), row_indices.len(), "values and row_indices must have same length");
        assert_eq!(values.len(), col_indices.len(), "values and col_indices must have same length");
        
        Self {
            values,
            row_indices,
            col_indices,
            shape,
        }
    }

    /// Convert from CSR format to COO format.
    pub fn from_csr(csr: CsrMatrix) -> Self {
        let nnz = csr.values.len();
        let mut row_indices = Vec::with_capacity(nnz);

        // Expand row_ptr to row indices
        for (row_idx, window) in csr.row_ptr.windows(2).enumerate() {
            let count = window[1] - window[0];
            row_indices.extend(std::iter::repeat(row_idx).take(count));
        }

        Self {
            values: csr.values,
            row_indices,
            col_indices: csr.col_indices,
            shape: csr.shape,
        }
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_creation() {
        let csr = CsrMatrix::new(
            vec![1.0, 2.0, 3.0],
            vec![0, 2, 1],
            vec![0, 2, 2, 3],
            (3, 3),
        );
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.shape, (3, 3));
    }

    #[test]
    fn test_coo_creation() {
        let coo = CooMatrix::new(
            vec![1.0, 2.0, 3.0],
            vec![0, 0, 2],
            vec![0, 2, 1],
            (3, 3),
        );
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.shape, (3, 3));
    }

    #[test]
    fn test_coo_to_csr() {
        let coo = CooMatrix::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0, 0, 1, 2, 2],
            vec![0, 2, 3, 0, 1],
            (3, 4),
        );

        let csr = CsrMatrix::from_coo(coo);
        
        assert_eq!(csr.values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(csr.col_indices, vec![0, 2, 3, 0, 1]);
        assert_eq!(csr.row_ptr, vec![0, 2, 3, 5]);
        assert_eq!(csr.shape, (3, 4));
    }

    #[test]
    fn test_csr_to_coo() {
        let csr = CsrMatrix::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0, 2, 3, 0, 1],
            vec![0, 2, 3, 5],
            (3, 4),
        );

        let coo = CooMatrix::from_csr(csr);
        
        assert_eq!(coo.values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(coo.row_indices, vec![0, 0, 1, 2, 2]);
        assert_eq!(coo.col_indices, vec![0, 2, 3, 0, 1]);
        assert_eq!(coo.shape, (3, 4));
    }

    #[test]
    fn test_empty_conversion() {
        let coo = CooMatrix::new(
            vec![],
            vec![],
            vec![],
            (3, 4),
        );

        let csr = CsrMatrix::from_coo(coo);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.row_ptr, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_roundtrip() {
        let original_coo = CooMatrix::new(
            vec![1.0, 2.0, 3.0],
            vec![0, 1, 2],
            vec![1, 2, 0],
            (3, 3),
        );

        let csr = CsrMatrix::from_coo(original_coo.clone());
        let roundtrip_coo = CooMatrix::from_csr(csr);
        
        assert_eq!(roundtrip_coo.values, original_coo.values);
        assert_eq!(roundtrip_coo.row_indices, original_coo.row_indices);
        assert_eq!(roundtrip_coo.col_indices, original_coo.col_indices);
        assert_eq!(roundtrip_coo.shape, original_coo.shape);
    }
}
