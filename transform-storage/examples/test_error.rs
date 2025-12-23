use transform_storage::{GraphMatrix, BlockFormat};

fn main() {
    // Test with values in reasonable dynamic range (~2 octaves)
    let test_values: Vec<f32> = vec![
        1.0, 1.2, 1.5, 1.8,           // Close to 1.0
        2.0, 2.2, 2.5, 2.8,           // Close to 2.0
        -1.0, -1.5, -2.0, -2.5,       // Negative values
        0.8, 0.9, 1.1, 1.3,           // Around 1.0
    ];
    
    let matrix = GraphMatrix::from_dense(&test_values, (1, 16), BlockFormat::B16x8);
    
    // Get log-sparse representation
    let log_sparse = matrix.to_log_sparse();
    
    // Convert back to linear
    let linear = log_sparse.to_linear();
    
    // Calculate errors
    let mut max_rel_error = 0.0f32;
    let mut total_rel_error = 0.0f32;
    
    for i in 0..linear.values.len() {
        let col = linear.col_indices[i];
        let original = test_values[col];
        let reconstructed = linear.values[i];
        
        let abs_error = (original - reconstructed).abs();
        let rel_error = abs_error / original.abs();
        
        max_rel_error = max_rel_error.max(rel_error);
        total_rel_error += rel_error;
        
        println!("Orig: {:6.3}, Recon: {:6.3}, Err: {:5.2}%", 
                 original, reconstructed, rel_error * 100.0);
    }
    
    let avg_rel_error = total_rel_error / linear.values.len() as f32;
    
    println!("\n=== SUMMARY ===");
    println!("Max relative error: {:.2}%", max_rel_error * 100.0);
    println!("Avg relative error: {:.2}%", avg_rel_error * 100.0);
}
