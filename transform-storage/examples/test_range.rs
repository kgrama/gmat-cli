use transform_storage::{GraphMatrix, BlockFormat};

fn main() {
    // Wide dynamic range data
    let data: Vec<f32> = vec![
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,
        0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0,
    ];
    
    println!("Original data log2 range: {:.1} to {:.1} = {:.1} octaves",
             0.001f32.log2(), 100.0f32.log2(), 
             100.0f32.log2() - 0.001f32.log2());
    
    // Split into 2 rows of 8 - each block handles ~8 octave subset
    let matrix = GraphMatrix::from_dense(&data, (2, 8), BlockFormat::B8x8);

    // Check what each block stores
    for (block_idx, block) in matrix.block_iter().enumerate() {
        println!("\nBlock {} scale_log: {:.2}", block_idx, block.scale_log().to_f32());
        for (idx, log_mag, sign) in block.log_iter() {
            let data_idx = block_idx * 8 + idx;
            let orig = data[data_idx];
            let reconstructed = if sign == 1 { -1.0 } else { 1.0 } * 2.0f32.powf(log_mag);
            let error = (orig - reconstructed).abs() / orig.abs() * 100.0;
            println!("  [{}] orig={:8.4} log_mag={:6.2} recon={:8.4} err={:.1}%",
                     idx, orig, log_mag, reconstructed, error);
        }
    }
}
