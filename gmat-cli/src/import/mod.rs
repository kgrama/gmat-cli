//! Import module - SafeTensors/GGUF to GMAT conversion.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use transform_storage::{BlockFormat, GmatMetadata, GraphMatrix};
use uuid::Uuid;

use crate::common::{bytes_to_f32, discover_safetensor_files, load_config};
use crate::config::import_config::{ImportConfig, ModelMetadata, TensorMapping};

/// Generate an import config template from a model file.
pub fn generate_config_template(model_path: &str) -> Result<()> {
    let path = Path::new(model_path);
    let safetensor_files = discover_safetensor_files(path)?;

    println!("Found {} safetensors file(s)", safetensor_files.len());

    let mut tensor_mappings = Vec::new();
    let metadata = ModelMetadata::default();

    for file_path in &safetensor_files {
        println!("Processing: {}", file_path.display());

        let data = fs::read(file_path)
            .with_context(|| format!("Failed to read: {}", file_path.display()))?;

        let st = SafeTensors::deserialize(&data)
            .with_context(|| format!("Failed to parse: {}", file_path.display()))?;

        for (name, tensor_view) in st.tensors() {
            let shape = tensor_view.shape();
            tensor_mappings.push(TensorMapping {
                source: name.to_string(),
                target: Uuid::new_v4().to_string(),
                include: true,
            });
            println!("  - {} (shape: {:?})", name, shape);
        }
    }

    let config = ImportConfig {
        source_format: "safetensors".to_string(),
        block_format: "B8x8".to_string(),
        tensor_map: tensor_mappings,
        metadata,
    };

    let json = serde_json::to_string_pretty(&config)?;
    fs::write("import_config.json", json)?;

    println!("\n=== Generated import_config.json ===");
    println!("Tensors: {}", config.tensor_map.len());
    println!("\nEdit the config, then run:");
    println!("  gmat import --model {} --config import_config.json", model_path);

    Ok(())
}

/// Run the import process.
pub fn run(model_path: &str, config_path: Option<&str>, output_path: Option<&str>) -> Result<()> {
    let config: ImportConfig = load_config(config_path, "import_config.json")?;

    let block_format = parse_block_format(&config.block_format)
        .with_context(|| format!("Invalid block format: {}", config.block_format))?;

    let output_dir = Path::new(output_path.unwrap_or("output")).join("model.gmat");
    let tensors_dir = output_dir.join("tensors");
    fs::create_dir_all(&tensors_dir)?;

    println!("Output: {}", output_dir.display());
    println!("Block format: {:?}", block_format);

    let path = Path::new(model_path);
    let safetensor_files = discover_safetensor_files(path)?;

    println!("Found {} safetensors file(s)", safetensor_files.len());

    let tensor_map: HashMap<&str, &str> = config.tensor_map.iter()
        .filter(|tm| tm.include)
        .map(|tm| (tm.source.as_str(), tm.target.as_str()))
        .collect();

    let mut converted = Vec::new();

    for file_path in &safetensor_files {
        println!("\nProcessing: {}", file_path.display());

        let data = fs::read(file_path)?;
        let st = SafeTensors::deserialize(&data)?;

        for (tensor_name, tensor_view) in st.tensors() {
            let Some(&target_uuid) = tensor_map.get(tensor_name.as_str()) else {
                continue;
            };

            let shape = tensor_view.shape();
            let dtype = tensor_view.dtype();

            if shape.len() != 2 {
                anyhow::bail!("Tensor '{}' must be 2D, got {:?}", tensor_name, shape);
            }

            let (rows, cols) = (shape[0], shape[1]);
            println!("  {} [{rows}x{cols}] {:?}", tensor_name, dtype);

            let f32_data = bytes_to_f32(tensor_view.data(), dtype, rows * cols)?;

            let mut matrix = GraphMatrix::from_dense(&f32_data, (rows, cols), block_format);

            let mut meta = GmatMetadata::new();
            meta.set_str("source_tensor_name", tensor_name.as_str());
            meta.set_str("tensor_uuid", target_uuid);
            meta.set_str("original_dtype", &format!("{:?}", dtype));
            meta.set_str("block_format", &config.block_format);
            meta.set_str("source_format", &config.source_format);

            if let Some(ref arch) = config.metadata.architecture {
                meta.set_str("model_architecture", arch);
            }
            if let Some(v) = config.metadata.vocab_size { meta.set_u64("vocab_size", v); }
            if let Some(v) = config.metadata.hidden_size { meta.set_u64("hidden_size", v); }
            if let Some(v) = config.metadata.num_layers { meta.set_u64("num_layers", v); }

            matrix.set_metadata(meta);

            let out_file = tensors_dir.join(format!("{}.gmat", target_uuid));
            matrix.save(&out_file)?;

            println!("    -> {} (nnz={}, sparsity={:.1}%)",
                out_file.file_name().unwrap().to_string_lossy(),
                matrix.nnz(), matrix.sparsity() * 100.0);

            converted.push((tensor_name.to_string(), target_uuid.to_string()));
        }
    }

    // Write metadata.json
    let metadata_json = serde_json::json!({
        "config": config,
        "tensor_name_map": converted.iter()
            .map(|(n, u)| (n.as_str(), u.as_str()))
            .collect::<HashMap<_, _>>(),
        "total_tensors": converted.len(),
    });

    fs::write(
        output_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata_json)?,
    )?;

    println!("\n=== Import Complete: {} tensors ===", converted.len());
    Ok(())
}

/// Parse block format string to BlockFormat enum.
fn parse_block_format(s: &str) -> Result<BlockFormat> {
    match s {
        "B8x4" => Ok(BlockFormat::B8x4),
        "B8x8" => Ok(BlockFormat::B8x8),
        "B16x4" => Ok(BlockFormat::B16x4),
        "B16x8" => Ok(BlockFormat::B16x8),
        "DualRow8x4" => Ok(BlockFormat::DualRow8x4),
        "DualRow8x8" => Ok(BlockFormat::DualRow8x8),
        "DualRow16x4" => Ok(BlockFormat::DualRow16x4),
        "DualRow16x8" => Ok(BlockFormat::DualRow16x8),
        _ => anyhow::bail!("Unknown block format: {}. Valid formats: B8x4, B8x8, B16x4, B16x8, DualRow8x4, DualRow8x8, DualRow16x4, DualRow16x8", s),
    }
}
