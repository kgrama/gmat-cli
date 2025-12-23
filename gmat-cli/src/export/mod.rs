//! Export module - GMAT to GGUF/SafeTensors conversion.

mod shard;
mod util;

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;
use transform_storage::conversions::gguf_quant::{
    compute_tensor_importance, quantize_to_gguf, GgufQuantType, ScaleOptimization,
};
use transform_storage::GraphMatrix;

use crate::common::{load_config, load_gmat_model};
use crate::config::export_config::{ExportConfig, QuantizationConfig, TensorExportMapping};

use shard::{run_sharded_consumer, run_single_file_consumer, ProcessedTensor, ShardResult};
use util::{num_cpus, parse_quant_type, recommend_quant_type, safetensor_to_gguf_name};

/// Tensor analysis result for config generation.
struct TensorAnalysis {
    source_name: String,
    uuid: String,
    gguf_name: String,
    rows: usize,
    cols: usize,
    importance: f32,
    quant_type: String,
}

/// Tensor processing job with all info needed for quantization.
struct QuantizeJob {
    source: String,
    target: String,
    tensor_path: PathBuf,
    quant_type: GgufQuantType,
}

/// Generate an export config template from a GMAT model.
pub fn generate_config_template(model_path: &str) -> Result<()> {
    let (model_dir, metadata) = load_gmat_model(model_path)?;

    let tensor_map = metadata["tensor_name_map"]
        .as_object()
        .context("tensor_name_map not found in metadata.json")?;

    println!("Analyzing {} tensors...", tensor_map.len());

    let tensors_dir = model_dir.join("tensors");

    // Collect tensor entries for parallel processing
    let tensor_entries: Vec<(&String, &serde_json::Value)> = tensor_map.iter().collect();

    // Analyze tensors in parallel
    let analyses: Vec<TensorAnalysis> = tensor_entries
        .par_iter()
        .filter_map(|(source_name, uuid_value)| {
            let uuid = uuid_value.as_str()?;
            let tensor_path = tensors_dir.join(format!("{}.gmat", uuid));

            if !tensor_path.exists() {
                eprintln!("Warning: missing tensor: {}", tensor_path.display());
                return None;
            }

            let matrix = GraphMatrix::load(&tensor_path).ok()?;
            let importance = compute_tensor_importance(&matrix);
            let (rows, cols) = matrix.shape();

            let gguf_name = safetensor_to_gguf_name(source_name);
            let quant_type = recommend_quant_type(source_name, importance, rows * cols);

            Some(TensorAnalysis {
                source_name: source_name.to_string(),
                uuid: uuid.to_string(),
                gguf_name,
                rows,
                cols,
                importance,
                quant_type,
            })
        })
        .collect();

    // Print results and build config
    let mut tensor_mappings = Vec::with_capacity(analyses.len());
    let mut per_tensor_quant = HashMap::with_capacity(analyses.len());

    for analysis in &analyses {
        println!(
            "  {} -> {} [{}x{}] imp={:.3} quant={}",
            analysis.source_name,
            analysis.gguf_name,
            analysis.rows,
            analysis.cols,
            analysis.importance,
            analysis.quant_type
        );

        tensor_mappings.push(TensorExportMapping {
            source: analysis.uuid.clone(),
            target: analysis.gguf_name.clone(),
        });
        per_tensor_quant.insert(analysis.uuid.clone(), analysis.quant_type.clone());
    }

    let config = ExportConfig {
        target_format: "gguf".to_string(),
        quantization: Some(QuantizationConfig {
            default_type: "q4_k_m".to_string(),
            scale_optimization: "trellis".to_string(),
            trellis_lambda: 0.3,
            per_tensor: per_tensor_quant,
        }),
        tensor_map: tensor_mappings,
        shard_size: None,
    };

    fs::write("export_config.json", serde_json::to_string_pretty(&config)?)?;

    println!(
        "\nGenerated export_config.json ({} tensors)",
        tensor_map.len()
    );
    println!(
        "Run: gmat export --model {} --config export_config.json -o model.gguf",
        model_path
    );

    Ok(())
}

/// Run the export process with parallel quantization and incremental building.
///
/// Uses a producer-consumer pattern:
/// - Producer threads (rayon pool): Load and quantize tensors in parallel
/// - Consumer thread: Adds tensors to GGUF builder as they complete
///
/// This limits memory usage to ~buffer_size tensors at a time instead of all tensors.
///
/// If `shard_size_override` is provided, it overrides the config file setting.
pub fn run(
    model_path: &str,
    config_path: Option<&str>,
    output_path: Option<&str>,
    shard_size_override: Option<u64>,
) -> Result<()> {
    let config: ExportConfig = load_config(config_path, "export_config.json")?;

    if config.target_format.to_lowercase() != "gguf" {
        anyhow::bail!("Only 'gguf' target format is currently supported");
    }

    let (model_dir, metadata) = load_gmat_model(model_path)?;
    let output_file = output_path.unwrap_or("model.gguf").to_string();

    // CLI override takes precedence over config file
    let shard_size = shard_size_override.or(config.shard_size);

    if let Some(size) = shard_size {
        println!(
            "Exporting to GGUF (sharded, max {} MB per shard): {}",
            size / 1_000_000,
            output_file
        );
    } else {
        println!("Exporting to GGUF: {}", output_file);
    }

    let quant_config = config
        .quantization
        .as_ref()
        .context("quantization config is required")?;
    let default_quant = parse_quant_type(&quant_config.default_type)?;

    // Parse scale optimization setting
    let scale_opt = match quant_config.scale_optimization.to_lowercase().as_str() {
        "trellis" => ScaleOptimization::Trellis {
            lambda: quant_config.trellis_lambda,
        },
        "standard" | _ => ScaleOptimization::Standard,
    };

    println!("Scale optimization: {:?}", scale_opt);

    let tensors_dir = model_dir.join("tensors");

    // Build job list with validation
    let jobs: Vec<QuantizeJob> = config
        .tensor_map
        .iter()
        .filter_map(|mapping| {
            let quant_type = quant_config
                .per_tensor
                .get(&mapping.source)
                .map(|s| parse_quant_type(s))
                .transpose()
                .ok()?
                .unwrap_or(default_quant);

            let tensor_path = tensors_dir.join(format!("{}.gmat", mapping.source));
            if !tensor_path.exists() {
                eprintln!("Warning: missing tensor: {}", tensor_path.display());
                return None;
            }

            Some(QuantizeJob {
                source: mapping.source.clone(),
                target: mapping.target.clone(),
                tensor_path,
                quant_type,
            })
        })
        .collect();

    let total = jobs.len();
    println!("\nProcessing {} tensors...", total);

    // Bounded channel to limit memory usage
    let buffer_size = num_cpus().saturating_mul(2).max(4);
    let (tx, rx) = mpsc::sync_channel::<Result<ProcessedTensor>>(buffer_size);

    let counter = AtomicUsize::new(0);

    // Clone metadata for consumer thread
    let metadata_clone = metadata.clone();
    let output_file_clone = output_file.clone();

    // Spawn consumer thread that builds GGUF incrementally (with sharding support)
    let consumer = thread::spawn(move || -> Result<ShardResult> {
        if let Some(max_shard_size) = shard_size {
            run_sharded_consumer(rx, &metadata_clone, &output_file_clone, max_shard_size)
        } else {
            run_single_file_consumer(rx, &metadata_clone, &output_file_clone)
        }
    });

    // Producer: quantize tensors in parallel and send to consumer
    jobs.into_par_iter().for_each(|job| {
        let result = (|| -> Result<ProcessedTensor> {
            let matrix = GraphMatrix::load(&job.tensor_path)
                .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", job.source, e))?;

            let (rows, cols) = matrix.shape();

            let quant_data = quantize_to_gguf(&matrix, job.quant_type, scale_opt, None)
                .map_err(|e| anyhow::anyhow!("Quantize failed for {}: {}", job.target, e))?;

            let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
            println!(
                "[{}/{}] {} [{}x{}] -> {:?}",
                count, total, job.target, rows, cols, job.quant_type
            );

            Ok(ProcessedTensor {
                target_name: job.target,
                shape: (rows, cols),
                quant_data,
            })
        })();

        // Send result (ok or error) to consumer
        let _ = tx.send(result);
    });

    // Drop sender to signal completion
    drop(tx);

    // Wait for consumer and get results
    let result = consumer
        .join()
        .map_err(|_| anyhow::anyhow!("Consumer thread panicked"))??;

    match result {
        ShardResult::Single {
            tensor_count,
            bytes_written,
        } => {
            println!("Done! {} tensors, {} bytes", tensor_count, bytes_written);
        }
        ShardResult::Sharded {
            shard_count,
            total_tensors,
            total_bytes,
        } => {
            println!(
                "Done! {} shards, {} tensors, {} bytes total",
                shard_count, total_tensors, total_bytes
            );
        }
    }

    Ok(())
}
