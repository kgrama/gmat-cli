//! Export module - GMAT to GGUF/SafeTensors conversion.
//!
//! Uses async pipeline with tokio-rayon for CPU-bound quantization.

mod config_gen;
mod shard;
mod tensor_overrides;
mod tensor_types;
mod token_metadata;
mod util;
mod validate;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use transform_storage::GraphMatrix;
use transform_storage::conversions::gguf_quant::{GgufQuantType, ScaleOptimization, quantize_to_gguf};

use crate::common::runtime::run_blocking;
use crate::common::load_config;
use crate::workqueue::{PipelineState, run_pipeline};

pub use config_gen::generate_config_template;
pub use validate::validate_gguf;

use crate::common::load_gmat_model;
use crate::config::export_config::ExportConfig;

use shard::{GgufStreamWriter, ProcessedTensor, ShardResult};
use tensor_types::{TensorEntry, QuantizeEntry, find_tensor_entry};
use token_metadata::TokenMetadata;
use util::{num_cpus, parse_quant_type};

/// Tensor processing job with all info needed for quantization.
struct QuantizeJob {
    source: String,
    target: String,
    quant_type: GgufQuantType,
    /// For simple 2D tensors: single path. For N-D: multiple plane paths.
    tensor_entry: QuantizeEntry,
}

/// Quantize a single tensor job (worker function for pipeline).
fn quantize_tensor_job(job: QuantizeJob, scale_opt: ScaleOptimization) -> Result<ProcessedTensor> {
    match &job.tensor_entry {
        QuantizeEntry::Simple { tensor_path } => {
            let matrix = GraphMatrix::load(tensor_path)
                .map_err(|e| anyhow::anyhow!("Failed to load {}: {}", job.source, e))?;

            let (rows, cols) = matrix.shape();

            let quant_data = quantize_to_gguf(&matrix, job.quant_type, scale_opt, None)
                .map_err(|e| anyhow::anyhow!("Quantize failed for {}: {}", job.target, e))?;

            Ok(ProcessedTensor {
                target_name: job.target,
                shape: (rows, cols),
                quant_data,
            })
        }
        QuantizeEntry::NdPlanes {
            original_shape,
            matrix_shape,
            plane_paths,
        } => {
            let (_rows, cols) = *matrix_shape;
            let mut all_quant_data: Vec<u8> = Vec::new();

            for (plane_idx, path) in plane_paths.iter().enumerate() {
                let matrix = GraphMatrix::load(path).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to load plane {} of {}: {}",
                        plane_idx,
                        job.source,
                        e
                    )
                })?;

                let plane_quant = quantize_to_gguf(&matrix, job.quant_type, scale_opt, None)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "Quantize failed for plane {} of {}: {}",
                            plane_idx,
                            job.target,
                            e
                        )
                    })?;

                all_quant_data.extend(plane_quant.data);
            }

            let total_elements: usize = original_shape.iter().product();
            let combined_rows = total_elements / cols;

            let quant_data = transform_storage::conversions::gguf_quant::GgufQuantizedData {
                data: all_quant_data,
                quant_type: job.quant_type,
                shape: (combined_rows, cols),
            };

            Ok(ProcessedTensor {
                target_name: job.target,
                shape: (combined_rows, cols),
                quant_data,
            })
        }
    }
}

/// Run the export process with async pipeline.
///
/// Uses tokio-rayon pattern:
/// - Producer: async, sends jobs to channel
/// - Workers: CPU-bound quantization on rayon pool
/// - Writer: async, writes results to GGUF
///
/// If `shard_size_override` is provided, it overrides the config file setting.
pub fn run(
    model_path: &str,
    config_path: Option<&str>,
    output_path: Option<&str>,
    shard_size_override: Option<u64>,
) -> Result<()> {
    // Run async pipeline
    run_blocking(run_async(
        model_path,
        config_path,
        output_path,
        shard_size_override,
    ))
}

/// Async implementation of export pipeline.
async fn run_async(
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
        _ => ScaleOptimization::Standard,
    };

    println!("Scale optimization: {:?}", scale_opt);

    let tensors_dir = model_dir.join("tensors");

    // Parse tensor entries from metadata
    let tensor_name_map = metadata["tensor_name_map"]
        .as_object()
        .context("tensor_name_map not found in metadata.json")?;

    // Build a lookup from source tensor name to entry
    let mut name_to_entry: HashMap<String, TensorEntry> = HashMap::new();
    for (name, value) in tensor_name_map {
        if let Some(entry) = TensorEntry::from_json(value) {
            name_to_entry.insert(name.clone(), entry);
        }
    }

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

            let tensor_entry = find_tensor_entry(&name_to_entry, &mapping.source, &tensors_dir)?;

            Some(QuantizeJob {
                source: mapping.source.clone(),
                target: mapping.target.clone(),
                quant_type,
                tensor_entry,
            })
        })
        .collect();

    let total = jobs.len();
    println!("\nProcessing {} tensors...", total);

    // Load tokenizer metadata if available
    let token_meta = match TokenMetadata::load(&model_dir).await {
        Ok(meta) => {
            println!("Loaded tokenizer: {} tokens", meta.vocab_size());
            Some(meta)
        }
        Err(e) => {
            eprintln!("Warning: Could not load tokenizer: {}", e);
            None
        }
    };

    let buffer_size = num_cpus().saturating_mul(2).max(4);

    // Capture values for closures - use pre-resolved config values
    let gguf_metadata = config.gguf_metadata.clone();
    let gguf_architecture = config.gguf_architecture.clone();
    let special_token_keys = config.special_token_keys.clone();
    let output_file_for_writer = output_file.clone();

    run_pipeline::<QuantizeJob, ProcessedTensor, anyhow::Error, _, _, _, _, _>(
        buffer_size,
        buffer_size,
        // Producer: send jobs
        move |tx: mpsc::Sender<QuantizeJob>, state: Arc<PipelineState>| async move {
            for job in jobs {
                state.inc_produced();
                if tx.send(job).await.is_err() {
                    break;
                }
            }
            Ok(())
        },
        // Worker: quantize tensor (runs on rayon)
        move |job: QuantizeJob| -> Result<ProcessedTensor> {
            quantize_tensor_job(job, scale_opt)
        },
        // Writer: stream-write to GGUF using pre-resolved config values
        move |mut rx: mpsc::Receiver<Result<ProcessedTensor>>, state: Arc<PipelineState>| async move {
            let mut writer = GgufStreamWriter::new(
                &gguf_metadata,
                &output_file_for_writer,
                shard_size,
                token_meta,
                &gguf_architecture,
                &special_token_keys,
            );

            while let Some(result) = rx.recv().await {
                let tensor = result?;
                let completed = state.inc_completed();
                println!(
                    "[{}/{}] {} [{},{}]",
                    completed, total, tensor.target_name, tensor.shape.0, tensor.shape.1
                );
                writer.add_tensor(tensor)?;
            }

            let result = writer.finish()?;

            match result {
                ShardResult::Single { tensor_count, bytes_written } => {
                    println!("Done! {} tensors, {} bytes", tensor_count, bytes_written);
                }
                ShardResult::Sharded { shard_count, total_tensors, total_bytes } => {
                    println!(
                        "Done! {} shards, {} tensors, {} bytes total",
                        shard_count, total_tensors, total_bytes
                    );
                }
            }

            Ok(())
        },
    ).await
}
