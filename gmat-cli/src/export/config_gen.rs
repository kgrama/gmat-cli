//! Export config generation from GMAT models.
//!
//! Analyzes tensors and generates export configuration files with
//! quantization recommendations based on importance scores.

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;
use transform_storage::GraphMatrix;
use transform_storage::conversions::gguf_quant::compute_tensor_importance;

use crate::common::runtime::{ProgressTracker, run_blocking};
use crate::config::export_config::{ExportConfig, GgufMetadataValues, QuantizationConfig, TensorExportMapping};

use super::tensor_overrides::TensorNameOverrides;
use super::tensor_types::TensorEntry;
use super::util::{ImportanceThresholds, recommend_quant_type};

/// Tensor analysis result for config generation.
#[allow(dead_code)]
pub struct TensorAnalysis {
    pub source_name: String,
    pub uuid: String,
    pub gguf_name: String,
    pub rows: usize,
    pub cols: usize,
    pub importance: f32,
    pub quant_type: String,
}

/// Export tensor statistics.
#[derive(Debug, Default)]
pub struct ExportTensorStats {
    total_tensors: usize,
    total_elements: u64,
    min_elements: u64,
    max_elements: u64,
    avg_importance: f32,
    min_importance: f32,
    max_importance: f32,
    quant_type_counts: HashMap<String, usize>,
}

impl ExportTensorStats {
    pub fn from_analyses(analyses: &[TensorAnalysis]) -> Self {
        if analyses.is_empty() {
            return Self::default();
        }

        let mut stats = Self {
            total_tensors: analyses.len(),
            min_elements: u64::MAX,
            min_importance: f32::MAX,
            max_importance: f32::MIN,
            ..Default::default()
        };

        let mut total_importance = 0.0f32;

        for analysis in analyses {
            let elements = (analysis.rows * analysis.cols) as u64;
            stats.total_elements += elements;
            stats.min_elements = stats.min_elements.min(elements);
            stats.max_elements = stats.max_elements.max(elements);

            total_importance += analysis.importance;
            stats.min_importance = stats.min_importance.min(analysis.importance);
            stats.max_importance = stats.max_importance.max(analysis.importance);

            *stats
                .quant_type_counts
                .entry(analysis.quant_type.clone())
                .or_insert(0) += 1;
        }

        stats.avg_importance = total_importance / analyses.len() as f32;
        stats
    }

    pub fn print_summary(&self) {
        println!("\n=== Export Tensor Statistics ===");
        println!("Total tensors: {}", self.total_tensors);
        println!(
            "Total elements: {} ({:.2} B)",
            self.total_elements,
            self.total_elements as f64 / 1e9
        );
        println!(
            "Elements range: {} - {} ({:.2} M max)",
            self.min_elements,
            self.max_elements,
            self.max_elements as f64 / 1e6
        );

        println!("\nImportance scores:");
        println!("  Min: {:.4}", self.min_importance);
        println!("  Max: {:.4}", self.max_importance);
        println!("  Avg: {:.4}", self.avg_importance);

        println!("\nQuantization types:");
        let mut quants: Vec<_> = self.quant_type_counts.iter().collect();
        quants.sort_by(|a, b| b.1.cmp(a.1));
        for (qtype, count) in quants {
            let pct = (*count as f64 / self.total_tensors as f64) * 100.0;
            println!("  {}: {} ({:.1}%)", qtype, count, pct);
        }
    }
}

/// Helper to get a value from nested metadata using key aliases.
/// Tries aliases first, then falls back to the canonical key.
fn get_metadata_value<T, F>(
    metadata: &serde_json::Value,
    text_cfg: Option<&serde_json::Value>,
    canonical: &str,
    aliases: Option<&[String]>,
    extractor: F,
) -> Option<T>
where
    F: Fn(&serde_json::Value) -> Option<T>,
{
    // Try aliases first
    if let Some(alias_list) = aliases {
        for alias in alias_list {
            let val = text_cfg
                .and_then(|cfg| cfg.get(alias))
                .or_else(|| metadata.get(alias))
                .and_then(&extractor);
            if val.is_some() {
                return val;
            }
        }
    }
    // Fallback to canonical key
    text_cfg
        .and_then(|cfg| cfg.get(canonical))
        .or_else(|| metadata.get(canonical))
        .and_then(extractor)
}

/// Extract GGUF metadata from model metadata.json using key aliases.
pub fn extract_gguf_metadata(
    metadata: &serde_json::Value,
    overrides: &TensorNameOverrides,
) -> GgufMetadataValues {
    // Nested config keys for VLMs (same as import)
    const NESTED_KEYS: &[&str] = &["text_config", "language_config", "llm_config"];

    let text_cfg = NESTED_KEYS
        .iter()
        .find_map(|key| metadata.get(*key))
        .filter(|v| v.is_object());

    // Helper closures using the generic function
    let get_num = |canonical: &str| -> Option<u64> {
        get_metadata_value(
            metadata,
            text_cfg,
            canonical,
            overrides.metadata_key_aliases(canonical),
            |v| v.as_u64(),
        )
    };

    let get_f64 = |canonical: &str| -> Option<f64> {
        get_metadata_value(
            metadata,
            text_cfg,
            canonical,
            overrides.metadata_key_aliases(canonical),
            |v| v.as_f64(),
        )
    };

    GgufMetadataValues {
        name: metadata.get("name").and_then(|v| v.as_str()).map(String::from),
        vocab_size: metadata
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .or_else(|| text_cfg.and_then(|cfg| cfg.get("vocab_size")).and_then(|v| v.as_u64())),
        hidden_size: get_num("hidden_size"),
        num_layers: get_num("num_layers"),
        num_attention_heads: get_num("num_attention_heads"),
        num_key_value_heads: get_num("num_key_value_heads"),
        intermediate_size: get_num("intermediate_size"),
        max_position_embeddings: get_num("max_position_embeddings"),
        rms_norm_eps: get_f64("rms_norm_eps"),
        rope_theta: get_f64("rope_theta"),
        num_experts: get_num("num_experts"),
        num_experts_per_tok: get_num("num_experts_per_tok"),
    }
}

/// Analyze tensors in parallel and compute importance/quantization recommendations.
pub fn analyze_model_tensors(
    tensor_entries: Vec<(&String, &serde_json::Value)>,
    tensors_dir: &Path,
    thresholds: &ImportanceThresholds,
    overrides: &TensorNameOverrides,
) -> Vec<TensorAnalysis> {
    let progress = ProgressTracker::new(tensor_entries.len(), "Analyzed tensors");

    let results: Vec<TensorAnalysis> = tensor_entries
        .par_iter()
        .filter_map(|(source_name, json_value)| {
            let entry = TensorEntry::from_json(json_value)?;

            // Get UUID, path, and shape info - load matrix once for both shape and importance
            let (uuid, matrix, shape) = match &entry {
                TensorEntry::Simple { uuid } => {
                    let path = tensors_dir.join(format!("{}.gmat", uuid));
                    if !path.exists() {
                        eprintln!("Warning: missing tensor: {}", path.display());
                        return None;
                    }
                    let matrix = GraphMatrix::load(&path).ok()?;
                    let shape = matrix.shape();
                    (uuid.clone(), matrix, shape)
                }
                TensorEntry::NdPlanes {
                    plane_uuids,
                    matrix_shape,
                    original_shape,
                    ..
                } => {
                    // For N-D tensors, analyze first plane and compute total size
                    let first_uuid = plane_uuids.first()?;
                    let path = tensors_dir.join(format!("{}.gmat", first_uuid));
                    if !path.exists() {
                        eprintln!("Warning: missing tensor plane: {}", path.display());
                        return None;
                    }
                    let matrix = GraphMatrix::load(&path).ok()?;
                    // Use original shape's last 2 dims times num_planes for total size estimation
                    let total_elements: usize = original_shape.iter().product();
                    let approx_rows = total_elements / matrix_shape.1;
                    (first_uuid.clone(), matrix, (approx_rows, matrix_shape.1))
                }
            };

            let importance = compute_tensor_importance(&matrix);
            let (rows, cols) = shape;

            let gguf_name = overrides.map_name(source_name).unwrap_or_else(|| source_name.to_string());
            let quant_type = recommend_quant_type(source_name, importance, rows, cols, thresholds, overrides);

            progress.increment_every(100);

            Some(TensorAnalysis {
                source_name: source_name.to_string(),
                uuid,
                gguf_name,
                rows,
                cols,
                importance,
                quant_type,
            })
        })
        .collect();

    progress.finish();
    results
}

/// Build export config from tensor analyses and write to file.
pub fn build_and_write_config(
    analyses: Vec<TensorAnalysis>,
    model_path: &str,
    overrides: &TensorNameOverrides,
    metadata: &serde_json::Value,
) -> Result<()> {
    let mut tensor_mappings = Vec::with_capacity(analyses.len());
    let mut per_tensor_quant = HashMap::with_capacity(analyses.len());

    for analysis in &analyses {
        tensor_mappings.push(TensorExportMapping {
            source: analysis.uuid.clone(),
            target: analysis.gguf_name.clone(),
        });
        per_tensor_quant.insert(analysis.uuid.clone(), analysis.quant_type.clone());
    }

    // Print statistics
    let stats = ExportTensorStats::from_analyses(&analyses);
    stats.print_summary();

    // Extract pre-resolved metadata values
    let gguf_metadata = extract_gguf_metadata(metadata, overrides);

    let config = ExportConfig {
        target_format: "gguf".to_string(),
        gguf_architecture: overrides.gguf_architecture().to_string(),
        gguf_metadata,
        quantization: Some(QuantizationConfig {
            default_type: "q4_k_m".to_string(),
            scale_optimization: "trellis".to_string(),
            trellis_lambda: 0.3,
            per_tensor: per_tensor_quant,
        }),
        tensor_map: tensor_mappings,
        shard_size: None,
        special_token_keys: overrides.special_token_keys().clone(),
    };

    fs::write("export_config.json", serde_json::to_string_pretty(&config)?)?;

    println!(
        "\n=== Generated export_config.json ({} tensors) ===",
        analyses.len()
    );
    println!(
        "Run: gmat export --model {} --config export_config.json -o model.gguf",
        model_path
    );

    Ok(())
}

/// Generate an export config template from a GMAT model.
///
/// Uses tokio-rayon for parallel tensor analysis.
/// The model_config parameter is REQUIRED - path to model config JSON.
pub fn generate_config_template(
    model_path: &str,
    importance_high: f32,
    importance_medium: f32,
    model_config: &str,
) -> Result<()> {
    run_blocking(generate_config_template_async(
        model_path,
        importance_high,
        importance_medium,
        model_config,
    ))
}

/// Async implementation of export config generation.
async fn generate_config_template_async(
    model_path: &str,
    importance_high: f32,
    importance_medium: f32,
    model_config: &str,
) -> Result<()> {
    use crate::common::load_gmat_model;

    let start_time = Instant::now();
    let (model_dir, metadata) = load_gmat_model(model_path)?;

    let tensor_map = metadata["tensor_name_map"]
        .as_object()
        .context("tensor_name_map not found in metadata.json")?;

    // Load tensor name overrides from required model config file
    let overrides = TensorNameOverrides::from_file(Path::new(model_config))
        .context("Failed to load model config")?;
    println!("Using model config: {} (architecture: {})", model_config, overrides.gguf_architecture());

    let thresholds = ImportanceThresholds::new(importance_high, importance_medium);
    println!(
        "Analyzing {} tensors (thresholds: high={}, medium={})...",
        tensor_map.len(),
        thresholds.high,
        thresholds.medium
    );

    let tensors_dir = model_dir.join("tensors");

    // Collect tensor entries for parallel processing (clone owned data for async)
    let tensor_entries: Vec<(String, serde_json::Value)> = tensor_map
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Run CPU-bound tensor analysis on rayon pool
    // Load overrides again for analysis (will be used in closure)
    let overrides_for_analysis = TensorNameOverrides::from_file(Path::new(model_config))
        .context("Failed to reload model config")?;
    let analysis_start = Instant::now();
    let analyses = tokio_rayon::spawn(move || {
        let entries_ref: Vec<(&String, &serde_json::Value)> =
            tensor_entries.iter().map(|(k, v)| (k, v)).collect();
        analyze_model_tensors(entries_ref, &tensors_dir, &thresholds, &overrides_for_analysis)
    })
    .await;
    let analysis_elapsed = analysis_start.elapsed();

    // Build and write config with overrides and metadata for gguf_metadata extraction
    let result = build_and_write_config(analyses, model_path, &overrides, &metadata);

    let total_elapsed = start_time.elapsed();

    println!("\n=== Timing ===");
    println!("Tensor analysis: {:.2}s", analysis_elapsed.as_secs_f64());
    println!("Total: {:.2}s", total_elapsed.as_secs_f64());

    result
}
