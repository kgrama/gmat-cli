//! Export module - GMAT to GGUF/SafeTensors conversion.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use transform_storage::GraphMatrix;
use transform_storage::conversions::gguf_quant::compute_tensor_importance;

use crate::common::{load_config, load_gmat_model};
use crate::config::export_config::{ExportConfig, QuantizationConfig, TensorExportMapping};

/// Generate an export config template from a GMAT model.
pub fn generate_config_template(model_path: &str) -> Result<()> {
    let (model_dir, metadata) = load_gmat_model(model_path)?;

    let tensor_map = metadata["tensor_name_map"].as_object()
        .context("tensor_name_map not found in metadata.json")?;

    println!("Analyzing {} tensors...", tensor_map.len());

    let mut tensor_mappings = Vec::new();
    let mut per_tensor_quant = HashMap::new();
    let tensors_dir = model_dir.join("tensors");

    for (source_name, uuid_value) in tensor_map {
        let uuid = uuid_value.as_str()
            .with_context(|| format!("UUID for {} is not a string", source_name))?;

        let tensor_path = tensors_dir.join(format!("{}.gmat", uuid));
        if !tensor_path.exists() {
            eprintln!("Warning: missing tensor: {}", tensor_path.display());
            continue;
        }

        let matrix = GraphMatrix::load(&tensor_path)?;
        let importance = compute_tensor_importance(&matrix);
        let (rows, cols) = matrix.shape();

        let gguf_name = safetensor_to_gguf_name(source_name);
        let quant_type = recommend_quant_type(source_name, importance, rows * cols);

        println!("  {} -> {} [{}x{}] imp={:.3} quant={}",
            source_name, gguf_name, rows, cols, importance, quant_type);

        tensor_mappings.push(TensorExportMapping {
            source: uuid.to_string(),
            target: gguf_name,
        });
        per_tensor_quant.insert(uuid.to_string(), quant_type);
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
    };

    fs::write("export_config.json", serde_json::to_string_pretty(&config)?)?;

    println!("\nGenerated export_config.json ({} tensors)", tensor_map.len());
    println!("Run: gmat export --model {} --config export_config.json -o model.gguf", model_path);

    Ok(())
}

/// Convert SafeTensor name to GGUF tensor name (llama.cpp convention).
fn safetensor_to_gguf_name(st_name: &str) -> String {
    // Handle token embeddings
    if st_name == "model.embed_tokens.weight" {
        return "token_embd.weight".to_string();
    }

    // Handle output layer and norm
    if st_name == "lm_head.weight" {
        return "output.weight".to_string();
    }
    if st_name == "model.norm.weight" {
        return "output_norm.weight".to_string();
    }

    // Handle layer-specific tensors
    if let Some(rest) = st_name.strip_prefix("model.layers.") {
        // Extract layer number
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let tensor_type = &rest[dot_pos + 1..];

            // Map tensor types
            let gguf_type = match tensor_type {
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                _ => tensor_type,
            };

            return format!("blk.{}.{}", layer_num, gguf_type);
        }
    }

    // Fallback: use original name if no mapping found
    st_name.to_string()
}

/// Recommend quantization type based on tensor name, importance, and size.
///
/// Uses heuristics from resume.txt:
/// - High-importance tensors (embeddings, attention output) → Q6_K
/// - Standard tensors → Q4_K_M
/// - Small tensors (<1M params) → Q8_0
fn recommend_quant_type(tensor_name: &str, importance: f32, size: usize) -> String {
    // Small tensors: use Q8_0 for minimal impact
    if size < 1_000_000 {
        return "q8_0".to_string();
    }

    // Check tensor type patterns
    let name_lower = tensor_name.to_lowercase();

    // Token embeddings and output layers: highest quality
    if name_lower.contains("embed_tokens") || name_lower.contains("lm_head") ||
       name_lower.contains("model.norm") {
        return if importance > 0.5 { "q8_0" } else { "q6_k" }.to_string();
    }

    // Attention output and value: sensitive to quantization
    if name_lower.contains("attn.o_proj") || name_lower.contains("attn.v_proj") {
        return if importance > 0.3 { "q6_k" } else { "q5_k_m" }.to_string();
    }

    // FFN down projection: output path, more sensitive
    if name_lower.contains("mlp.down_proj") {
        return if importance > 0.3 { "q6_k" } else { "q5_k_m" }.to_string();
    }

    // Attention Q/K: can tolerate more compression
    if name_lower.contains("attn.q_proj") || name_lower.contains("attn.k_proj") {
        return "q4_k_m".to_string();
    }

    // FFN gate and up: large, compressible
    if name_lower.contains("mlp.gate_proj") || name_lower.contains("mlp.up_proj") {
        return "q4_k_m".to_string();
    }

    // Default: Q4_K_M for standard tensors
    "q4_k_m".to_string()
}

/// Run the export process.
pub fn run(model_path: &str, config_path: Option<&str>, output_path: Option<&str>) -> Result<()> {
    use transform_storage::conversions::gguf_quant::{quantize_to_gguf, ScaleOptimization};
    use gguf_rs_lib::builder::GGUFBuilder;
    use gguf_rs_lib::format::MetadataValue;

    let config: ExportConfig = load_config(config_path, "export_config.json")?;

    if config.target_format.to_lowercase() != "gguf" {
        anyhow::bail!("Only 'gguf' target format is currently supported");
    }

    let (model_dir, metadata) = load_gmat_model(model_path)?;
    let output_file = output_path.unwrap_or("model.gguf");

    println!("Exporting to GGUF: {}", output_file);

    let quant_config = config.quantization.as_ref()
        .context("quantization config is required")?;
    let default_quant = parse_quant_type(&quant_config.default_type)?;

    // Parse scale optimization setting
    let scale_opt = match quant_config.scale_optimization.to_lowercase().as_str() {
        "trellis" => ScaleOptimization::Trellis { lambda: quant_config.trellis_lambda },
        "standard" | _ => ScaleOptimization::Standard,
    };

    println!("Scale optimization: {:?}", scale_opt);

    let mut builder = GGUFBuilder::new();

    if let Some(arch) = metadata.get("architecture").and_then(|v| v.as_str()) {
        builder = builder.add_metadata("general.architecture", MetadataValue::String(arch.to_string()));
    }
    if let Some(name) = metadata.get("name").and_then(|v| v.as_str()) {
        builder = builder.add_metadata("general.name", MetadataValue::String(name.to_string()));
    }

    let tensors_dir = model_dir.join("tensors");
    println!("\nProcessing {} tensors...", config.tensor_map.len());

    for mapping in &config.tensor_map {
        let quant_type = quant_config.per_tensor
            .get(&mapping.source)
            .map(|s| parse_quant_type(s))
            .transpose()?
            .unwrap_or(default_quant);

        let tensor_path = tensors_dir.join(format!("{}.gmat", mapping.source));
        if !tensor_path.exists() {
            eprintln!("Warning: missing tensor: {}", tensor_path.display());
            continue;
        }

        let matrix = GraphMatrix::load(&tensor_path)?;
        let (rows, cols) = matrix.shape();

        println!("  {} [{}x{}] -> {:?}", mapping.target, rows, cols, quant_type);

        let quant_data = quantize_to_gguf(&matrix, quant_type, scale_opt, None)
            .map_err(|e| anyhow::anyhow!("Quantize failed for {}: {}", mapping.target, e))?;

        builder = builder.add_quantized_tensor(
            mapping.target.clone(),
            vec![cols as u64, rows as u64],
            quant_type_to_gguf_tensor_type(&quant_data.quant_type),
            quant_data.data,
        );
    }

    println!("\nWriting: {}", output_file);
    let result = builder.build_to_file(output_file)
        .map_err(|e| anyhow::anyhow!("Failed to write GGUF: {}", e))?;

    println!("Done! {} tensors, {} bytes", result.tensor_results.len(), result.total_bytes_written);
    Ok(())
}

/// Parse quantization type string to GgufQuantType enum.
fn parse_quant_type(s: &str) -> Result<transform_storage::conversions::gguf_quant::GgufQuantType> {
    use transform_storage::conversions::gguf_quant::GgufQuantType;

    match s.to_lowercase().as_str() {
        "q8_0" => Ok(GgufQuantType::Q8_0),
        "q4_0" => Ok(GgufQuantType::Q4_0),
        "q4_1" => Ok(GgufQuantType::Q4_1),
        "q5_0" => Ok(GgufQuantType::Q5_0),
        "q5_1" => Ok(GgufQuantType::Q5_1),
        "q2_k" => Ok(GgufQuantType::Q2_K),
        "q3_k_s" => Ok(GgufQuantType::Q3_K_S),
        "q3_k_m" | "q3_k" => Ok(GgufQuantType::Q3_K_M),
        "q3_k_l" => Ok(GgufQuantType::Q3_K_L),
        "q4_k_s" => Ok(GgufQuantType::Q4_K_S),
        "q4_k_m" | "q4_k" => Ok(GgufQuantType::Q4_K_M),
        "q5_k_s" => Ok(GgufQuantType::Q5_K_S),
        "q5_k_m" | "q5_k" => Ok(GgufQuantType::Q5_K_M),
        "q6_k" => Ok(GgufQuantType::Q6_K),
        "iq4_xs" => Ok(GgufQuantType::IQ4_XS),
        "iq4_nl" => Ok(GgufQuantType::IQ4_NL),
        _ => Err(anyhow::anyhow!("Unknown quantization type: {}", s)),
    }
}

/// Convert GgufQuantType to gguf-rs-lib TensorType.
fn quant_type_to_gguf_tensor_type(quant_type: &transform_storage::conversions::gguf_quant::GgufQuantType) -> gguf_rs_lib::format::GGUFTensorType {
    use transform_storage::conversions::gguf_quant::GgufQuantType;
    use gguf_rs_lib::format::GGUFTensorType;

    match quant_type {
        GgufQuantType::Q4_0 => GGUFTensorType::Q4_0,
        GgufQuantType::Q4_1 => GGUFTensorType::Q4_1,
        GgufQuantType::Q5_0 => GGUFTensorType::Q5_0,
        GgufQuantType::Q5_1 => GGUFTensorType::Q5_1,
        GgufQuantType::Q8_0 => GGUFTensorType::Q8_0,
        GgufQuantType::Q2_K => GGUFTensorType::Q2_K,
        GgufQuantType::Q3_K_S => GGUFTensorType::Q3_K,
        GgufQuantType::Q3_K_M => GGUFTensorType::Q3_K,
        GgufQuantType::Q3_K_L => GGUFTensorType::Q3_K,
        GgufQuantType::Q4_K_S => GGUFTensorType::Q4_K,
        GgufQuantType::Q4_K_M => GGUFTensorType::Q4_K,
        GgufQuantType::Q5_K_S => GGUFTensorType::Q5_K,
        GgufQuantType::Q5_K_M => GGUFTensorType::Q5_K,
        GgufQuantType::Q6_K => GGUFTensorType::Q6_K,
        GgufQuantType::IQ2_XXS => GGUFTensorType::IQ2_XXS,
        GgufQuantType::IQ2_XS => GGUFTensorType::IQ2_XS,
        GgufQuantType::IQ3_XXS => GGUFTensorType::IQ3_XXS,
        GgufQuantType::IQ1_S => GGUFTensorType::IQ1_S,
        GgufQuantType::IQ4_NL => GGUFTensorType::IQ4_NL,
        GgufQuantType::IQ3_S => GGUFTensorType::IQ3_S,
        GgufQuantType::IQ2_S => GGUFTensorType::IQ2_S,
        GgufQuantType::IQ4_XS => GGUFTensorType::IQ4_XS,
        GgufQuantType::IQ1_M => GGUFTensorType::IQ1_M,
    }
}
