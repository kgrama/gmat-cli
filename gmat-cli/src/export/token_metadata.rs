//! Token tree to GGUF tokenizer metadata conversion.
//!
//! Loads TokenIdTree from tokens.bin and converts to GGUF metadata format.

use anyhow::{Context, Result};
use gguf_rs_lib::format::{GGUFValueType, MetadataArray, MetadataValue};
use std::collections::HashMap;
use std::path::Path;

use crate::config::export_config::default_special_token_keys;
use crate::tokens::{load_token_tree, TokenEntry, TokenIdTree, TokenizerType};

/// GGUF token type constants (matches llama.cpp LLAMA_TOKEN_TYPE_*).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTokenType {
    /// Normal vocabulary token.
    Normal = 1,
    /// Unknown token.
    Unknown = 2,
    /// Control/special token (BOS, EOS, etc.).
    Control = 3,
    /// Byte-level token (for byte fallback).
    Byte = 6,
}

impl GgufTokenType {
    /// Determine token type from TokenEntry.
    fn from_entry(entry: &TokenEntry) -> Self {
        if entry.special {
            match entry.special_type.as_str() {
                "unk" | "unk_token" => GgufTokenType::Unknown,
                _ => GgufTokenType::Control,
            }
        } else if entry.token.starts_with("<0x") && entry.token.ends_with('>') {
            // Byte tokens like <0x00>, <0x0A>
            GgufTokenType::Byte
        } else {
            GgufTokenType::Normal
        }
    }
}

/// Tokenizer metadata for GGUF export.
pub struct TokenMetadata {
    /// Tokenizer model type (bpe, unigram, etc.).
    pub model: String,
    /// Token strings sorted by embedding row.
    pub tokens: Vec<String>,
    /// Token scores (for Unigram, else 0.0).
    pub scores: Vec<f32>,
    /// Token types (Normal, Control, etc.).
    pub token_types: Vec<u32>,
    /// Special token IDs mapped by their type (e.g., "bos" -> row).
    pub special_token_ids: HashMap<String, u32>,
}

impl TokenMetadata {
    /// Load token metadata from a model directory containing tokens.bin.
    pub async fn load(model_dir: &Path) -> Result<Self> {
        let tree = load_token_tree(model_dir)
            .await
            .context("Failed to load tokens.bin")?;

        Ok(Self::from_tree(&tree))
    }

    /// Convert a TokenIdTree to TokenMetadata.
    pub fn from_tree(tree: &TokenIdTree) -> Self {
        let vocab_size = tree.vocab_size();

        // Collect entries sorted by embedding_row
        let mut sorted: Vec<&TokenEntry> = tree.collect_entries();
        sorted.sort_by_key(|e| e.embedding_row);

        // Build parallel arrays
        let mut tokens = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut token_types = Vec::with_capacity(vocab_size);

        for entry in &sorted {
            tokens.push(entry.token.clone());
            scores.push(entry.score.unwrap_or(0.0));
            token_types.push(GgufTokenType::from_entry(entry) as u32);
        }

        // Extract special token IDs from the tree
        let mut special_token_ids = HashMap::new();
        for entry in tree.special_tokens() {
            if !entry.special_type.is_empty() {
                // Normalize: strip "_token" suffix if present
                let key = entry.special_type.trim_end_matches("_token").to_string();
                special_token_ids.insert(key, entry.embedding_row);
            }
        }

        // Map tokenizer type to GGUF model name
        let model = match tree.tokenizer_type {
            TokenizerType::Bpe => "gpt2",
            TokenizerType::Unigram => "llama",
            TokenizerType::WordPiece => "bert",
            TokenizerType::WordLevel => "gpt2",
            TokenizerType::Tiktoken => "gpt2",
        };

        Self {
            model: model.to_string(),
            tokens,
            scores,
            token_types,
            special_token_ids,
        }
    }

    /// Add tokenizer metadata to a GGUF builder.
    ///
    /// Uses default special token key mappings. For custom mappings, use `to_gguf_metadata_with_keys`.
    pub fn to_gguf_metadata(&self) -> Result<Vec<(String, MetadataValue)>> {
        self.to_gguf_metadata_with_keys(&HashMap::new())
    }

    /// Add tokenizer metadata to a GGUF builder with custom special token key mappings.
    ///
    /// The `overrides` map is merged with defaults - use empty string value to disable a mapping.
    pub fn to_gguf_metadata_with_keys(
        &self,
        overrides: &HashMap<String, String>,
    ) -> Result<Vec<(String, MetadataValue)>> {
        let mut metadata = Vec::new();

        // Tokenizer model type
        metadata.push((
            "tokenizer.ggml.model".to_string(),
            MetadataValue::String(self.model.clone()),
        ));

        // Token list (string array)
        let token_values: Vec<MetadataValue> = self
            .tokens
            .iter()
            .map(|t| MetadataValue::String(t.clone()))
            .collect();
        let tokens_array =
            MetadataArray::new(GGUFValueType::String, token_values).context("tokens array")?;
        metadata.push((
            "tokenizer.ggml.tokens".to_string(),
            MetadataValue::Array(Box::new(tokens_array)),
        ));

        // Scores (float array)
        let score_values: Vec<MetadataValue> =
            self.scores.iter().map(|&s| MetadataValue::F32(s)).collect();
        let scores_array =
            MetadataArray::new(GGUFValueType::F32, score_values).context("scores array")?;
        metadata.push((
            "tokenizer.ggml.scores".to_string(),
            MetadataValue::Array(Box::new(scores_array)),
        ));

        // Token types (u32 array)
        let type_values: Vec<MetadataValue> = self
            .token_types
            .iter()
            .map(|&t| MetadataValue::U32(t))
            .collect();
        let types_array =
            MetadataArray::new(GGUFValueType::U32, type_values).context("token_type array")?;
        metadata.push((
            "tokenizer.ggml.token_type".to_string(),
            MetadataValue::Array(Box::new(types_array)),
        ));

        // Build effective key mapping: defaults + overrides
        let mut key_map = default_special_token_keys();
        for (k, v) in overrides {
            key_map.insert(k.clone(), v.clone());
        }

        // Add special token IDs using the key mapping
        for (token_type, id) in &self.special_token_ids {
            if let Some(gguf_key) = key_map.get(token_type).filter(|k| !k.is_empty()) {
                metadata.push((gguf_key.clone(), MetadataValue::U32(*id)));
            }
        }

        Ok(metadata)
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::{SourceFormat, TokenizerType};

    fn make_test_tree() -> TokenIdTree {
        // Build a simple test tree using right field directly
        let world = TokenEntry::new(2u32, "world".to_string(), 2);
        let mut hello = TokenEntry::new(1u32, "hello".to_string(), 1);
        hello.right = Some(Box::new(world));
        let mut bos = TokenEntry::special(0u32, "<s>".to_string(), 0, "bos");
        bos.right = Some(Box::new(hello));

        TokenIdTree::new(SourceFormat::HuggingFace, TokenizerType::Bpe, bos)
    }

    #[test]
    fn test_from_tree_basic() {
        let tree = make_test_tree();
        let meta = TokenMetadata::from_tree(&tree);

        assert_eq!(meta.vocab_size(), 3);
        assert_eq!(meta.model, "gpt2");
        assert_eq!(meta.tokens, vec!["<s>", "hello", "world"]);
        assert_eq!(meta.special_token_ids.get("bos"), Some(&0));
    }

    #[test]
    fn test_token_types() {
        let tree = make_test_tree();
        let meta = TokenMetadata::from_tree(&tree);

        // First token is special (Control), others are Normal
        assert_eq!(meta.token_types[0], GgufTokenType::Control as u32);
        assert_eq!(meta.token_types[1], GgufTokenType::Normal as u32);
        assert_eq!(meta.token_types[2], GgufTokenType::Normal as u32);
    }

    #[test]
    fn test_byte_token_detection() {
        let byte_token = TokenEntry::new(0u32, "<0x0A>".to_string(), 0);
        assert_eq!(
            GgufTokenType::from_entry(&byte_token),
            GgufTokenType::Byte
        );

        let normal = TokenEntry::new(1u32, "hello".to_string(), 1);
        assert_eq!(GgufTokenType::from_entry(&normal), GgufTokenType::Normal);
    }

    #[test]
    fn test_unk_token_type() {
        let unk = TokenEntry::special(0u32, "<unk>".to_string(), 0, "unk");
        assert_eq!(GgufTokenType::from_entry(&unk), GgufTokenType::Unknown);
    }

    #[test]
    fn test_to_gguf_metadata() {
        let tree = make_test_tree();
        let meta = TokenMetadata::from_tree(&tree);
        let gguf_meta = meta.to_gguf_metadata().unwrap();

        // Check we have the expected keys
        let keys: Vec<&str> = gguf_meta.iter().map(|(k, _)| k.as_str()).collect();
        assert!(keys.contains(&"tokenizer.ggml.model"));
        assert!(keys.contains(&"tokenizer.ggml.tokens"));
        assert!(keys.contains(&"tokenizer.ggml.scores"));
        assert!(keys.contains(&"tokenizer.ggml.token_type"));
        assert!(keys.contains(&"tokenizer.ggml.bos_token_id"));
    }

    #[test]
    fn test_unigram_model_type() {
        let entry = TokenEntry::new(0u32, "test".to_string(), 0);
        let tree = TokenIdTree::new(SourceFormat::HuggingFace, TokenizerType::Unigram, entry);
        let meta = TokenMetadata::from_tree(&tree);

        assert_eq!(meta.model, "llama");
    }
}
