//! Tokenizer format parsers.
//!
//! Parses various tokenizer formats into the unified TokenIdTree structure.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use crate::tokens::{SourceFormat, SpecialTokenMapping, TokenEntry, TokenIdTree, TokenizerType};

/// Parse a HuggingFace tokenizer.json file.
pub fn parse_hf_tokenizer(path: &Path, special_mapping: &SpecialTokenMapping) -> Result<TokenIdTree> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read tokenizer.json: {}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&content).context("Failed to parse tokenizer.json")?;

    let tokenizer_type = detect_hf_tokenizer_type(&json)?;
    let root = parse_hf_vocab(&json, tokenizer_type, special_mapping)?;

    Ok(TokenIdTree::new(SourceFormat::HuggingFace, tokenizer_type, root))
}

/// Parse a HuggingFace tokenizer_config.json for special tokens.
///
/// Dynamically extracts all fields that contain a token string.
pub fn parse_hf_tokenizer_config(path: &Path) -> Result<ParsedSpecialTokens> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read tokenizer_config.json: {}", path.display()))?;
    let json: serde_json::Value =
        serde_json::from_str(&content).context("Failed to parse tokenizer_config.json")?;

    let mut tokens = std::collections::HashMap::new();

    if let Some(obj) = json.as_object() {
        for (key, _) in obj {
            if let Some(token_str) = extract_token_string(&json, key) {
                tokens.insert(key.clone(), token_str);
            }
        }
    }

    Ok(ParsedSpecialTokens { tokens })
}

/// Parse a tiktoken .tiktoken file (base64-encoded mergeable ranks).
pub fn parse_tiktoken(path: &Path) -> Result<TokenIdTree> {
    use base64::{engine::general_purpose::STANDARD, Engine};

    let content = fs::read_to_string(path)
        .with_context(|| format!("Failed to read tiktoken file: {}", path.display()))?;

    let root = build_tree(content.lines().filter_map(|line| {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }
        let mut parts = line.split_whitespace();
        let b64 = parts.next()?;
        let rank: u32 = parts.next()?.parse().ok()?;
        let token_bytes = STANDARD.decode(b64).ok()?;
        let token = String::from_utf8_lossy(&token_bytes).to_string();
        Some(TokenEntry::new(rank, token, rank))
    }))?;

    Ok(TokenIdTree::new(
        SourceFormat::Tiktoken,
        TokenizerType::Tiktoken,
        root,
    ))
}

/// Special tokens configuration extracted from tokenizer_config.json.
///
/// Stores field_name → token_string mappings dynamically,
/// capturing any field ending in `_token`.
#[derive(Debug, Default)]
pub struct ParsedSpecialTokens {
    /// Map from field name (e.g., "bos_token") → token string (e.g., "<s>")
    pub tokens: std::collections::HashMap<String, String>,
}

impl ParsedSpecialTokens {
    /// Build a SpecialTokenMapping (token_string → field_name) for lookup.
    pub fn to_mapping(&self) -> SpecialTokenMapping {
        let mut mapping = SpecialTokenMapping::new();
        for (field_name, token_string) in &self.tokens {
            mapping.insert(token_string.clone(), field_name.clone());
        }
        mapping
    }
}

/// Detect the tokenizer type from HuggingFace tokenizer.json.
fn detect_hf_tokenizer_type(json: &serde_json::Value) -> Result<TokenizerType> {
    let model = json.get("model").context("Missing 'model' field in tokenizer.json")?;
    let model_type = model
        .get("type")
        .and_then(|v| v.as_str())
        .context("Missing 'model.type' field")?;

    match model_type {
        "BPE" => Ok(TokenizerType::Bpe),
        "Unigram" => Ok(TokenizerType::Unigram),
        "WordPiece" => Ok(TokenizerType::WordPiece),
        "WordLevel" => Ok(TokenizerType::WordLevel),
        other => anyhow::bail!("Unknown tokenizer model type: {}", other),
    }
}

/// Parse vocabulary from HuggingFace tokenizer.json.
fn parse_hf_vocab(
    json: &serde_json::Value,
    tokenizer_type: TokenizerType,
    special_mapping: &SpecialTokenMapping,
) -> Result<TokenEntry> {
    let model = json.get("model").context("Missing 'model' field")?;

    match tokenizer_type {
        TokenizerType::Bpe => parse_bpe_vocab(model, json, special_mapping),
        TokenizerType::Unigram => parse_unigram_vocab(model, special_mapping),
        TokenizerType::WordPiece | TokenizerType::WordLevel => {
            parse_simple_vocab(model, special_mapping)
        }
        TokenizerType::Tiktoken => anyhow::bail!("Use parse_tiktoken for tiktoken files"),
    }
}

/// Build a BST from an iterator, with id 0 as root.
fn build_tree<I>(mut iter: I) -> Result<TokenEntry>
where
    I: Iterator<Item = TokenEntry>,
{
    let mut root = iter.next().context("Empty vocabulary")?;
    for entry in iter {
        insert_bst(&mut root, entry);
    }
    Ok(reroot_at_zero(root))
}

/// Insert entry into BST by id.
fn insert_bst(node: &mut TokenEntry, entry: TokenEntry) {
    if entry.id.as_u32() < node.id.as_u32() {
        match &mut node.left {
            Some(left) => insert_bst(left, entry),
            None => node.left = Some(Box::new(entry)),
        }
    } else {
        match &mut node.right {
            Some(right) => insert_bst(right, entry),
            None => node.right = Some(Box::new(entry)),
        }
    }
}

/// Reroot tree so id 0 becomes the root via rotations.
fn reroot_at_zero(mut root: TokenEntry) -> TokenEntry {
    // Keep rotating right until root is id 0
    while root.id.as_u32() != 0 {
        root = rotate_right(root);
    }
    root
}

/// Rotate right: left child becomes new root.
fn rotate_right(mut node: TokenEntry) -> TokenEntry {
    let mut new_root = *node.left.take().expect("rotate_right requires left child");
    node.left = new_root.right.take();
    new_root.right = Some(Box::new(node));
    new_root
}

/// Create a TokenEntry, marking it as special if it's in the mapping.
#[inline]
fn make_entry(id: u32, token: &str, special_mapping: &SpecialTokenMapping) -> TokenEntry {
    match special_mapping.get_type(token) {
        Some(special_type) => TokenEntry::special(id, token.to_string(), id, special_type.to_string()),
        None => TokenEntry::new(id, token.to_string(), id),
    }
}

/// Parse BPE vocabulary with merge information.
fn parse_bpe_vocab(
    model: &serde_json::Value,
    _json: &serde_json::Value,
    special_mapping: &SpecialTokenMapping,
) -> Result<TokenEntry> {
    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .context("Missing 'model.vocab' object")?;

    build_tree(vocab.iter().filter_map(|(token, id)| {
        id.as_u64().map(|id| make_entry(id as u32, token, special_mapping))
    }))
}

/// Parse Unigram vocabulary with scores.
fn parse_unigram_vocab(
    model: &serde_json::Value,
    special_mapping: &SpecialTokenMapping,
) -> Result<TokenEntry> {
    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_array())
        .context("Missing 'model.vocab' array for Unigram")?;

    build_tree(vocab.iter().enumerate().filter_map(|(idx, item)| {
        let arr = item.as_array()?;
        let token = arr.get(0)?.as_str()?;
        let score = arr.get(1)?.as_f64()? as f32;
        Some(make_entry(idx as u32, token, special_mapping).with_score(score))
    }))
}

/// Parse simple vocab (WordPiece, WordLevel).
fn parse_simple_vocab(
    model: &serde_json::Value,
    special_mapping: &SpecialTokenMapping,
) -> Result<TokenEntry> {
    let vocab = model
        .get("vocab")
        .and_then(|v| v.as_object())
        .context("Missing 'model.vocab' object")?;

    build_tree(vocab.iter().filter_map(|(token, id)| {
        id.as_u64().map(|id| make_entry(id as u32, token, special_mapping))
    }))
}

/// Extract a token string from JSON (handles both string and object formats).
fn extract_token_string(json: &serde_json::Value, key: &str) -> Option<String> {
    json.get(key).and_then(|v| {
        if let Some(s) = v.as_str() {
            Some(s.to_string())
        } else if let Some(obj) = v.as_object() {
            obj.get("content").and_then(|c| c.as_str()).map(String::from)
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_bpe_type() {
        let json: serde_json::Value = serde_json::json!({
            "model": { "type": "BPE", "vocab": {} }
        });
        assert_eq!(detect_hf_tokenizer_type(&json).unwrap(), TokenizerType::Bpe);
    }

    #[test]
    fn test_detect_unigram_type() {
        let json: serde_json::Value = serde_json::json!({
            "model": { "type": "Unigram", "vocab": [] }
        });
        assert_eq!(detect_hf_tokenizer_type(&json).unwrap(), TokenizerType::Unigram);
    }

    #[test]
    fn test_extract_token_string_simple() {
        let json: serde_json::Value = serde_json::json!({
            "bos_token": "<s>"
        });
        assert_eq!(extract_token_string(&json, "bos_token"), Some("<s>".to_string()));
    }

    #[test]
    fn test_extract_token_string_object() {
        let json: serde_json::Value = serde_json::json!({
            "bos_token": { "content": "<|begin|>", "lstrip": false }
        });
        assert_eq!(extract_token_string(&json, "bos_token"), Some("<|begin|>".to_string()));
    }

    #[test]
    fn test_parse_simple_vocab() {
        let model: serde_json::Value = serde_json::json!({
            "vocab": {
                "hello": 0,
                "world": 1,
                "test": 2
            }
        });
        let mapping = SpecialTokenMapping::new();
        let root = parse_simple_vocab(&model, &mapping).unwrap();
        assert_eq!(root.token, "hello");
        assert_eq!(root.embedding_row, 0);
    }

    #[test]
    fn test_parse_simple_vocab_with_special_tokens() {
        let model: serde_json::Value = serde_json::json!({
            "vocab": {
                "<s>": 0,
                "hello": 1,
                "</s>": 2
            }
        });
        let mut mapping = SpecialTokenMapping::new();
        mapping.insert("<s>".to_string(), "bos_token".to_string());
        mapping.insert("</s>".to_string(), "eos_token".to_string());

        let root = parse_simple_vocab(&model, &mapping).unwrap();

        // First token should be <s> with special_type = "bos_token"
        assert_eq!(root.token, "<s>");
        assert!(root.special);
        assert_eq!(root.special_type, "bos_token");

        // Traverse to find </s>
        let mut current = &root;
        while let Some(ref next) = current.right {
            current = next;
        }
        assert_eq!(current.token, "</s>");
        assert!(current.special);
        assert_eq!(current.special_type, "eos_token");
    }

    #[test]
    fn test_parsed_special_tokens_to_mapping() {
        let mut tokens = std::collections::HashMap::new();
        tokens.insert("bos_token".to_string(), "<s>".to_string());
        tokens.insert("eos_token".to_string(), "</s>".to_string());
        tokens.insert("pad_token".to_string(), "<pad>".to_string());

        let parsed = ParsedSpecialTokens { tokens };
        let mapping = parsed.to_mapping();

        // Inverse mapping: token_string -> field_name
        assert_eq!(mapping.get_type("<s>"), Some("bos_token"));
        assert_eq!(mapping.get_type("</s>"), Some("eos_token"));
        assert_eq!(mapping.get_type("<pad>"), Some("pad_token"));
        assert_eq!(mapping.get_type("<unk>"), None);
    }

    #[test]
    fn test_parse_bpe_vocab_with_special_tokens() {
        let model: serde_json::Value = serde_json::json!({
            "vocab": {
                "<s>": 0,
                "Ġhello": 1,
                "Ġworld": 2,
                "</s>": 3
            }
        });
        let json: serde_json::Value = serde_json::json!({
            "model": model
        });

        let mut mapping = SpecialTokenMapping::new();
        mapping.insert("<s>".to_string(), "bos_token".to_string());
        mapping.insert("</s>".to_string(), "eos_token".to_string());

        let root = parse_bpe_vocab(&model, &json, &mapping).unwrap();

        assert_eq!(root.token, "<s>");
        assert!(root.special);
        assert_eq!(root.special_type, "bos_token");
    }

    #[test]
    fn test_parse_unigram_vocab_with_special_tokens() {
        let model: serde_json::Value = serde_json::json!({
            "vocab": [
                ["<s>", 0.0],
                ["hello", -1.5],
                ["</s>", 0.0]
            ]
        });

        let mut mapping = SpecialTokenMapping::new();
        mapping.insert("<s>".to_string(), "bos_token".to_string());
        mapping.insert("</s>".to_string(), "eos_token".to_string());

        let root = parse_unigram_vocab(&model, &mapping).unwrap();

        assert_eq!(root.token, "<s>");
        assert!(root.special);
        assert_eq!(root.special_type, "bos_token");
        assert_eq!(root.score, Some(0.0));
    }
}
