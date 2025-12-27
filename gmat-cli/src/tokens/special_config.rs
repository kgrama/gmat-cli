//! Special tokens configuration.
//!
//! Provides utilities for mapping token strings to their semantic roles.

use std::collections::HashMap;

/// Mapping from token strings to their special type.
///
/// Built from tokenizer_config.json fields like bos_token, eos_token, etc.
#[derive(Debug, Default)]
pub struct SpecialTokenMapping {
    /// Map from token string → special_type (field name from tokenizer_config.json)
    token_to_type: HashMap<String, String>,
}

impl SpecialTokenMapping {
    /// Create a new empty mapping.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the special type for a token string, if known.
    pub fn get_type(&self, token: &str) -> Option<&str> {
        self.token_to_type.get(token).map(|s| s.as_str())
    }

    /// Add a mapping from token string to special type.
    pub fn insert(&mut self, token: String, special_type: String) {
        self.token_to_type.insert(token, special_type);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_token_mapping() {
        let mut mapping = SpecialTokenMapping::new();
        mapping.insert("<s>".to_string(), "bos_token".to_string());
        mapping.insert("</s>".to_string(), "eos_token".to_string());

        assert_eq!(mapping.get_type("<s>"), Some("bos_token"));
        assert_eq!(mapping.get_type("</s>"), Some("eos_token"));
        assert_eq!(mapping.get_type("<unk>"), None);
    }
}
