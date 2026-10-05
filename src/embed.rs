//! Vector memory: embed concept labels with a local BERT sentence-transformer
//! (candle) so the graph can be recalled by meaning, not only by exact labels.
//! Embeddings are L2-normalized, so cosine similarity is a plain dot product.
//! With no model directory the embedder is `None` and the system runs unchanged.

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use rusqlite::Connection;
use tokenizers::Tokenizer;

use crate::db;

/// A loaded embedding model: a BERT encoder plus its tokenizer, on CPU.
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    /// Load from `dir` (expecting `config.json`, `tokenizer.json`, and
    /// `model.safetensors`). `Ok(None)` when `dir` is unset or holds no model, so
    /// vector memory is absent rather than a hard dependency, mirroring the
    /// optional LLM backend (a configured-but-missing path is not a fatal error).
    pub fn load(dir: Option<&str>) -> Result<Option<Self>> {
        let Some(dir) = dir else {
            return Ok(None);
        };
        let dir = Path::new(dir);
        if !dir.join("config.json").exists() {
            return Ok(None);
        }
        let config: Config = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).context("read embedding config.json")?,
        )?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let device = Device::Cpu;
        let tensors = candle_core::safetensors::load(dir.join("model.safetensors"), &device)
            .context("load embedding weights")?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = BertModel::load(vb, &config)?;
        Ok(Some(Self {
            model,
            tokenizer,
            device,
        }))
    }

    /// Embed one text into an L2-normalized vector. A single text carries no
    /// padding, so mean-pooling over every token position needs no attention mask.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize: {e}"))?;
        let ids = Tensor::new(encoding.get_ids(), &self.device)?.unsqueeze(0)?;
        let type_ids = ids.zeros_like()?;
        let hidden = self.model.forward(&ids, &type_ids, None)?;
        let tokens = hidden.dim(1)? as f64;
        let pooled = hidden.sum(1)?.affine(1.0 / tokens, 0.0)?;
        let norm = pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = pooled.broadcast_div(&norm)?;
        Ok(normalized.squeeze(0)?.to_vec1::<f32>()?)
    }
}

/// Embed every node that has no embedding yet, in one transaction. Returns how
/// many were embedded. Cheap to call repeatedly: embedded nodes are skipped.
pub fn backfill(embedder: &Embedder, conn: &mut Connection) -> Result<usize> {
    let labels = db::unembedded_labels(conn)?;
    if labels.is_empty() {
        return Ok(0);
    }
    let embedded: Vec<(String, Vec<f32>)> = labels
        .into_iter()
        .map(|label| {
            let vector = embedder.embed(&label)?;
            Ok((label, vector))
        })
        .collect::<Result<_>>()?;
    let count = embedded.len();
    let tx = conn.transaction()?;
    for (label, vector) in &embedded {
        db::set_node_embedding(&tx, label, vector)?;
    }
    tx.commit()?;
    Ok(count)
}

/// Cosine similarity of two L2-normalized vectors, i.e. their dot product.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The `k` nodes whose embeddings are nearest `query` by cosine, above
/// `threshold`, sorted most-similar first. The semantic-recall primitive: it
/// surfaces concepts related by meaning even when no label token matches.
pub fn semantic_neighbors(
    query: &[f32],
    nodes: &[(String, Vec<f32>)],
    k: usize,
    threshold: f32,
) -> Vec<(String, f32)> {
    let mut scored: Vec<(String, f32)> = nodes
        .iter()
        .map(|(label, vector)| (label.clone(), cosine(query, vector)))
        .filter(|(_, score)| *score >= threshold)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_neighbors_rank_by_cosine_above_threshold() {
        let query = [1.0, 0.0];
        let nodes = vec![
            ("near".to_string(), vec![0.99, 0.14]),
            ("mid".to_string(), vec![0.7, 0.71]),
            ("far".to_string(), vec![0.0, 1.0]),
        ];
        let neighbors = semantic_neighbors(&query, &nodes, 5, 0.5);
        let labels: Vec<&str> = neighbors.iter().map(|(label, _)| label.as_str()).collect();
        assert_eq!(labels, ["near", "mid"]);
    }

    #[test]
    fn semantic_neighbors_cap_at_k() {
        let query = [1.0, 0.0];
        let nodes = vec![
            ("a".to_string(), vec![1.0, 0.0]),
            ("b".to_string(), vec![0.99, 0.1]),
            ("c".to_string(), vec![0.98, 0.2]),
        ];
        assert_eq!(semantic_neighbors(&query, &nodes, 2, 0.0).len(), 2);
    }
}
