//! Causal extraction (seed-on-delegation). A delegated answer ends with a
//! machine-readable ```` ```numen-causal ```` block holding the causal relations
//! Claude asserts. We parse that block deterministically, strip it from the
//! prose shown to the user, and hand the relations to the persistence layer.

use serde::Deserialize;

use crate::graph::Relation;

/// The split of a delegation answer: human-facing prose (block removed) and the
/// relations to seed into the graph.
pub struct Extraction {
    pub prose: String,
    pub relations: Vec<Relation>,
}

#[derive(Deserialize)]
struct RawRelation {
    cause: String,
    effect: String,
    #[serde(default = "default_kind")]
    relation: String,
    #[serde(default = "default_strength")]
    strength: f64,
}

fn default_kind() -> String {
    "causes".to_string()
}

fn default_strength() -> f64 {
    0.5
}

const FENCE: &str = "```numen-causal";

/// Max length of a canonicalized concept label; longer ones are dropped to bound
/// storage, prompt bloat, and any payload smuggled through the model's output.
const MAX_LABEL_LEN: usize = 64;

/// The only relation kinds we store. An off-list kind is dropped, not defaulted,
/// so it cannot silently bypass the polarity contradiction screen.
const KNOWN_KINDS: [&str; 5] = ["causes", "enables", "prevents", "requires", "part_of"];

/// Parse the causal block out of `answer`. A missing or malformed block yields
/// no relations and the full answer as prose, so delegation never fails on a
/// non-compliant response.
pub fn extract(answer: &str) -> Extraction {
    let Some(start) = answer.find(FENCE) else {
        return no_relations(answer);
    };
    let after = &answer[start + FENCE.len()..];
    let Some(rel_end) = after.find("```") else {
        return no_relations(answer);
    };
    let relations = parse(after[..rel_end].trim());

    let block_end = start + FENCE.len() + rel_end + "```".len();
    let prose = format!("{}{}", &answer[..start], &answer[block_end..]);
    Extraction {
        prose: prose.trim().to_string(),
        relations,
    }
}

fn no_relations(answer: &str) -> Extraction {
    Extraction {
        prose: answer.trim().to_string(),
        relations: Vec::new(),
    }
}

fn parse(json: &str) -> Vec<Relation> {
    let raw: Vec<RawRelation> = match serde_json::from_str(json) {
        Ok(parsed) => parsed,
        Err(error) => {
            tracing::warn!(%error, "numen-causal block is not valid JSON; no relations extracted");
            return Vec::new();
        }
    };
    raw.into_iter()
        .map(|r| Relation {
            cause: canonical_label(&r.cause),
            effect: canonical_label(&r.effect),
            kind: r.relation.trim().to_lowercase(),
            strength: r.strength.clamp(0.0, 1.0),
        })
        .filter(|r| {
            !r.cause.is_empty()
                && !r.effect.is_empty()
                && r.cause != r.effect
                && r.cause.len() <= MAX_LABEL_LEN
                && r.effect.len() <= MAX_LABEL_LEN
                && KNOWN_KINDS.contains(&r.kind.as_str())
        })
        .collect()
}

/// Canonicalize a concept label so trivial variants collapse to one node:
/// lowercase, whitespace-collapsed, surrounding punctuation and a leading
/// article dropped, and the last token de-pluralized by a guarded heuristic
/// (`data races` -> `data race`). The heuristic may over-fold rare words like
/// `physics`; a real lemmatizer is a Phase 1 concern. Exposed so query inputs
/// (e.g. `/explain`) are matched against stored labels under the same rule.
pub fn canonical_label(label: &str) -> String {
    let lowered = label.trim().to_lowercase();
    let collapsed = lowered.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim_matches(|c: char| !c.is_alphanumeric());
    let dearticled = ["the ", "a ", "an "]
        .iter()
        .find_map(|article| trimmed.strip_prefix(article))
        .unwrap_or(trimmed);
    singularize_last(dearticled)
}

/// De-pluralize only the final token, which carries the head noun.
fn singularize_last(label: &str) -> String {
    let mut tokens: Vec<&str> = label.split(' ').collect();
    if let Some(&last) = tokens.last() {
        let idx = tokens.len() - 1;
        tokens[idx] = fold_plural(last);
    }
    tokens.join(" ")
}

/// Drop a trailing `s` when it is plausibly a plural, guarding common
/// singular-looking endings (`process`, `status`, `axis`, `chaos`).
fn fold_plural(word: &str) -> &str {
    let protected = word.ends_with("ss")
        || word.ends_with("us")
        || word.ends_with("is")
        || word.ends_with("os");
    if word.len() > 4 && word.ends_with('s') && !protected {
        &word[..word.len() - 1]
    } else {
        word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANSWER: &str = "Ownership drives lifetimes.\n\n\
```numen-causal\n\
[{\"cause\":\"Ownership\",\"effect\":\"Lifetime\",\"relation\":\"Causes\",\"strength\":0.9},\n\
 {\"cause\":\"borrow\",\"effect\":\"lifetime\",\"relation\":\"requires\",\"strength\":0.8}]\n\
```";

    #[test]
    fn extracts_relations_and_strips_block() {
        let e = extract(ANSWER);
        assert_eq!(e.prose, "Ownership drives lifetimes.");
        assert_eq!(e.relations.len(), 2);
        assert_eq!(e.relations[0].cause, "ownership");
        assert_eq!(e.relations[0].effect, "lifetime");
        assert_eq!(e.relations[0].kind, "causes");
        assert!((e.relations[0].strength - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn no_block_returns_full_prose() {
        let e = extract("Just an answer, no block.");
        assert_eq!(e.prose, "Just an answer, no block.");
        assert!(e.relations.is_empty());
    }

    #[test]
    fn malformed_json_yields_no_relations() {
        let e = extract("text\n```numen-causal\nnot json at all\n```");
        assert_eq!(e.prose, "text");
        assert!(e.relations.is_empty());
    }

    #[test]
    fn normalize_canonicalizes_variants() {
        assert_eq!(canonical_label("  The  Ownership "), "ownership");
        assert_eq!(canonical_label("data races"), "data race");
        assert_eq!(canonical_label("Lifetimes"), "lifetime");
        assert_eq!(canonical_label("a borrow checker"), "borrow checker");
    }

    #[test]
    fn normalize_guards_singular_looking_words() {
        assert_eq!(canonical_label("process"), "process");
        assert_eq!(canonical_label("status"), "status");
        assert_eq!(canonical_label("axis"), "axis");
        assert_eq!(canonical_label("gas"), "gas");
    }

    #[test]
    fn drops_overlong_labels_and_off_enum_kinds() {
        let long = "a".repeat(80);
        let e = extract(&format!(
            "x\n```numen-causal\n[{{\"cause\":\"{long}\",\"effect\":\"b\",\"relation\":\"causes\"}},\
             {{\"cause\":\"a\",\"effect\":\"b\",\"relation\":\"frobnicates\"}}]\n```"
        ));
        assert!(e.relations.is_empty());
    }

    #[test]
    fn self_loops_and_blanks_are_dropped() {
        let e = extract(
            "x\n```numen-causal\n[{\"cause\":\"a\",\"effect\":\"a\"},{\"cause\":\"\",\"effect\":\"b\"}]\n```",
        );
        assert!(e.relations.is_empty());
    }
}
