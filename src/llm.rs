//! LLM backends Numen can generate from. Claude (the cloud fallback) is always
//! available via the `claude -p` subprocess and needs no state, so only the
//! optional local `mistral.rs` engine lives in `Backend`. The meta-router picks
//! between them per query (see `router::route`).

use std::path::Path;
use std::process::Stdio;

use anyhow::{anyhow, bail, Result};
use mistralrs::{GgufModelBuilder, Model, TextMessageRole, TextMessages};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Upper bound (in characters) on the subprocess stderr embedded in a
/// `claude -p` error. A failing CLI can emit verbose diagnostics; capping the
/// captured prefix keeps the returned error (and the log line built from it)
/// from ballooning.
const MAX_STDERR_LEN: usize = 500;

/// Holds the local inference engine when one is loaded.
pub struct Backend {
    local: Option<Model>,
}

impl Backend {
    /// Load the local GGUF model if `model_path` points to an existing file;
    /// otherwise the backend is cloud-only (Claude). Loading reads tens of GB
    /// into memory and can take a while - it runs once at startup.
    pub async fn load(model_path: Option<&str>) -> Result<Self> {
        let raw = match model_path {
            Some(raw) if Path::new(raw).is_file() => raw,
            Some(raw) => {
                tracing::warn!(
                    path = raw,
                    "NUMEN_MODEL_PATH is not a file, using Claude only"
                );
                return Ok(Self::cloud_only());
            }
            None => {
                tracing::info!("no NUMEN_MODEL_PATH set, using Claude only");
                return Ok(Self::cloud_only());
            }
        };

        let path = Path::new(raw);
        let dir = path
            .parent()
            .ok_or_else(|| anyhow!("model path has no parent"))?;
        let file = path
            .file_name()
            .ok_or_else(|| anyhow!("model path has no file name"))?;
        tracing::info!(model = %path.display(), "loading local GGUF model (may take a while)");
        let model = GgufModelBuilder::new(
            dir.to_string_lossy().into_owned(),
            vec![file.to_string_lossy().into_owned()],
        )
        .with_logging()
        .build()
        .await?;
        tracing::info!("local model loaded");
        Ok(Self { local: Some(model) })
    }

    /// A cloud-only backend with no local model (used in tests and as the
    /// fallback when no model is configured).
    pub fn cloud_only() -> Self {
        Self { local: None }
    }

    /// Whether a local model is loaded. Drives routing - the router only picks
    /// the local route when this is true.
    pub fn has_local(&self) -> bool {
        self.local.is_some()
    }

    /// Generate with the local model. The router guarantees a local model exists
    /// before selecting this path.
    pub async fn generate_local(&self, prompt: &str) -> Result<String> {
        let model = self
            .local
            .as_ref()
            .ok_or_else(|| anyhow!("no local model loaded"))?;
        let messages = TextMessages::new().add_message(TextMessageRole::User, prompt);
        let response = model.send_chat_request(messages).await?;
        response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| anyhow!("local model returned no content"))
    }
}

/// Delegate to Claude via `claude -p`. Always available (stateless subprocess),
/// authenticated headlessly via `CLAUDE_CODE_OAUTH_TOKEN`. Errors if the CLI is
/// missing, unauthenticated, or exits non-zero.
///
/// The prompt is written to the child's stdin rather than passed as an argv
/// argument: the delegation prompt grows with injected graph context and a large
/// one would exceed the OS argv length limit (`ARG_MAX`). `claude -p` with no
/// prompt argument reads it from stdin, equivalent to `echo "..." | claude -p`.
pub async fn claude(prompt: &str) -> Result<String> {
    let mut child = Command::new("claude")
        .arg("-p")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| anyhow!("claude -p child stdin was not piped"))?;
    stdin.write_all(prompt.as_bytes()).await?;
    stdin.shutdown().await?;
    drop(stdin);
    let output = child.wait_with_output().await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let trimmed = stderr.trim();
        let bounded: String = trimmed.chars().take(MAX_STDERR_LEN).collect();
        bail!("claude -p exited with {}: {}", output.status, bounded);
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}
