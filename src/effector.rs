//! Effector: proactive outbound notifications. Numen pushes what it discovers on
//! its own to the operator over Telegram, so the autonomous Active Inference loop is not
//! silent. Configured via `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`; with either
//! unset it is a no-op (the agent still explores, it just does not notify).

use anyhow::{bail, Result};

/// A configured Telegram push target.
#[derive(Clone)]
pub struct Telegram {
    token: String,
    chat_id: String,
    client: reqwest::Client,
}

impl Telegram {
    /// Build from `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, reusing the shared
    /// HTTP client. `None` if either variable is unset or empty.
    pub fn from_env(client: reqwest::Client) -> Option<Self> {
        Self::new(
            std::env::var("TELEGRAM_BOT_TOKEN").ok()?,
            std::env::var("TELEGRAM_CHAT_ID").ok()?,
            client,
        )
    }

    fn new(token: String, chat_id: String, client: reqwest::Client) -> Option<Self> {
        if token.is_empty() || chat_id.is_empty() {
            return None;
        }
        Some(Self {
            token,
            chat_id,
            client,
        })
    }

    /// Send a plain-text message to the configured chat.
    pub async fn send(&self, text: &str) -> Result<()> {
        let url = format!("https://api.telegram.org/bot{}/sendMessage", self.token);
        let response = self
            .client
            .post(url)
            .json(&serde_json::json!({ "chat_id": self.chat_id, "text": text }))
            .send()
            .await?;
        if !response.status().is_success() {
            bail!("telegram sendMessage returned {}", response.status());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telegram_requires_both_token_and_chat_id() {
        let client = reqwest::Client::new();
        assert!(Telegram::new(String::new(), "chat".to_string(), client.clone()).is_none());
        assert!(Telegram::new("token".to_string(), String::new(), client.clone()).is_none());
        assert!(Telegram::new("token".to_string(), "chat".to_string(), client).is_some());
    }
}
