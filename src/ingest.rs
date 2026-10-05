//! Corpus ingestion. Splits a document into prompt-sized chunks so each can be
//! run through the same `claude -p` causal extractor as live delegation. The
//! orchestration (delegate -> extract -> seed) lives in the `/learn` handler;
//! this module owns the deterministic chunking and the RSS/Atom feed parsing of
//! the World Ingestion Layer.

use std::net::IpAddr;

use anyhow::{bail, Context, Result};

/// Pack paragraphs (blank-line separated) greedily into chunks of at most
/// `max_chars`. A single paragraph longer than `max_chars` becomes its own
/// oversized chunk rather than being split mid-sentence.
pub fn chunk(text: &str, max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in text.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        if current.is_empty() {
            current.push_str(paragraph);
        } else if current.len() + 2 + paragraph.len() <= max_chars {
            current.push_str("\n\n");
            current.push_str(paragraph);
        } else {
            chunks.push(std::mem::take(&mut current));
            current.push_str(paragraph);
        }
        if current.len() >= max_chars {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Fetch an RSS or Atom feed and return each entry as `(id, text)`: the feed's
/// stable entry id (for dedup across scheduled re-fetches) and its title-plus-body
/// text for the learn pipeline to extract causal relations from.
pub async fn fetch_feed(client: &reqwest::Client, url: &str) -> Result<Vec<(String, String)>> {
    validate_feed_url(url)?;
    let body = client
        .get(url)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    parse_feed(&body)
}

/// Reject feed URLs that are an SSRF risk: only `http`/`https`, and no literal
/// internal target (loopback, private, link-local incl. cloud metadata, or
/// `localhost`/`.local`). A hostname that resolves to an internal IP via DNS
/// rebinding is a residual a network egress policy must cover; this blocks the
/// direct cases without a blocking DNS lookup on the async path.
fn validate_feed_url(url: &str) -> Result<()> {
    let parsed = reqwest::Url::parse(url).context("invalid feed URL")?;
    if !matches!(parsed.scheme(), "http" | "https") {
        bail!("feed URL scheme '{}' is not allowed", parsed.scheme());
    }
    let host = parsed.host_str().context("feed URL has no host")?;
    // host_str keeps the brackets on an IPv6 literal (`[::1]`); strip them so the
    // address parses, otherwise an internal IPv6 literal would slip past as a
    // non-IP host. A domain never contains brackets, so this is a no-op for it.
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_internal_ip(&ip) {
            bail!("feed host is an internal address");
        }
    } else if host.eq_ignore_ascii_case("localhost") || host.ends_with(".local") {
        bail!("feed host is internal");
    }
    Ok(())
}

/// Whether an IP is loopback, private, link-local (incl. `169.254.0.0/16`
/// metadata), or unspecified - never a legitimate feed origin. IPv6 covers the
/// same ground (unique-local `fc00::/7`, link-local `fe80::/10`) and unwraps an
/// IPv4-mapped address so `::ffff:169.254.169.254` cannot smuggle a metadata IP
/// past the v4 checks.
fn is_internal_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_private() || v4.is_loopback() || v4.is_link_local() || v4.is_unspecified()
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_internal_ip(&IpAddr::V4(v4));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unique_local()
                || v6.is_unicast_link_local()
        }
    }
}

/// Parse RSS 2.0 or Atom feed bytes into one `(id, text)` pair per entry.
fn parse_feed(bytes: &[u8]) -> Result<Vec<(String, String)>> {
    let feed = feed_rs::parser::parse(bytes)?;
    Ok(feed
        .entries
        .iter()
        .map(|entry| (entry.id.clone(), entry_text(entry)))
        .filter(|(_, text)| !text.is_empty())
        .collect())
}

/// An entry's title joined with its summary (or, failing that, its content
/// body), with HTML tags stripped and entities decoded so the extractor sees
/// clean prose instead of markup.
fn entry_text(entry: &feed_rs::model::Entry) -> String {
    let title = entry
        .title
        .as_ref()
        .map(|t| strip_html(&t.content))
        .unwrap_or_default();
    let body = entry
        .summary
        .as_ref()
        .map(|s| s.content.as_str())
        .or_else(|| entry.content.as_ref().and_then(|c| c.body.as_deref()))
        .map(strip_html)
        .unwrap_or_default();
    match (title.is_empty(), body.is_empty()) {
        (true, true) => String::new(),
        (false, true) => title,
        (true, false) => body,
        (false, false) => format!("{title}. {body}"),
    }
}

/// Strip HTML tags, decode the common named and numeric entities, and collapse
/// whitespace, turning an HTML feed body into a single clean line of prose.
/// Tags are removed before entities are decoded, so a decoded `&lt;` stays text.
fn strip_html(input: &str) -> String {
    let mut text = String::with_capacity(input.len());
    let mut in_tag = false;
    for c in input.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    let decoded = text
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&");
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_paragraphs_pack_into_one_chunk() {
        let text = "first para.\n\nsecond para.\n\nthird para.";
        let chunks = chunk(text, 1000);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("first") && chunks[0].contains("third"));
    }

    #[test]
    fn paragraphs_split_when_budget_exceeded() {
        let text = "aaaa\n\nbbbb\n\ncccc";
        let chunks = chunk(text, 6);
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn blank_paragraphs_are_skipped() {
        let text = "only.\n\n\n\n   \n\nreal.";
        let chunks = chunk(text, 1000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "only.\n\nreal.");
    }

    #[test]
    fn empty_input_yields_no_chunks() {
        assert!(chunk("   ", 1000).is_empty());
    }

    #[test]
    fn parse_feed_extracts_title_and_summary_per_entry() {
        let rss = r#"<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>Rain causes floods</title><description>Heavy rain leads to flooding.</description></item>
<item><title>Sun dries soil</title><description>Sunshine evaporates moisture.</description></item>
</channel></rss>"#;
        let entries = parse_feed(rss.as_bytes()).unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries[0].1.contains("Rain causes floods"));
        assert!(entries[0].1.contains("flooding"));
    }

    #[test]
    fn entry_html_is_stripped_to_clean_prose() {
        let rss = r#"<?xml version="1.0"?>
<rss version="2.0"><channel><title>Test feed</title>
<item><title>Storms</title><description>&lt;p&gt;Heavy &lt;b&gt;rain&lt;/b&gt;   causes floods &amp; damage.&lt;/p&gt;</description></item>
</channel></rss>"#;
        let entries = parse_feed(rss.as_bytes()).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(
            !entries[0].1.contains('<'),
            "tags must be stripped: {}",
            entries[0].1
        );
        assert!(entries[0].1.contains("rain"));
        assert!(entries[0].1.contains("causes floods & damage"));
    }

    #[test]
    fn feed_url_validation_blocks_internal_and_non_http() {
        assert!(validate_feed_url("https://feeds.bbci.co.uk/news.xml").is_ok());
        assert!(validate_feed_url("http://localhost/x").is_err());
        assert!(validate_feed_url("http://127.0.0.1/x").is_err());
        assert!(validate_feed_url("http://169.254.169.254/latest/meta-data").is_err());
        assert!(validate_feed_url("http://10.0.0.5/x").is_err());
        assert!(validate_feed_url("file:///etc/passwd").is_err());
    }

    #[test]
    fn feed_url_validation_blocks_internal_ipv6_and_v4_mapped() {
        assert!(validate_feed_url("http://[::1]/x").is_err()); // loopback
        assert!(validate_feed_url("http://[fe80::1]/x").is_err()); // link-local
        assert!(validate_feed_url("http://[fc00::1]/x").is_err()); // unique-local
                                                                   // An IPv4-mapped IPv6 literal must not smuggle the metadata IP past v4 checks.
        assert!(validate_feed_url("http://[::ffff:169.254.169.254]/latest").is_err());
        // A public IPv6 literal is still allowed.
        assert!(validate_feed_url("http://[2606:4700:4700::1111]/x").is_ok());
    }
}
