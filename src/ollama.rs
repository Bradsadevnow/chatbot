//! Ollama API client with token streaming.
//!
//! WHY THIS FILE EXISTS: this is where serde and async earn their reputation.
//! The JSON handling is ~15 lines of struct definitions with zero parsing code,
//! and the streaming loop is ordinary-looking sequential code that happens to be
//! non-blocking.

use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error::{ChatError, Result};

/// One turn in the conversation. Shared between what we send and what we store.
///
/// `derive(Serialize, Deserialize)` generates the JSON conversion code at compile
/// time. No reflection, no runtime type inspection -- the generated code is as
/// fast as one you'd write by hand, and it can't drift out of sync with the struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: "assistant".into(),
            content: content.into(),
        }
    }

    /// Instructions and grounding evidence, not conversation.
    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: "system".into(),
            content: content.into(),
        }
    }
}

/// The request body we POST to /api/chat.
///
/// The `<'a>` is a *lifetime parameter*. It says this struct borrows data that must
/// outlive it -- we serialize the caller's messages directly out of their `Vec`
/// rather than cloning the entire conversation on every turn. This is the kind of
/// zero-copy move that's routine in Rust and nerve-wracking in C.
#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
    /// Reasoning models stream their chain-of-thought in a separate `thinking`
    /// field, with `content` empty until the thinking finishes. We only read
    /// `content`, so leaving this on has a failure mode worse than the lost
    /// visibility: a model that spends its whole generation budget thinking
    /// emits NO content at all, and the answer arrives empty.
    ///
    /// Measured on qwen3.5:9b against this corpus: 1,505 of 1,620 streamed
    /// chunks were thinking, and "What are the core principles?" reproducibly
    /// returned zero content. With this off, the same question answers normally.
    /// Models that don't support it ignore the field.
    think: bool,
}

/// One newline-delimited JSON object from the streaming response.
///
/// Serde ignores fields we didn't declare, so we only name the three we care
/// about even though Ollama sends a dozen.
#[derive(Deserialize)]
struct ChatChunk {
    message: Option<ChunkMessage>,
    #[serde(default)]
    done: bool,
    error: Option<String>,
}

#[derive(Deserialize)]
struct ChunkMessage {
    content: String,
}

/// A thin client over the Ollama HTTP API.
pub struct Ollama {
    http: reqwest::Client,
    base_url: String,
}

impl Ollama {
    pub fn new(base_url: impl Into<String>) -> Self {
        Ollama {
            // reqwest::Client holds an internal connection pool, so build it once
            // and reuse it. Cloning it is cheap (it's an Arc inside).
            http: reqwest::Client::new(),
            base_url: base_url.into(),
        }
    }

    /// Send the conversation and stream the reply back.
    ///
    /// `on_token` is called with each fragment as it arrives, so the caller decides
    /// how to display it. Taking `impl FnMut(&str)` means the closure is passed as a
    /// generic parameter and inlined at the call site -- no allocation, no vtable,
    /// no indirect call. "Zero-cost abstraction" is the marketing term; this is what
    /// it actually looks like.
    ///
    /// Returns the fully assembled reply.
    pub async fn chat_stream<F>(
        &self,
        model: &str,
        messages: &[Message],
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let body = ChatRequest {
            model,
            messages,
            stream: true,
            think: false,
        };

        let response = self
            .http
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await?; // <-- `?` converts reqwest::Error into ChatError via our From impl

        // A non-2xx status isn't an Err by default in reqwest; check it explicitly.
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ChatError::Ollama(format!("HTTP {status}: {text}")));
        }

        let mut stream = response.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        let mut full_reply = String::new();

        // Ollama streams newline-delimited JSON. Crucially, TCP chunks do NOT line
        // up with JSON objects -- one chunk can hold two objects, or half of one,
        // or split a multi-byte UTF-8 character down the middle. So we accumulate
        // raw bytes and only decode once we have a complete line.
        while let Some(chunk) = stream.next().await {
            buffer.extend_from_slice(&chunk?);

            while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                // `drain` removes the range and hands it to us -- one pass, no
                // second allocation for the remainder.
                let line: Vec<u8> = buffer.drain(..=newline_pos).collect();
                let line = String::from_utf8_lossy(&line);
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                let parsed: ChatChunk = serde_json::from_str(line)?;

                if let Some(err) = parsed.error {
                    return Err(ChatError::Ollama(err));
                }

                if let Some(msg) = parsed.message {
                    if !msg.content.is_empty() {
                        on_token(&msg.content);
                        full_reply.push_str(&msg.content);
                    }
                }

                if parsed.done {
                    return Ok(full_reply);
                }
            }
        }

        Ok(full_reply)
    }

    /// List locally available models, for `/model` with no argument.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        #[derive(Deserialize)]
        struct Tags {
            models: Vec<Tag>,
        }
        #[derive(Deserialize)]
        struct Tag {
            name: String,
        }

        let tags: Tags = self
            .http
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await?
            .json()
            .await?;

        Ok(tags.models.into_iter().map(|t| t.name).collect())
    }

    /// Turn a batch of texts into meaning-vectors.
    ///
    /// Batching matters a lot: one request with 24 texts is dramatically faster
    /// than 24 requests, because the model only gets loaded and scheduled once.
    pub async fn embed(&self, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        #[derive(Serialize)]
        struct EmbedRequest<'a> {
            model: &'a str,
            input: &'a [String],
        }

        #[derive(Deserialize)]
        struct EmbedResponse {
            embeddings: Vec<Vec<f32>>,
        }

        let response = self
            .http
            .post(format!("{}/api/embed", self.base_url))
            .json(&EmbedRequest {
                model,
                input: inputs,
            })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ChatError::Ollama(format!("HTTP {status}: {text}")));
        }

        let parsed: EmbedResponse = response.json().await?;

        // A silent length mismatch here would misattribute every citation, so
        // catch it loudly instead.
        if parsed.embeddings.len() != inputs.len() {
            return Err(ChatError::Ollama(format!(
                "expected {} embeddings, got {}",
                inputs.len(),
                parsed.embeddings.len()
            )));
        }

        Ok(parsed.embeddings)
    }
}
