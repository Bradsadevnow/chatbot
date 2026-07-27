//! Error handling.
//!
//! WHY THIS FILE EXISTS: Rust has no exceptions. A function that can fail says so
//! in its type: `Result<T, E>`. That sounds tedious until you notice the payoff --
//! you cannot accidentally ignore a failure, because you literally can't get at the
//! `T` without acknowledging the `E`.
//!
//! The tedium is removed by two things: the `?` operator, and `From` conversions.
//! `?` means "if this is an error, convert it into MY error type and return early."
//! The `impl From<...>` blocks at the bottom are what make that conversion happen.
//! Write them once here, and every `?` in the codebase just works.

use std::fmt;

/// Every way this program can fail, enumerated.
///
/// An enum in Rust is a *tagged union* -- a value is exactly one of these variants,
/// and each variant can carry different data. This is much closer to Haskell's
/// sum types than to a C enum or a Java enum.
#[derive(Debug)]
pub enum ChatError {
    /// The HTTP request itself failed (connection refused, timeout, ...).
    Http(reqwest::Error),
    /// Ollama sent us something that wasn't the JSON we expected.
    Json(serde_json::Error),
    /// Reading stdin or writing stdout failed.
    Io(std::io::Error),
    /// Ollama accepted the request but reported an error in the response body.
    Ollama(String),
}

/// `Display` is the human-readable rendering. `{}` in a format string uses this.
/// (`Debug`, which we derived above, is the programmer-facing `{:?}` rendering.)
impl fmt::Display for ChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `match` must be exhaustive. If you add a variant to ChatError above and
        // forget to handle it here, this does not compile. That property -- the
        // compiler finding every place you need to update -- is most of why
        // refactoring in Rust feels safe.
        match self {
            ChatError::Http(e) => write!(f, "http error: {e}"),
            ChatError::Json(e) => write!(f, "could not parse response: {e}"),
            ChatError::Io(e) => write!(f, "i/o error: {e}"),
            ChatError::Ollama(msg) => write!(f, "ollama error: {msg}"),
        }
    }
}

/// Opting into the standard error trait so this composes with the wider ecosystem.
impl std::error::Error for ChatError {}

// --- The `?` plumbing -------------------------------------------------------
// Each of these says: "a reqwest::Error can become a ChatError." Once that's
// known, `some_reqwest_call()?` inside a function returning Result<_, ChatError>
// compiles, and the conversion is automatic.

impl From<reqwest::Error> for ChatError {
    fn from(e: reqwest::Error) -> Self {
        ChatError::Http(e)
    }
}

impl From<serde_json::Error> for ChatError {
    fn from(e: serde_json::Error) -> Self {
        ChatError::Json(e)
    }
}

impl From<std::io::Error> for ChatError {
    fn from(e: std::io::Error) -> Self {
        ChatError::Io(e)
    }
}

/// A type alias so signatures read `Result<String>` instead of
/// `Result<String, ChatError>`. Very common Rust convention.
pub type Result<T> = std::result::Result<T, ChatError>;
