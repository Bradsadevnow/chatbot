//! A local chatbot that answers questions about your own files.
//!
//! Browser UI (default):  cargo run
//! Terminal:              cargo run -- --terminal
//! Different port:        cargo run -- --port 8080

mod command;
mod error;
mod governance;
mod knowledge;
mod ollama;
mod repl;
mod web;

use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::governance::Store;
use crate::knowledge::{Knowledge, EMBED_MODEL};
use crate::ollama::Ollama;

const DEFAULT_MODEL: &str = "qwen3.5:9b";
const DEFAULT_HOST: &str = "http://localhost:11434";
const DEFAULT_PORT: u16 = 4141;

/// Where the learned index is cached between runs.
const INDEX_FILE: &str = ".knowledge.json";

const BOLD: &str = "\x1b[1m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";

/// Everything both front ends need, resolved once at startup.
pub struct Startup {
    pub client: Ollama,
    pub host: String,
    pub model: String,
    pub available: Vec<String>,
    pub knowledge: Knowledge,
    pub index_path: PathBuf,
    /// The only folder this process is allowed to read files from.
    pub store: Store,
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    let terminal = args.iter().any(|a| a == "--terminal" || a == "-t");
    let port = parse_port(&args).unwrap_or(DEFAULT_PORT);

    let startup = match bootstrap().await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{RED}fatal:{RESET} {e}");
            std::process::exit(1);
        }
    };

    let result = if terminal {
        repl::run(startup).await
    } else {
        web::serve(startup, port).await
    };

    if let Err(e) = result {
        eprintln!("{RED}fatal:{RESET} {e}");
        std::process::exit(1);
    }
}

/// Connect to Ollama, check the models exist, and load any saved index.
async fn bootstrap() -> Result<Startup> {
    let host = std::env::var("OLLAMA_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
    let model = std::env::var("CHAT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());

    let client = Ollama::new(&host);

    let available = match client.list_models().await {
        Ok(models) => models,
        Err(e) => {
            eprintln!("{RED}Could not reach Ollama at {host}{RESET}");
            eprintln!("  {e}");
            eprintln!("  Is it running? Try: {BOLD}ollama serve{RESET}");
            std::process::exit(1);
        }
    };

    if !available.iter().any(|m| m == &model) {
        eprintln!("{RED}warning:{RESET} model {BOLD}{model}{RESET} not found locally.");
        eprintln!("  available: {}", available.join(", "));
        eprintln!("  pull it with: {BOLD}ollama pull {model}{RESET}\n");
    }

    if !available.iter().any(|m| m.starts_with(EMBED_MODEL)) {
        eprintln!("{YELLOW}note:{RESET} {EMBED_MODEL} isn't installed, so indexing won't work.");
        eprintln!("  get it with: {BOLD}ollama pull {EMBED_MODEL}{RESET}\n");
    }

    // Deny by default. Without a knowledge store there is no admitted corpus, so
    // there is nothing this program is permitted to read -- and starting anyway
    // would leave it willing to index whatever it was pointed at, which is the
    // posture the governance layer exists to remove. Refuse, and say how to fix it.
    let base = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let store = match Store::discover(&base) {
        Ok(store) => store,
        Err(refusal) => {
            eprintln!("{RED}refused:{RESET} {}", refusal.detail);
            eprintln!("  {BOLD}{}{RESET}", refusal.remedy);
            eprintln!("  ({})", refusal.code.as_str());
            std::process::exit(1);
        }
    };

    let index_path = PathBuf::from(INDEX_FILE);
    let knowledge = Knowledge::load(&index_path).unwrap_or_default();

    Ok(Startup {
        client,
        host,
        model,
        available,
        knowledge,
        index_path,
        store,
    })
}

/// `--port 8080`
fn parse_port(args: &[String]) -> Option<u16> {
    let position = args.iter().position(|a| a == "--port" || a == "-p")?;
    args.get(position + 1)?.parse().ok()
}

/// Expand a leading `~` to the home directory.
///
/// The shell normally does this, but a path typed into our own prompt or the
/// browser never touches a shell -- so `~/notes` would otherwise be read as a
/// folder literally named "~".
pub fn expand_path(raw: &str) -> PathBuf {
    let raw = raw.trim();

    if let Some(rest) = raw.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    if raw == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(raw)
}

fn print_usage() {
    println!(
        "\
chatbot -- ask questions about your own files, locally

  cargo run                    open the browser UI (default)
  cargo run -- --terminal      use the terminal instead
  cargo run -- --port 8080     serve the UI on a different port

environment:
  CHAT_MODEL     which model answers      (default {DEFAULT_MODEL})
  OLLAMA_HOST    where Ollama lives       (default {DEFAULT_HOST})"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_flag_is_parsed() {
        let args = vec!["--port".to_string(), "8080".to_string()];
        assert_eq!(parse_port(&args), Some(8080));
    }

    #[test]
    fn missing_or_bad_port_falls_back() {
        assert_eq!(parse_port(&["--port".to_string()]), None);
        assert_eq!(
            parse_port(&["--port".to_string(), "banana".to_string()]),
            None
        );
        assert_eq!(parse_port(&[]), None);
    }

    #[test]
    fn tilde_expands_to_home() {
        std::env::set_var("HOME", "/home/someone");
        assert_eq!(expand_path("~/docs"), PathBuf::from("/home/someone/docs"));
        assert_eq!(expand_path("~"), PathBuf::from("/home/someone"));
    }

    #[test]
    fn other_paths_are_left_alone() {
        assert_eq!(expand_path("/tmp/x"), PathBuf::from("/tmp/x"));
        assert_eq!(expand_path("relative/x"), PathBuf::from("relative/x"));
        // A '~' that isn't a home reference stays literal.
        assert_eq!(expand_path("./~weird"), PathBuf::from("./~weird"));
    }
}
