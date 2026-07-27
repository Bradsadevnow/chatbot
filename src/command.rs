//! Slash-command parsing.
//!
//! WHY THIS FILE EXISTS: to show off the single feature most people fall in love
//! with first -- enums plus exhaustive pattern matching. Compare to a stringly-typed
//! `if line == "/help" ... else if ...` chain, where nothing stops you from
//! forgetting a case. Here, the set of things the user can say is a *type*, and the
//! compiler audits every place that consumes it.

/// One parsed line of user input.
///
/// Note `Model(Option<String>)`: Rust has no `null`. "Maybe a string" is spelled
/// `Option<String>`, and it's an ordinary enum (`Some(x)` or `None`) with no
/// special compiler magic. Because it's a real type, you cannot dereference it
/// without handling the `None` case -- which is the whole "no null pointer
/// exceptions" thing people talk about.
#[derive(Debug, PartialEq)]
pub enum Command {
    /// User just hit enter.
    Empty,
    /// `/help`
    Help,
    /// `/clear` -- wipe conversation history.
    Clear,
    /// `/model` (query current) or `/model llama3` (switch).
    Model(Option<String>),
    /// `/quit` or `/exit`
    Quit,
    /// `/index <folder>` -- read a folder and learn what's in it.
    Index(String),
    /// `/sources` -- what's currently loaded.
    Sources,
    /// `/forget` -- drop the indexed knowledge.
    Forget,
    /// A slash-command we don't recognize. Carries what they typed, for the error.
    Unknown(String),
    /// Ordinary text to send to the model.
    Say(String),
}

impl Command {
    /// Parse a raw input line.
    ///
    /// The `&str` parameter is a *borrowed* string slice -- we're reading the
    /// caller's buffer, not taking ownership of it and not copying it. The returned
    /// `Command` owns its own `String`s, so it can outlive the input buffer. That
    /// split between "borrow to look at" and "own to keep" is the core idea of
    /// the language, and it's why Rust needs no garbage collector.
    pub fn parse(line: &str) -> Command {
        let line = line.trim();

        if line.is_empty() {
            return Command::Empty;
        }

        // Anything not starting with '/' is just a message.
        if !line.starts_with('/') {
            return Command::Say(line.to_string());
        }

        // Split "/model qwen3.5:9b" into the verb and the rest.
        // `split_once` returns Option<(&str, &str)> -- None if there's no space.
        let (verb, rest) = match line.split_once(char::is_whitespace) {
            Some((v, r)) => (v, r.trim()),
            None => (line, ""),
        };

        match verb {
            "/help" | "/h" | "/?" => Command::Help,
            "/clear" | "/reset" => Command::Clear,
            "/quit" | "/exit" | "/q" => Command::Quit,
            "/model" | "/m" => {
                if rest.is_empty() {
                    Command::Model(None)
                } else {
                    Command::Model(Some(rest.to_string()))
                }
            }
            "/index" | "/i" => Command::Index(rest.to_string()),
            "/sources" | "/src" => Command::Sources,
            "/forget" => Command::Forget,
            other => Command::Unknown(other.to_string()),
        }
    }
}

/// Unit tests live next to the code they test, in the same file.
/// `cargo test` finds and runs them. No separate test project, no config.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_is_a_message() {
        assert_eq!(
            Command::parse("hello there"),
            Command::Say("hello there".into())
        );
    }

    #[test]
    fn whitespace_only_is_empty() {
        assert_eq!(Command::parse("   \t "), Command::Empty);
    }

    #[test]
    fn aliases_resolve() {
        assert_eq!(Command::parse("/q"), Command::Quit);
        assert_eq!(Command::parse("/exit"), Command::Quit);
    }

    #[test]
    fn model_with_and_without_argument() {
        assert_eq!(Command::parse("/model"), Command::Model(None));
        assert_eq!(
            Command::parse("/model qwen3.5:9b"),
            Command::Model(Some("qwen3.5:9b".into()))
        );
    }

    #[test]
    fn index_captures_the_path_including_spaces() {
        assert_eq!(
            Command::parse("/index /home/me/my notes"),
            Command::Index("/home/me/my notes".into())
        );
        assert_eq!(Command::parse("/index"), Command::Index("".into()));
    }

    #[test]
    fn knowledge_commands_resolve() {
        assert_eq!(Command::parse("/sources"), Command::Sources);
        assert_eq!(Command::parse("/forget"), Command::Forget);
    }

    #[test]
    fn unknown_slash_command_is_captured() {
        assert_eq!(
            Command::parse("/bogus x"),
            Command::Unknown("/bogus".into())
        );
    }
}
