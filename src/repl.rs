//! The terminal front end. Still here for anyone who wants it -- run with
//! `--terminal`. The browser UI is the default.

use std::io::{self, Write};

use tokio::io::{AsyncBufReadExt, BufReader};

use crate::command::Command;
use crate::error::Result;
use crate::knowledge::{Knowledge, EMBED_MODEL};
use crate::ollama::Message;
use crate::Startup;

const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const CYAN: &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";

pub async fn run(startup: Startup) -> Result<()> {
    let Startup {
        client,
        host,
        mut model,
        available,
        mut knowledge,
        index_path,
    } = startup;

    println!("{BOLD}rust chatbot{RESET} {DIM}// {model} @ {host}{RESET}");
    if knowledge.is_empty() {
        println!("{DIM}no files indexed yet -- try /index ~/some/folder{RESET}");
    } else {
        println!(
            "{DIM}knows {} chunks from {} files in {}{RESET}",
            knowledge.chunks.len(),
            knowledge.file_count(),
            knowledge.root
        );
    }
    println!("{DIM}/help for commands, /quit to exit{RESET}\n");

    let mut history: Vec<Message> = Vec::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();

    loop {
        print!("{CYAN}you ›{RESET} ");
        io::stdout().flush()?;

        let Some(line) = lines.next_line().await? else {
            println!();
            break;
        };

        match Command::parse(&line) {
            Command::Empty => continue,

            Command::Quit => break,

            Command::Help => print_help(),

            Command::Clear => {
                let n = history.len();
                history.clear();
                println!("{DIM}cleared {n} messages{RESET}\n");
            }

            Command::Model(None) => {
                println!("{DIM}current: {BOLD}{model}{RESET}");
                for m in &available {
                    let marker = if m == &model { "*" } else { " " };
                    println!("{DIM}  {marker} {m}{RESET}");
                }
                println!();
            }

            Command::Model(Some(new_model)) => {
                if available.iter().any(|m| m == &new_model) {
                    model = new_model;
                    println!("{DIM}switched to {BOLD}{model}{RESET}\n");
                } else {
                    println!("{RED}no such model:{RESET} {new_model}");
                    println!("{DIM}available: {}{RESET}\n", available.join(", "));
                }
            }

            Command::Index(raw_path) => {
                if raw_path.is_empty() {
                    println!("{RED}usage:{RESET} /index <folder>\n");
                    continue;
                }

                let path = crate::expand_path(&raw_path);
                if !path.is_dir() {
                    println!("{RED}not a folder:{RESET} {}\n", path.display());
                    continue;
                }

                println!("{DIM}reading {}...{RESET}", path.display());

                let started = std::time::Instant::now();
                let result = Knowledge::build(&client, &path, |done, total| {
                    print!("\r{DIM}  embedded {done}/{total} chunks{RESET}");
                    let _ = io::stdout().flush();
                })
                .await;
                println!();

                match result {
                    Ok(built) if built.is_empty() => {
                        println!("{YELLOW}nothing readable found there.{RESET}\n");
                    }
                    Ok(built) => {
                        println!(
                            "{GREEN}learned{RESET} {} chunks from {} files in {:.1}s",
                            built.chunks.len(),
                            built.file_count(),
                            started.elapsed().as_secs_f32()
                        );
                        if let Err(e) = built.save(&index_path) {
                            eprintln!("{YELLOW}could not save index:{RESET} {e}");
                        }
                        knowledge = built;
                        println!();
                    }
                    Err(e) => eprintln!("{RED}indexing failed:{RESET} {e}\n"),
                }
            }

            Command::Sources => {
                if knowledge.is_empty() {
                    println!("{DIM}nothing indexed. try /index ~/some/folder{RESET}\n");
                } else {
                    println!(
                        "{DIM}{} chunks from {} files, indexed from {}{RESET}",
                        knowledge.chunks.len(),
                        knowledge.file_count(),
                        knowledge.root
                    );
                    let mut names: Vec<&str> =
                        knowledge.chunks.iter().map(|c| c.source.as_str()).collect();
                    names.sort_unstable();
                    names.dedup();
                    for name in names.iter().take(20) {
                        println!("{DIM}  {name}{RESET}");
                    }
                    if names.len() > 20 {
                        println!("{DIM}  ... and {} more{RESET}", names.len() - 20);
                    }
                    println!();
                }
            }

            Command::Forget => {
                knowledge = Knowledge::default();
                let _ = std::fs::remove_file(&index_path);
                println!("{DIM}forgot everything from the indexed files{RESET}\n");
            }

            Command::Unknown(verb) => {
                println!("{RED}unknown command:{RESET} {verb} {DIM}(try /help){RESET}\n");
            }

            Command::Say(text) => {
                let (context, cited) = if knowledge.is_empty() {
                    (None, Vec::new())
                } else {
                    match client.embed(EMBED_MODEL, std::slice::from_ref(&text)).await {
                        Ok(vectors) if !vectors.is_empty() => knowledge.context_for(&vectors[0]),
                        Ok(_) => (None, Vec::new()),
                        Err(e) => {
                            eprintln!("{YELLOW}lookup failed, answering without files:{RESET} {e}");
                            (None, Vec::new())
                        }
                    }
                };

                history.push(Message::user(text));

                // The grounding context is prepended per-request rather than stored,
                // so it never accumulates in the conversation.
                let mut request: Vec<Message> = Vec::new();
                if let Some(ctx) = context {
                    request.push(Message::system(ctx));
                }
                request.extend(history.iter().cloned());

                print!("{GREEN}bot ›{RESET} ");
                io::stdout().flush()?;

                let result = client
                    .chat_stream(&model, &request, |token| {
                        print!("{token}");
                        let _ = io::stdout().flush();
                    })
                    .await;

                println!();

                match result {
                    Ok(reply) => {
                        if !cited.is_empty() {
                            let list = cited
                                .iter()
                                .map(|(src, score)| format!("{src} ({score:.2})"))
                                .collect::<Vec<_>>()
                                .join(", ");
                            println!("{DIM}  sources: {list}{RESET}");
                        }
                        println!();
                        history.push(Message::assistant(reply));
                    }
                    Err(e) => {
                        eprintln!("{RED}error:{RESET} {e}\n");
                        history.pop();
                    }
                }
            }
        }
    }

    println!("{DIM}bye{RESET}");
    Ok(())
}

fn print_help() {
    println!(
        "\
{DIM}  /index <folder>  read a folder and learn what's in it
  /sources         what's currently loaded
  /forget          drop everything it learned
  /clear           forget the conversation (keeps indexed files)
  /model           list available models
  /model <name>    switch model
  /help            this message
  /quit            exit (ctrl-D works too){RESET}
"
    );
}
