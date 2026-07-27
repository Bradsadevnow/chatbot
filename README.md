# rust chatbot

A local chatbot that answers questions about your own files, in your browser.
Runs entirely on your machine against local Ollama — no API key, no account, nothing
leaves the box.

## Use it

```bash
cd ~/chatbot
cargo run
```

Then open **http://127.0.0.1:4141**.

Click **📁 Choose folder…**, click through to the folder you want, hit **Index this
folder** — then ask questions. It answers from your files and shows which ones it used,
with a match score for each.

The picker shows how many readable files sit in each folder before you commit, and it
hides exactly what the indexer skips (`.git`, `target`, `node_modules`, hidden folders),
so it can't offer you a folder that would come back empty. Typing a path still works if
you prefer; `~` is expanded for you. Open the page at `#pick` to jump straight to the
picker.

```bash
cargo run -- --terminal      # the old terminal interface, still works
cargo run -- --port 8080     # different port
cargo run -- --help
```

Defaults to `qwen3.5:9b`; switch models from the dropdown. Override the startup default
with `CHAT_MODEL` / `OLLAMA_HOST`. Indexing needs `nomic-embed-text`.

The server binds to `127.0.0.1` only. It has no authentication and will read any folder
it's pointed at, so it has no business being reachable from the network.

### Terminal commands

Only relevant with `--terminal`:

```
/index <folder>  read a folder and learn what's in it
/sources         what's currently loaded
/forget          drop everything it learned
/clear           forget the conversation (keeps indexed files)
/model [name]    list or switch models
/help  /quit
```

## What it actually does

1. Walks the folder, collecting readable text files (skips binaries, `.git`, `target`,
   `node_modules`, anything over 2 MB).
2. Cuts each file into overlapping ~1200-character chunks, split on paragraph
   boundaries so a thought is rarely severed mid-sentence.
3. Turns each chunk into 768 numbers encoding its meaning, via Ollama.
4. Saves all of it to `.knowledge.json`, so it still knows your files next launch.
5. On a question: encodes the question the same way, finds the closest chunks, and
   hands only those to the model — with instructions to answer from the excerpts or
   admit it can't.

That last instruction is load-bearing. Verified behavior against the governance docs:

```
you › What are the monthly subscription pricing tiers for Halcyon?
bot › ...there is no information about monthly subscription pricing tiers
      for Halcyon. These excerpts do not contain any commercial or pricing
      information.
  sources: DEPLOYMENT_TRUTH.md (0.56), CONSTITUTIONAL_PROTOCOL.md (0.50)
```

It declined to invent an answer rather than producing a plausible one.

## Known limits

- **The index is a snapshot.** Edit the files and it won't know until you re-index.
- **Re-indexing is all-or-nothing.** No incremental update. (77 chunks / 5 files ≈ 15 s.)
- **Search is brute force.** Every chunk is compared to every question. Fine into the
  tens of thousands of chunks; it will drag well beyond that.
- **One conversation, held in memory.** No saved or named chats; restarting the server
  clears the thread (indexed files survive).
- **Only text files.** No PDFs, no Word docs, no images.
- **Retrieval is similarity-only.** It finds chunks that *sound* related to the
  question, which is not the same as chunks that *answer* it. The scores are the tell —
  low numbers mean it was reaching.

## The code

Seven files — ~1620 lines of Rust plus a 510-line page — and no cleverness:

| file | job |
|---|---|
| `src/main.rs` | startup, flags, choosing a front end |
| `src/web.rs` | the HTTP server and streaming endpoints |
| `src/ui.html` | the whole browser UI, baked into the binary |
| `src/knowledge.rs` | reading files, chunking, embedding, searching |
| `src/ollama.rs` | talking to Ollama (chat streaming + embeddings) |
| `src/command.rs` | parsing terminal commands |
| `src/error.rs` | how failures are represented |

Both front ends call the same `Knowledge::context_for`, so the terminal and the browser
can never drift apart on what the model is actually told.

```bash
cargo test      # 18 tests
cargo clippy    # clean
cargo fmt
```

### Why Rust, honestly

The usual pitch is about protecting the person typing: the compiler forces you to handle
every case, there is no null, and anything that can fail says so in its type. That's
real — adding the browser UI meant restructuring how the app starts up, and the compiler
listed every place that needed to change.

But if you aren't the one writing the code, that benefit lands differently: it's a free
reviewer of whatever the author produced, catching a class of mistake before anyone runs
anything.

What it does **not** catch is "this built the wrong thing." No compiler does. That's
still on testing — which is why every behavior claimed above was verified by running it.

The other honest reason: `cargo build --release` produces one file, with the entire web
UI inside it, no runtime and no system dependencies. Copy it to a machine with the same
libc and it runs.
