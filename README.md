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

Everything it reads lives in `knowledgestore/`. Put your documents there — subfolders
are fine, and each one can be indexed on its own. That folder is the whole world as
far as this program is concerned: it will not read outside it, and it refuses to
start if it's missing rather than quietly falling back to somewhere permissive.

Click **📁 Choose folder…**, click through to the folder you want, hit **Index this
folder** — then ask questions. It answers from your files and shows which ones it used,
with a match score for each.

The picker shows how many readable files sit in each folder before you commit, and it
hides exactly what the indexer skips (`.git`, `target`, `node_modules`, hidden folders),
so it can't offer you a folder that would come back empty. It starts at the knowledge
store and cannot climb above it — the breadcrumbs stop there. Typing a path still works
if you prefer, but a path outside the store is refused rather than followed. Open the
page at `#pick` to jump straight to the picker.

```bash
cargo run -- --terminal      # the old terminal interface, still works
cargo run -- --port 8080     # different port
cargo run -- --help
```

Defaults to `gpt-oss:20b`; switch models from the dropdown. Override the startup default
with `CHAT_MODEL` / `OLLAMA_HOST`. Indexing needs `nomic-embed-text`.

The default was picked by running the same four questions through every locally
installed model against byte-identical excerpts. `gpt-oss:20b` recovered detail the
others dropped — a five-object model and a set of tier definitions that two other
models missed completely. It is the slowest of the three and it reasons even when told
not to, which is why it isn't the obvious pick on a stopwatch. Every model tested
declined the unsupported question correctly, so that wasn't a differentiator.

The server binds to `127.0.0.1` only and has no authentication, so it still has no
business being reachable from the network — but it is no longer true that it will read
any folder it's pointed at. See below.

### Terminal commands

Only relevant with `--terminal`:

```
/index <folder>  read a folder inside knowledgestore/ and learn what's in it
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

## What it refuses to do

There is one module, `src/governance.rs`, that decides what may be read and what may
be answered. Both front ends route through it, so the terminal and the browser cannot
drift on what's allowed any more than they can drift on what the model is told.

It follows one rule borrowed from a larger governed runtime: *evidence nominates,
policy admits, admission creates canonical state, canonical state stays revisable,
every transition leaves a receipt.*

Three gates, each verified by running it:

| gate | refuses | reason code |
|---|---|---|
| **store** | anything resolving outside `knowledgestore/` | `outside_store` |
| **budget** | more than 500 files or ~4000 chunks | `over_budget` |
| **evidence** | questions nothing indexed can speak to | `no_evidence_retrieved` |

```
$ curl -X POST localhost:4141/api/index -d '{"path":"knowledgestore/../.."}'
{"type":"refused","code":"outside_store",
 "message":"/home/you is outside the knowledge store",
 "remedy":"put the files inside /home/you/chatbot/knowledgestore first"}
```

Note what that traversal did: it *resolved* to `/home/you` and was then refused. Paths
are canonicalized before they're compared, so `..` and symlinks are checked against
where they actually land, not how they're spelled. A symlink inside the store pointing
out of it is refused for the same reason.

Two properties are deliberate and easy to lose:

- **A refusal is not a failure.** `Refusal` is not a variant of `ChatError`, and the
  browser renders it in its own style, not the error style. Every refusal carries a
  remedy, because one that doesn't tell you how to satisfy it reads like a bug — and
  people route around things they can't understand, which is how a boundary stops
  being one.
- **Deny by default.** No `knowledgestore/` means it refuses to start and prints the
  `mkdir` that fixes it. It does not create the folder, and it does not fall back to
  your home directory. Governance that quietly degrades into permissiveness isn't
  governance.

### On the evidence gate, honestly

The evidence gate is a similarity floor, and a similarity floor **cannot** tell you
whether excerpts support an answer. The refusal quoted above is the proof: it scored
**0.56** — high — and the correct answer was still "not in these documents." Cosine
similarity measures topical relatedness, and a pricing question against governance docs
is highly related and entirely unsupported. Any floor tuned to catch that would reject
legitimate answers too.

So the floor claims only what it can back: *did retrieval return anything at all.* It
reports the real pre-filter score when it declines (`best match 0.33, floor 0.35`)
rather than implying it saw nothing. The genuine support verdict — one that comes from
*reading* the excerpts — would have to come from the model emitting a structured
verdict, and `EvidenceGate` is the seam where that swaps in without touching either
front end.

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
  low numbers mean it was reaching. This is also why the evidence gate is honest about
  being a floor and not a support verdict.
- **It will not answer off-corpus questions at all.** Since the evidence gate went in,
  a question nothing indexed can speak to is refused rather than answered from the
  model's general knowledge. That is the intent — it only speaks about your files — but
  it does mean this is no longer usable as a general chatbot with files attached.
- **The budget is a guess before the fact.** The pre-index walk estimates chunks from
  file size, so a folder of dense prose can be admitted and still take a while. It
  bounds the disaster, not the runtime.

## The code

Nine files — ~2080 lines of Rust plus an 822-line page — and no cleverness:

| file | job |
|---|---|
| `src/main.rs` | startup, flags, choosing a front end |
| `src/web.rs` | the HTTP server and streaming endpoints |
| `src/ui.html` | the whole browser UI, baked into the binary |
| `src/repl.rs` | the terminal front end and its command loop |
| `src/governance.rs` | what may be read, and what may be answered |
| `src/knowledge.rs` | reading files, chunking, embedding, searching |
| `src/ollama.rs` | talking to Ollama (chat streaming + embeddings) |
| `src/command.rs` | parsing terminal commands |
| `src/error.rs` | how failures are represented |

Both front ends call the same `Knowledge::context_for`, so the terminal and the browser
can never drift apart on what the model is actually told.

```bash
cargo test      # 30 tests
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
