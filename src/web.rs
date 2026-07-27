//! The browser front end.
//!
//! Serves one self-contained HTML page and a handful of JSON endpoints. Replies
//! and indexing progress are pushed to the page as Server-Sent Events, so text
//! appears word by word instead of arriving all at once.

use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Query, State},
    response::sse::{Event, Sse},
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::governance::{self, Store};
use crate::knowledge::{self, Knowledge, EMBED_MODEL};
use crate::ollama::{Message, Ollama};
use crate::Startup;

/// Everything the handlers share.
///
/// Each mutable piece gets its own lock so that, say, indexing a folder doesn't
/// block reading the conversation.
struct AppState {
    client: Ollama,
    available: Vec<String>,
    model: Mutex<String>,
    history: Mutex<Vec<Message>>,
    knowledge: Mutex<Knowledge>,
    index_path: PathBuf,
    /// The admitted corpus boundary. Every path that reaches the filesystem from a
    /// request goes through this first.
    store: Store,
}

/// What the page receives while a reply is being generated.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum ChatEvent {
    Token {
        text: String,
    },
    Done {
        sources: Vec<(String, f32)>,
    },
    /// A governance decision, not a failure -- kept as its own variant so the page
    /// can render "I declined, and here is what would fix it" differently from
    /// "something broke".
    Refused {
        code: &'static str,
        message: String,
        remedy: String,
    },
    Error {
        message: String,
    },
}

/// What the page receives while a folder is being indexed.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum IndexEvent {
    Progress {
        done: usize,
        total: usize,
    },
    Done {
        chunks: usize,
        files: usize,
        seconds: f32,
    },
    /// Refused before any embedding happened -- outside the store, or over budget.
    Refused {
        code: &'static str,
        message: String,
        remedy: String,
    },
    Error {
        message: String,
    },
}

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
}

#[derive(Deserialize)]
struct IndexRequest {
    path: String,
}

#[derive(Deserialize)]
struct ModelRequest {
    model: String,
}

#[derive(Deserialize)]
struct BrowseQuery {
    path: Option<String>,
}

#[derive(Serialize)]
struct BrowseEntry {
    name: String,
    path: String,
    /// Indexable files directly inside, so you can see what's worth picking.
    files: usize,
}

#[derive(Serialize)]
struct BrowseResponse {
    path: String,
    /// `None` at the filesystem root, which is how the UI hides "go up".
    parent: Option<String>,
    /// Each ancestor, for clickable breadcrumbs.
    crumbs: Vec<BrowseEntry>,
    dirs: Vec<BrowseEntry>,
    files_here: usize,
    error: Option<String>,
}

#[derive(Serialize)]
struct Status {
    model: String,
    models: Vec<String>,
    chunks: usize,
    files: usize,
    root: String,
}

pub async fn serve(startup: Startup, port: u16) -> crate::error::Result<()> {
    let state = Arc::new(AppState {
        client: startup.client,
        available: startup.available,
        model: Mutex::new(startup.model),
        history: Mutex::new(Vec::new()),
        knowledge: Mutex::new(startup.knowledge),
        index_path: startup.index_path,
        store: startup.store,
    });

    let app = Router::new()
        .route("/", get(page))
        .route("/api/status", get(status))
        .route("/api/browse", get(browse))
        .route("/api/chat", post(chat))
        .route("/api/index", post(index))
        .route("/api/model", post(set_model))
        .route("/api/clear", post(clear))
        .route("/api/forget", post(forget))
        .with_state(state);

    // Bind to loopback only. This is a personal tool with no auth, and it can
    // read any folder you point it at -- it has no business being reachable
    // from the rest of the network.
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    println!("\n  \x1b[1mchatbot\x1b[0m is running");
    println!("  \x1b[36mhttp://{addr}\x1b[0m");
    println!("\n  \x1b[2mctrl-C to stop\x1b[0m\n");

    axum::serve(listener, app).await?;
    Ok(())
}

/// The page itself, baked into the binary so there are no loose files to ship.
async fn page() -> Html<&'static str> {
    Html(include_str!("ui.html"))
}

/// List the folders inside a directory, for the picker.
///
/// Directories only -- this never returns file contents. It is also confined to the
/// knowledge store: an unadmitted path does not fall back to showing the requester
/// something else, it falls back to the store root. Before that confinement this
/// endpoint would enumerate the whole filesystem for anything that could reach the
/// port, which made the picker a directory-listing oracle as well as a picker.
async fn browse(
    State(state): State<Arc<AppState>>,
    Query(query): Query<BrowseQuery>,
) -> Json<BrowseResponse> {
    let root = state.store.root();

    // No path means "start at the store root" -- there is nowhere else to start.
    let requested = query
        .path
        .map(|p| crate::expand_path(&p))
        .unwrap_or_else(|| root.to_path_buf());

    let (path, error) = match state.store.admit(&requested) {
        Ok(admitted) => (admitted, None),
        Err(refusal) => (root.to_path_buf(), Some(refusal.detail)),
    };

    // Breadcrumbs stop at the store root rather than walking up to `/`, so the
    // picker never offers a way out of the boundary.
    let crumbs = path
        .ancestors()
        .take_while(|ancestor| ancestor.starts_with(root))
        .skip(1)
        .map(|ancestor| BrowseEntry {
            name: ancestor
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| crate::governance::STORE_DIR.to_string()),
            path: ancestor.display().to_string(),
            files: 0,
        })
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();

    Json(BrowseResponse {
        // `None` at the store root, which is how the UI hides "go up".
        parent: if path == root {
            None
        } else {
            path.parent()
                .filter(|p| p.starts_with(root))
                .map(|p| p.display().to_string())
        },
        crumbs,
        dirs: knowledge::listable_dirs(&path)
            .into_iter()
            .map(entry_for)
            .collect(),
        files_here: knowledge::count_indexable(&path),
        path: path.display().to_string(),
        error,
    })
}

fn entry_for(path: PathBuf) -> BrowseEntry {
    BrowseEntry {
        name: path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        files: knowledge::count_indexable(&path),
        path: path.display().to_string(),
    }
}

async fn status(State(state): State<Arc<AppState>>) -> Json<Status> {
    let knowledge = state.knowledge.lock().await;
    Json(Status {
        model: state.model.lock().await.clone(),
        models: state.available.clone(),
        chunks: knowledge.chunks.len(),
        files: knowledge.file_count(),
        root: knowledge.root.clone(),
    })
}

async fn set_model(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ModelRequest>,
) -> Json<bool> {
    if state.available.contains(&req.model) {
        *state.model.lock().await = req.model;
        Json(true)
    } else {
        Json(false)
    }
}

async fn clear(State(state): State<Arc<AppState>>) -> Json<bool> {
    state.history.lock().await.clear();
    Json(true)
}

async fn forget(State(state): State<Arc<AppState>>) -> Json<bool> {
    *state.knowledge.lock().await = Knowledge::default();
    let _ = std::fs::remove_file(&state.index_path);
    Json(true)
}

/// Stream a reply.
///
/// The work happens in a spawned task that pushes events into a channel; the
/// channel is handed back to axum as a stream. That indirection is what lets
/// tokens reach the browser as they're produced rather than after the fact.
async fn chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Sse<impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel::<ChatEvent>();

    tokio::spawn(async move {
        let question = req.message;

        // Embed the question first, without holding the knowledge lock across
        // the network call.
        let query_vector = if state.knowledge.lock().await.is_empty() {
            None
        } else {
            match state
                .client
                .embed(EMBED_MODEL, std::slice::from_ref(&question))
                .await
            {
                Ok(mut v) if !v.is_empty() => Some(v.remove(0)),
                Ok(_) => None,
                Err(e) => {
                    let _ = tx.send(ChatEvent::Error {
                        message: format!("could not search your files: {e}"),
                    });
                    None
                }
            }
        };

        let retrieval = match query_vector {
            Some(v) => state.knowledge.lock().await.context_for(&v),
            None => knowledge::Retrieval::default(),
        };
        let (context, sources) = (retrieval.context, retrieval.sources);

        // Evidence nominates; policy admits. The model is never asked a question we
        // have no grounds to answer -- an empty or off-corpus retrieval is refused
        // here rather than handed over as an empty context for it to improvise from.
        if let Err(refusal) = governance::EvidenceGate::admit(&sources, retrieval.best_score) {
            let _ = tx.send(ChatEvent::Refused {
                code: refusal.code.as_str(),
                message: refusal.detail,
                remedy: refusal.remedy,
            });
            return;
        }

        let request = {
            let mut history = state.history.lock().await;
            history.push(Message::user(question));

            let mut request: Vec<Message> = Vec::new();
            if let Some(ctx) = context {
                request.push(Message::system(ctx));
            }
            request.extend(history.iter().cloned());
            request
        };

        let model = state.model.lock().await.clone();

        let sender = tx.clone();
        let result = state
            .client
            .chat_stream(&model, &request, move |token| {
                let _ = sender.send(ChatEvent::Token {
                    text: token.to_string(),
                });
            })
            .await;

        match result {
            Ok(reply) => {
                state.history.lock().await.push(Message::assistant(reply));
                let _ = tx.send(ChatEvent::Done { sources });
            }
            Err(e) => {
                // Drop the unanswered user turn so history stays coherent.
                state.history.lock().await.pop();
                let _ = tx.send(ChatEvent::Error {
                    message: e.to_string(),
                });
            }
        }
    });

    Sse::new(to_event_stream(rx))
}

/// Index a folder, reporting progress as it goes.
async fn index(
    State(state): State<Arc<AppState>>,
    Json(req): Json<IndexRequest>,
) -> Sse<impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>>> {
    let (tx, rx) = mpsc::unbounded_channel::<IndexEvent>();

    tokio::spawn(async move {
        let requested = crate::expand_path(&req.path);

        // Policy admits before anything is read. Both gates refuse *before* a
        // single embedding call, which is the point: the expensive, unresumable
        // work never starts on a request that was never going to be allowed.
        let path = match state.store.admit(&requested) {
            Ok(admitted) => admitted,
            Err(refusal) => {
                let _ = tx.send(IndexEvent::Refused {
                    code: refusal.code.as_str(),
                    message: refusal.detail,
                    remedy: refusal.remedy,
                });
                return;
            }
        };

        let (files, estimated_chunks) = knowledge::survey(&path);
        if let Err(refusal) = governance::Budget::admit(files, estimated_chunks) {
            let _ = tx.send(IndexEvent::Refused {
                code: refusal.code.as_str(),
                message: refusal.detail,
                remedy: refusal.remedy,
            });
            return;
        }

        let started = Instant::now();
        let progress_tx = tx.clone();

        let built = Knowledge::build(&state.client, &path, move |done, total| {
            let _ = progress_tx.send(IndexEvent::Progress { done, total });
        })
        .await;

        match built {
            Ok(built) if built.is_empty() => {
                let _ = tx.send(IndexEvent::Error {
                    message: "nothing readable found there".to_string(),
                });
            }
            Ok(built) => {
                // Every transition leaves a receipt.
                let receipt = governance::IndexReceipt {
                    root: path.clone(),
                    files: built.file_count(),
                    chunks: built.chunks.len(),
                    seconds: started.elapsed().as_secs_f32(),
                };
                println!("  indexed {receipt}");

                let event = IndexEvent::Done {
                    chunks: receipt.chunks,
                    files: receipt.files,
                    seconds: receipt.seconds,
                };
                if let Err(e) = built.save(&state.index_path) {
                    eprintln!("could not save index: {e}");
                }
                *state.knowledge.lock().await = built;
                let _ = tx.send(event);
            }
            Err(e) => {
                let _ = tx.send(IndexEvent::Error {
                    message: e.to_string(),
                });
            }
        }
    });

    Sse::new(to_event_stream(rx))
}

/// Shared plumbing: turn a channel of serializable events into an SSE stream.
fn to_event_stream<T: Serialize + Send + 'static>(
    rx: mpsc::UnboundedReceiver<T>,
) -> impl tokio_stream::Stream<Item = std::result::Result<Event, Infallible>> {
    UnboundedReceiverStream::new(rx).map(|event| {
        // Serializing our own small enums cannot fail; if it somehow did, an
        // empty frame is better than tearing down the whole connection.
        Ok(Event::default()
            .json_data(event)
            .unwrap_or_else(|_| Event::default().data("{}")))
    })
}
