//! The "know your stuff" layer: read a folder of files, remember what's in them,
//! and pull up the relevant bits when a question comes in.
//!
//! How it works, in order:
//!   1. Walk a folder and collect readable text files.
//!   2. Cut each file into overlapping chunks small enough to reason about.
//!   3. Ask Ollama to turn each chunk into a list of 768 numbers (an "embedding")
//!      that captures its meaning. Similar meaning -> similar numbers.
//!   4. Save all of that to disk so it survives a restart.
//!   5. At question time, embed the question the same way and find the chunks
//!      whose numbers point in the most similar direction.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ollama::Ollama;

/// File types worth reading. Anything else is skipped.
const TEXT_EXTENSIONS: &[&str] = &[
    "md", "txt", "rst", "org", "rs", "py", "js", "ts", "jsx", "tsx", "go", "java", "c", "h", "cpp",
    "hpp", "sh", "bash", "sql", "toml", "yaml", "yml", "json", "html", "css", "csv", "ini", "cfg",
];

/// Directories that are never worth indexing.
const SKIP_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "dist",
    "build",
    ".git",
];

/// Files bigger than this are skipped (usually data dumps, not prose).
const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;

/// Roughly how big each chunk should get before we cut it.
const CHUNK_TARGET_CHARS: usize = 1200;

/// How many chunks to embed per request to Ollama.
const EMBED_BATCH: usize = 24;

/// The embedding model. Small, fast, and already installed.
pub const EMBED_MODEL: &str = "nomic-embed-text";

/// How many excerpts to feed the model per question.
pub const TOP_K: usize = 5;

/// Below this similarity, an excerpt is more likely noise than signal.
pub const MIN_SCORE: f32 = 0.35;

/// One piece of one file, plus its meaning-vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// Where this text came from, for citations.
    pub source: String,
    pub text: String,
    /// Pre-normalized to length 1, so similarity is just a dot product.
    pub embedding: Vec<f32>,
}

/// Everything we know, plus where it came from.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Knowledge {
    pub root: String,
    pub chunks: Vec<Chunk>,
}

impl Knowledge {
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// How many distinct files are represented.
    pub fn file_count(&self) -> usize {
        let mut seen: Vec<&str> = self.chunks.iter().map(|c| c.source.as_str()).collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len()
    }

    /// Build an index from a folder. `progress` is called with (done, total).
    pub async fn build<F>(client: &Ollama, root: &Path, mut progress: F) -> Result<Knowledge>
    where
        F: FnMut(usize, usize),
    {
        let files = collect_files(root);

        // Cut every file into chunks first, so we know the total up front and can
        // report honest progress.
        let mut pending: Vec<(String, String)> = Vec::new();
        for path in &files {
            let Ok(text) = fs::read_to_string(path) else {
                continue; // not valid UTF-8 -- almost certainly binary
            };
            let label = display_path(root, path);
            for piece in split_into_chunks(&text) {
                pending.push((label.clone(), piece));
            }
        }

        let total = pending.len();
        let mut chunks: Vec<Chunk> = Vec::with_capacity(total);

        for batch in pending.chunks(EMBED_BATCH) {
            let texts: Vec<String> = batch.iter().map(|(_, t)| t.clone()).collect();
            let mut vectors = client.embed(EMBED_MODEL, &texts).await?;

            for ((source, text), embedding) in batch.iter().zip(vectors.drain(..)) {
                chunks.push(Chunk {
                    source: source.clone(),
                    text: text.clone(),
                    embedding: normalize(embedding),
                });
            }

            progress(chunks.len(), total);
        }

        Ok(Knowledge {
            root: root.display().to_string(),
            chunks,
        })
    }

    /// Find the `top_k` chunks most related to an already-embedded question.
    ///
    /// Both sides are unit-length, so the dot product IS the cosine similarity:
    /// 1.0 means "pointing the same way", 0.0 means unrelated.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(&Chunk, f32)> {
        let query = normalize(query.to_vec());

        let mut scored: Vec<(&Chunk, f32)> = self
            .chunks
            .iter()
            .map(|c| {
                let score = c
                    .embedding
                    .iter()
                    .zip(query.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                (c, score)
            })
            .collect();

        // Highest score first. `partial_cmp` because floats have no total order
        // (NaN exists), so Rust makes us acknowledge that rather than pretend.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Turn an embedded question into grounding text for the model, plus the
    /// (file, score) pairs to show the user.
    ///
    /// Both are empty when nothing is indexed or nothing matched well enough --
    /// in which case the bot just answers from general knowledge.
    ///
    /// This is deliberately separate from the front end: the terminal and the
    /// browser both call it, so they can never drift apart on what the model
    /// actually gets told.
    pub fn context_for(&self, query: &[f32]) -> (Option<String>, Vec<(String, f32)>) {
        if self.is_empty() {
            return (None, Vec::new());
        }

        let hits: Vec<_> = self
            .search(query, TOP_K)
            .into_iter()
            .filter(|(_, score)| *score >= MIN_SCORE)
            .collect();

        if hits.is_empty() {
            return (None, Vec::new());
        }

        // Tell the model to stay inside the evidence, and to admit it when the
        // evidence doesn't cover the question. A useful "I don't know" beats a
        // confident invention.
        let mut context = String::from(
            "Answer the user's question using only the excerpts from their files below. \
             If the excerpts don't contain the answer, say so plainly and do not guess. \
             Mention which file you're drawing from when it helps.\n\n",
        );

        let mut cited: Vec<(String, f32)> = Vec::new();
        for (chunk, score) in &hits {
            context.push_str(&format!(
                "--- from {} ---\n{}\n\n",
                chunk.source, chunk.text
            ));

            // Keep the best score per file, and only list each file once.
            match cited.iter_mut().find(|(src, _)| src == &chunk.source) {
                Some(existing) => existing.1 = existing.1.max(*score),
                None => cited.push((chunk.source.clone(), *score)),
            }
        }

        (Some(context), cited)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Option<Knowledge> {
        let text = fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }
}

/// Scale a vector to length 1 so comparisons are about direction, not magnitude.
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let magnitude = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for x in v.iter_mut() {
            *x /= magnitude;
        }
    }
    v
}

/// Show paths relative to the indexed root, so citations stay readable.
fn display_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Recursively gather indexable files, skipping hidden and build directories.
fn collect_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();

            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }

            // symlink_metadata doesn't follow links, so we can't loop forever.
            let Ok(meta) = fs::symlink_metadata(&path) else {
                continue;
            };

            if meta.is_dir() {
                stack.push(path);
            } else if meta.is_file() && meta.len() <= MAX_FILE_BYTES && is_text_file(&path) {
                found.push(path);
            }
        }
    }

    found.sort();
    found
}

/// Subfolders the indexer would actually descend into, for the folder picker.
///
/// This deliberately reuses the same skip rules as `collect_files`, so the picker
/// can never offer you a folder the indexer would silently ignore.
pub fn listable_dirs(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };

    let mut dirs: Vec<PathBuf> = entries
        .flatten()
        .filter(|entry| {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || SKIP_DIRS.contains(&name.as_str()) {
                return false;
            }
            fs::symlink_metadata(entry.path())
                .map(|m| m.is_dir())
                .unwrap_or(false)
        })
        .map(|entry| entry.path())
        .collect();

    dirs.sort();
    dirs
}

/// How many files *directly inside* this folder would be indexed.
///
/// Deliberately not recursive: it stays instant even on a huge tree, and it
/// answers the question the picker actually needs ("is there anything here?").
pub fn count_indexable(dir: &Path) -> usize {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };

    entries
        .flatten()
        .filter(|entry| {
            let path = entry.path();
            fs::symlink_metadata(&path)
                .map(|m| m.is_file() && m.len() <= MAX_FILE_BYTES)
                .unwrap_or(false)
                && is_text_file(&path)
        })
        .count()
}

fn is_text_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| TEXT_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Cut text into chunks on paragraph boundaries, with one paragraph of overlap
/// so a thought split across a boundary is still findable from either side.
///
/// Working in whole paragraphs (rather than slicing at a character count) means
/// we can never cut a multi-byte character in half.
fn split_into_chunks(text: &str) -> Vec<String> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    let mut chunks = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut current_len = 0usize;

    for para in paragraphs {
        // A single huge paragraph still has to be broken up somewhere; do it on
        // line boundaries, which is the next-safest seam.
        if para.len() > CHUNK_TARGET_CHARS * 2 {
            if !current.is_empty() {
                chunks.push(current.join("\n\n"));
                current.clear();
                current_len = 0;
            }
            for piece in split_long_block(para) {
                chunks.push(piece);
            }
            continue;
        }

        current.push(para);
        current_len += para.len();

        if current_len >= CHUNK_TARGET_CHARS {
            chunks.push(current.join("\n\n"));
            // Carry the last paragraph forward as overlap.
            let overlap = current.pop();
            current.clear();
            current_len = 0;
            if let Some(tail) = overlap {
                current.push(tail);
                current_len = tail.len();
            }
        }
    }

    if !current.is_empty() {
        chunks.push(current.join("\n\n"));
    }

    chunks.retain(|c| !c.trim().is_empty());
    chunks
}

/// Break an oversized paragraph on line boundaries.
fn split_long_block(block: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for line in block.lines() {
        if current.len() + line.len() > CHUNK_TARGET_CHARS && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }

    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_gives_unit_length() {
        let v = normalize(vec![3.0, 4.0]);
        let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_survives_all_zeros() {
        let v = normalize(vec![0.0, 0.0]);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[test]
    fn short_text_is_one_chunk() {
        let chunks = split_into_chunks("hello world");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn long_text_splits_and_overlaps() {
        let para = "x".repeat(700);
        let text = format!("{para}\n\n{para}\n\n{para}");
        let chunks = split_into_chunks(&text);
        assert!(chunks.len() >= 2, "expected a split, got {}", chunks.len());
    }

    #[test]
    fn multibyte_text_is_never_cut_in_half() {
        // If we sliced by byte offset this would panic or corrupt.
        let para = "日本語のテキストです。".repeat(200);
        let text = format!("{para}\n\n{para}");
        let chunks = split_into_chunks(&text);
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(c.chars().count() > 0);
        }
    }

    #[test]
    fn empty_input_yields_nothing() {
        assert!(split_into_chunks("   \n\n  \n\n ").is_empty());
    }

    #[test]
    fn only_text_extensions_are_indexed() {
        assert!(is_text_file(Path::new("a/b/notes.md")));
        assert!(is_text_file(Path::new("main.RS")));
        assert!(!is_text_file(Path::new("photo.png")));
        assert!(!is_text_file(Path::new("noextension")));
    }
}
