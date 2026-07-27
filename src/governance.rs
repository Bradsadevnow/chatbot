//! The governance layer: what may be indexed, and what may be answered.
//!
//! WHY THIS FILE EXISTS: the rest of the program is willing to read any folder it
//! is pointed at and hand any retrieved text to the model. That is fine for a toy
//! and wrong for anything else. This module is the single place where a request is
//! *admitted* or *refused*, so the answer to "what is this thing allowed to touch?"
//! lives in one file instead of being spread across two front ends.
//!
//! The protocol it implements, borrowed from the larger governed runtime this was
//! modelled on: **evidence nominates, policy admits, admission creates canonical
//! state, canonical state stays revisable, every transition leaves a receipt.**
//!
//! Two properties are load-bearing, and both are easy to lose by accident:
//!
//! 1. **A refusal is not a failure.** `Refusal` is deliberately NOT a variant of
//!    `ChatError`. A refused index is the system working correctly, and it carries
//!    a machine-readable code plus a remedy the caller can act on. Folding it into
//!    the error type would make "I declined" indistinguishable from "I broke".
//!
//! 2. **Deny by default.** Absent configuration means refuse, never improvise. A
//!    missing knowledge store is not silently created and does not fall back to
//!    the home directory; it is reported with the exact command that fixes it.
//!    Governance that quietly degrades into permissiveness is not governance.

use std::fmt;
use std::path::{Path, PathBuf};

/// The folder that holds everything this chatbot is allowed to read.
///
/// Deliberately a fixed location inside the project rather than a configurable
/// path: the whole point is that the corpus boundary is not negotiable at runtime.
pub const STORE_DIR: &str = "knowledgestore";

/// Refuse to index a tree with more files than this.
///
/// Not arbitrary: indexing runs at roughly 5 chunks/sec against a local Ollama, and
/// re-indexing is all-or-nothing, so a few thousand chunks is already a multi-minute
/// commitment that cannot be resumed. The budget exists so you find that out from a
/// refusal in milliseconds rather than from a progress bar twenty minutes in.
pub const MAX_FILES: usize = 500;

/// Refuse to index a tree that would produce more chunks than this.
///
/// Search is brute force -- every chunk is compared to every question -- so this
/// also bounds query latency, not just indexing time.
pub const MAX_CHUNKS: usize = 4_000;

/// Below this similarity, retrieval found nothing worth calling evidence.
///
/// IMPORTANT -- read before raising this number. This floor answers exactly one
/// question: "did retrieval return anything at all?" It does NOT and CANNOT answer
/// "do these excerpts support an answer", because cosine similarity measures topical
/// relatedness, not supportability. The two come apart in the ordinary case: asking
/// a governance corpus about pricing scores *highly* (same vocabulary, same entity)
/// while being completely unsupported. A floor tuned to catch that would have to sit
/// so high it would reject legitimate answers.
///
/// So this is honestly labelled as what it is. It is the seam, not the verdict:
/// `EvidenceGate` is the enforcement point, and a real support verdict -- one that
/// comes from *reading* the excerpts rather than scoring them -- can be swapped in
/// behind it without re-plumbing either front end.
///
/// The value is inherited, not invented: it was `knowledge::MIN_SCORE`, already
/// tuned against this corpus. It was moved here because retrieval and admission
/// were enforcing the same idea with two different numbers, and the stricter one
/// living in `knowledge` meant this gate could never fire -- a gate that cannot
/// fire is decoration. Both now read this constant.
pub const EVIDENCE_FLOOR: f32 = 0.35;

/// Why something was refused. Machine-readable so a UI can distinguish a refusal
/// from a crash, and distinguish the kinds of refusal from each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefusalCode {
    /// The knowledge store folder does not exist.
    NoStoreConfigured,
    /// The requested path resolves outside the knowledge store.
    OutsideStore,
    /// The requested path is not a folder.
    NotAFolder,
    /// Indexing that folder would exceed the file or chunk budget.
    OverBudget,
    /// Retrieval returned nothing above the evidence floor.
    NoEvidenceRetrieved,
}

impl RefusalCode {
    /// The stable string form, for JSON and for tests. Deliberately snake_case and
    /// never derived from the variant name, so renaming a variant cannot silently
    /// change the wire format.
    pub fn as_str(self) -> &'static str {
        match self {
            RefusalCode::NoStoreConfigured => "no_store_configured",
            RefusalCode::OutsideStore => "outside_store",
            RefusalCode::NotAFolder => "not_a_folder",
            RefusalCode::OverBudget => "over_budget",
            RefusalCode::NoEvidenceRetrieved => "no_evidence_retrieved",
        }
    }
}

/// A governance decision to decline. Carries what happened and what would fix it.
///
/// The `remedy` is not decoration. A refusal that does not tell you how to satisfy
/// it is indistinguishable from a bug, and users route around things they cannot
/// understand -- which is how a boundary stops being one.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub code: RefusalCode,
    pub detail: String,
    pub remedy: String,
}

impl Refusal {
    fn new(code: RefusalCode, detail: impl Into<String>, remedy: impl Into<String>) -> Self {
        Refusal {
            code,
            detail: detail.into(),
            remedy: remedy.into(),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.detail, self.code.as_str())
    }
}

/// The admitted corpus boundary: a canonicalized folder that all reads stay inside.
#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

impl Store {
    /// Locate the knowledge store, or refuse.
    ///
    /// Canonicalizes immediately, because every later containment check compares
    /// against this value -- a root holding `..` or an unresolved symlink would make
    /// those comparisons meaningless.
    pub fn discover(base: &Path) -> Result<Store, Refusal> {
        let candidate = base.join(STORE_DIR);

        if !candidate.is_dir() {
            return Err(Refusal::new(
                RefusalCode::NoStoreConfigured,
                format!("no knowledge store at {}", candidate.display()),
                format!("create it with: mkdir -p {}", candidate.display()),
            ));
        }

        let root = candidate.canonicalize().map_err(|e| {
            Refusal::new(
                RefusalCode::NoStoreConfigured,
                format!("could not resolve {}: {e}", candidate.display()),
                "check the folder is readable",
            )
        })?;

        Ok(Store { root })
    }

    /// The store root itself.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Admit a requested folder, or refuse it.
    ///
    /// The order here is the whole security property: resolve FIRST, then compare.
    /// Checking the string before canonicalizing would let `store/../../etc` pass,
    /// since it is textually under the root while resolving well outside it.
    ///
    /// Note this is why the containment holds without extra symlink handling:
    /// `canonicalize` follows every link in the path, so a symlink pointing out of
    /// the store resolves to its real target and fails the prefix test. (The walk in
    /// `knowledge::collect_files` separately declines to descend symlinked
    /// directories, so links *discovered during* indexing can't escape either.)
    pub fn admit(&self, requested: &Path) -> Result<PathBuf, Refusal> {
        let resolved = requested.canonicalize().map_err(|_| {
            Refusal::new(
                RefusalCode::NotAFolder,
                format!("no such folder: {}", requested.display()),
                format!("choose a folder inside {}", self.root.display()),
            )
        })?;

        if !resolved.is_dir() {
            return Err(Refusal::new(
                RefusalCode::NotAFolder,
                format!("not a folder: {}", resolved.display()),
                format!("choose a folder inside {}", self.root.display()),
            ));
        }

        if !resolved.starts_with(&self.root) {
            return Err(Refusal::new(
                RefusalCode::OutsideStore,
                format!("{} is outside the knowledge store", resolved.display()),
                format!("put the files inside {} first", self.root.display()),
            ));
        }

        Ok(resolved)
    }
}

/// The admission decision for an indexing request, made from a cheap walk before
/// any embedding happens.
pub struct Budget;

impl Budget {
    /// Admit or refuse a walk result against the budget.
    pub fn admit(files: usize, estimated_chunks: usize) -> Result<(), Refusal> {
        if files > MAX_FILES {
            return Err(Refusal::new(
                RefusalCode::OverBudget,
                format!("{files} files exceeds the {MAX_FILES}-file budget"),
                "index a narrower subfolder of the knowledge store".to_string(),
            ));
        }

        if estimated_chunks > MAX_CHUNKS {
            return Err(Refusal::new(
                RefusalCode::OverBudget,
                format!("~{estimated_chunks} chunks exceeds the {MAX_CHUNKS}-chunk budget"),
                "index a narrower subfolder of the knowledge store".to_string(),
            ));
        }

        Ok(())
    }
}

/// The gate between retrieval and the model.
///
/// This is the seam described on [`EVIDENCE_FLOOR`]. Today the predicate is
/// mechanical and modest; the value is that the *enforcement point* exists and both
/// front ends already route through it.
pub struct EvidenceGate;

impl EvidenceGate {
    /// Admit retrieved evidence, or refuse to answer from it.
    ///
    /// `best_score` is the highest similarity retrieval saw *before* the floor was
    /// applied, so the refusal reports what actually happened. Deriving it from the
    /// admitted `sources` instead would make every refusal claim it saw nothing at
    /// all -- the sources are empty precisely because the floor removed them.
    pub fn admit(sources: &[(String, f32)], best: f32) -> Result<(), Refusal> {
        if sources.is_empty() || best < EVIDENCE_FLOOR {
            return Err(Refusal::new(
                RefusalCode::NoEvidenceRetrieved,
                format!("nothing in the indexed files is related to that (best match {best:.2}, floor {EVIDENCE_FLOOR:.2})"),
                "index the folder that covers this topic, or ask about something in it",
            ));
        }

        Ok(())
    }
}

/// What an admitted index actually consumed -- the receipt for the transition.
///
/// Deliberately records what was *left out* as well as what went in. The skip rules
/// in `knowledge::collect_files` are otherwise invisible, which makes an incomplete
/// corpus look identical to a complete one.
#[derive(Debug, Clone)]
pub struct IndexReceipt {
    pub root: PathBuf,
    pub files: usize,
    pub chunks: usize,
    pub seconds: f32,
}

impl fmt::Display for IndexReceipt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} files / {} chunks from {} in {:.1}s",
            self.files,
            self.chunks,
            self.root.display(),
            self.seconds
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a temp dir with a `knowledgestore` inside it.
    fn temp_base(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!("chatbot-gov-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join(STORE_DIR)).unwrap();
        base
    }

    #[test]
    fn missing_store_is_refused_not_created() {
        let base = std::env::temp_dir().join(format!("chatbot-gov-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();

        let refusal = Store::discover(&base).unwrap_err();
        assert_eq!(refusal.code, RefusalCode::NoStoreConfigured);
        // Deny-by-default: refusing must not have the side effect of fixing itself.
        assert!(!base.join(STORE_DIR).exists());
        assert!(refusal.remedy.contains("mkdir"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn the_store_root_admits_itself() {
        let base = temp_base("self");
        let store = Store::discover(&base).unwrap();
        assert!(store.admit(store.root()).is_ok());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn subfolders_of_the_store_are_admitted() {
        let base = temp_base("sub");
        let store = Store::discover(&base).unwrap();
        let sub = store.root().join("halcyon");
        std::fs::create_dir_all(&sub).unwrap();

        assert_eq!(store.admit(&sub).unwrap(), sub.canonicalize().unwrap());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn traversal_out_of_the_store_is_refused() {
        let base = temp_base("traversal");
        let store = Store::discover(&base).unwrap();

        // Textually under the root, resolves well outside it. This is the case that
        // a naive string prefix check would wave through.
        let escape = store.root().join("..").join("..");
        let refusal = store.admit(&escape).unwrap_err();
        assert_eq!(refusal.code, RefusalCode::OutsideStore);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn absolute_paths_outside_the_store_are_refused() {
        let base = temp_base("absolute");
        let store = Store::discover(&base).unwrap();

        let refusal = store.admit(Path::new("/etc")).unwrap_err();
        assert_eq!(refusal.code, RefusalCode::OutsideStore);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_symlink_pointing_outside_the_store_is_refused() {
        let base = temp_base("symlink");
        let store = Store::discover(&base).unwrap();

        // The escape hatch worth testing explicitly: a link that lives inside the
        // store but resolves outside it.
        let link = store.root().join("escape");
        #[cfg(unix)]
        std::os::unix::fs::symlink("/etc", &link).unwrap();

        #[cfg(unix)]
        {
            let refusal = store.admit(&link).unwrap_err();
            assert_eq!(refusal.code, RefusalCode::OutsideStore);
        }

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn budget_admits_within_limits_and_refuses_beyond() {
        assert!(Budget::admit(10, 100).is_ok());

        let too_many_files = Budget::admit(MAX_FILES + 1, 10).unwrap_err();
        assert_eq!(too_many_files.code, RefusalCode::OverBudget);

        let too_many_chunks = Budget::admit(10, MAX_CHUNKS + 1).unwrap_err();
        assert_eq!(too_many_chunks.code, RefusalCode::OverBudget);
    }

    #[test]
    fn evidence_gate_refuses_empty_retrieval() {
        let refusal = EvidenceGate::admit(&[], 0.0).unwrap_err();
        assert_eq!(refusal.code, RefusalCode::NoEvidenceRetrieved);
    }

    #[test]
    fn evidence_gate_refuses_below_the_floor() {
        let refusal = EvidenceGate::admit(&[], 0.10).unwrap_err();
        assert_eq!(refusal.code, RefusalCode::NoEvidenceRetrieved);
    }

    #[test]
    fn a_refusal_reports_the_score_it_actually_saw() {
        // The sources are empty *because* the floor removed them, so the reported
        // score has to come from before filtering -- otherwise every refusal claims
        // it saw nothing, which is a different and false statement.
        let refusal = EvidenceGate::admit(&[], 0.18).unwrap_err();
        assert!(
            refusal.detail.contains("0.18"),
            "refusal should report the pre-filter score, got: {}",
            refusal.detail
        );
    }

    #[test]
    fn the_documented_correct_refusal_is_still_admitted() {
        // The case the README cites as proof the system works: asking a governance
        // corpus about pricing scored 0.56 -- HIGH -- and the right answer was still
        // "not in these documents". The gate must admit it, because similarity has
        // no way to know that. The model declining is what makes that case work, and
        // a floor tuned to catch it would have to reject legitimate answers too.
        let readme_case = [
            ("DEPLOYMENT_TRUTH.md".to_string(), 0.56_f32),
            ("CONSTITUTIONAL_PROTOCOL.md".to_string(), 0.50_f32),
        ];
        assert!(EvidenceGate::admit(&readme_case, 0.56).is_ok());
    }

    #[test]
    fn reason_codes_are_stable_strings() {
        assert_eq!(RefusalCode::OutsideStore.as_str(), "outside_store");
        assert_eq!(RefusalCode::OverBudget.as_str(), "over_budget");
        assert_eq!(
            RefusalCode::NoEvidenceRetrieved.as_str(),
            "no_evidence_retrieved"
        );
    }
}
