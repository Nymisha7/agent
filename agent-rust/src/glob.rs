use anyhow::{bail, Context, Result};
use globset::GlobBuilder;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};

use crate::ripgrep::{ripgrep_paths, RipgrepFilesOptions, RipgrepPathKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GlobKind {
    Any,
    File,
    Directory,
}

#[derive(Debug, Clone)]
pub struct GlobOptions {
    pub pattern: String,
    pub root: PathBuf,
    pub limit: usize,
    pub include_hidden: bool,
    pub include_generated: bool,
    pub kind: GlobKind,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlobMatch {
    pub path: PathBuf,
    pub kind: GlobKind,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlobResult {
    pub matches: Vec<GlobMatch>,
    pub truncated: bool,
    pub backend: String,
}

pub fn glob_files(options: GlobOptions) -> Result<GlobResult> {
    let pattern = options.pattern.trim();
    if pattern.is_empty() {
        bail!("glob pattern is empty");
    }

    let root = normalize_root(&options.root)?;
    if !root.is_dir() {
        bail!("Glob root is not a directory: {}", root.display());
    }

    let matcher = GlobBuilder::new(pattern)
        .literal_separator(true)
        .backslash_escape(false)
        .build()
        .with_context(|| format!("Invalid glob pattern: {pattern}"))?
        .compile_matcher();

    let limit = options.limit.max(1);
    let mut matches = BinaryHeap::new();
    let mut truncated = false;
    for item in ripgrep_paths(
        &root,
        RipgrepFilesOptions {
            include_hidden: options.include_hidden,
            follow_links: false,
            include_ignored: options.include_generated,
            threads: None,
        },
    )? {
        let kind = match item.kind {
            RipgrepPathKind::File => GlobKind::File,
            RipgrepPathKind::Directory => GlobKind::Directory,
        };
        if !matches_kind(options.kind, kind) || !matcher.is_match(&item.relative) {
            continue;
        }
        let candidate = RankedGlobMatch(GlobMatch {
            path: root.join(item.relative),
            kind,
        });
        if matches.len() < limit {
            matches.push(candidate);
        } else {
            truncated = true;
            if matches.peek().is_some_and(|worst| candidate < *worst) {
                matches.pop();
                matches.push(candidate);
            }
        }
    }
    let mut matches = matches.into_vec();
    matches.sort();
    let matches = matches.into_iter().map(|item| item.0).collect();

    Ok(GlobResult {
        matches,
        truncated,
        backend: "ripgrep+globset".to_string(),
    })
}

#[derive(Debug)]
struct RankedGlobMatch(GlobMatch);

impl Ord for RankedGlobMatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .path
            .cmp(&other.0.path)
            .then_with(|| glob_kind_rank(self.0.kind).cmp(&glob_kind_rank(other.0.kind)))
    }
}

impl PartialOrd for RankedGlobMatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RankedGlobMatch {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for RankedGlobMatch {}

fn glob_kind_rank(kind: GlobKind) -> u8 {
    match kind {
        GlobKind::Any => 0,
        GlobKind::File => 1,
        GlobKind::Directory => 2,
    }
}

fn normalize_root(root: &Path) -> Result<PathBuf> {
    expand_tilde(root)
        .canonicalize()
        .with_context(|| format!("Invalid glob root: {}", root.display()))
}

fn expand_tilde(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    path.to_path_buf()
}

fn matches_kind(expected: GlobKind, actual: GlobKind) -> bool {
    expected == GlobKind::Any || expected == actual
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn recursive_glob_uses_globset_over_ripgrep_inventory() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agent-glob-{suffix}"));
        fs::create_dir_all(root.join("src/nested")).expect("create fixture");
        fs::write(root.join("src/nested/lib.rs"), "pub fn value() {}\n").expect("write source");
        fs::write(root.join("src/readme.md"), "docs\n").expect("write docs");

        let result = glob_files(GlobOptions {
            pattern: "**/*.rs".to_string(),
            root: root.clone(),
            limit: 10,
            include_hidden: false,
            include_generated: false,
            kind: GlobKind::File,
        })
        .expect("glob");

        assert_eq!(result.backend, "ripgrep+globset");
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].path, root.join("src/nested/lib.rs"));
        fs::remove_dir_all(root).expect("clean fixture");
    }
}
