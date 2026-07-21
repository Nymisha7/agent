use anyhow::{bail, Context, Result};
use ignore::WalkBuilder;
use serde::Serialize;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

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

    let limit = options.limit.max(1);
    let root = normalize_root(&options.root)?;

    let mut matches = Vec::new();
    let mut backend = "ripgrep".to_string();

    if options.kind != GlobKind::Directory {
        if let Ok(rg_matches) =
            rg_glob_matches(&root, pattern, options.include_hidden, options.kind)
        {
            matches.extend(rg_matches);
        }
    }

    if matches.is_empty() || options.kind == GlobKind::Directory {
        backend = "recursive_glob".to_string();
        matches = recursive_glob_matches(&root, pattern, options.include_hidden, options.kind)?;
    }

    dedupe_matches(&mut matches);
    matches.sort_by(|a, b| a.path.cmp(&b.path));

    let truncated = matches.len() > limit;
    matches.truncate(limit);

    Ok(GlobResult {
        matches,
        truncated,
        backend,
    })
}

fn normalize_root(root: &Path) -> Result<PathBuf> {
    let root = expand_tilde(root);
    root.canonicalize()
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

fn rg_glob_matches(
    root: &Path,
    pattern: &str,
    include_hidden: bool,
    kind: GlobKind,
) -> Result<Vec<GlobMatch>> {
    let mut command = Command::new("rg");
    command.arg("--files");
    command.arg("--no-messages");
    if include_hidden {
        command.arg("--hidden");
    }
    command.arg("--glob");
    command.arg(pattern);
    command.arg(root);

    let output = command
        .output()
        .context("Failed to run ripgrep for glob discovery")?;

    if !output.status.success() && output.status.code() != Some(1) {
        bail!(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let mut matches = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let candidate = root.join(line);
        let resolved = candidate.canonicalize().unwrap_or(candidate);
        if matches_kind(&resolved, kind) && glob_matches(pattern, line) {
            let kind = path_kind(&resolved);
            matches.push(GlobMatch {
                path: resolved,
                kind,
            });
        }
    }

    Ok(matches)
}

fn recursive_glob_matches(
    root: &Path,
    pattern: &str,
    include_hidden: bool,
    kind: GlobKind,
) -> Result<Vec<GlobMatch>> {
    let walker = WalkBuilder::new(root)
        .hidden(!include_hidden)
        .follow_links(false)
        .git_ignore(false)
        .git_exclude(false)
        .parents(false)
        .build();

    let mut matches = Vec::new();
    for entry in walker {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };

        let path = entry.path();
        if path == root {
            continue;
        }

        let relative = match path.strip_prefix(root) {
            Ok(relative) => relative,
            Err(_) => continue,
        };

        let relative_text = relative.to_string_lossy().replace('\\', "/");
        if !glob_matches(pattern, &relative_text) {
            continue;
        }

        let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if !matches_kind(&resolved, kind) {
            continue;
        }

        matches.push(GlobMatch {
            path: resolved,
            kind: path_kind(path),
        });
    }

    Ok(matches)
}

fn dedupe_matches(matches: &mut Vec<GlobMatch>) {
    let mut seen = HashSet::new();
    matches.retain(|item| seen.insert(item.path.clone()));
}

fn matches_kind(path: &Path, kind: GlobKind) -> bool {
    match kind {
        GlobKind::Any => true,
        GlobKind::File => path.is_file(),
        GlobKind::Directory => path.is_dir(),
    }
}

fn path_kind(path: &Path) -> GlobKind {
    if path.is_dir() {
        GlobKind::Directory
    } else {
        GlobKind::File
    }
}

pub(crate) fn glob_matches(pattern: &str, candidate: &str) -> bool {
    let pattern = normalize_pattern(pattern);
    let candidate = normalize_candidate(candidate);

    let pattern_segments: Vec<&str> = pattern.split('/').collect();
    let candidate_segments: Vec<&str> = if candidate.is_empty() {
        Vec::new()
    } else {
        candidate.split('/').collect()
    };

    match_segments(&pattern_segments, 0, &candidate_segments, 0)
}

fn normalize_pattern(pattern: &str) -> String {
    pattern.replace('\\', "/")
}

fn normalize_candidate(candidate: &str) -> String {
    candidate.replace('\\', "/")
}

fn match_segments(pattern: &[&str], pi: usize, candidate: &[&str], ci: usize) -> bool {
    if pi == pattern.len() {
        return ci == candidate.len();
    }

    if pattern[pi] == "**" {
        let mut idx = ci;
        while idx <= candidate.len() {
            if match_segments(pattern, pi + 1, candidate, idx) {
                return true;
            }
            idx += 1;
        }
        return false;
    }

    if ci >= candidate.len() {
        return false;
    }

    if !match_component(pattern[pi], candidate[ci]) {
        return false;
    }

    match_segments(pattern, pi + 1, candidate, ci + 1)
}

fn match_component(pattern: &str, candidate: &str) -> bool {
    let pattern = pattern.as_bytes();
    let candidate = candidate.as_bytes();

    let mut p = 0;
    let mut c = 0;
    let mut star: Option<usize> = None;
    let mut match_idx = 0;

    while c < candidate.len() {
        if p < pattern.len() && (pattern[p] == candidate[c] || pattern[p] == b'?') {
            p += 1;
            c += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = Some(p);
            p += 1;
            match_idx = c;
        } else if let Some(star_idx) = star {
            p = star_idx + 1;
            match_idx += 1;
            c = match_idx;
        } else {
            return false;
        }
    }

    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }

    p == pattern.len()
}
