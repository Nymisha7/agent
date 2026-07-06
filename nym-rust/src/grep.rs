use anyhow::{Context, Result};
use ignore::WalkBuilder;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::glob::glob_matches;

#[derive(Debug, Clone)]
pub struct GrepOptions {
    pub pattern: String,
    pub root: PathBuf,
    pub include: Option<String>,
    pub limit: usize,
    pub literal_text: bool,
    pub include_hidden: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct GrepMatch {
    pub path: PathBuf,
    pub line_number: usize,
    pub line: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GrepResult {
    pub matches: Vec<GrepMatch>,
    pub truncated: bool,
    pub backend: String,
}

pub fn grep_files(options: GrepOptions) -> Result<GrepResult> {
    let pattern = options.pattern.trim();
    if pattern.is_empty() {
        anyhow::bail!("grep pattern is empty");
    }

    let limit = options.limit.max(1);
    let root = normalize_root(&options.root)?;

    match rg_grep_matches(&root, pattern, &options) {
        Ok(mut matches) if !matches.is_empty() => {
            let truncated = matches.len() > limit;
            matches.truncate(limit);
            return Ok(GrepResult {
                matches,
                truncated,
                backend: "ripgrep".to_string(),
            });
        }
        Ok(_) => {}
        Err(_) => {}
    }

    let mut matches = recursive_grep_matches(&root, pattern, &options)?;
    let truncated = matches.len() > limit;
    matches.truncate(limit);

    Ok(GrepResult {
        matches,
        truncated,
        backend: "recursive_walk".to_string(),
    })
}

fn normalize_root(root: &Path) -> Result<PathBuf> {
    let root = expand_tilde(root);
    root.canonicalize()
        .with_context(|| format!("Invalid grep root: {}", root.display()))
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

fn rg_grep_matches(root: &Path, pattern: &str, options: &GrepOptions) -> Result<Vec<GrepMatch>> {
    let mut command = Command::new("rg");
    command.arg("--line-number");
    command.arg("--with-filename");
    command.arg("--no-heading");
    command.arg("--no-messages");
    if options.include_hidden {
        command.arg("--hidden");
    }
    if options.literal_text {
        command.arg("--fixed-strings");
    }
    if let Some(include) = &options.include {
        command.arg("--glob");
        command.arg(include);
    }
    command.arg(pattern);
    command.arg(root);

    let output = command.output().context("Failed to run ripgrep for content search")?;
    if !output.status.success() && output.status.code() != Some(1) {
        anyhow::bail!(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let mut matches = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let mut parts = line.splitn(3, ':');
        let path = parts.next().unwrap_or_default();
        let line_number = parts.next().unwrap_or_default();
        let text = parts.next().unwrap_or_default();

        let line_number = line_number.parse::<usize>().unwrap_or(1);
        let path = root.join(path).canonicalize().unwrap_or_else(|_| root.join(path));

        matches.push(GrepMatch {
            path,
            line_number,
            line: text.to_string(),
        });
    }

    Ok(matches)
}

fn recursive_grep_matches(root: &Path, pattern: &str, options: &GrepOptions) -> Result<Vec<GrepMatch>> {
    let walker = WalkBuilder::new(root)
        .hidden(!options.include_hidden)
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
        if path == root || !path.is_file() {
            continue;
        }

        let relative = match path.strip_prefix(root) {
            Ok(relative) => relative,
            Err(_) => continue,
        };

        if let Some(include) = &options.include {
            let relative_text = relative.to_string_lossy().replace('\\', "/");
            if !glob_matches(include, &relative_text) {
                continue;
            }
        }

        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(_) => continue,
        };

        for (index, line) in content.lines().enumerate() {
            let is_match = if options.literal_text {
                line.contains(pattern)
            } else {
                line.contains(pattern)
            };

            if is_match {
                matches.push(GrepMatch {
                    path: path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
                    line_number: index + 1,
                    line: line.to_string(),
                });
            }
        }
    }

    Ok(matches)
}
