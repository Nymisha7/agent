use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

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
        bail!("grep pattern is empty");
    }

    let root = normalize_root(&options.root)?;
    if !root.is_dir() {
        bail!("Grep root is not a directory: {}", root.display());
    }

    let limit = options.limit.max(1);
    let mut command = Command::new("rg");
    command
        .current_dir(&root)
        .arg("--json")
        .arg("--line-number")
        .arg("--no-messages")
        .arg("--max-count")
        .arg(limit.saturating_add(1).to_string());

    if options.include_hidden {
        command.arg("--hidden");
    }
    if options.literal_text {
        command.arg("--fixed-strings");
    }
    if let Some(include) = &options.include {
        command.arg("--glob").arg(include);
    }
    command.arg(pattern).arg(".");

    let output = command
        .output()
        .context("Failed to run ripgrep for content search")?;
    if !output.status.success() && output.status.code() != Some(1) {
        bail!(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    let mut matches = Vec::new();
    for line in output.stdout.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        let Ok(event) = serde_json::from_slice::<Value>(line) else {
            continue;
        };
        if event.get("type").and_then(Value::as_str) != Some("match") {
            continue;
        }
        let Some(data) = event.get("data") else {
            continue;
        };
        let Some(path) = data
            .pointer("/path/text")
            .and_then(Value::as_str)
            .map(PathBuf::from)
        else {
            continue;
        };
        let Some(line_number) = data.get("line_number").and_then(Value::as_u64) else {
            continue;
        };
        let Some(text) = data.pointer("/lines/text").and_then(Value::as_str) else {
            continue;
        };

        matches.push(GrepMatch {
            path: root.join(strip_dot_prefix(&path)),
            line_number: line_number as usize,
            line: text.trim_end_matches(['\r', '\n']).to_string(),
        });
        if matches.len() > limit {
            break;
        }
    }

    let truncated = matches.len() > limit;
    matches.truncate(limit);
    Ok(GrepResult {
        matches,
        truncated,
        backend: "ripgrep".to_string(),
    })
}

fn normalize_root(root: &Path) -> Result<PathBuf> {
    expand_tilde(root)
        .canonicalize()
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

fn strip_dot_prefix(path: &Path) -> &Path {
    path.strip_prefix(".").unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn content_search_preserves_regex_semantics() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agent-grep-{suffix}"));
        fs::create_dir_all(&root).expect("create fixture");
        fs::write(root.join("sample.rs"), "alpha 42\nalpha forty-two\n").expect("write source");

        let result = grep_files(GrepOptions {
            pattern: r"alpha \d+".to_string(),
            root: root.clone(),
            include: Some("*.rs".to_string()),
            limit: 10,
            literal_text: false,
            include_hidden: false,
        })
        .expect("grep");

        assert_eq!(result.backend, "ripgrep");
        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].line_number, 1);
        assert_eq!(result.matches[0].line, "alpha 42");
        fs::remove_dir_all(root).expect("clean fixture");
    }
}
