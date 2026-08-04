use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::{write_file, WriteFileOptions, WriteFileResult};

const NO_MATCH_PREVIEW_LINES: usize = 20;

#[derive(Debug, Clone)]
pub struct EditFileOptions {
    pub path: PathBuf,
    pub workspace_root: PathBuf,
    pub old_text: String,
    pub new_text: String,
    pub replace_all: bool,
    pub expected_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EditFileResult {
    pub path: PathBuf,
    pub resource: String,
    pub replaced: bool,
    pub occurrences: usize,
    pub bytes_written: usize,
    pub line_count: usize,
    pub before_sha256: String,
    pub after_sha256: String,
    pub line_ending: String,
}

pub fn edit_file(options: EditFileOptions) -> Result<EditFileResult> {
    if options.old_text.is_empty() {
        bail!("old_text must not be empty");
    }

    let workspace_root = options.workspace_root.canonicalize().with_context(|| {
        format!(
            "Invalid workspace root: {}",
            options.workspace_root.display()
        )
    })?;
    let target = resolve_target_path(&options.path, &workspace_root)?;
    let resource = relative_display(&target, &workspace_root);

    let metadata = fs::symlink_metadata(&target).with_context(|| {
        format!(
            "Path does not exist or cannot be inspected: {}",
            target.display()
        )
    })?;
    if metadata.file_type().is_symlink() {
        bail!("Refusing to edit symlink target: {}", target.display());
    }
    if metadata.is_dir() {
        bail!("Path is a directory: {}", target.display());
    }

    let bytes =
        fs::read(&target).with_context(|| format!("Failed to read file: {}", target.display()))?;
    let before_sha256 = sha256_hex(&bytes);
    if let Some(expected_sha256) = options.expected_sha256.as_deref() {
        if before_sha256 != expected_sha256 {
            bail!(
                "File changed since it was read: expected {}, found {}",
                expected_sha256,
                before_sha256
            );
        }
    }

    let original = String::from_utf8(bytes)
        .with_context(|| format!("File is not valid UTF-8: {}", target.display()))?;
    let matches = original.matches(&options.old_text).count();

    let updated = if options.replace_all {
        if matches == 0 {
            return Err(anyhow::Error::msg(
                string_replace(&original, &options.old_text, &options.new_text)
                    .expect_err("zero matches must return an error"),
            ));
        }
        original.replace(&options.old_text, &options.new_text)
    } else {
        string_replace(&original, &options.old_text, &options.new_text)
            .map_err(anyhow::Error::msg)?
    };

    let result: WriteFileResult = write_file(WriteFileOptions {
        path: target,
        workspace_root,
        content: updated,
        create_dirs: false,
        overwrite: true,
        preserve_line_endings: true,
        expected_sha256: Some(before_sha256.clone()),
    })?;

    Ok(EditFileResult {
        path: result.path,
        resource,
        replaced: true,
        occurrences: if options.replace_all { matches } else { 1 },
        bytes_written: result.bytes_written,
        line_count: result.line_count,
        before_sha256,
        after_sha256: result.after_sha256,
        line_ending: result.line_ending,
    })
}

fn resolve_target_path(path: &Path, workspace_root: &Path) -> Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    };
    let candidate = normalize_lexical_path(&candidate);

    let parent = candidate
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Path has no parent: {}", candidate.display()))?;
    let canonical_parent = parent
        .canonicalize()
        .with_context(|| format!("Failed to resolve parent path: {}", parent.display()))?;
    ensure_within_workspace(&canonical_parent, workspace_root)?;

    let suffix = candidate
        .strip_prefix(parent)
        .unwrap_or_else(|_| Path::new(""));
    let resolved = normalize_lexical_path(&canonical_parent.join(suffix));
    ensure_within_workspace(&resolved, workspace_root)?;
    Ok(resolved)
}

fn ensure_within_workspace(path: &Path, workspace_root: &Path) -> Result<()> {
    path.strip_prefix(workspace_root).map(|_| ()).map_err(|_| {
        anyhow::anyhow!(
            "Path '{}' is outside workspace root '{}'",
            path.display(),
            workspace_root.display()
        )
    })
}

fn normalize_lexical_path(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = normalized.pop();
            }
            Component::RootDir | Component::Prefix(_) => normalized.push(component.as_os_str()),
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

fn relative_display(path: &Path, workspace_root: &Path) -> String {
    path.strip_prefix(workspace_root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02x}", byte);
    }
    out
}

fn string_replace(content: &str, before: &str, after: &str) -> std::result::Result<String, String> {
    let mut matches = content.match_indices(before);
    let Some((first_position, _)) = matches.next() else {
        let suggestion = find_similar_context(content, before);
        let mut message = "No match found for the specified text.".to_string();
        if let Some(hint) = suggestion {
            message.push_str(&format!("\n\nDid you mean:\n```\n{hint}\n```"));
        }
        let preview = build_file_preview(content, NO_MATCH_PREVIEW_LINES);
        message.push_str(&format!("\n\nFile preview:\n```\n{preview}\n```"));
        return Err(message);
    };
    let Some((second_position, _)) = matches.next() else {
        return Ok(content.replacen(before, after, 1));
    };
    let count = 2 + matches.count();
    let mut message =
        format!("Found {count} matches. Please provide more context to identify a unique match:\n");
    for (index, position) in [first_position, second_position].into_iter().enumerate() {
        let line_number = count_lines_before(content, position);
        let context = get_line_context(content, line_number, 1);
        message.push_str(&format!(
            "\nMatch {} (line {}):\n```\n{}\n```",
            index + 1,
            line_number,
            context
        ));
    }
    if count > 2 {
        message.push_str(&format!("\n\n...and {} more", count - 2));
    }
    Err(message)
}

fn count_lines_before(content: &str, byte_position: usize) -> usize {
    content
        .char_indices()
        .take_while(|(index, _)| *index < byte_position)
        .filter(|(_, character)| *character == '\n')
        .count()
        + 1
}

fn get_line_context(content: &str, target_line: usize, context: usize) -> String {
    let start = target_line.saturating_sub(context + 1);
    let count = target_line.saturating_add(context).saturating_sub(start);
    let mut output = String::new();
    for line in content.lines().skip(start).take(count) {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(line);
    }
    output
}

fn find_similar_context(content: &str, search: &str) -> Option<String> {
    let first_line = search.lines().next()?.trim();
    if first_line.is_empty() {
        return None;
    }

    for (index, line) in content.lines().enumerate() {
        if line.contains(first_line) || first_line.contains(line.trim()) {
            return Some(get_line_context(content, index + 1, 2));
        }
    }
    None
}

fn build_file_preview(content: &str, max_lines: usize) -> String {
    if content.is_empty() {
        return "(file is empty)".to_string();
    }

    use std::fmt::Write as _;

    let mut preview = String::new();
    let mut total_lines = 0;
    for (index, line) in content.lines().enumerate() {
        if index < max_lines {
            if !preview.is_empty() {
                preview.push('\n');
            }
            let _ = write!(&mut preview, "{:>4}: {}", index + 1, line);
        }
        total_lines = index + 1;
    }
    if total_lines > max_lines {
        let _ = write!(
            &mut preview,
            "\n... ({} more lines)",
            total_lines - max_lines
        );
    }
    preview
}

#[cfg(test)]
mod tests {
    use super::string_replace;

    #[test]
    fn edit_diagnostics_show_context_for_ambiguous_matches() {
        let error =
            string_replace("foo\nbar\nfoo\n", "foo", "baz").expect_err("ambiguous edit must fail");

        assert!(error.contains("Found 2 matches"));
        assert!(error.contains("Match 1 (line 1)"));
        assert!(error.contains("Match 2 (line 3)"));
    }

    #[test]
    fn edit_diagnostics_preview_file_when_text_is_missing() {
        let error =
            string_replace("alpha\nbeta\n", "gamma", "delta").expect_err("missing edit must fail");

        assert!(error.contains("No match found"));
        assert!(error.contains("File preview:"));
        assert!(error.contains("1: alpha"));
    }
}
