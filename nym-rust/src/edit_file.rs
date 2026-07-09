use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::{write_file, WriteFileOptions, WriteFileResult};

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
    if matches == 0 {
        bail!("Target text not found in {}", target.display());
    }
    if !options.replace_all && matches > 1 {
        bail!(
            "Target text appears {} times in {}; set replace_all=true or provide a more specific match",
            matches,
            target.display()
        );
    }

    let updated = if options.replace_all {
        original.replace(&options.old_text, &options.new_text)
    } else {
        original.replacen(&options.old_text, &options.new_text, 1)
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
