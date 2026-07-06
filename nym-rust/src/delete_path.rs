use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::fs;
use std::path::{Component, Path, PathBuf};

#[derive(Debug, Clone)]
pub struct DeletePathOptions {
    pub path: PathBuf,
    pub workspace_root: PathBuf,
    pub recursive: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeletePathResult {
    pub path: PathBuf,
    pub resource: String,
    pub deleted: bool,
    pub kind: String,
}

pub fn delete_path(options: DeletePathOptions) -> Result<DeletePathResult> {
    let workspace_root = options
        .workspace_root
        .canonicalize()
        .with_context(|| format!("Invalid workspace root: {}", options.workspace_root.display()))?;
    let target = resolve_target_path(&options.path, &workspace_root)?;
    let resource = relative_display(&target, &workspace_root);

    let metadata = fs::symlink_metadata(&target)
        .with_context(|| format!("Path does not exist or cannot be inspected: {}", target.display()))?;

    if metadata.file_type().is_symlink() {
        bail!("Refusing to delete symlink target: {}", target.display());
    }

    let kind = if metadata.is_dir() { "directory" } else { "file" };

    if metadata.is_dir() {
        let mut entries = fs::read_dir(&target)
            .with_context(|| format!("Failed to read directory: {}", target.display()))?;
        if !options.recursive && entries.next().is_some() {
            bail!("Directory is not empty: {}", target.display());
        }
        fs::remove_dir_all(&target)
            .with_context(|| format!("Failed to remove directory: {}", target.display()))?;
    } else {
        fs::remove_file(&target)
            .with_context(|| format!("Failed to remove file: {}", target.display()))?;
    }

    Ok(DeletePathResult {
        path: target.clone(),
        resource,
        deleted: true,
        kind: kind.to_string(),
    })
}

fn resolve_target_path(path: &Path, workspace_root: &Path) -> Result<PathBuf> {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    };
    let candidate = normalize_lexical_path(&candidate);

    if candidate.exists() || fs::symlink_metadata(&candidate).is_ok() {
        let parent = candidate
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Path has no parent: {}", candidate.display()))?;
        let canonical_parent = parent
            .canonicalize()
            .with_context(|| format!("Failed to resolve parent path: {}", parent.display()))?;
        ensure_within_workspace(&canonical_parent, workspace_root)?;

        let suffix = candidate.strip_prefix(parent).unwrap_or_else(|_| Path::new(""));
        let resolved = normalize_lexical_path(&canonical_parent.join(suffix));
        ensure_within_workspace(&resolved, workspace_root)?;
        return Ok(resolved);
    }

    let mut current = candidate.as_path();
    while let Some(parent) = current.parent() {
        if parent.exists() {
            let canonical_parent = parent
                .canonicalize()
                .with_context(|| format!("Failed to resolve parent path: {}", parent.display()))?;
            ensure_within_workspace(&canonical_parent, workspace_root)?;

            let suffix = candidate.strip_prefix(parent).unwrap_or_else(|_| Path::new(""));
            let resolved = normalize_lexical_path(&canonical_parent.join(suffix));
            ensure_within_workspace(&resolved, workspace_root)?;
            return Ok(resolved);
        }
        current = parent;
    }

    bail!("Could not resolve path: {}", path.display())
}

fn ensure_within_workspace(path: &Path, workspace_root: &Path) -> Result<()> {
    path.strip_prefix(workspace_root)
        .map(|_| ())
        .map_err(|_| anyhow::anyhow!("Path '{}' is outside workspace root '{}'", path.display(), workspace_root.display()))
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
