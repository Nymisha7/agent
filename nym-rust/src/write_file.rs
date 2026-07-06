use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct WriteFileOptions {
    pub path: PathBuf,
    pub workspace_root: PathBuf,
    pub content: String,
    pub create_dirs: bool,
    pub overwrite: bool,
    pub preserve_line_endings: bool,
    pub expected_sha256: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WriteFileResult {
    pub path: PathBuf,
    pub resource: String,
    pub created: bool,
    pub bytes_written: usize,
    pub line_count: usize,
    pub before_sha256: Option<String>,
    pub after_sha256: String,
    pub line_ending: String,
}

pub fn write_file(options: WriteFileOptions) -> Result<WriteFileResult> {
    let workspace_root = options
        .workspace_root
        .canonicalize()
        .with_context(|| format!("Invalid workspace root: {}", options.workspace_root.display()))?;
    let target = resolve_target_path(&options.path, &workspace_root)?;
    let resource = relative_display(&target, &workspace_root);

    if target.exists() {
        let metadata = fs::symlink_metadata(&target)
            .with_context(|| format!("Failed to inspect target: {}", target.display()))?;

        if metadata.file_type().is_symlink() {
            bail!("Refusing to write to symlink target: {}", target.display());
        }

        if metadata.is_dir() {
            bail!("Path is a directory: {}", target.display());
        }
    }

    let existing = if target.exists() {
        Some(
            fs::read(&target)
                .with_context(|| format!("Failed to read existing file: {}", target.display()))?,
        )
    } else {
        None
    };

    if let Some(expected_sha256) = options.expected_sha256.as_deref() {
        let current = existing
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Expected file hash provided, but target does not exist: {}", target.display()))?;
        let current_sha256 = sha256_hex(current);
        if current_sha256 != expected_sha256 {
            bail!(
                "File changed since it was read: expected {}, found {}",
                expected_sha256,
                current_sha256
            );
        }
    }

    if existing.is_some() && !options.overwrite {
        bail!("Path already exists: {}", target.display());
    }

    let parent = target
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Path has no parent: {}", target.display()))?;
    if !parent.exists() {
        if options.create_dirs {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create parent directories: {}", parent.display()))?;
        } else {
            bail!("Parent directory does not exist: {}", parent.display());
        }
    }

    let normalized_content = normalize_line_endings(&options.content);
    let existing_has_bom = existing
        .as_ref()
        .map(|bytes| has_utf8_bom(bytes))
        .unwrap_or(false);
    let input_has_bom = normalized_content.starts_with('\u{feff}');
    let content_without_bom = strip_utf8_bom(&normalized_content);
    let line_ending = if options.preserve_line_endings && existing_has_bom {
        detect_line_ending(existing.as_deref().unwrap_or_default())
    } else if options.preserve_line_endings {
        detect_line_ending(existing.as_deref().unwrap_or_default())
    } else {
        "\n"
    };
    let content = if options.preserve_line_endings && existing.is_some() {
        convert_line_endings(content_without_bom, line_ending)
    } else {
        normalized_content
            .strip_prefix('\u{feff}')
            .map(str::to_string)
            .unwrap_or_else(|| normalized_content.clone())
    };
    let final_text = if existing_has_bom || input_has_bom {
        format!("\u{feff}{}", content)
    } else {
        content
    };
    let final_bytes = final_text.as_bytes().to_vec();
    let after_sha256 = sha256_hex(&final_bytes);

    let temp_path = temp_write_path(parent)?;
    write_temp_file(&temp_path, &final_bytes)?;

    if existing.is_some() {
        let _ = fs::remove_file(&target);
    }

    fs::rename(&temp_path, &target).with_context(|| {
        format!(
            "Failed to move temporary file into place: {} -> {}",
            temp_path.display(),
            target.display()
        )
    })?;

    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }

    Ok(WriteFileResult {
        path: target.clone(),
        resource,
        created: existing.is_none(),
        bytes_written: final_bytes.len(),
        line_count: line_count(&final_text),
        before_sha256: existing.as_ref().map(|bytes| sha256_hex(bytes)),
        after_sha256,
        line_ending: line_ending.to_string(),
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
    let resolved_path = path.to_path_buf();
    resolved_path
        .strip_prefix(workspace_root)
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

fn normalize_line_endings(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn convert_line_endings(text: &str, line_ending: &str) -> String {
    if line_ending == "\n" {
        return normalize_line_endings(text);
    }
    normalize_line_endings(text).replace('\n', line_ending)
}

fn detect_line_ending(bytes: &[u8]) -> &'static str {
    if bytes.windows(2).any(|window| window == b"\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

fn strip_utf8_bom(text: &str) -> &str {
    text.strip_prefix('\u{feff}').unwrap_or(text)
}

fn has_utf8_bom(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xef && bytes[1] == 0xbb && bytes[2] == 0xbf
}

fn line_count(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.lines().count()
    }
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

fn temp_write_path(parent: &Path) -> Result<PathBuf> {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for attempt in 0..16 {
        let candidate = parent.join(format!(".nym-write-{pid}-{nanos}-{attempt}.tmp"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }

    bail!("Failed to allocate temporary write path in {}", parent.display())
}

fn write_temp_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("Failed to create temporary file: {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("Failed to write temporary file: {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("Failed to flush temporary file: {}", path.display()))?;
    drop(file);
    Ok(())
}

fn relative_display(path: &Path, workspace_root: &Path) -> String {
    path.strip_prefix(workspace_root)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.to_string_lossy().replace('\\', "/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nym-write-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn cleanup(dir: &Path) {
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn writes_new_file_and_creates_dirs() {
        let dir = temp_dir();
        let result = write_file(WriteFileOptions {
            path: PathBuf::from("src/nested/file.txt"),
            workspace_root: dir.clone(),
            content: "hello\nworld".to_string(),
            create_dirs: true,
            overwrite: true,
            preserve_line_endings: true,
            expected_sha256: None,
        })
        .unwrap();

        assert_eq!(fs::read_to_string(dir.join("src/nested/file.txt")).unwrap(), "hello\nworld");
        assert!(!result.created || result.before_sha256.is_none());
        assert_eq!(result.resource, "src/nested/file.txt");
        cleanup(&dir);
    }

    #[test]
    fn overwrites_existing_file_and_preserves_bom_and_crlf() {
        let dir = temp_dir();
        let path = dir.join("existing.txt");
        fs::write(&path, b"\xef\xbb\xbfone\r\ntwo\r\n").unwrap();

        let result = write_file(WriteFileOptions {
            path: PathBuf::from("existing.txt"),
            workspace_root: dir.clone(),
            content: "alpha\nbeta\n".to_string(),
            create_dirs: true,
            overwrite: true,
            preserve_line_endings: true,
            expected_sha256: None,
        })
        .unwrap();

        assert_eq!(fs::read(&path).unwrap(), b"\xef\xbb\xbfalpha\r\nbeta\r\n");
        assert!(result.created == false);
        assert_eq!(result.line_ending, "\r\n");
        cleanup(&dir);
    }

    #[test]
    fn rejects_hash_mismatch() {
        let dir = temp_dir();
        let path = dir.join("stale.txt");
        fs::write(&path, "current").unwrap();

        let error = write_file(WriteFileOptions {
            path: PathBuf::from("stale.txt"),
            workspace_root: dir.clone(),
            content: "next".to_string(),
            create_dirs: true,
            overwrite: true,
            preserve_line_endings: true,
            expected_sha256: Some("deadbeef".to_string()),
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("File changed since it was read"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "current");
        cleanup(&dir);
    }

    #[test]
    fn rejects_directory_target() {
        let dir = temp_dir();
        fs::create_dir_all(dir.join("folder")).unwrap();

        let error = write_file(WriteFileOptions {
            path: PathBuf::from("folder"),
            workspace_root: dir.clone(),
            content: "nope".to_string(),
            create_dirs: true,
            overwrite: true,
            preserve_line_endings: true,
            expected_sha256: None,
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("directory"));
        cleanup(&dir);
    }

    #[test]
    fn rejects_outside_workspace() {
        let dir = temp_dir();
        let outside = std::env::temp_dir().join(format!(
            "nym-write-outside-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&outside).unwrap();

        let error = write_file(WriteFileOptions {
            path: outside.join("file.txt"),
            workspace_root: dir.clone(),
            content: "nope".to_string(),
            create_dirs: true,
            overwrite: true,
            preserve_line_endings: true,
            expected_sha256: None,
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("outside workspace root"));
        cleanup(&dir);
        cleanup(&outside);
    }

    #[test]
    fn rejects_symlink_target() {
        let dir = temp_dir();
        let target = dir.join("real.txt");
        fs::write(&target, "real").unwrap();
        let link = dir.join("link.txt");
        symlink(&target, &link).unwrap();

        let error = write_file(WriteFileOptions {
            path: PathBuf::from("link.txt"),
            workspace_root: dir.clone(),
            content: "nope".to_string(),
            create_dirs: true,
            overwrite: true,
            preserve_line_endings: true,
            expected_sha256: None,
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("symlink"));
        cleanup(&dir);
    }
}
