use crate::ripgrep::{ripgrep_paths, RipgrepFilesOptions, RipgrepPath, RipgrepPathKind};
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const HIDDEN_DIR_ALLOWLIST: &[&str] = &[
    ".circleci",
    ".config",
    ".github",
    ".gitlab",
    ".idea",
    ".vscode",
];
const SHALLOW_SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".mypy_cache",
    ".packages",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
];
const DEEP_SKIP_DIRS: &[&str] = &[".git", ".hg", ".svn", "__pycache__", "node_modules"];
const TEXT_SUFFIXES: &[&str] = &[
    "cfg",
    "css",
    "csv",
    "env",
    "gitignore",
    "html",
    "ini",
    "js",
    "json",
    "jsx",
    "md",
    "py",
    "rs",
    "sh",
    "sql",
    "toml",
    "ts",
    "tsx",
    "txt",
    "yaml",
    "yml",
];

#[derive(Debug, Clone)]
pub struct InspectTreeOptions {
    pub root: PathBuf,
    pub workspace_root: PathBuf,
    pub max_files: usize,
    pub max_entries: usize,
    pub max_bytes_per_file: usize,
    pub max_total_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectTreeEntry {
    pub path: PathBuf,
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectedTreeFile {
    pub path: PathBuf,
    pub bytes: u64,
    pub truncated: bool,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkippedTreeEntry {
    pub path: PathBuf,
    pub reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectTreeResult {
    pub path: PathBuf,
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub direct_children: Vec<InspectTreeEntry>,
    pub tree: Vec<InspectTreeEntry>,
    pub files: Vec<InspectedTreeFile>,
    pub skipped: Vec<SkippedTreeEntry>,
    pub file_count: usize,
    pub read_file_count: usize,
    pub bytes_read: usize,
    pub truncated: bool,
    pub tree_truncated: bool,
    pub tree_entry_limit: usize,
}

pub fn inspect_tree(options: InspectTreeOptions) -> Result<InspectTreeResult> {
    let root = options
        .root
        .canonicalize()
        .with_context(|| format!("Path does not exist: {}", options.root.display()))?;
    let workspace_root = options
        .workspace_root
        .canonicalize()
        .unwrap_or_else(|_| options.workspace_root.clone());
    if root.is_file() {
        return inspect_single_file(&root, &workspace_root, &options);
    }
    if !root.is_dir() {
        anyhow::bail!(
            "Path is not a regular file or directory: {}",
            root.display()
        );
    }

    let inventory = ripgrep_paths(
        &root,
        RipgrepFilesOptions {
            include_hidden: true,
            follow_links: false,
            include_ignored: false,
            threads: None,
        },
    )?;
    let mut eligible = inventory
        .into_iter()
        .filter(path_allowed)
        .collect::<Vec<_>>();
    eligible.sort_by(entry_order);
    let direct_children = direct_children(&root, &workspace_root, &eligible);
    let tree_truncated = eligible.len() > options.max_entries;
    eligible.truncate(options.max_entries);

    let mut tree = Vec::with_capacity(eligible.len());
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    let mut total_bytes = 0usize;
    let mut truncated = tree_truncated;

    for entry in eligible {
        let absolute = root.join(&entry.relative);
        let display = relative_display(&absolute, &workspace_root);
        if entry.kind == RipgrepPathKind::Directory {
            tree.push(InspectTreeEntry {
                path: display,
                kind: "directory",
                bytes: None,
            });
            continue;
        }

        let size = absolute
            .metadata()
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        tree.push(InspectTreeEntry {
            path: display.clone(),
            kind: "file",
            bytes: Some(size),
        });
        if files.len() >= options.max_files {
            skipped.push(SkippedTreeEntry {
                path: display,
                reason: "max file count reached",
            });
            truncated = true;
            continue;
        }
        if !looks_readable_text_file(&absolute) {
            skipped.push(SkippedTreeEntry {
                path: display,
                reason: "binary or unsupported file type",
            });
            continue;
        }
        let remaining = options.max_total_bytes.saturating_sub(total_bytes);
        if remaining == 0 {
            skipped.push(SkippedTreeEntry {
                path: display,
                reason: "max total content bytes reached",
            });
            truncated = true;
            continue;
        }
        let limit = options.max_bytes_per_file.min(remaining);
        let inspected = read_bounded_text(&absolute, limit)?;
        total_bytes += inspected.bytes_read;
        truncated |= inspected.truncated;
        files.push(InspectedTreeFile {
            path: display,
            bytes: size,
            truncated: inspected.truncated,
            content: inspected.content,
        });
    }

    let file_count = tree.iter().filter(|entry| entry.kind == "file").count();
    Ok(InspectTreeResult {
        path: root,
        kind: "directory",
        direct_children,
        tree,
        read_file_count: files.len(),
        files,
        skipped,
        file_count,
        bytes_read: total_bytes,
        truncated,
        tree_truncated,
        tree_entry_limit: options.max_entries,
    })
}

fn inspect_single_file(
    path: &Path,
    workspace_root: &Path,
    options: &InspectTreeOptions,
) -> Result<InspectTreeResult> {
    let display = relative_display(path, workspace_root);
    let size = path.metadata()?.len();
    if !looks_readable_text_file(path) {
        return Ok(InspectTreeResult {
            path: path.to_path_buf(),
            kind: "file",
            direct_children: Vec::new(),
            tree: Vec::new(),
            files: Vec::new(),
            skipped: vec![SkippedTreeEntry {
                path: display,
                reason: "binary or unsupported file type",
            }],
            file_count: 1,
            read_file_count: 0,
            bytes_read: 0,
            truncated: false,
            tree_truncated: false,
            tree_entry_limit: options.max_entries,
        });
    }
    let inspected = read_bounded_text(path, options.max_bytes_per_file)?;
    Ok(InspectTreeResult {
        path: path.to_path_buf(),
        kind: "file",
        direct_children: Vec::new(),
        tree: Vec::new(),
        files: vec![InspectedTreeFile {
            path: display,
            bytes: size,
            truncated: inspected.truncated,
            content: inspected.content,
        }],
        skipped: Vec::new(),
        file_count: 1,
        read_file_count: 1,
        bytes_read: inspected.bytes_read,
        truncated: inspected.truncated,
        tree_truncated: false,
        tree_entry_limit: options.max_entries,
    })
}

struct BoundedText {
    content: String,
    bytes_read: usize,
    truncated: bool,
}

fn read_bounded_text(path: &Path, limit: usize) -> Result<BoundedText> {
    let mut bytes = Vec::with_capacity(limit.saturating_add(1));
    File::open(path)
        .with_context(|| format!("Could not read file: {}", path.display()))?
        .take(limit.saturating_add(1) as u64)
        .read_to_end(&mut bytes)?;
    let truncated = bytes.len() > limit;
    bytes.truncate(limit);
    let bytes_read = bytes.len();
    Ok(BoundedText {
        content: String::from_utf8_lossy(&bytes).into_owned(),
        bytes_read,
        truncated,
    })
}

fn looks_readable_text_file(path: &Path) -> bool {
    if path.file_name().and_then(|name| name.to_str()) == Some(".DS_Store") {
        return false;
    }
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase);
    if extension
        .as_deref()
        .is_some_and(|value| TEXT_SUFFIXES.contains(&value))
    {
        return true;
    }
    if extension.is_none()
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                matches!(
                    name.to_ascii_lowercase().as_str(),
                    "dockerfile" | "makefile" | "procfile"
                )
            })
    {
        return true;
    }
    let mut sample = [0u8; 4096];
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let Ok(read) = file.read(&mut sample) else {
        return false;
    };
    let sample = &sample[..read];
    !sample.contains(&0) && std::str::from_utf8(sample).is_ok()
}

fn path_allowed(entry: &RipgrepPath) -> bool {
    let components = entry
        .relative
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .collect::<Vec<_>>();
    if components.is_empty() || components.last() == Some(&".DS_Store") {
        return false;
    }
    for (index, name) in components.iter().enumerate() {
        let is_directory = index + 1 < components.len() || entry.kind == RipgrepPathKind::Directory;
        if !is_directory {
            continue;
        }
        if name.starts_with('.') && !HIDDEN_DIR_ALLOWLIST.contains(name) {
            return false;
        }
        let depth = index + 1;
        let skip_names = if depth > 2 {
            DEEP_SKIP_DIRS
        } else {
            SHALLOW_SKIP_DIRS
        };
        if skip_names.contains(name) || name.ends_with(".egg-info") {
            return false;
        }
    }
    true
}

fn entry_order(left: &RipgrepPath, right: &RipgrepPath) -> std::cmp::Ordering {
    left.relative
        .parent()
        .cmp(&right.relative.parent())
        .then_with(|| match (left.kind, right.kind) {
            (RipgrepPathKind::File, RipgrepPathKind::Directory) => std::cmp::Ordering::Less,
            (RipgrepPathKind::Directory, RipgrepPathKind::File) => std::cmp::Ordering::Greater,
            _ => left.relative.cmp(&right.relative),
        })
}

fn direct_children(
    root: &Path,
    workspace_root: &Path,
    eligible: &[RipgrepPath],
) -> Vec<InspectTreeEntry> {
    let mut children = BTreeMap::<PathBuf, RipgrepPathKind>::new();
    for entry in eligible {
        let Some(first) = entry.relative.components().next() else {
            continue;
        };
        let relative = PathBuf::from(first.as_os_str());
        let kind = if entry.relative == relative {
            entry.kind
        } else {
            RipgrepPathKind::Directory
        };
        children.entry(relative).or_insert(kind);
    }
    for entry in fs::read_dir(root).into_iter().flatten().flatten() {
        let path = entry.path();
        if !path.is_dir() || path.is_symlink() {
            continue;
        }
        let Ok(relative) = path.strip_prefix(root).map(Path::to_path_buf) else {
            continue;
        };
        let candidate = RipgrepPath {
            relative: relative.clone(),
            kind: RipgrepPathKind::Directory,
        };
        if path_allowed(&candidate) {
            children
                .entry(relative)
                .or_insert(RipgrepPathKind::Directory);
        }
    }
    children
        .into_iter()
        .map(|(relative, kind)| {
            let path = root.join(relative);
            let is_file = kind == RipgrepPathKind::File;
            InspectTreeEntry {
                path: relative_display(&path, workspace_root),
                kind: if is_file { "file" } else { "directory" },
                bytes: is_file
                    .then(|| path.metadata().ok().map(|metadata| metadata.len()))
                    .flatten(),
            }
        })
        .collect()
}

fn relative_display(path: &Path, workspace_root: &Path) -> PathBuf {
    path.strip_prefix(workspace_root)
        .unwrap_or(path)
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agent-inspect-tree-{suffix}"));
        fs::create_dir_all(root.join(".github/workflows")).expect("create hidden config");
        fs::create_dir_all(root.join(".secret")).expect("create hidden secret");
        fs::create_dir_all(root.join("build")).expect("create shallow build");
        fs::create_dir_all(root.join("src/features/build")).expect("create deep build");
        fs::write(root.join("main.py"), "print('ok')\n").expect("write source");
        fs::write(root.join("large.txt"), "abcdefghij").expect("write bounded source");
        fs::write(root.join(".github/workflows/ci.yml"), "name: ci\n").expect("write workflow");
        fs::write(root.join(".secret/token.txt"), "hidden\n").expect("write hidden");
        fs::write(root.join("build/output.txt"), "skip\n").expect("write build output");
        fs::write(root.join("src/features/build/source.txt"), "keep\n").expect("write deep source");
        root
    }

    #[test]
    fn inspection_is_bounded_and_preserves_walk_policy() {
        let root = fixture();
        let result = inspect_tree(InspectTreeOptions {
            root: root.clone(),
            workspace_root: root.clone(),
            max_files: 20,
            max_entries: 100,
            max_bytes_per_file: 5,
            max_total_bytes: 100,
        })
        .expect("inspect tree");
        let paths = result
            .tree
            .iter()
            .map(|entry| entry.path.as_path())
            .collect::<Vec<_>>();

        assert!(paths.contains(&Path::new("main.py")));
        assert!(paths.contains(&Path::new(".github/workflows/ci.yml")));
        assert!(paths.contains(&Path::new("src/features/build/source.txt")));
        assert!(!paths.iter().any(|path| path.starts_with(".secret")));
        assert!(!paths.iter().any(|path| path.starts_with("build")));
        let large = result
            .files
            .iter()
            .find(|file| file.path == Path::new("large.txt"))
            .expect("large file");
        assert_eq!(large.content, "abcde");
        assert!(large.truncated);
        fs::remove_dir_all(root).expect("clean fixture");
    }
}
