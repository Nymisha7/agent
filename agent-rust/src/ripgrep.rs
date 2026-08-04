use anyhow::{bail, Context, Result};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(unix)]
use std::os::unix::ffi::OsStringExt;

#[derive(Debug, Clone, Copy)]
pub(crate) struct RipgrepFilesOptions {
    pub include_hidden: bool,
    pub follow_links: bool,
    pub include_ignored: bool,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RipgrepPathKind {
    File,
    Directory,
}

#[derive(Debug, Clone)]
pub(crate) struct RipgrepPath {
    pub relative: PathBuf,
    pub kind: RipgrepPathKind,
}

pub(crate) fn ripgrep_paths(root: &Path, options: RipgrepFilesOptions) -> Result<Vec<RipgrepPath>> {
    let mut command = Command::new("rg");
    command
        .current_dir(root)
        .arg("--files")
        .arg("--null")
        .arg("--no-messages");

    if options.include_hidden {
        command.arg("--hidden");
    }
    if options.follow_links {
        command.arg("--follow");
    }
    if options.include_ignored {
        command.arg("--no-ignore");
    }
    if let Some(threads) = options.threads {
        command.arg("--threads").arg(threads.max(1).to_string());
    }
    command.arg(".");

    let output = command
        .output()
        .context("Failed to run ripgrep for file discovery")?;
    if !output.status.success() && output.status.code() != Some(1) {
        bail!(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }

    // Keep the final representation while deriving parent directories.  The former
    // files -> directories -> paths pipeline held three overlapping collections.
    let mut paths = Vec::new();
    for bytes in output.stdout.split(|byte| *byte == 0) {
        if bytes.is_empty() {
            continue;
        }
        let relative = strip_dot_prefix(&path_from_bytes(bytes));
        if relative.as_os_str().is_empty() {
            continue;
        }
        paths.extend(
            relative
                .ancestors()
                .skip(1)
                .filter(|path| !path.as_os_str().is_empty())
                .map(|path| RipgrepPath {
                    relative: path.to_path_buf(),
                    kind: RipgrepPathKind::Directory,
                }),
        );
        paths.push(RipgrepPath {
            relative,
            kind: RipgrepPathKind::File,
        });
    }
    paths.sort_by(|left, right| {
        ripgrep_kind_rank(left.kind)
            .cmp(&ripgrep_kind_rank(right.kind))
            .then_with(|| left.relative.cmp(&right.relative))
    });
    paths.dedup_by(|left, right| left.kind == right.kind && left.relative == right.relative);
    Ok(paths)
}

fn ripgrep_kind_rank(kind: RipgrepPathKind) -> u8 {
    match kind {
        RipgrepPathKind::Directory => 0,
        RipgrepPathKind::File => 1,
    }
}

fn strip_dot_prefix(path: &Path) -> PathBuf {
    path.strip_prefix(".").unwrap_or(path).to_path_buf()
}

#[cfg(unix)]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(OsString::from_vec(bytes.to_vec()))
}

#[cfg(not(unix))]
fn path_from_bytes(bytes: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agent-ripgrep-{suffix}"));
        fs::create_dir_all(root.join("src")).expect("create source directory");
        fs::create_dir_all(root.join("ignored")).expect("create ignored directory");
        fs::write(root.join("src/main.rs"), "fn main() {}\n").expect("write source");
        fs::write(root.join("ignored/cache.rs"), "ignored\n").expect("write ignored source");
        fs::write(root.join(".ignore"), "ignored/\n").expect("write ignore file");
        root
    }

    #[test]
    fn inventory_uses_ripgrep_ignore_rules_and_derives_directories() {
        let root = fixture();
        let paths = ripgrep_paths(
            &root,
            RipgrepFilesOptions {
                include_hidden: false,
                follow_links: false,
                include_ignored: false,
                threads: None,
            },
        )
        .expect("inventory");

        assert!(paths.iter().any(|item| {
            item.relative == Path::new("src") && item.kind == RipgrepPathKind::Directory
        }));
        assert!(paths.iter().any(|item| {
            item.relative == Path::new("src/main.rs") && item.kind == RipgrepPathKind::File
        }));
        assert!(!paths
            .iter()
            .any(|item| item.relative.starts_with("ignored")));
        fs::remove_dir_all(root).expect("clean fixture");
    }

    #[test]
    fn inventory_can_include_ignored_paths_without_name_lists() {
        let root = fixture();
        let paths = ripgrep_paths(
            &root,
            RipgrepFilesOptions {
                include_hidden: false,
                follow_links: false,
                include_ignored: true,
                threads: None,
            },
        )
        .expect("inventory");

        assert!(paths
            .iter()
            .any(|item| item.relative == Path::new("ignored/cache.rs")));
        fs::remove_dir_all(root).expect("clean fixture");
    }
}
