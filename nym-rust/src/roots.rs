use anyhow::{Context, Result};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const LOCATOR_ROOTS_ENV: &str = "NYM_LOCATOR_ROOTS";
const LOCATOR_ROOTS_FILE_ENV: &str = "NYM_LOCATOR_ROOTS_FILE";
const ROOTS_ENV: &str = "NYM_SEARCH_ROOTS";
const ROOTS_FILE_ENV: &str = "NYM_ROOTS_FILE";

pub fn resolve_search_roots(cli_roots: Vec<PathBuf>) -> Result<Vec<PathBuf>> {
    if !cli_roots.is_empty() {
        return resolve_roots(cli_roots, true);
    }

    let mut roots = Vec::new();
    roots.extend(env_roots());
    roots.extend(config_roots()?);
    roots.push(current_dir_root()?);

    resolve_roots(roots, false)
}

pub fn resolve_locator_roots() -> Result<Vec<PathBuf>> {
    let mut roots = Vec::new();

    roots.extend(locator_env_roots());
    roots.extend(locator_config_roots()?);

    let resolved = resolve_roots(roots, false)?;

    if resolved.is_empty() {
        anyhow::bail!(
            "No locator roots configured. Add roots using NYM_LOCATOR_ROOTS or ~/.config/nym/locator_roots"
        );
    }

    Ok(resolved)
}

fn config_roots() -> Result<Vec<PathBuf>> {
    let Some(path) = config_path() else {
        return Ok(Vec::new());
    };

    if !path.exists() {
        return Ok(Vec::new());
    }

    let text = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read roots config: {}", path.display()))?;

    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect())
}

fn config_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os(ROOTS_FILE_ENV) {
        return Some(expand_tilde(Path::new(&path)));
    }

    if let Some(config_home) = env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(config_home).join("nym").join("roots"));
    }

    env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("nym")
            .join("roots")
    })
}

fn current_dir_root() -> Result<PathBuf> {
    env::current_dir().context("Failed to read current directory")
}

fn locator_env_roots() -> Vec<PathBuf> {
    let Some(value) = env::var_os(LOCATOR_ROOTS_ENV) else {
        return Vec::new();
    };

    env::split_paths(&value).collect()
}

fn env_roots() -> Vec<PathBuf> {
    let Some(value) = env::var_os(ROOTS_ENV) else {
        return Vec::new();
    };

    env::split_paths(&value).collect()
}

fn locator_config_roots() -> Result<Vec<PathBuf>> {
    let Some(path) = locator_config_path() else {
        return Ok(Vec::new());
    };

    if !path.exists() {
        return Ok(Vec::new());
    }

    let text = fs::read_to_string(&path)
        .with_context(|| format!("Failed to read locator roots config: {}", path.display()))?;

    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(PathBuf::from)
        .collect())
}

fn locator_config_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os(LOCATOR_ROOTS_FILE_ENV) {
        return Some(expand_tilde(Path::new(&path)));
    }

    if let Some(config_home) = env::var_os("XDG_CONFIG_HOME") {
        return Some(PathBuf::from(config_home).join("nym").join("locator_roots"));
    }

    env::var_os("HOME").map(|home| {
        PathBuf::from(home)
            .join(".config")
            .join("nym")
            .join("locator_roots")
    })
}

fn resolve_roots(roots: Vec<PathBuf>, strict: bool) -> Result<Vec<PathBuf>> {
    let mut resolved = Vec::new();
    let mut seen = HashSet::new();

    for root in roots {
        let root = expand_tilde(&root);
        let root = match root.canonicalize() {
            Ok(root) => root,
            Err(error) if strict => {
                return Err(error)
                    .with_context(|| format!("Invalid search root: {}", root.display()));
            }
            Err(_) => continue,
        };

        if !root.is_dir() {
            if strict {
                anyhow::bail!("Search root is not a directory: {}", root.display());
            }
            continue;
        }

        if seen.insert(root.clone()) {
            resolved.push(root);
        }
    }

    Ok(resolved)
}

fn expand_tilde(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();

    if raw == "~" {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }

    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = env::var_os("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }

    path.to_path_buf()
}

pub fn system_search_roots() -> Result<Vec<PathBuf>> {
    resolve_locator_roots()
}
