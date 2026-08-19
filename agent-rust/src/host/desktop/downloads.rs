use super::*;

pub(super) fn downloads_directory() -> Option<PathBuf> {
    let home = env::var_os("HOME").map(PathBuf::from)?;
    let config_home = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".config"));
    let configured = fs::read_to_string(config_home.join("user-dirs.dirs"))
        .ok()
        .and_then(|content| {
            content.lines().find_map(|line| {
                let value = line.trim().strip_prefix("XDG_DOWNLOAD_DIR=")?.trim();
                let value = value.strip_prefix('"')?.strip_suffix('"')?;
                if value == "$HOME" {
                    Some(home.clone())
                } else if let Some(relative) = value.strip_prefix("$HOME/") {
                    Some(home.join(relative))
                } else {
                    let path = PathBuf::from(value);
                    path.is_absolute().then_some(path)
                }
            })
        });
    configured.or_else(|| Some(home.join("Downloads")))
}

pub(super) fn download_inventory(limit: usize) -> Result<Value> {
    let Some(directory) = downloads_directory() else {
        return Ok(json!({
            "ok": false,
            "reason": "downloads_directory_unavailable",
            "items": [],
        }));
    };
    download_inventory_at(&directory, limit)
}

pub(super) fn download_inventory_at(directory: &Path, limit: usize) -> Result<Value> {
    if !directory.is_dir() {
        return Ok(json!({
            "ok": false,
            "reason": "downloads_directory_missing",
            "directory": directory,
            "items": [],
        }));
    }
    let mut items = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        let name = entry.file_name().to_string_lossy().into_owned();
        let modified_unix_ms = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|duration| duration.as_millis())
            .unwrap_or(0);
        let partial = is_partial_download_name(&name);
        items.push(json!({
            "name": name,
            "path": entry.path(),
            "kind": if metadata.is_file() { "file" } else if metadata.is_dir() { "directory" } else { "other" },
            "byte_count": metadata.is_file().then_some(metadata.len()),
            "modified_unix_ms": modified_unix_ms,
            "status": if partial { "partial" } else { "complete" },
        }));
    }
    items.sort_by(|left, right| {
        let timestamp = |value: &Value| {
            value
                .get("modified_unix_ms")
                .and_then(Value::as_u64)
                .unwrap_or(0)
        };
        timestamp(right).cmp(&timestamp(left)).then_with(|| {
            left.get("name")
                .and_then(Value::as_str)
                .cmp(&right.get("name").and_then(Value::as_str))
        })
    });
    items.truncate(limit.clamp(1, 200));
    let partial_count = items
        .iter()
        .filter(|item| item.get("status").and_then(Value::as_str) == Some("partial"))
        .count();
    Ok(json!({
        "ok": true,
        "directory": directory,
        "count": items.len(),
        "partial_count": partial_count,
        "items": items,
    }))
}

pub(super) fn is_partial_download_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [".crdownload", ".part", ".partial", ".download"]
        .iter()
        .any(|suffix| lower.ends_with(suffix))
}
