use super::*;

pub(super) fn resolve_applications(query: &str, limit: usize) -> Result<Vec<Value>> {
    let apps = installed_applications(200)?;
    let mut candidates = Vec::new();
    if let Some(items) = apps.get("items").and_then(Value::as_array) {
        for app in items {
            let name = app.get("name").and_then(Value::as_str).unwrap_or_default();
            let id = app.get("id").and_then(Value::as_str).unwrap_or_default();
            let exec = app.get("exec").and_then(Value::as_str).unwrap_or_default();
            let target = app.get("target").and_then(Value::as_str).unwrap_or(id);
            let score = match_score(query, &[name, id, exec]);
            if score > 0 {
                candidates.push(json!({
                    "kind": "application",
                    "score": score,
                    "id": id,
                    "name": name,
                    "exec": exec,
                    "target": target,
                    "action": "launch_application",
                    "backend": app.get("backend").cloned().or_else(|| apps.get("backend").cloned()),
                }));
            }
        }
    }
    if candidates.len() < limit && windows_host_powershell_available() {
        for app in windows_host_registered_application_matches(query, limit - candidates.len())? {
            let name = app.get("name").and_then(Value::as_str).unwrap_or_default();
            let id = app.get("id").and_then(Value::as_str).unwrap_or_default();
            let exec = app.get("exec").and_then(Value::as_str).unwrap_or_default();
            let score = match_score(query, &[name, id, exec]);
            if score > 0
                && !candidates
                    .iter()
                    .any(|candidate| candidate.get("target") == app.get("target"))
            {
                candidates.push(json!({
                    "kind": "application",
                    "score": score,
                    "id": id,
                    "name": name,
                    "exec": exec,
                    "target": app.get("target").and_then(Value::as_str).unwrap_or(id),
                    "action": "launch_application",
                    "backend": app.get("backend").cloned(),
                }));
            }
        }
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate_score(candidate)));
    candidates.truncate(limit);
    Ok(candidates)
}

pub(super) fn resolve_windows(query: &str, limit: usize) -> Result<Vec<Value>> {
    let windows = visible_windows(200)?;
    let mut candidates = Vec::new();
    if windows.get("ok").and_then(Value::as_bool) != Some(true) {
        return Ok(candidates);
    }
    if let Some(items) = windows.get("items").and_then(Value::as_array) {
        for window in items {
            let title = window
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let class = window
                .get("class")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let process = window
                .get("process")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let id = window.get("id").and_then(Value::as_str).unwrap_or_default();
            let score = match_score(query, &[title, class, process, id]);
            if score > 0 {
                candidates.push(json!({
                    "kind": "window",
                    "score": score,
                    "id": id,
                    "title": title,
                    "class": class,
                    "process": process,
                    "target": id,
                    "action": "focus_window",
                    "backend": windows.get("backend").cloned(),
                }));
            }
        }
    }
    candidates.sort_by_key(|candidate| std::cmp::Reverse(candidate_score(candidate)));
    candidates.truncate(limit);
    Ok(candidates)
}

pub(super) fn match_score(query: &str, fields: &[&str]) -> i64 {
    let mut best = 0;
    let query_token_count = query.split_whitespace().count().max(1);
    for field in fields {
        let normalized = normalize_match_text(field);
        if normalized.is_empty() {
            continue;
        }
        let score = if normalized == query {
            100
        } else if normalized.starts_with(query) {
            80
        } else if normalized.contains(query) {
            60
        } else {
            let matched = query
                .split_whitespace()
                .filter(|token| normalized.contains(token))
                .count();
            if matched > 0 {
                (matched as i64 * 40) / query_token_count as i64
            } else {
                0
            }
        };
        best = best.max(score);
    }
    best
}

pub(super) fn normalize_match_text(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut pending_space = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            if pending_space && !normalized.is_empty() {
                normalized.push(' ');
            }
            normalized.push(character.to_ascii_lowercase());
            pending_space = false;
        } else {
            pending_space = !normalized.is_empty();
        }
    }
    normalized
}

pub(super) fn candidate_score(candidate: &Value) -> i64 {
    candidate.get("score").and_then(Value::as_i64).unwrap_or(0)
}

pub(super) fn candidate_label(candidate: &Value) -> String {
    candidate
        .get("name")
        .or_else(|| candidate.get("title"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}
