use super::*;

pub(super) fn launch_desktop_target(
    action: &str,
    target: &str,
    use_xdg_open: bool,
) -> Result<Value> {
    if let Some(shortcut) = decode_windows_shortcut_target(target)? {
        return launch_windows_host_shortcut(action, target, &shortcut);
    }
    if let Some(app_id) = decode_windows_app_target(target)? {
        return launch_windows_host_app(action, target, &app_id);
    }

    let (program, argument) = if use_xdg_open {
        ("xdg-open", target)
    } else if command_exists("gtk-launch") {
        if !valid_identifier(target) {
            return Err(anyhow::anyhow!(
                "launch_application requires an application ID without spaces"
            ));
        }
        ("gtk-launch", target)
    } else {
        if !valid_identifier(target) || !command_exists(target) {
            return Ok(json!({
                "ok": false,
                "tool": "desktop_action",
                "action": action,
                "target": target,
                "reason": "application_unavailable",
                "error": "No matching desktop application launcher was found.",
            }));
        }
        (target, "")
    };
    if !command_exists(program) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "reason": "dependency_unavailable",
            "error": format!("Required desktop command `{program}` is not installed."),
        }));
    }
    let before = launch_observation(None)?;
    let mut command = ProcessCommand::new(program);
    if !argument.is_empty() {
        command.arg(argument);
    }
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = command.spawn()?;
    let pid = child.id();
    let mut after = launch_observation(Some(pid))?;
    let direct_process_verifies = !use_xdg_open && program == target;
    let mut verified = launch_observation_changed(&before, &after, direct_process_verifies);
    let mut waited_ms = 0u64;
    while !verified && waited_ms < 500 {
        thread::sleep(Duration::from_millis(100));
        waited_ms += 100;
        after = launch_observation(Some(pid))?;
        verified = launch_observation_changed(&before, &after, direct_process_verifies);
    }
    let focus = focus_launched_window(&before, &after, Some(target))?;

    Ok(json!({
        "ok": true,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "program": program,
        "argument": if argument.is_empty() { None } else { Some(argument) },
        "pid": pid,
        "before": before,
        "after": after,
        "verified": verified,
        "verification": if verified { "confirmed" } else { "not_confirmed" },
        "waited_ms": waited_ms,
        "focus": focus,
    }))
}

pub(super) fn open_path_in_application(
    action: &str,
    target: &str,
    application: &str,
) -> Result<Value> {
    let path = fs::canonicalize(Path::new(target))
        .map_err(|error| anyhow::anyhow!("Cannot open local path `{target}`: {error}"))?;
    let application = application.trim();
    if let Some(shortcut) = decode_windows_shortcut_target(application)? {
        return open_path_with_windows_shortcut(action, &path, application, &shortcut);
    }
    if !valid_identifier(application) {
        return Err(anyhow::anyhow!(
            "{action} requires an application identifier"
        ));
    }
    let direct = command_exists(application);
    let gtk = !direct && command_exists("gtk-launch");
    if !direct && !gtk {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "application": application,
            "reason": "application_unavailable",
            "recoverable": true,
            "error": format!("Application executable `{application}` was not found on PATH."),
            "guidance": "Select an installed executable available on PATH.",
        }));
    }
    let mut command = ProcessCommand::new(if direct { application } else { "gtk-launch" });
    if gtk {
        command.arg(application);
    }
    let child = command
        .arg(&path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    let mut child = match child {
        Ok(child) => child,
        Err(error) => {
            return Ok(json!({
                "ok": false,
                "tool": "desktop_action",
                "action": action,
                "target": path,
                "application": application,
                "reason": "application_launch_failed",
                "recoverable": true,
                "error": error.to_string(),
                "guidance": "Verify that the selected executable can launch this path.",
            }));
        }
    };
    thread::sleep(Duration::from_millis(50));
    let early_status = child.try_wait()?;
    let accepted = early_status.is_none_or(|status| status.success());
    Ok(json!({
        "ok": accepted,
        "tool": "desktop_action",
        "action": action,
        "target": path,
        "application": application,
        "backend": if direct { "executable" } else { "gtk-launch" },
        "pid": child.id(),
        "verified": accepted,
        "verification": if accepted { "invocation_accepted" } else { "command_failed" },
        "verification_scope": "application_invocation",
        "exit_code": early_status.and_then(|status| status.code()),
        "waited_ms": 50,
    }))
}

pub(super) fn open_path_with_windows_shortcut(
    action: &str,
    path: &Path,
    application: &str,
    shortcut: &str,
) -> Result<Value> {
    if !windows_host_powershell_available() || !command_exists("wslpath") {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": path,
            "application": application,
            "reason": "dependency_unavailable",
            "recoverable": true,
            "error": "Opening a WSL path in a Windows application requires PowerShell and wslpath.",
        }));
    }
    let path_text = path.to_string_lossy();
    let windows_path = run_capture_dynamic(&["wslpath", "-w", path_text.as_ref()])?;
    if windows_path.status != 0 || windows_path.stdout.is_empty() {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": path,
            "application": application,
            "reason": "path_translation_failed",
            "recoverable": true,
            "stderr": windows_path.stderr,
        }));
    }
    let payload = serde_json::to_string(&json!({
        "shortcut": shortcut,
        "argument": windows_path.stdout,
    }))?;
    let output = run_capture_with_stdin(
        &[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-Command",
            windows_host_open_shortcut_script(),
        ],
        &payload,
    )?;
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "target": path,
        "application": application,
        "backend": "windows_host_powershell",
        "exit_code": output.status,
        "stdout": output.stdout,
        "stderr": output.stderr,
        "verified": output.status == 0,
        "verification": if output.status == 0 { "invocation_accepted" } else { "failed" },
        "verification_scope": "application_invocation",
    }))
}

pub(super) fn windows_host_open_shortcut_script() -> &'static str {
    r#"
$payload = [Console]::In.ReadToEnd() | ConvertFrom-Json
$shortcut = [string]$payload.shortcut
$argument = [string]$payload.argument
if (-not $shortcut.EndsWith('.lnk', [System.StringComparison]::OrdinalIgnoreCase)) { Write-Error 'Expected a Start Menu shortcut'; exit 2 }
if (-not (Test-Path -LiteralPath $shortcut -PathType Leaf)) { Write-Error 'Shortcut not found'; exit 3 }
if ($argument.Contains('"')) { Write-Error 'Invalid path argument'; exit 4 }
$link = (New-Object -ComObject WScript.Shell).CreateShortcut($shortcut)
if (-not $link.TargetPath -or -not (Test-Path -LiteralPath $link.TargetPath -PathType Leaf)) { Write-Error 'Shortcut target not found'; exit 5 }
$arguments = @()
if ($link.Arguments) { $arguments += $link.Arguments }
$arguments += ('"' + $argument + '"')
$params = @{ FilePath = $link.TargetPath; ArgumentList = ($arguments -join ' '); PassThru = $true }
if ($link.WorkingDirectory -and (Test-Path -LiteralPath $link.WorkingDirectory -PathType Container)) { $params.WorkingDirectory = $link.WorkingDirectory }
$process = Start-Process @params
@{ pid = $process.Id } | ConvertTo-Json -Compress
"#
}

pub(super) fn launch_windows_host_shortcut(
    action: &str,
    target: &str,
    shortcut: &str,
) -> Result<Value> {
    if !windows_host_powershell_available() {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "reason": "dependency_unavailable",
            "error": "Launching Windows shortcuts requires reachable Windows PowerShell under WSL.",
        }));
    }
    let before = launch_observation(None)?;
    let output = windows_host_start_shortcut(shortcut)?;
    let mut after = launch_observation(None)?;
    let mut verified = launch_observation_changed(&before, &after, false);
    let mut waited_ms = 0u64;
    while !verified && waited_ms < 1500 {
        thread::sleep(Duration::from_millis(150));
        waited_ms += 150;
        after = launch_observation(None)?;
        verified = launch_observation_changed(&before, &after, false);
    }
    let focus = if output.status == 0 {
        let query = windows_shortcut_focus_query(shortcut);
        focus_launched_window(&before, &after, Some(&query))?
    } else {
        None
    };

    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "backend": "windows_host_powershell",
        "exit_code": output.status,
        "before": before,
        "after": after,
        "verified": output.status == 0 && verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "waited_ms": waited_ms,
        "focus": focus,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_start_shortcut(shortcut: &str) -> Result<CommandOutput> {
    run_capture_with_stdin(
        &[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            windows_host_start_shortcut_script(),
        ],
        shortcut,
    )
}

pub(super) fn windows_host_start_shortcut_script() -> &'static str {
    r#"
$path = [Console]::In.ReadToEnd().Trim()
if (-not $path.EndsWith('.lnk', [System.StringComparison]::OrdinalIgnoreCase)) { Write-Error 'Expected a Start Menu shortcut'; exit 2 }
if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Write-Error 'Shortcut not found'; exit 3 }
Start-Process -FilePath $path
"#
}

pub(super) fn launch_windows_host_app(action: &str, target: &str, app_id: &str) -> Result<Value> {
    if !windows_host_powershell_available() {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "reason": "dependency_unavailable",
            "error": "Launching Windows apps requires reachable Windows PowerShell under WSL.",
        }));
    }
    let before = launch_observation(None)?;
    let output = windows_host_start_app(app_id)?;
    let mut after = launch_observation(None)?;
    let mut verified = launch_observation_changed(&before, &after, false);
    let mut waited_ms = 0u64;
    while !verified && waited_ms < 1500 {
        thread::sleep(Duration::from_millis(150));
        waited_ms += 150;
        after = launch_observation(None)?;
        verified = launch_observation_changed(&before, &after, false);
    }
    let focus = if output.status == 0 {
        focus_launched_window(&before, &after, Some(app_id))?
    } else {
        None
    };

    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "backend": "windows_host_powershell",
        "exit_code": output.status,
        "before": before,
        "after": after,
        "verified": output.status == 0 && verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "waited_ms": waited_ms,
        "focus": focus,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_start_app(app_id: &str) -> Result<CommandOutput> {
    run_capture_with_stdin(
        &[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            windows_host_start_app_script(),
        ],
        app_id,
    )
}

pub(super) fn windows_host_start_app_script() -> &'static str {
    r#"
$appId = [Console]::In.ReadToEnd().Trim()
if ($appId.Length -eq 0 -or $appId.Length -gt 512 -or $appId -match '[\r\n]') { Write-Error 'Invalid app id'; exit 2 }
Start-Process -FilePath ("shell:AppsFolder\" + $appId)
"#
}

pub(super) fn launch_observation(pid: Option<u32>) -> Result<Value> {
    let windows = launch_window_summary()?;
    Ok(json!({
        "pid": pid,
        "process": pid.map(|value| process_state(value as i32)),
        "windows": windows,
    }))
}

pub(super) fn launch_observation_changed(
    before: &Value,
    after: &Value,
    direct_process_verifies: bool,
) -> bool {
    let process_running = after
        .get("process")
        .and_then(|process| process.get("running"))
        .and_then(Value::as_bool)
        == Some(true);
    let before_windows = before.get("windows").unwrap_or(&Value::Null);
    let after_windows = after.get("windows").unwrap_or(&Value::Null);
    (direct_process_verifies && process_running)
        || window_summary_changed(before_windows, after_windows)
}

pub(super) fn launch_window_summary() -> Result<Value> {
    let observed = visible_windows(200)?;
    let items = observed
        .get("items")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let ids: Vec<Value> = items
        .iter()
        .filter_map(|item| item.get("id").cloned())
        .collect();
    let titles: Vec<Value> = items
        .iter()
        .filter_map(|item| item.get("title").cloned())
        .collect();
    Ok(json!({
        "ok": observed.get("ok").and_then(Value::as_bool).unwrap_or(false),
        "backend": observed.get("backend").cloned(),
        "reason": observed.get("reason").cloned(),
        "count": items.len(),
        "ids": ids,
        "titles": titles,
    }))
}

pub(super) fn window_summary_changed(before: &Value, after: &Value) -> bool {
    if before.get("ok").and_then(Value::as_bool) != Some(true)
        || after.get("ok").and_then(Value::as_bool) != Some(true)
    {
        return false;
    }
    before.get("count").and_then(Value::as_u64) != after.get("count").and_then(Value::as_u64)
        || before.get("ids") != after.get("ids")
        || before.get("titles") != after.get("titles")
}

pub(super) fn focus_launched_window(
    before: &Value,
    after: &Value,
    query: Option<&str>,
) -> Result<Option<Value>> {
    if let Some(id) = launched_window_id(before, after) {
        return Ok(Some(window_control_action("focus_window", &id)?));
    }
    if let Some(id) = matching_window_id(query)? {
        return Ok(Some(window_control_action("focus_window", &id)?));
    };
    Ok(None)
}

pub(super) fn launched_window_id(before: &Value, after: &Value) -> Option<String> {
    let before_ids: HashSet<&str> = before
        .pointer("/windows/ids")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    after
        .pointer("/windows/ids")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(Value::as_str)
        .find(|id| !before_ids.contains(id))
        .map(ToString::to_string)
}

pub(super) fn matching_window_id(query: Option<&str>) -> Result<Option<String>> {
    let Some(query) = query
        .map(normalize_match_text)
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    Ok(resolve_windows(&query, 1)?
        .first()
        .filter(|candidate| candidate_score(candidate) >= 40)
        .and_then(|candidate| candidate.get("target"))
        .and_then(Value::as_str)
        .map(ToString::to_string))
}

pub(super) fn windows_shortcut_focus_query(shortcut: &str) -> String {
    shortcut
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(shortcut)
        .strip_suffix(".lnk")
        .unwrap_or(shortcut)
        .to_string()
}

pub(super) fn installed_applications(limit: usize) -> Result<Value> {
    let mut apps = Vec::new();
    let mut seen = HashSet::new();
    for dir in desktop_application_dirs() {
        collect_desktop_entries(&dir, &dir, limit, &mut seen, &mut apps)?;
        if apps.len() >= limit {
            break;
        }
    }
    if apps.len() < limit && windows_host_powershell_available() {
        for app in windows_host_start_menu_applications(limit - apps.len())? {
            let id = app
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if seen.insert(id) {
                apps.push(app);
            }
            if apps.len() >= limit {
                break;
            }
        }
    }
    if apps.len() < limit && windows_host_powershell_available() {
        for app in windows_host_registered_applications(limit - apps.len())? {
            let id = app
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if seen.insert(id) {
                apps.push(app);
            }
            if apps.len() >= limit {
                break;
            }
        }
    }
    Ok(json!({
        "ok": true,
        "backend": if apps.iter().any(|app| app.get("backend").and_then(Value::as_str) == Some("windows_host_powershell")) { "mixed" } else { "freedesktop_desktop_entries" },
        "count": apps.len(),
        "items": apps,
        "truncated": apps.len() >= limit,
    }))
}

pub(super) fn windows_host_registered_applications(limit: usize) -> Result<Vec<Value>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let script = windows_host_registered_applications_script(limit);
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])?;
    if output.status != 0 {
        return Ok(Vec::new());
    }
    let raw = serde_json::from_str::<Value>(&output.stdout).unwrap_or(Value::Null);
    Ok(json_values(raw)
        .into_iter()
        .filter_map(|item| {
            let name = item.get("Name").and_then(Value::as_str)?;
            let app_id = item.get("AppID").and_then(Value::as_str)?;
            let target = windows_app_target(app_id)?;
            Some(json!({
                "id": target,
                "target": target,
                "name": name,
                "exec": app_id,
                "path": format!("shell:AppsFolder\\{app_id}"),
                "no_display": false,
                "terminal": false,
                "categories": ["Windows", "StartApps"],
                "backend": "windows_host_powershell",
            }))
        })
        .collect())
}

pub(super) fn windows_host_registered_application_matches(
    query: &str,
    limit: usize,
) -> Result<Vec<Value>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let output = run_capture_with_stdin(
        &[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            &windows_host_registered_application_matches_script(limit),
        ],
        query,
    )?;
    if output.status != 0 {
        return Ok(Vec::new());
    }
    let raw = serde_json::from_str::<Value>(&output.stdout).unwrap_or(Value::Null);
    Ok(json_values(raw)
        .into_iter()
        .filter_map(windows_host_registered_application_value)
        .collect())
}

pub(super) fn windows_host_registered_application_value(item: Value) -> Option<Value> {
    let name = item.get("Name").and_then(Value::as_str)?;
    let app_id = item.get("AppID").and_then(Value::as_str)?;
    let target = windows_app_target(app_id)?;
    Some(json!({
        "id": target,
        "target": target,
        "name": name,
        "exec": app_id,
        "path": format!("shell:AppsFolder\\{app_id}"),
        "no_display": false,
        "terminal": false,
        "categories": ["Windows", "StartApps"],
        "backend": "windows_host_powershell",
    }))
}

pub(super) fn windows_host_registered_application_matches_script(limit: usize) -> String {
    r#"
$query = [Console]::In.ReadToEnd().Trim().ToLowerInvariant()
if ($query.Length -eq 0) { @() | ConvertTo-Json -Compress; exit 0 }
Get-StartApps |
  Where-Object { $_.Name.ToLowerInvariant().Contains($query) -or $_.AppID.ToLowerInvariant().Contains($query) } |
  Select-Object -First __LIMIT__ Name,AppID |
  ConvertTo-Json -Compress
"#
    .replace("__LIMIT__", &limit.to_string())
}

pub(super) fn windows_host_registered_applications_script(limit: usize) -> String {
    "Get-StartApps | Select-Object -First __LIMIT__ Name,AppID | ConvertTo-Json -Compress"
        .replace("__LIMIT__", &limit.to_string())
}

pub(super) fn windows_host_start_menu_applications(limit: usize) -> Result<Vec<Value>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let script = windows_host_start_menu_applications_script(limit);
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])?;
    if output.status != 0 {
        return Ok(Vec::new());
    }
    let raw = serde_json::from_str::<Value>(&output.stdout).unwrap_or(Value::Null);
    Ok(json_values(raw)
        .into_iter()
        .filter_map(|item| {
            let name = item.get("Name").and_then(Value::as_str)?;
            let path = item.get("Path").and_then(Value::as_str)?;
            let target = windows_shortcut_target(path)?;
            Some(json!({
                "id": target,
                "target": target,
                "name": name,
                "exec": path,
                "path": path,
                "no_display": false,
                "terminal": false,
                "categories": ["Windows", "StartMenu"],
                "backend": "windows_host_powershell",
            }))
        })
        .collect())
}

pub(super) fn windows_host_start_menu_applications_script(limit: usize) -> String {
    r#"
$dirs = @(
  "$env:ProgramData\Microsoft\Windows\Start Menu\Programs",
  "$env:APPDATA\Microsoft\Windows\Start Menu\Programs"
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
if (-not $dirs) { @() | ConvertTo-Json -Compress; exit 0 }
Get-ChildItem -LiteralPath $dirs -Filter '*.lnk' -File -Recurse -ErrorAction SilentlyContinue |
  Select-Object -First __LIMIT__ @{Name='Name';Expression={$_.BaseName}},@{Name='Path';Expression={$_.FullName}} |
  ConvertTo-Json -Compress
"#
    .replace("__LIMIT__", &limit.to_string())
}

pub(super) fn windows_shortcut_target(path: &str) -> Option<String> {
    if !valid_windows_shortcut_path(path) {
        return None;
    }
    Some(format!(
        "{WINDOWS_SHORTCUT_PREFIX}{}",
        hex_encode(path.as_bytes())
    ))
}

pub(super) fn windows_app_target(app_id: &str) -> Option<String> {
    if !valid_windows_app_id(app_id) {
        return None;
    }
    Some(format!(
        "{WINDOWS_APP_PREFIX}{}",
        hex_encode(app_id.as_bytes())
    ))
}

pub(super) fn decode_windows_app_target(target: &str) -> Result<Option<String>> {
    let Some(encoded) = target.strip_prefix(WINDOWS_APP_PREFIX) else {
        return Ok(None);
    };
    let bytes = hex_decode(encoded)?;
    let app_id = String::from_utf8(bytes)?;
    if !valid_windows_app_id(&app_id) {
        return Err(anyhow::anyhow!("invalid Windows app target"));
    }
    Ok(Some(app_id))
}

pub(super) fn valid_windows_app_id(app_id: &str) -> bool {
    !app_id.is_empty() && app_id.len() <= 512 && !app_id.chars().any(char::is_control)
}

pub(super) fn decode_windows_shortcut_target(target: &str) -> Result<Option<String>> {
    let Some(encoded) = target.strip_prefix(WINDOWS_SHORTCUT_PREFIX) else {
        return Ok(None);
    };
    let bytes = hex_decode(encoded)?;
    let path = String::from_utf8(bytes)?;
    if !valid_windows_shortcut_path(&path) {
        return Err(anyhow::anyhow!("invalid Windows shortcut target"));
    }
    Ok(Some(path))
}

pub(super) fn valid_windows_shortcut_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 1024
        && path.ends_with(".lnk")
        && !path.chars().any(char::is_control)
}

pub(super) fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let high = (byte >> 4) as usize;
        let low = (byte & 0x0f) as usize;
        // SAFETY: shifting or masking one byte yields a value in 0..=15,
        // exactly the valid index range of HEX.
        unsafe {
            encoded.push(*HEX.get_unchecked(high) as char);
            encoded.push(*HEX.get_unchecked(low) as char);
        }
    }
    encoded
}

pub(super) fn hex_decode(value: &str) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2) {
        return Err(anyhow::anyhow!("invalid hex target"));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for chunk in value.as_bytes().chunks_exact(2) {
        // SAFETY: chunks_exact(2) yields slices whose length is exactly two.
        let (high_byte, low_byte) = unsafe { (*chunk.get_unchecked(0), *chunk.get_unchecked(1)) };
        let high = hex_value(high_byte).ok_or_else(|| anyhow::anyhow!("invalid hex target"))?;
        let low = hex_value(low_byte).ok_or_else(|| anyhow::anyhow!("invalid hex target"))?;
        bytes.push((high << 4) | low);
    }
    Ok(bytes)
}

pub(super) fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

pub(super) fn desktop_application_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(data_home) = env::var("XDG_DATA_HOME") {
        dirs.push(PathBuf::from(data_home).join("applications"));
    } else if let Ok(home) = env::var("HOME") {
        dirs.push(PathBuf::from(home).join(".local/share/applications"));
    }

    let data_dirs =
        env::var("XDG_DATA_DIRS").unwrap_or_else(|_| "/usr/local/share:/usr/share".to_string());
    for raw in data_dirs.split(':').filter(|item| !item.is_empty()) {
        dirs.push(PathBuf::from(raw).join("applications"));
    }
    dirs
}

pub(super) fn collect_desktop_entries(
    root: &Path,
    dir: &Path,
    limit: usize,
    seen: &mut HashSet<String>,
    apps: &mut Vec<Value>,
) -> Result<()> {
    if apps.len() >= limit || !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_desktop_entries(root, &path, limit, seen, apps)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("desktop") {
            if let Some(app) = parse_desktop_entry(root, &path)? {
                let id = app
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if seen.insert(id) {
                    apps.push(app);
                }
            }
        }
        if apps.len() >= limit {
            break;
        }
    }
    Ok(())
}

pub(super) fn parse_desktop_entry(root: &Path, path: &Path) -> Result<Option<Value>> {
    let text = fs::read_to_string(path)?;
    let mut in_desktop_entry = false;
    let mut is_application = false;
    let mut name = None;
    let mut exec = None;
    let mut no_display = false;
    let mut hidden = false;
    let mut terminal = false;
    let mut categories = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_desktop_entry = line == "[Desktop Entry]";
            continue;
        }
        if !in_desktop_entry {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        match key {
            "Type" => is_application = value == "Application",
            "Name" => name = Some(value.to_string()),
            "Exec" => exec = Some(value.to_string()),
            "NoDisplay" => no_display = value.eq_ignore_ascii_case("true"),
            "Hidden" => hidden = value.eq_ignore_ascii_case("true"),
            "Terminal" => terminal = value.eq_ignore_ascii_case("true"),
            "Categories" => {
                categories = value
                    .split(';')
                    .filter(|item| !item.is_empty())
                    .map(ToString::to_string)
                    .collect();
            }
            _ => {}
        }
    }

    if !is_application || name.is_none() || hidden || no_display {
        return Ok(None);
    }
    let relative = path.strip_prefix(root).unwrap_or(path);
    let id = relative
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "-");
    Ok(Some(json!({
        "id": id,
        "name": name,
        "exec": exec,
        "path": path.to_string_lossy(),
        "no_display": no_display,
        "terminal": terminal,
        "categories": categories,
    })))
}
