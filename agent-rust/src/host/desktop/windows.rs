use super::*;

pub(super) fn window_control_action(action: &str, target: &str) -> Result<Value> {
    if !valid_window_id(target) {
        return Err(anyhow::anyhow!(
            "{action} requires a concrete window id from desktop_observe"
        ));
    }
    match action {
        "focus_window" => {
            if command_exists("wmctrl") {
                window_control_receipt(action, target, &["wmctrl", "-ia", target], |_, after| {
                    window_state_bool(after, "active") == Some(true)
                })
            } else if command_exists("xdotool") {
                window_control_receipt(
                    action,
                    target,
                    &["xdotool", "windowactivate", target],
                    |_, after| window_state_bool(after, "active") == Some(true),
                )
            } else if is_wsl_runtime() && windows_host_powershell_available() {
                windows_host_focus_window_receipt(action, target)
            } else {
                Ok(json!({
                    "ok": false,
                    "tool": "desktop_action",
                    "action": action,
                    "target": target,
                    "reason": "dependency_unavailable",
                    "error": "Window focus requires wmctrl, xdotool, or reachable Windows PowerShell under WSL.",
                }))
            }
        }
        "minimize_window" => window_control_receipt(
            action,
            target,
            &["xdotool", "windowminimize", target],
            |_, after| window_has_state(after, "_NET_WM_STATE_HIDDEN"),
        ),
        "maximize_window" => window_control_receipt(
            action,
            target,
            &[
                "wmctrl",
                "-ir",
                target,
                "-b",
                "add,maximized_vert,maximized_horz",
            ],
            |_, after| {
                window_has_state(after, "_NET_WM_STATE_MAXIMIZED_VERT")
                    && window_has_state(after, "_NET_WM_STATE_MAXIMIZED_HORZ")
            },
        ),
        "restore_window" => window_control_receipt(
            action,
            target,
            &[
                "wmctrl",
                "-ir",
                target,
                "-b",
                "remove,maximized_vert,maximized_horz",
            ],
            |_, after| {
                !window_has_state(after, "_NET_WM_STATE_MAXIMIZED_VERT")
                    && !window_has_state(after, "_NET_WM_STATE_MAXIMIZED_HORZ")
            },
        ),
        "close_window" => {
            if command_exists("wmctrl") {
                window_control_receipt(
                    action,
                    target,
                    &["wmctrl", "-ic", target],
                    |before, after| {
                        window_state_bool(before, "visible") == Some(true)
                            && window_state_bool(after, "visible") == Some(false)
                    },
                )
            } else if command_exists("xdotool") {
                window_control_receipt(
                    action,
                    target,
                    &["xdotool", "windowclose", target],
                    |before, after| {
                        window_state_bool(before, "visible") == Some(true)
                            && window_state_bool(after, "visible") == Some(false)
                    },
                )
            } else if is_wsl_runtime() && windows_host_powershell_available() {
                windows_host_close_window_receipt(action, target)
            } else {
                Ok(json!({
                    "ok": false,
                    "tool": "desktop_action",
                    "action": action,
                    "target": target,
                    "reason": "dependency_unavailable",
                    "error": "Window close requires wmctrl, xdotool, or reachable Windows PowerShell under WSL.",
                }))
            }
        }
        _ => Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "reason": "unsupported_action",
            "error": format!("Unsupported window action: {action}"),
        })),
    }
}

pub(super) fn window_control_receipt<F>(
    action: &str,
    target: &str,
    argv: &[&str],
    verify: F,
) -> Result<Value>
where
    F: Fn(&Value, &Value) -> bool,
{
    let Some((program, _)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty window command"));
    };
    if !command_exists(program) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "reason": "dependency_unavailable",
            "error": format!("Required window command `{program}` is not installed."),
        }));
    }
    let before = window_state(target)?;
    let output = run_capture_dynamic(argv)?;
    let (after, verified) = if output.status == 0 {
        wait_for_window_verification(target, &before, &verify)?
    } else {
        (window_state(target)?, false)
    };
    let (ok, verification, reason) = window_action_outcome(output.status, verified);
    Ok(json!({
        "ok": ok,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "backend": program,
        "exit_code": output.status,
        "before": before,
        "after": after,
        "verified": verified,
        "verification": verification,
        "reason": reason,
        "recoverable": !ok,
        "guidance": if output.status == 0 && !verified {
            Some("The window action was dispatched, but its postcondition was not observed. Re-observe the window before reporting success.")
        } else {
            None
        },
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn wait_for_window_verification<F>(
    target: &str,
    before: &Value,
    verify: &F,
) -> Result<(Value, bool)>
where
    F: Fn(&Value, &Value) -> bool,
{
    const ATTEMPTS: usize = 20;
    const INTERVAL: Duration = Duration::from_millis(100);

    let mut after = window_state(target)?;
    for attempt in 0..ATTEMPTS {
        if verify(before, &after) {
            return Ok((after, true));
        }
        if attempt + 1 < ATTEMPTS {
            thread::sleep(INTERVAL);
            after = window_state(target)?;
        }
    }
    Ok((after, false))
}

pub(super) fn window_action_outcome(
    status: i32,
    verified: bool,
) -> (bool, &'static str, Option<&'static str>) {
    if status != 0 {
        return (false, "failed", Some("command_failed"));
    }
    if !verified {
        return (false, "not_confirmed", Some("verification_failed"));
    }
    (true, "confirmed", None)
}

pub(super) fn window_state(target: &str) -> Result<Value> {
    let windows = visible_windows(200)?;
    let target_normalized = normalize_window_id(target);
    let matched = windows
        .get("items")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find(|item| {
                item.get("id")
                    .and_then(Value::as_str)
                    .and_then(normalize_window_id)
                    == target_normalized
            })
        });
    let active = active_window_id()?;
    let states = net_wm_states(target)?;
    Ok(json!({
        "id": target,
        "visible": matched.is_some(),
        "title": matched.and_then(|item| item.get("title")).cloned(),
        "active": target_normalized.is_some() && active.is_some() && target_normalized == active,
        "net_wm_state": states,
        "windows_backend": windows.get("backend").cloned(),
    }))
}

pub(super) fn active_window_id() -> Result<Option<String>> {
    if command_exists("xdotool") {
        let output = run_capture_dynamic(&["xdotool", "getactivewindow"])?;
        if output.status == 0 {
            return Ok(normalize_window_id(output.stdout.trim()));
        }
    }
    if command_exists("xprop") {
        let output = run_capture_dynamic(&["xprop", "-root", "_NET_ACTIVE_WINDOW"])?;
        if output.status == 0 {
            if let Some((_, raw)) = output.stdout.rsplit_once(' ') {
                return Ok(normalize_window_id(raw.trim().trim_end_matches(',')));
            }
        }
    }
    if is_wsl_runtime() && windows_host_powershell_available() {
        let active = windows_host_active_window()?;
        return Ok(active
            .pointer("/window/id")
            .and_then(Value::as_str)
            .and_then(normalize_window_id));
    }
    Ok(None)
}

pub(super) fn net_wm_states(target: &str) -> Result<Value> {
    if !command_exists("xprop") {
        return Ok(json!({
            "available": false,
            "backend": "unavailable",
            "states": [],
            "reason": "dependency_unavailable",
        }));
    }
    let output = run_capture_dynamic(&["xprop", "-id", target, "_NET_WM_STATE"])?;
    let states = if output.status == 0 {
        output
            .stdout
            .split_once('=')
            .map(|(_, value)| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|state| state.starts_with("_NET_WM_STATE"))
                    .map(ToString::to_string)
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    Ok(json!({
        "available": output.status == 0,
        "backend": "xprop",
        "states": states,
        "stderr": output.stderr,
    }))
}

pub(super) fn window_state_bool(state: &Value, key: &str) -> Option<bool> {
    state.get(key).and_then(Value::as_bool)
}

pub(super) fn window_has_state(state: &Value, name: &str) -> bool {
    state
        .get("net_wm_state")
        .and_then(|value| value.get("states"))
        .and_then(Value::as_array)
        .is_some_and(|states| states.iter().any(|state| state.as_str() == Some(name)))
}

pub(super) fn valid_window_id(value: &str) -> bool {
    normalize_window_id(value).is_some()
}

pub(super) fn normalize_window_id(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let parsed = if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()?
    } else {
        trimmed.parse::<u64>().ok()?
    };
    Some(format!("0x{parsed:x}"))
}

pub(super) fn visible_windows(limit: usize) -> Result<Value> {
    if command_exists("wmctrl") {
        return linux_windows(limit);
    }
    if windows_host_powershell_available() {
        return windows_host_windows(limit);
    }
    Ok(json!({
        "ok": false,
        "backend": "unavailable",
        "reason": "backend_unavailable",
        "items": [],
        "count": 0,
        "error": "Window listing requires wmctrl, or Windows PowerShell under WSL.",
    }))
}

pub(super) fn linux_windows(limit: usize) -> Result<Value> {
    let output = run_capture_dynamic(&["wmctrl", "-lx"])?;
    let mut items = Vec::new();
    for line in output.stdout.lines().take(limit) {
        let mut fields = line.split_whitespace();
        let Some(id) = fields.next() else {
            continue;
        };
        let Some(desktop) = fields.next() else {
            continue;
        };
        let Some(pid) = fields.next() else {
            continue;
        };
        let Some(class) = fields.next() else {
            continue;
        };
        if fields.next().is_none() {
            continue;
        }
        let title_start = nth_field_start(line, 4).unwrap_or(line.len());
        items.push(json!({
            "id": id,
            "desktop": desktop,
            "pid": pid.parse::<u32>().ok(),
            "class": class,
            "title": line[title_start..].trim(),
            "backend": "wmctrl",
        }));
    }
    Ok(json!({
        "ok": output.status == 0,
        "backend": "wmctrl",
        "count": items.len(),
        "items": items,
        "truncated": output.stdout.lines().count() > limit,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_windows(limit: usize) -> Result<Value> {
    let script = windows_host_window_list_script(limit);
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])?;
    let raw = serde_json::from_str::<Value>(&output.stdout).unwrap_or(Value::Null);
    let values = json_values(raw);
    let items: Vec<Value> = values
        .into_iter()
        .map(|item| {
            json!({
                "id": item.get("Id").cloned(),
                "pid": item.get("Pid").cloned(),
                "process": item.get("ProcessName").cloned(),
                "title": item.get("MainWindowTitle").cloned(),
                "path": item.get("Path").cloned(),
                "backend": "windows_host_powershell",
            })
        })
        .collect();
    Ok(json!({
        "ok": output.status == 0,
        "backend": "windows_host_powershell",
        "count": items.len(),
        "items": items,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_window_list_script(limit: usize) -> String {
    r#"Get-Process | Where-Object {$_.MainWindowTitle -and $_.MainWindowHandle -ne 0} | Select-Object -First __LIMIT__ @{Name='Id';Expression={'0x{0:x}' -f $_.MainWindowHandle.ToInt64()}},@{Name='Pid';Expression={$_.Id}},ProcessName,MainWindowTitle,@{Name='Path';Expression={try {$_.Path} catch {$null}}} | ConvertTo-Json -Compress"#
        .replace("__LIMIT__", &limit.to_string())
}

pub(super) fn active_window() -> Result<Value> {
    if command_exists("xdotool") {
        return linux_active_window();
    }
    if windows_host_powershell_available() {
        return windows_host_active_window();
    }
    Ok(json!({
        "ok": false,
        "backend": "unavailable",
        "reason": "backend_unavailable",
        "error": "Active-window observation requires xdotool, or Windows PowerShell under WSL.",
    }))
}

pub(super) fn linux_active_window() -> Result<Value> {
    let id = run_capture_dynamic(&["xdotool", "getactivewindow"])?;
    if id.status != 0 || id.stdout.is_empty() {
        return Ok(json!({
            "ok": false,
            "backend": "xdotool",
            "reason": "active_window_unavailable",
            "stderr": id.stderr,
        }));
    }
    let name = run_capture_dynamic(&["xdotool", "getwindowname", &id.stdout])?;
    let pid = run_capture_dynamic(&["xdotool", "getwindowpid", &id.stdout]).ok();
    Ok(json!({
        "ok": name.status == 0,
        "backend": "xdotool",
        "id": id.stdout,
        "title": name.stdout,
        "pid": pid.as_ref().and_then(|output| output.stdout.parse::<u32>().ok()),
        "stderr": name.stderr,
    }))
}

pub(super) fn windows_host_active_window() -> Result<Value> {
    let script = r#"
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class AgentUser32 {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}
'@
$hwnd = [AgentUser32]::GetForegroundWindow()
$builder = New-Object System.Text.StringBuilder 1024
[void][AgentUser32]::GetWindowText($hwnd, $builder, $builder.Capacity)
$processId = 0
[void][AgentUser32]::GetWindowThreadProcessId($hwnd, [ref]$processId)
$process = if ($processId) { Get-Process -Id $processId -ErrorAction SilentlyContinue } else { $null }
@{ id = ('0x{0:x}' -f $hwnd.ToInt64()); pid = $processId; title = $builder.ToString(); process = $process.ProcessName; path = $process.Path } | ConvertTo-Json -Compress
"#;
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        script,
    ])?;
    let item = serde_json::from_str::<Value>(&output.stdout).unwrap_or(Value::Null);
    Ok(json!({
        "ok": output.status == 0 && !item.is_null(),
        "backend": "windows_host_powershell",
        "window": item,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_focus_window_receipt(action: &str, target: &str) -> Result<Value> {
    let before = window_state(target)?;
    let output = windows_host_focus_window(target)?;
    thread::sleep(Duration::from_millis(100));
    let after = window_state(target)?;
    let verified = output.status == 0 && window_state_bool(&after, "active") == Some(true);
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "backend": "windows_host_powershell",
        "exit_code": output.status,
        "before": before,
        "after": after,
        "verified": verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_focus_window(target: &str) -> Result<CommandOutput> {
    let script = windows_host_focus_window_script(target)?;
    run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])
}

pub(super) fn windows_host_focus_window_script(target: &str) -> Result<String> {
    let normalized = normalize_window_id(target)
        .ok_or_else(|| anyhow::anyhow!("focus_window requires a concrete window id"))?;
    let handle = u64::from_str_radix(normalized.trim_start_matches("0x"), 16)?;
    Ok(format!(
        r#"
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class AgentUser32 {{
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
}}
'@
$hwnd = [IntPtr]{handle}
if (-not [AgentUser32]::IsWindow($hwnd)) {{ Write-Error "Invalid window handle"; exit 2 }}
[void][AgentUser32]::ShowWindowAsync($hwnd, 9)
Start-Sleep -Milliseconds 50
[void][AgentUser32]::SetForegroundWindow($hwnd)
Start-Sleep -Milliseconds 50
$active = [AgentUser32]::GetForegroundWindow().ToInt64()
@{{ target = {handle}; active = $active; focused = ($active -eq {handle}) }} | ConvertTo-Json -Compress
"#
    ))
}

pub(super) fn windows_host_terminate_process_receipt(
    action: &str,
    raw_pid: &str,
    pid: i32,
) -> Result<Value> {
    let before = windows_host_process_state(pid)?;
    let output = windows_host_terminate_process(pid)?;
    thread::sleep(Duration::from_millis(250));
    let after = windows_host_process_state(pid)?;
    let after_available = after.get("available").and_then(Value::as_bool) == Some(true);
    let after_running = after.get("running").and_then(Value::as_bool) == Some(true);
    let before_running = before.get("running").and_then(Value::as_bool) == Some(true);
    let verified = output.status == 0 && after_available && !after_running;
    Ok(json!({
        "ok": output.status == 0 && verified,
        "tool": "desktop_action",
        "action": action,
        "target": raw_pid,
        "backend": "windows_host_powershell",
        "exit_code": output.status,
        "reason": if before_running { None::<&str> } else { Some("already_not_running") },
        "before": before,
        "after": after,
        "verified": verified,
        "verification": if verified { "confirmed" } else if output.status != 0 { "failed" } else { "not_confirmed" },
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_terminate_process(pid: i32) -> Result<CommandOutput> {
    let script = windows_host_terminate_process_script(pid)?;
    run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])
}

pub(super) fn windows_host_terminate_process_script(pid: i32) -> Result<String> {
    if pid <= 1 {
        return Err(anyhow::anyhow!("Refusing to terminate protected PID {pid}"));
    }
    Ok(format!(
        r#"
$processId = {pid}
$process = Get-Process -Id $processId -ErrorAction SilentlyContinue
if ($null -eq $process) {{ Write-Output "not_running"; exit 0 }}
Stop-Process -Id $processId -Force -ErrorAction Stop
Write-Output "stopped"
"#
    ))
}

pub(super) fn windows_host_process_state(pid: i32) -> Result<Value> {
    let script = windows_host_process_state_script(pid)?;
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])?;
    let item = serde_json::from_str::<Value>(&output.stdout).unwrap_or(Value::Null);
    if output.status == 0 && item.is_object() {
        return Ok(item);
    }
    Ok(json!({
        "pid": pid,
        "running": false,
        "available": false,
        "backend": "windows_host_powershell",
        "error": output.stderr,
    }))
}

pub(super) fn windows_host_process_state_script(pid: i32) -> Result<String> {
    if pid <= 0 {
        return Err(anyhow::anyhow!("PID must be positive"));
    }
    Ok(format!(
        r#"
$processId = {pid}
$process = Get-Process -Id $processId -ErrorAction SilentlyContinue
if ($null -eq $process) {{
    @{{ pid = $processId; running = $false; name = $null; path = $null; available = $true; backend = "windows_host_powershell" }} | ConvertTo-Json -Compress
    exit 0
}}
$processPath = try {{ $process.Path }} catch {{ $null }}
@{{ pid = $process.Id; running = $true; name = $process.ProcessName; path = $processPath; available = $true; backend = "windows_host_powershell" }} | ConvertTo-Json -Compress
"#
    ))
}

pub(super) fn windows_host_close_window_receipt(action: &str, target: &str) -> Result<Value> {
    let before = window_state(target)?;
    let output = windows_host_close_window(target)?;
    let (after, verified) = if output.status == 0 {
        wait_for_window_verification(target, &before, &|_, after| {
            window_state_bool(after, "visible") == Some(false)
        })?
    } else {
        (window_state(target)?, false)
    };
    let (ok, verification, reason) = window_action_outcome(output.status, verified);
    Ok(json!({
        "ok": ok,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "backend": "windows_host_powershell",
        "exit_code": output.status,
        "before": before,
        "after": after,
        "verified": verified,
        "verification": verification,
        "reason": reason,
        "recoverable": !ok,
        "guidance": if output.status == 0 && !verified {
            Some("The close request was dispatched, but the window is still visible. Re-observe it before reporting success.")
        } else {
            None
        },
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

pub(super) fn windows_host_close_window(target: &str) -> Result<CommandOutput> {
    let script = windows_host_close_window_script(target)?;
    run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])
}

pub(super) fn windows_host_close_window_script(target: &str) -> Result<String> {
    let normalized = normalize_window_id(target)
        .ok_or_else(|| anyhow::anyhow!("close_window requires a concrete window id"))?;
    let handle = u64::from_str_radix(normalized.trim_start_matches("0x"), 16)?;
    Ok(format!(
        r#"
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class AgentUser32 {{
    [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hWnd);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}}
'@
$hwnd = [IntPtr]{handle}
if (-not [AgentUser32]::IsWindow($hwnd)) {{ Write-Error "Invalid window handle"; exit 2 }}
$processId = 0
[void][AgentUser32]::GetWindowThreadProcessId($hwnd, [ref]$processId)
$ok = [AgentUser32]::PostMessage($hwnd, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)
Start-Sleep -Milliseconds 500
$forced = $false
if ([AgentUser32]::IsWindow($hwnd) -and $processId -gt 0) {{
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
    $forced = $true
    Start-Sleep -Milliseconds 250
}}
@{{ target = {handle}; pid = $processId; close_requested = $ok; forced = $forced }} | ConvertTo-Json -Compress
"#
    ))
}
