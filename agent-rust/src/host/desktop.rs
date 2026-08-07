use super::{
    bluetooth_info_value, command_exists, is_wsl_runtime, read_trimmed, required_target,
    run_capture_dynamic,
};
use anyhow::Result;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

const WINDOWS_SHORTCUT_PREFIX: &str = "windows-shortcut:";
const WINDOWS_APP_PREFIX: &str = "windows-app:";

pub(crate) fn desktop_capabilities() -> Result<Value> {
    let runtime = desktop_runtime();
    let linux_supported = cfg!(target_os = "linux");
    let wsl = is_wsl_runtime();
    let has_pactl = command_exists("pactl");
    let has_powershell_host = windows_host_powershell_available();
    let has_gtk_launch = command_exists("gtk-launch");
    let has_xdg_open = command_exists("xdg-open");
    let accessibility = accessibility_backend();

    Ok(json!({
        "ok": true,
        "tool": "desktop_capabilities",
        "schema_version": 1,
        "runtime": runtime,
        "platform": std::env::consts::OS,
        "wsl": wsl,
        "backends": {
            "desktop_observation": backend_status(
                linux_supported
                    && (command_exists("wmctrl")
                        || command_exists("xdotool")
                        || desktop_application_dirs().iter().any(|path| path.exists())
                        || (wsl && has_powershell_host)),
                desktop_observation_backend(wsl),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "audio": backend_status(
                linux_supported && (has_pactl || (wsl && has_powershell_host)),
                if has_pactl {
                    "pactl"
                } else if wsl && has_powershell_host {
                    "windows_host_powershell"
                } else {
                    "unavailable"
                },
                if linux_supported {
                    None
                } else {
                    Some("unsupported_platform")
                },
            ),
            "brightness": backend_status(
                linux_supported && command_exists("brightnessctl"),
                "brightnessctl",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "bluetooth": backend_status(
                linux_supported && command_exists("bluetoothctl"),
                "bluetoothctl",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "network": backend_status(
                linux_supported && command_exists("ip") && PathBuf::from("/sys/class/net").exists(),
                "ip",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "storage": backend_status(
                linux_supported && command_exists("udisksctl"),
                "udisksctl",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "process": backend_status(
                linux_supported,
                "libc_signal",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "application_launch": backend_status(
                linux_supported && (has_gtk_launch || has_xdg_open),
                if has_gtk_launch {
                    "gtk-launch"
                } else if has_xdg_open {
                    "xdg-open"
                } else {
                    "path_lookup"
                },
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "open": backend_status(
                linux_supported && has_xdg_open,
                "xdg-open",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "window_control": backend_status(
                linux_supported
                    && (command_exists("wmctrl")
                        || command_exists("xdotool")
                        || (wsl && has_powershell_host)),
                window_control_backend(),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "clipboard": backend_status(
                linux_supported && clipboard_backend() != "unavailable",
                clipboard_backend(),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "keyboard_input": backend_status(
                linux_supported && keyboard_backend() != "unavailable",
                keyboard_backend(),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "pointer_input": backend_status(
                linux_supported && pointer_backend() != "unavailable",
                pointer_backend(),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "accessibility_tree": backend_status(
                linux_supported && accessibility != "unavailable",
                accessibility,
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "display_observation": backend_status(
                linux_supported && display_backend() != "unavailable",
                display_backend(),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
        },
        "actions": [
            action_capability("set_volume", linux_supported && (has_pactl || (wsl && has_powershell_host)), if has_pactl { "pactl" } else if wsl && has_powershell_host { "windows_host_powershell" } else { "unavailable" }, "approval_required", None),
            action_capability("set_mute", linux_supported && has_pactl, "pactl", "approval_required", None),
            action_capability("set_brightness", linux_supported && command_exists("brightnessctl"), "brightnessctl", "approval_required", None),
            action_capability("bluetooth_connect", linux_supported && command_exists("bluetoothctl"), "bluetoothctl", "approval_required", Some("target_required")),
            action_capability("bluetooth_disconnect", linux_supported && command_exists("bluetoothctl"), "bluetoothctl", "approval_required", Some("target_required")),
            action_capability("network_connect", linux_supported && command_exists("ip"), "ip", "approval_required", Some("target_required")),
            action_capability("network_disconnect", linux_supported && command_exists("ip"), "ip", "approval_required", Some("target_required")),
            action_capability("eject_storage", linux_supported && command_exists("udisksctl"), "udisksctl", "approval_required", Some("target_required")),
            action_capability(
                "terminate_process",
                linux_supported,
                if wsl && has_powershell_host { "libc_signal/windows_host_powershell" } else { "libc_signal" },
                "approval_required",
                Some("target_required"),
            ),
            action_capability("launch_application", linux_supported && (has_gtk_launch || has_xdg_open || (wsl && has_powershell_host)), if has_gtk_launch { "gtk-launch" } else if has_xdg_open { "xdg-open" } else if wsl && has_powershell_host { "windows_host_powershell" } else { "path_lookup" }, "direct", Some("target_required")),
            action_capability("open_path", linux_supported && has_xdg_open, "xdg-open", "approval_required", Some("target_required")),
            action_capability("open_url", linux_supported && has_xdg_open, "xdg-open", "approval_required", Some("target_required")),
            action_capability("focus_window", linux_supported && (command_exists("wmctrl") || command_exists("xdotool") || (wsl && has_powershell_host)), window_control_backend(), "approval_required", Some("target_required")),
            action_capability("minimize_window", linux_supported && command_exists("xdotool"), "xdotool", "approval_required", Some("target_required")),
            action_capability("maximize_window", linux_supported && command_exists("wmctrl"), "wmctrl", "approval_required", Some("target_required")),
            action_capability("restore_window", linux_supported && command_exists("wmctrl"), "wmctrl", "approval_required", Some("target_required")),
            action_capability("close_window", linux_supported && (command_exists("wmctrl") || command_exists("xdotool") || (wsl && has_powershell_host)), window_control_backend(), "direct", Some("target_required")),
            action_capability("clipboard_write", linux_supported && clipboard_backend() != "unavailable", clipboard_backend(), "approval_required", None),
            action_capability("clipboard_files", linux_supported && file_clipboard_backend() != "unavailable", file_clipboard_backend(), "approval_required", Some("existing_paths_required")),
            action_capability("send_key", linux_supported && command_exists("xdotool"), "xdotool", "approval_required", None),
            action_capability("type_text", linux_supported && keyboard_backend() != "unavailable", keyboard_backend(), "approval_required", None),
            action_capability("mouse_click", linux_supported && pointer_backend() != "unavailable", pointer_backend(), "approval_required", Some("target_required")),
            action_capability("scroll", linux_supported && pointer_backend() != "unavailable", pointer_backend(), "approval_required", None),
            action_capability("focus_element", linux_supported && accessibility == "atspi_dbus", accessibility, "approval_required", Some("observed_element_required")),
            action_capability("invoke_element", linux_supported && accessibility == "atspi_dbus", accessibility, "approval_required", Some("observed_element_and_action_required")),
            action_capability("set_field_text", linux_supported && accessibility == "atspi_dbus", accessibility, "approval_required", Some("observed_editable_element_required")),
        ],
        "limitations": desktop_limitations(linux_supported, wsl),
    }))
}

pub(crate) fn desktop_observe(scope: &str, limit: usize) -> Result<Value> {
    let normalized_scope = match scope {
        "all" | "applications" | "windows" | "active_window" | "clipboard" | "ui_tree"
        | "displays" | "audio" | "dialogs" | "downloads" => scope,
        "active-window" => "active_window",
        "clipboard_metadata" | "clipboard-metadata" => "clipboard",
        "accessibility" | "accessibility_tree" | "accessibility-tree" => "ui_tree",
        "display" | "screen" | "screens" | "monitor" | "monitors" => "displays",
        _ => "all",
    };
    let bounded_limit = limit.clamp(1, 200);
    let linux_supported = cfg!(target_os = "linux");
    let wsl = is_wsl_runtime();
    let has_powershell_host = windows_host_powershell_available();
    let accessibility = accessibility_backend();
    let snapshot_id = format!("desktop-{}", now_unix_millis());
    let mut payload = json!({
        "ok": linux_supported,
        "tool": "desktop_observe",
        "schema_version": 1,
        "scope": normalized_scope,
        "runtime": desktop_runtime(),
        "platform": std::env::consts::OS,
        "wsl": wsl,
        "snapshot_id": snapshot_id,
        "observed_at_unix_ms": now_unix_millis(),
        "backends": {
            "applications": backend_status(
                linux_supported && desktop_application_dirs().iter().any(|path| path.exists()),
                "freedesktop_desktop_entries",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "windows": backend_status(
                linux_supported && (command_exists("wmctrl") || (wsl && has_powershell_host)),
                if command_exists("wmctrl") {
                    "wmctrl"
                } else if wsl && has_powershell_host {
                    "windows_host_powershell"
                } else {
                    "unavailable"
                },
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "active_window": backend_status(
                linux_supported && (command_exists("xdotool") || (wsl && has_powershell_host)),
                if command_exists("xdotool") {
                    "xdotool"
                } else if wsl && has_powershell_host {
                    "windows_host_powershell"
                } else {
                    "unavailable"
                },
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "clipboard": backend_status(
                linux_supported && clipboard_backend() != "unavailable",
                clipboard_backend(),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "ui_tree": backend_status(
                linux_supported && accessibility != "unavailable",
                accessibility,
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "displays": backend_status(
                linux_supported && display_backend() != "unavailable",
                display_backend(),
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "audio": backend_status(
                linux_supported && command_exists("pactl"),
                if command_exists("pactl") { "pactl" } else { "unavailable" },
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "dialogs": backend_status(
                linux_supported && accessibility != "unavailable",
                accessibility,
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
            "downloads": backend_status(
                linux_supported && downloads_directory().is_some_and(|path| path.is_dir()),
                "freedesktop_user_dirs",
                if linux_supported { None } else { Some("unsupported_platform") },
            ),
        },
        "limitations": desktop_observation_limitations(linux_supported, wsl),
    });

    if !linux_supported {
        if let Some(object) = payload.as_object_mut() {
            object.insert("reason".to_string(), json!("unsupported_platform"));
            object.insert(
                "error".to_string(),
                json!("Desktop observation is currently implemented for Linux and WSL runtimes."),
            );
        }
        return Ok(payload);
    }

    if matches!(normalized_scope, "all" | "applications") {
        insert_payload_value(
            &mut payload,
            "applications",
            json!(installed_applications(bounded_limit)?),
        );
    }
    if matches!(normalized_scope, "all" | "windows") {
        insert_payload_value(
            &mut payload,
            "windows",
            json!(visible_windows(bounded_limit)?),
        );
    }
    if matches!(normalized_scope, "all" | "active_window") {
        insert_payload_value(&mut payload, "active_window", active_window()?);
    }
    if matches!(normalized_scope, "all" | "clipboard") {
        insert_payload_value(&mut payload, "clipboard", clipboard_metadata()?);
    }
    if normalized_scope == "ui_tree" {
        insert_payload_value(
            &mut payload,
            "ui_tree",
            accessibility_tree(&snapshot_id, bounded_limit)?,
        );
    }
    if matches!(normalized_scope, "all" | "displays") {
        insert_payload_value(&mut payload, "displays", display_inventory(bounded_limit)?);
    }
    if normalized_scope == "audio" {
        insert_payload_value(&mut payload, "audio", audio_observation()?);
    }
    if normalized_scope == "dialogs" {
        insert_payload_value(
            &mut payload,
            "dialogs",
            dialog_inventory(&snapshot_id, bounded_limit)?,
        );
    }
    if normalized_scope == "downloads" {
        insert_payload_value(
            &mut payload,
            "downloads",
            download_inventory(bounded_limit)?,
        );
    }

    Ok(payload)
}

pub(crate) fn desktop_resolve(query: &str, kind: &str, limit: usize) -> Result<Value> {
    let normalized_kind = match kind {
        "application" | "window" | "any" => kind,
        _ => "any",
    };
    let normalized_query = normalize_match_text(query);
    if normalized_query.is_empty() {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_resolve",
            "reason": "query_required",
            "error": "desktop_resolve requires a non-empty query.",
        }));
    }
    let bounded_limit = limit.clamp(1, 50);
    let mut candidates = Vec::new();

    if matches!(normalized_kind, "application" | "any") {
        candidates.extend(resolve_applications(&normalized_query, bounded_limit)?);
    }
    if matches!(normalized_kind, "window" | "any") {
        candidates.extend(resolve_windows(&normalized_query, bounded_limit)?);
    }

    candidates.sort_by(|left, right| {
        candidate_score(right)
            .cmp(&candidate_score(left))
            .then_with(|| candidate_label(left).cmp(&candidate_label(right)))
    });
    candidates.truncate(bounded_limit);
    let top_score = candidates.first().map(candidate_score).unwrap_or(0);
    let top_count = candidates
        .iter()
        .filter(|candidate| candidate_score(candidate) == top_score && top_score > 0)
        .count();

    Ok(json!({
        "ok": true,
        "tool": "desktop_resolve",
        "schema_version": 1,
        "query": query,
        "kind": normalized_kind,
        "count": candidates.len(),
        "ambiguous": top_count > 1,
        "candidates": candidates,
    }))
}

pub(crate) fn desktop_action(
    action: &str,
    target: Option<&str>,
    value: Option<&str>,
    backend_bus: Option<&str>,
    backend_path: Option<&str>,
) -> Result<Value> {
    if !cfg!(target_os = "linux") {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "unsupported_platform",
            "runtime": std::env::consts::OS,
            "error": "Desktop actions are currently implemented for Linux and WSL runtimes.",
        }));
    }

    match action {
        "set_volume" => {
            let percent = required_percent(action, value)?;
            if command_exists("pactl") {
                desktop_command_receipt(
                    action,
                    target,
                    value,
                    &[
                        "pactl",
                        "set-sink-volume",
                        "@DEFAULT_SINK@",
                        &format!("{percent}%"),
                    ],
                    || audio_state("volume"),
                )
            } else if windows_host_powershell_available() {
                windows_host_set_volume(percent, target, value)
            } else {
                Ok(json!({
                    "ok": false,
                    "tool": "desktop_action",
                    "action": action,
                    "target": target,
                    "value": value,
                    "reason": "dependency_unavailable",
                    "error": "Volume control requires `pactl`, or reachable Windows PowerShell when running under WSL.",
                }))
            }
        }
        "set_mute" => {
            let setting = value.unwrap_or("toggle");
            if !matches!(setting, "true" | "false" | "toggle") {
                return Err(anyhow::anyhow!(
                    "set_mute value must be true, false, or toggle"
                ));
            }
            desktop_command_receipt(
                action,
                target,
                Some(setting),
                &["pactl", "set-sink-mute", "@DEFAULT_SINK@", setting],
                || audio_state("mute"),
            )
        }
        "set_brightness" => {
            let percent = required_percent(action, value)?;
            desktop_command_receipt(
                action,
                target,
                value,
                &["brightnessctl", "set", &format!("{percent}%")],
                brightness_state,
            )
        }
        "bluetooth_connect" | "bluetooth_disconnect" => {
            let address = required_target(action, target)?;
            if !valid_bluetooth_address(address) {
                return Err(anyhow::anyhow!("{action} requires a Bluetooth MAC address"));
            }
            let verb = if action == "bluetooth_connect" {
                "connect"
            } else {
                "disconnect"
            };
            desktop_command_receipt(
                action,
                target,
                value,
                &["bluetoothctl", verb, address],
                || bluetooth_connection_state(address),
            )
        }
        "network_connect" | "network_disconnect" => {
            let interface = required_target(action, target)?;
            if !valid_identifier(interface) {
                return Err(anyhow::anyhow!(
                    "{action} requires a valid network interface name"
                ));
            }
            let state = if action == "network_connect" {
                "up"
            } else {
                "down"
            };
            desktop_command_receipt(
                action,
                target,
                value,
                &["ip", "link", "set", "dev", interface, state],
                || network_connection_state(interface),
            )
        }
        "eject_storage" => {
            let device = required_target(action, target)?;
            if !device.starts_with("/dev/") || !valid_path_token(device) {
                return Err(anyhow::anyhow!(
                    "eject_storage requires a concrete /dev device path"
                ));
            }
            desktop_command_receipt(
                action,
                target,
                value,
                &["udisksctl", "unmount", "-b", device],
                || storage_mount_state(device),
            )
        }
        "terminate_process" => {
            let raw_pid = required_target(action, target)?;
            let pid = raw_pid
                .parse::<i32>()
                .map_err(|_| anyhow::anyhow!("terminate_process requires a numeric PID"))?;
            terminate_process_action(action, raw_pid, pid)
        }
        "launch_application" => {
            launch_desktop_target(action, required_target(action, target)?, false)
        }
        "open_path" => launch_desktop_target(action, required_target(action, target)?, true),
        "open_url" => {
            let url = required_target(action, target)?;
            if !(url.starts_with("https://") || url.starts_with("http://")) {
                return Err(anyhow::anyhow!(
                    "open_url only accepts http:// or https:// URLs"
                ));
            }
            launch_desktop_target(action, url, true)
        }
        "focus_window" | "minimize_window" | "maximize_window" | "restore_window"
        | "close_window" => window_control_action(action, required_target(action, target)?),
        "clipboard_write" => clipboard_write_action(action, value),
        "send_key" => keyboard_action(action, value),
        "type_text" => keyboard_action(action, value),
        "mouse_click" => pointer_action(action, target, value),
        "scroll" => pointer_action(action, target, value),
        "focus_element" | "invoke_element" | "set_field_text" => ui_element_action(
            action,
            required_target(action, target)?,
            value,
            backend_bus,
            backend_path,
        ),
        _ => Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "unsupported_action",
            "error": format!("Unsupported desktop action: {action}"),
        })),
    }
}

fn terminate_process_action(action: &str, raw_pid: &str, pid: i32) -> Result<Value> {
    if pid <= 1 || pid == std::process::id() as i32 {
        return Err(anyhow::anyhow!("Refusing to terminate protected PID {pid}"));
    }

    let before = process_state(pid);
    let before_running = before.get("running").and_then(Value::as_bool) == Some(true);
    if before_running {
        let status = unsafe { libc::kill(pid, libc::SIGTERM) };
        thread::sleep(Duration::from_millis(75));
        let after = process_state(pid);
        let stopped = after.get("running").and_then(Value::as_bool) == Some(false);
        return Ok(json!({
            "ok": status == 0,
            "tool": "desktop_action",
            "action": action,
            "target": raw_pid,
            "backend": "libc_signal",
            "before": before,
            "after": after,
            "verified": status == 0 && stopped,
            "verification": if status == 0 && stopped { "confirmed" } else { "not_confirmed" },
            "error": if status == 0 { None } else { Some(std::io::Error::last_os_error().to_string()) },
        }));
    }

    if is_wsl_runtime() && windows_host_powershell_available() {
        return windows_host_terminate_process_receipt(action, raw_pid, pid);
    }

    Ok(json!({
        "ok": true,
        "tool": "desktop_action",
        "action": action,
        "target": raw_pid,
        "backend": "libc_signal",
        "reason": "already_not_running",
        "before": before,
        "after": process_state(pid),
        "verified": true,
        "verification": "confirmed",
        "error": None::<String>,
    }))
}

fn desktop_runtime() -> &'static str {
    if is_wsl_runtime() {
        "wsl"
    } else {
        std::env::consts::OS
    }
}

fn desktop_observation_backend(wsl: bool) -> &'static str {
    if command_exists("wmctrl") || command_exists("xdotool") {
        "linux_desktop_tools"
    } else if wsl && windows_host_powershell_available() {
        "windows_host_powershell"
    } else if desktop_application_dirs().iter().any(|path| path.exists()) {
        "freedesktop_desktop_entries"
    } else {
        "unavailable"
    }
}

fn window_control_backend() -> &'static str {
    if command_exists("wmctrl") {
        "wmctrl"
    } else if command_exists("xdotool") {
        "xdotool"
    } else if is_wsl_runtime() && windows_host_powershell_available() {
        "windows_host_powershell"
    } else {
        "unavailable"
    }
}

fn backend_status(available: bool, backend: &str, reason: Option<&str>) -> Value {
    json!({
        "available": available,
        "backend": backend,
        "reason": if available { None } else { reason.or(Some("dependency_unavailable")) },
    })
}

fn insert_payload_value(payload: &mut Value, key: &str, value: Value) {
    if let Some(object) = payload.as_object_mut() {
        object.insert(key.to_string(), value);
    }
}

fn action_capability(
    action: &str,
    available: bool,
    backend: &str,
    safety: &str,
    requirement: Option<&str>,
) -> Value {
    json!({
        "action": action,
        "available": available,
        "backend": backend,
        "safety": safety,
        "requirement": requirement,
        "reason": if available { None } else { Some("dependency_unavailable") },
    })
}

fn desktop_limitations(linux_supported: bool, wsl: bool) -> Vec<&'static str> {
    let mut limitations = Vec::new();
    if !linux_supported {
        limitations
            .push("Desktop actions are currently implemented only for Linux and WSL runtimes.");
    }
    if wsl {
        limitations.push("WSL can see Linux runtime state and selected Windows host actions when host tools are reachable.");
    }
    if !command_exists("pactl") {
        limitations.push("PulseAudio volume and mute controls require pactl unless WSL Windows host audio fallback is available.");
    }
    if !command_exists("xdg-open") {
        limitations.push("Opening paths and URLs requires xdg-open.");
    }
    limitations
}

fn desktop_observation_limitations(linux_supported: bool, wsl: bool) -> Vec<&'static str> {
    let mut limitations = Vec::new();
    if !linux_supported {
        limitations
            .push("Desktop observation is currently implemented only for Linux and WSL runtimes.");
        return limitations;
    }
    if !command_exists("wmctrl") {
        limitations.push("Linux window listing requires wmctrl unless WSL Windows host observation is available.");
    }
    if !command_exists("xdotool") {
        limitations.push("Linux active-window observation requires xdotool unless WSL Windows host observation is available.");
    }
    if wsl && windows_host_powershell_available() {
        limitations.push("WSL desktop observation combines Linux runtime app entries with selected Windows host windows when PowerShell is reachable.");
    } else if wsl {
        limitations.push("WSL Windows host observation is unavailable because PowerShell cannot be reached from this runtime.");
    }
    limitations
}

fn windows_host_powershell_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        if !is_wsl_runtime() || !command_exists("powershell.exe") {
            return false;
        }
        run_capture_dynamic(&[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$PSVersionTable.PSVersion.Major",
        ])
        .map(|output| output.status == 0 && !output.stdout.is_empty())
        .unwrap_or(false)
    })
}

fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn downloads_directory() -> Option<PathBuf> {
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

fn download_inventory(limit: usize) -> Result<Value> {
    let Some(directory) = downloads_directory() else {
        return Ok(json!({
            "ok": false,
            "reason": "downloads_directory_unavailable",
            "items": [],
        }));
    };
    download_inventory_at(&directory, limit)
}

fn download_inventory_at(directory: &Path, limit: usize) -> Result<Value> {
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

fn is_partial_download_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [".crdownload", ".part", ".partial", ".download"]
        .iter()
        .any(|suffix| lower.ends_with(suffix))
}

fn windows_host_set_volume(
    percent: u8,
    target: Option<&str>,
    value: Option<&str>,
) -> Result<Value> {
    let script = windows_host_volume_script(percent);
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])?;
    let state = output
        .stdout
        .lines()
        .rev()
        .find_map(|line| serde_json::from_str::<Value>(line.trim()).ok());
    let after_percent = state
        .as_ref()
        .and_then(|item| item.get("after_percent"))
        .and_then(Value::as_u64);
    let verified = output.status == 0 && after_percent == Some(u64::from(percent));

    Ok(json!({
        "ok": verified,
        "tool": "desktop_action",
        "action": "set_volume",
        "target": target,
        "value": value,
        "runtime": "windows_host",
        "exit_code": output.status,
        "before": state.as_ref().and_then(|item| item.get("before_percent")).cloned(),
        "after": after_percent,
        "verified": verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "reason": if output.status != 0 { Some("command_failed") } else if !verified { Some("verification_failed") } else { None },
        "stdout": if state.is_none() { Some(output.stdout) } else { None },
        "stderr": output.stderr,
    }))
}

fn windows_host_volume_script(percent: u8) -> String {
    const CORE_AUDIO_TYPE: &str = r#"using System;
using System.Runtime.InteropServices;

public static class AgentCoreAudio {
    [ComImport]
    [Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
    private class MMDeviceEnumeratorComObject { }

    private enum EDataFlow { Render, Capture, All }
    private enum ERole { Console, Multimedia, Communications }

    [ComImport]
    [Guid("A95664D2-9614-4F35-A746-DE8DB63617E6")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMMDeviceEnumerator {
        int EnumAudioEndpoints(EDataFlow dataFlow, int stateMask, out IntPtr devices);
        [PreserveSig]
        int GetDefaultAudioEndpoint(EDataFlow dataFlow, ERole role, out IMMDevice device);
    }

    [ComImport]
    [Guid("D666063F-1587-4E43-81F1-B948E807363F")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IMMDevice {
        [PreserveSig]
        int Activate(ref Guid iid, int classContext, IntPtr activationParameters,
            [MarshalAs(UnmanagedType.IUnknown)] out object instance);
    }

    [ComImport]
    [Guid("5CDF2C82-841E-4546-9722-0CF74078229A")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IAudioEndpointVolume {
        int RegisterControlChangeNotify(IntPtr notify);
        int UnregisterControlChangeNotify(IntPtr notify);
        int GetChannelCount(ref uint channelCount);
        int SetMasterVolumeLevel(float levelDb, Guid eventContext);
        int SetMasterVolumeLevelScalar(float level, Guid eventContext);
        int GetMasterVolumeLevel(ref float levelDb);
        int GetMasterVolumeLevelScalar(ref float level);
    }

    private static IAudioEndpointVolume Endpoint() {
        var enumerator = (IMMDeviceEnumerator)new MMDeviceEnumeratorComObject();
        IMMDevice device;
        Marshal.ThrowExceptionForHR(enumerator.GetDefaultAudioEndpoint(
            EDataFlow.Render, ERole.Multimedia, out device));
        Guid iid = typeof(IAudioEndpointVolume).GUID;
        object endpoint;
        Marshal.ThrowExceptionForHR(device.Activate(ref iid, 23, IntPtr.Zero, out endpoint));
        return (IAudioEndpointVolume)endpoint;
    }

    public static float GetVolume() {
        float value = 0;
        Marshal.ThrowExceptionForHR(Endpoint().GetMasterVolumeLevelScalar(ref value));
        return value;
    }

    public static void SetVolume(float value) {
        Marshal.ThrowExceptionForHR(Endpoint().SetMasterVolumeLevelScalar(value, Guid.Empty));
    }
}"#;

    let scalar = f32::from(percent) / 100.0;
    format!(
        "$ErrorActionPreference = 'Stop'\nAdd-Type -TypeDefinition @'\n{}\n'@\n$before = [AgentCoreAudio]::GetVolume()\n[AgentCoreAudio]::SetVolume({scalar})\nStart-Sleep -Milliseconds 100\n$after = [AgentCoreAudio]::GetVolume()\n@{{before_percent = [math]::Round($before * 100); after_percent = [math]::Round($after * 100)}} | ConvertTo-Json -Compress",
        CORE_AUDIO_TYPE
    )
}

fn desktop_command_receipt<F>(
    action: &str,
    target: Option<&str>,
    value: Option<&str>,
    argv: &[&str],
    observe: F,
) -> Result<Value>
where
    F: Fn() -> Value,
{
    let Some((program, _)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty desktop command"));
    };
    if !command_exists(program) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "value": value,
            "reason": "dependency_unavailable",
            "error": format!("Required desktop command `{program}` is not installed."),
        }));
    }
    let before = observe();
    let output = run_capture_dynamic(argv)?;
    thread::sleep(Duration::from_millis(75));
    let after = observe();
    let changed = before != after;
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "value": value,
        "exit_code": output.status,
        "before": before,
        "after": after,
        "verified": output.status == 0 && changed,
        "verification": if output.status != 0 { "failed" } else if changed { "confirmed" } else { "not_confirmed" },
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

fn launch_desktop_target(action: &str, target: &str, use_xdg_open: bool) -> Result<Value> {
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

fn launch_windows_host_shortcut(action: &str, target: &str, shortcut: &str) -> Result<Value> {
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

fn windows_host_start_shortcut(shortcut: &str) -> Result<DesktopCommandOutput> {
    run_desktop_capture_with_stdin(
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

fn windows_host_start_shortcut_script() -> &'static str {
    r#"
$path = [Console]::In.ReadToEnd().Trim()
if (-not $path.EndsWith('.lnk', [System.StringComparison]::OrdinalIgnoreCase)) { Write-Error 'Expected a Start Menu shortcut'; exit 2 }
if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { Write-Error 'Shortcut not found'; exit 3 }
Start-Process -FilePath $path
"#
}

fn launch_windows_host_app(action: &str, target: &str, app_id: &str) -> Result<Value> {
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

fn windows_host_start_app(app_id: &str) -> Result<DesktopCommandOutput> {
    run_desktop_capture_with_stdin(
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

fn windows_host_start_app_script() -> &'static str {
    r#"
$appId = [Console]::In.ReadToEnd().Trim()
if ($appId.Length -eq 0 -or $appId.Length -gt 512 -or $appId -match '[\r\n]') { Write-Error 'Invalid app id'; exit 2 }
Start-Process -FilePath ("shell:AppsFolder\" + $appId)
"#
}

fn launch_observation(pid: Option<u32>) -> Result<Value> {
    let windows = launch_window_summary()?;
    Ok(json!({
        "pid": pid,
        "process": pid.map(|value| process_state(value as i32)),
        "windows": windows,
    }))
}

fn launch_observation_changed(
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

fn launch_window_summary() -> Result<Value> {
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

fn window_summary_changed(before: &Value, after: &Value) -> bool {
    if before.get("ok").and_then(Value::as_bool) != Some(true)
        || after.get("ok").and_then(Value::as_bool) != Some(true)
    {
        return false;
    }
    before.get("count").and_then(Value::as_u64) != after.get("count").and_then(Value::as_u64)
        || before.get("ids") != after.get("ids")
        || before.get("titles") != after.get("titles")
}

fn focus_launched_window(
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

fn launched_window_id(before: &Value, after: &Value) -> Option<String> {
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

fn matching_window_id(query: Option<&str>) -> Result<Option<String>> {
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

fn windows_shortcut_focus_query(shortcut: &str) -> String {
    shortcut
        .rsplit(['\\', '/'])
        .next()
        .unwrap_or(shortcut)
        .strip_suffix(".lnk")
        .unwrap_or(shortcut)
        .to_string()
}

fn clipboard_backend() -> &'static str {
    // In WSL the interactive terminal and screenshot tools normally use the
    // Windows host clipboard. Linux clipboard binaries may still be installed
    // while their display socket is unreachable, so selecting them first makes
    // Ctrl+V fail before the working host backend is attempted.
    select_clipboard_backend(
        is_wsl_runtime() && windows_host_powershell_available(),
        command_exists("wl-copy") && command_exists("wl-paste"),
        command_exists("xclip"),
        command_exists("xsel"),
    )
}

fn select_clipboard_backend(
    windows_host: bool,
    wayland: bool,
    xclip: bool,
    xsel: bool,
) -> &'static str {
    if windows_host {
        "windows_host_powershell"
    } else if wayland {
        "wl-clipboard"
    } else if xclip {
        "xclip"
    } else if xsel {
        "xsel"
    } else {
        "unavailable"
    }
}

fn keyboard_backend() -> &'static str {
    if command_exists("wtype") {
        "wtype"
    } else if command_exists("xdotool") {
        "xdotool"
    } else {
        "unavailable"
    }
}

fn pointer_backend() -> &'static str {
    if command_exists("xdotool") {
        "xdotool"
    } else {
        "unavailable"
    }
}

fn accessibility_backend() -> &'static str {
    if env::var("AT_SPI_BUS_ADDRESS").is_ok_and(|value| !value.trim().is_empty())
        || (command_exists("busctl") && atspi_bus_address().is_ok())
    {
        "atspi_dbus"
    } else {
        "unavailable"
    }
}

fn display_backend() -> &'static str {
    if command_exists("xrandr") {
        "xrandr"
    } else {
        "unavailable"
    }
}

fn display_inventory(limit: usize) -> Result<Value> {
    if display_backend() == "unavailable" {
        return Ok(json!({
            "ok": false,
            "backend": "unavailable",
            "reason": "backend_unavailable",
            "error": "Display observation requires xrandr.",
            "items": [],
            "count": 0,
        }));
    }

    let output = run_capture_dynamic(&["xrandr", "--query"])?;
    let mut items = parse_xrandr_displays(&output.stdout);
    let total = items.len();
    items.truncate(limit);
    Ok(json!({
        "ok": output.status == 0,
        "backend": "xrandr",
        "count": items.len(),
        "items": items,
        "truncated": total > limit,
        "stderr": output.stderr,
    }))
}

fn parse_xrandr_displays(output: &str) -> Vec<Value> {
    output.lines().filter_map(parse_xrandr_display).collect()
}

fn parse_xrandr_display(line: &str) -> Option<Value> {
    let mut fields = line.split_whitespace();
    let name = fields.next()?;
    let status = fields.next()?;
    if !matches!(status, "connected" | "disconnected") {
        return None;
    }
    let mut primary = false;
    let mut geometry = None;
    for field in fields {
        primary |= field == "primary";
        geometry = geometry.or_else(|| parse_xrandr_geometry(field));
    }
    Some(json!({
        "name": name,
        "status": status,
        "connected": status == "connected",
        "primary": primary,
        "geometry": geometry.map(|(width, height, x, y)| json!({
            "width": width,
            "height": height,
            "x": x,
            "y": y,
        })),
    }))
}

fn parse_xrandr_geometry(value: &str) -> Option<(u64, u64, i64, i64)> {
    let (width, rest) = value.split_once('x')?;
    let offset_start = rest.find(['+', '-'])?;
    let (height, offsets) = rest.split_at(offset_start);
    let second_offset = offsets[1..].find(['+', '-'])? + 1;
    let (x, y) = offsets.split_at(second_offset);
    Some((
        width.parse().ok()?,
        height.parse().ok()?,
        x.parse().ok()?,
        y.parse().ok()?,
    ))
}

fn audio_observation() -> Result<Value> {
    if !command_exists("pactl") {
        return Ok(json!({
            "ok": false,
            "backend": "unavailable",
            "reason": "backend_unavailable",
            "error": "Audio observation requires pactl.",
        }));
    }

    let volume = run_capture_dynamic(&["pactl", "get-sink-volume", "@DEFAULT_SINK@"])?;
    let mute = run_capture_dynamic(&["pactl", "get-sink-mute", "@DEFAULT_SINK@"])?;
    let sink = run_capture_dynamic(&["pactl", "get-default-sink"])?;
    let volume_percent = parse_pactl_volume_percent(&volume.stdout);
    let muted = parse_pactl_mute(&mute.stdout);
    let ok = volume.status == 0
        && mute.status == 0
        && sink.status == 0
        && volume_percent.is_some()
        && muted.is_some();
    let mut stderr = String::new();
    for value in [volume.stderr, mute.stderr, sink.stderr] {
        if value.is_empty() {
            continue;
        }
        if !stderr.is_empty() {
            stderr.push('\n');
        }
        stderr.push_str(&value);
    }
    Ok(json!({
        "ok": ok,
        "backend": "pactl",
        "sink": if sink.stdout.is_empty() { None } else { Some(sink.stdout) },
        "volume_percent": volume_percent,
        "muted": muted,
        "reason": if ok { None } else { Some("observation_failed") },
        "stderr": stderr,
    }))
}

fn parse_pactl_volume_percent(output: &str) -> Option<u8> {
    output.split_whitespace().find_map(|field| {
        field
            .strip_suffix('%')
            .and_then(|value| value.parse::<u8>().ok())
    })
}

fn parse_pactl_mute(output: &str) -> Option<bool> {
    match output
        .split_whitespace()
        .last()?
        .to_ascii_lowercase()
        .as_str()
    {
        "yes" => Some(true),
        "no" => Some(false),
        _ => None,
    }
}

fn accessibility_tree(snapshot_id: &str, limit: usize) -> Result<Value> {
    match native_accessibility_tree(snapshot_id, limit) {
        Ok(value) => Ok(value),
        Err(native_error) if command_exists("busctl") => {
            accessibility_tree_fallback(limit, &native_error.to_string())
        }
        Err(error) => Ok(json!({
            "ok": false,
            "backend": "unavailable",
            "reason": "backend_unavailable",
            "error": error.to_string(),
            "items": [],
            "count": 0,
        })),
    }
}

fn native_atspi_address() -> Result<String> {
    if let Ok(address) = env::var("AT_SPI_BUS_ADDRESS") {
        if !address.trim().is_empty() {
            return Ok(address);
        }
    }
    atspi_bus_address()
}

fn native_accessibility_tree(snapshot_id: &str, limit: usize) -> Result<Value> {
    let address = native_atspi_address()?;
    let connection = zbus::blocking::connection::Builder::address(address.as_str())?
        .method_timeout(Duration::from_secs(2))
        .build()?;
    let root = AccessibilityRef {
        bus: "org.a11y.atspi.Registry".to_string(),
        path: "/org/a11y/atspi/accessible/root".to_string(),
        depth: 0,
        parent_id: None,
    };
    let mut queue = VecDeque::from([root]);
    let mut visited = HashSet::new();
    let mut items = Vec::new();
    let mut skipped = 0usize;

    while let Some(reference) = queue.pop_front() {
        if items.len() >= limit {
            break;
        }
        let key = format!("{}\0{}", reference.bus, reference.path);
        if !visited.insert(key) {
            continue;
        }
        match accessibility_element(&connection, snapshot_id, &reference) {
            Ok((item, children)) => {
                let parent_id = item.get("id").and_then(Value::as_str).map(str::to_string);
                items.push(item);
                queue.extend(children.into_iter().map(|(bus, path)| AccessibilityRef {
                    bus,
                    path,
                    depth: reference.depth.saturating_add(1),
                    parent_id: parent_id.clone(),
                }));
            }
            Err(_) => skipped += 1,
        }
    }

    Ok(json!({
        "ok": true,
        "backend": "atspi_dbus",
        "snapshot_id": snapshot_id,
        "content": "accessible_metadata_only",
        "count": items.len(),
        "items": items,
        "truncated": !queue.is_empty(),
        "skipped_defunct_or_unreachable": skipped,
    }))
}

#[derive(Debug)]
struct AccessibilityRef {
    bus: String,
    path: String,
    depth: usize,
    parent_id: Option<String>,
}

fn accessibility_element(
    connection: &zbus::blocking::Connection,
    snapshot_id: &str,
    reference: &AccessibilityRef,
) -> Result<(Value, Vec<(String, String)>)> {
    let accessible = zbus::blocking::Proxy::new(
        connection,
        reference.bus.as_str(),
        reference.path.as_str(),
        "org.a11y.atspi.Accessible",
    )?;
    let name = accessible
        .get_property::<String>("Name")
        .unwrap_or_default();
    let description = accessible
        .get_property::<String>("Description")
        .unwrap_or_default();
    let role = accessible
        .call::<_, _, String>("GetRoleName", &())
        .unwrap_or_else(|_| "unknown".to_string());
    let state_ids = accessible
        .call::<_, _, Vec<u32>>("GetState", &())
        .unwrap_or_default();
    let interfaces = accessible
        .call::<_, _, Vec<String>>("GetInterfaces", &())
        .unwrap_or_default();
    let children = accessible
        .call::<_, _, Vec<(String, zbus::zvariant::OwnedObjectPath)>>("GetChildren", &())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(bus, path)| {
            let path = path.to_string();
            (path != "/org/a11y/atspi/null").then_some((bus, path))
        })
        .collect::<Vec<_>>();
    let bounds = accessibility_bounds(connection, reference, &interfaces);
    let actions = accessibility_actions(connection, reference, &interfaces);
    let target_id = accessibility_target_id(snapshot_id, &reference.bus, &reference.path);

    Ok((
        json!({
            "id": target_id,
            "snapshot_id": snapshot_id,
            "depth": reference.depth,
            "parent_id": reference.parent_id,
            "name": bounded_accessible_text(&name),
            "description": bounded_accessible_text(&description),
            "role": role,
            "state_ids": state_ids,
            "interfaces": interfaces,
            "actions": actions,
            "bounds": bounds,
            "child_count": children.len(),
            "backend_ref": {
                "bus": reference.bus,
                "path": reference.path,
            },
        }),
        children,
    ))
}

fn accessibility_bounds(
    connection: &zbus::blocking::Connection,
    reference: &AccessibilityRef,
    interfaces: &[String],
) -> Option<Value> {
    if !interfaces.iter().any(|item| item.ends_with(".Component")) {
        return None;
    }
    let proxy = zbus::blocking::Proxy::new(
        connection,
        reference.bus.as_str(),
        reference.path.as_str(),
        "org.a11y.atspi.Component",
    )
    .ok()?;
    let (x, y, width, height): (i32, i32, i32, i32) = proxy.call("GetExtents", &0u32).ok()?;
    Some(json!({ "x": x, "y": y, "width": width, "height": height }))
}

fn accessibility_actions(
    connection: &zbus::blocking::Connection,
    reference: &AccessibilityRef,
    interfaces: &[String],
) -> Vec<String> {
    if !interfaces.iter().any(|item| item.ends_with(".Action")) {
        return Vec::new();
    }
    let Ok(proxy) = zbus::blocking::Proxy::new(
        connection,
        reference.bus.as_str(),
        reference.path.as_str(),
        "org.a11y.atspi.Action",
    ) else {
        return Vec::new();
    };
    let count = proxy
        .call::<_, _, i32>("GetNActions", &())
        .unwrap_or(0)
        .clamp(0, 32);
    (0..count)
        .filter_map(|index| proxy.call::<_, _, String>("GetName", &index).ok())
        .map(|name| bounded_accessible_text(&name))
        .filter(|name| !name.is_empty())
        .collect()
}

fn accessibility_target_id(snapshot_id: &str, bus: &str, path: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(snapshot_id.as_bytes());
    hasher.update([0]);
    hasher.update(bus.as_bytes());
    hasher.update([0]);
    hasher.update(path.as_bytes());
    let digest = format!("{:x}", hasher.finalize());
    format!("ui-{}", &digest[..16])
}

fn bounded_accessible_text(value: &str) -> String {
    value.chars().take(256).collect()
}

fn dialog_inventory(snapshot_id: &str, limit: usize) -> Result<Value> {
    let tree_limit = limit.saturating_mul(10).clamp(1, 200);
    let tree = accessibility_tree(snapshot_id, tree_limit)?;
    if tree.get("ok").and_then(Value::as_bool) != Some(true) {
        return Ok(json!({
            "ok": false,
            "backend": tree.get("backend").cloned().unwrap_or(json!("unavailable")),
            "reason": tree.get("reason").cloned().unwrap_or(json!("backend_unavailable")),
            "error": tree.get("error").cloned(),
            "items": [],
            "count": 0,
        }));
    }
    let mut dialogs = dialog_items(
        tree.get("items")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default(),
        limit,
    );
    let total = dialogs.len();
    dialogs.truncate(limit);
    Ok(json!({
        "ok": true,
        "backend": tree.get("backend").cloned().unwrap_or(json!("atspi_dbus")),
        "snapshot_id": snapshot_id,
        "count": total.min(limit),
        "items": dialogs,
        "truncated": total > limit,
        "tree_truncated": tree.get("truncated").cloned().unwrap_or(json!(false)),
    }))
}

fn dialog_items(items: &[Value], control_limit: usize) -> Vec<Value> {
    let parents = items
        .iter()
        .filter_map(|item| {
            Some((
                item.get("id")?.as_str()?,
                item.get("parent_id")
                    .and_then(Value::as_str)
                    .filter(|parent_id| !parent_id.is_empty()),
            ))
        })
        .collect::<HashMap<_, _>>();

    items
        .iter()
        .filter_map(|item| {
            let role = item.get("role").and_then(Value::as_str)?;
            let kind = dialog_kind(role)?;
            let dialog_id = item.get("id").and_then(Value::as_str)?;
            let controls = items
                .iter()
                .filter(|candidate| {
                    candidate.get("id").and_then(Value::as_str) != Some(dialog_id)
                        && accessibility_descends_from(candidate, dialog_id, &parents)
                })
                .take(control_limit)
                .cloned()
                .collect::<Vec<_>>();
            let mut dialog = item.clone();
            let object = dialog.as_object_mut()?;
            object.insert("dialog_kind".to_string(), json!(kind));
            object.insert("control_count".to_string(), json!(controls.len()));
            object.insert("controls".to_string(), json!(controls));
            Some(dialog)
        })
        .collect()
}

fn accessibility_descends_from(
    item: &Value,
    ancestor_id: &str,
    parents: &HashMap<&str, Option<&str>>,
) -> bool {
    let mut current = item.get("parent_id").and_then(Value::as_str);
    let mut visited = HashSet::new();
    while let Some(id) = current {
        if id == ancestor_id {
            return true;
        }
        if !visited.insert(id) {
            return false;
        }
        current = parents.get(id).copied().flatten();
    }
    false
}

fn dialog_kind(role: &str) -> Option<&'static str> {
    let normalized = role.trim().to_ascii_lowercase().replace(['_', '-'], " ");
    if normalized.contains("file chooser") || normalized.contains("file picker") {
        Some("file_picker")
    } else if normalized == "alert" || normalized.contains("alert dialog") {
        Some("alert")
    } else if normalized == "dialog" || normalized.ends_with(" dialog") {
        Some("dialog")
    } else {
        None
    }
}

fn ui_element_action(
    action: &str,
    target_id: &str,
    value: Option<&str>,
    backend_bus: Option<&str>,
    backend_path: Option<&str>,
) -> Result<Value> {
    let bus =
        backend_bus.ok_or_else(|| anyhow::anyhow!("{action} requires an observed UI element"))?;
    let path =
        backend_path.ok_or_else(|| anyhow::anyhow!("{action} requires an observed UI element"))?;
    validate_accessibility_reference(bus, path)?;
    let address = native_atspi_address()?;
    let connection = zbus::blocking::connection::Builder::address(address.as_str())?
        .method_timeout(Duration::from_secs(2))
        .build()?;

    match action {
        "focus_element" => focus_accessibility_element(&connection, target_id, bus, path),
        "invoke_element" => invoke_accessibility_element(
            &connection,
            target_id,
            bus,
            path,
            value.ok_or_else(|| {
                anyhow::anyhow!("invoke_element requires an advertised action name")
            })?,
        ),
        "set_field_text" => set_accessibility_text(
            &connection,
            target_id,
            bus,
            path,
            value.ok_or_else(|| anyhow::anyhow!("set_field_text requires text"))?,
        ),
        _ => unreachable!("semantic action was validated by caller"),
    }
}

fn validate_accessibility_reference(bus: &str, path: &str) -> Result<()> {
    zbus::names::BusName::try_from(bus)
        .map_err(|_| anyhow::anyhow!("Invalid accessibility bus name"))?;
    zbus::zvariant::ObjectPath::try_from(path)
        .map_err(|_| anyhow::anyhow!("Invalid accessibility object path"))?;
    Ok(())
}

fn focus_accessibility_element(
    connection: &zbus::blocking::Connection,
    target_id: &str,
    bus: &str,
    path: &str,
) -> Result<Value> {
    const ATSPI_STATE_FOCUSED: u32 = 12;
    let before = accessibility_state(connection, bus, path);
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Component")?;
    let dispatched: bool = proxy.call("GrabFocus", &())?;
    thread::sleep(Duration::from_millis(75));
    let after = accessibility_state(connection, bus, path);
    let verified = dispatched
        && after
            .as_ref()
            .is_some_and(|states| states.contains(&ATSPI_STATE_FOCUSED));
    Ok(json!({
        "ok": dispatched,
        "tool": "desktop_action",
        "action": "focus_element",
        "target": target_id,
        "backend": "atspi_dbus",
        "before": { "state_ids": before },
        "after": { "state_ids": after },
        "verified": verified,
        "verification": if verified { "confirmed" } else { "not_confirmed" },
        "reason": if dispatched && !verified { Some("verification_failed") } else if !dispatched { Some("dispatch_rejected") } else { None },
    }))
}

fn invoke_accessibility_element(
    connection: &zbus::blocking::Connection,
    target_id: &str,
    bus: &str,
    path: &str,
    requested_action: &str,
) -> Result<Value> {
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Action")?;
    let count = proxy.call::<_, _, i32>("GetNActions", &())?.clamp(0, 32);
    let mut index = None;
    let mut advertised = Vec::new();
    for candidate in 0..count {
        if let Ok(name) = proxy.call::<_, _, String>("GetName", &candidate) {
            if name == requested_action {
                index = Some(candidate);
            }
            advertised.push(name);
        }
    }
    let index = index.ok_or_else(|| {
        anyhow::anyhow!(
            "Element does not advertise action {requested_action:?}; available: {}",
            advertised.join(", ")
        )
    })?;
    let before = accessibility_state(connection, bus, path);
    let dispatched: bool = proxy.call("DoAction", &index)?;
    thread::sleep(Duration::from_millis(100));
    let after = accessibility_state(connection, bus, path);
    let observable_change = before != after || (before.is_some() && after.is_none());
    let verified = dispatched && observable_change;
    Ok(json!({
        "ok": dispatched,
        "tool": "desktop_action",
        "action": "invoke_element",
        "target": target_id,
        "element_action": requested_action,
        "backend": "atspi_dbus",
        "before": { "state_ids": before },
        "after": { "state_ids": after },
        "verified": verified,
        "verification": if verified { "confirmed" } else if dispatched { "dispatch_confirmed" } else { "failed" },
        "reason": if dispatched && !verified { Some("effect_not_observable_on_target") } else if !dispatched { Some("dispatch_rejected") } else { None },
    }))
}

fn set_accessibility_text(
    connection: &zbus::blocking::Connection,
    target_id: &str,
    bus: &str,
    path: &str,
    text: &str,
) -> Result<Value> {
    if text.len() > 64 * 1024 {
        return Err(anyhow::anyhow!(
            "set_field_text is limited to 65536 UTF-8 bytes"
        ));
    }
    let before_hash = accessibility_text_hash(connection, bus, path);
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.EditableText")?;
    let dispatched: bool = proxy.call("SetTextContents", &text)?;
    thread::sleep(Duration::from_millis(75));
    let after_hash = accessibility_text_hash(connection, bus, path);
    let expected_hash = sha256_text(text);
    let verified = dispatched && after_hash.as_deref() == Some(expected_hash.as_str());
    Ok(json!({
        "ok": dispatched,
        "tool": "desktop_action",
        "action": "set_field_text",
        "target": target_id,
        "backend": "atspi_dbus",
        "value": { "redacted": true, "sha256": expected_hash, "bytes": text.len(), "characters": text.chars().count() },
        "before": { "sha256": before_hash },
        "after": { "sha256": after_hash },
        "verified": verified,
        "verification": if verified { "confirmed" } else { "not_confirmed" },
        "reason": if dispatched && !verified { Some("verification_failed") } else if !dispatched { Some("dispatch_rejected") } else { None },
    }))
}

fn accessibility_state(
    connection: &zbus::blocking::Connection,
    bus: &str,
    path: &str,
) -> Option<Vec<u32>> {
    let proxy =
        zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Accessible").ok()?;
    proxy.call("GetState", &()).ok()
}

fn accessibility_text_hash(
    connection: &zbus::blocking::Connection,
    bus: &str,
    path: &str,
) -> Option<String> {
    let proxy = zbus::blocking::Proxy::new(connection, bus, path, "org.a11y.atspi.Text").ok()?;
    let value: String = proxy.call("GetText", &(0i32, -1i32)).ok()?;
    Some(sha256_text(&value))
}

fn sha256_text(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn accessibility_tree_fallback(limit: usize, native_error: &str) -> Result<Value> {
    let address = match atspi_bus_address() {
        Ok(address) => address,
        Err(error) => {
            return Ok(json!({
                "ok": false,
                "backend": "unavailable",
                "reason": "backend_unavailable",
                "error": error.to_string(),
                "items": [],
                "count": 0,
            }));
        }
    };
    let address_arg = format!("--address={address}");
    let output = run_capture_dynamic(&[
        "busctl",
        &address_arg,
        "--no-pager",
        "--list",
        "tree",
        "org.a11y.atspi.Registry",
        "/org/a11y/atspi/accessible/root",
    ])?;
    let items: Vec<Value> = output
        .stdout
        .lines()
        .take(limit)
        .filter_map(accessibility_tree_item)
        .collect();
    Ok(json!({
        "ok": output.status == 0,
        "backend": "atspi_busctl_fallback",
        "content": "object_paths_only",
        "count": items.len(),
        "items": items,
        "truncated": output.stdout.lines().count() > limit,
        "stderr": output.stderr,
        "native_backend_error": native_error,
    }))
}

fn atspi_bus_address() -> Result<String> {
    if !command_exists("busctl") {
        return Err(anyhow::anyhow!("AT-SPI observation requires busctl."));
    }
    let output = run_capture_dynamic(&[
        "busctl",
        "--user",
        "call",
        "org.a11y.Bus",
        "/org/a11y/bus",
        "org.a11y.Bus",
        "GetAddress",
    ])?;
    if output.status != 0 {
        return Err(anyhow::anyhow!(
            "AT-SPI bus is unavailable: {}",
            if output.stderr.is_empty() {
                output.stdout
            } else {
                output.stderr
            }
        ));
    }
    parse_busctl_string(&output.stdout)
        .ok_or_else(|| anyhow::anyhow!("AT-SPI bus address could not be parsed."))
}

fn accessibility_tree_item(line: &str) -> Option<Value> {
    let path = line.trim();
    if !path.starts_with("/org/a11y/atspi/accessible") {
        return None;
    }
    Some(json!({
        "path": path,
        "kind": "accessible_object",
    }))
}

fn parse_busctl_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    let start = trimmed.find('"')?;
    let end = trimmed.rfind('"')?;
    if end <= start {
        return None;
    }
    let text = &trimmed[start + 1..end];
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn pointer_action(action: &str, target: Option<&str>, value: Option<&str>) -> Result<Value> {
    match action {
        "mouse_click" => {
            let (x, y) = parse_pointer_coordinates(required_target(action, target)?)?;
            let button = pointer_button(value)?;
            let x_arg = x.to_string();
            let y_arg = y.to_string();
            let button_arg = button.to_string();
            let argv = [
                "xdotool",
                "mousemove",
                "--sync",
                x_arg.as_str(),
                y_arg.as_str(),
                "click",
                button_arg.as_str(),
            ];
            pointer_receipt(action, json!({ "x": x, "y": y, "button": button }), &argv)
        }
        "scroll" => {
            let steps = parse_scroll_steps(value)?;
            let button = if steps < 0 { "5" } else { "4" };
            let repetitions = steps.unsigned_abs().min(25);
            let mut argv = Vec::with_capacity(1 + repetitions as usize * 2);
            argv.push(String::from("xdotool"));
            for _ in 0..repetitions {
                argv.push(String::from("click"));
                argv.push(String::from(button));
            }
            pointer_receipt(
                action,
                json!({ "steps": steps, "button": button, "repetitions": repetitions }),
                &argv,
            )
        }
        _ => Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "unsupported_action",
            "error": format!("Unsupported pointer action: {action}"),
        })),
    }
}

fn pointer_receipt<T: AsRef<std::ffi::OsStr>>(
    action: &str,
    value: Value,
    argv: &[T],
) -> Result<Value> {
    if !command_exists("xdotool") {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "backend": "xdotool",
            "reason": "dependency_unavailable",
            "error": "Pointer input requires xdotool.",
        }));
    }
    let before = active_window()?;
    let output = run_capture_dynamic(argv)?;
    thread::sleep(Duration::from_millis(50));
    let after = active_window()?;
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "backend": "xdotool",
        "value": value,
        "before": before,
        "after": after,
        "verified": false,
        "verification": if output.status == 0 { "dispatched_unverified" } else { "failed" },
        "exit_code": output.status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

fn parse_pointer_coordinates(value: &str) -> Result<(i32, i32)> {
    let Some((raw_x, raw_y)) = value.split_once(',') else {
        return Err(anyhow::anyhow!(
            "mouse_click target must be coordinates formatted as x,y"
        ));
    };
    let x = raw_x
        .trim()
        .parse::<i32>()
        .map_err(|_| anyhow::anyhow!("mouse_click x coordinate must be an integer"))?;
    let y = raw_y
        .trim()
        .parse::<i32>()
        .map_err(|_| anyhow::anyhow!("mouse_click y coordinate must be an integer"))?;
    if !(0..=100_000).contains(&x) || !(0..=100_000).contains(&y) {
        return Err(anyhow::anyhow!(
            "mouse_click coordinates must be between 0 and 100000"
        ));
    }
    Ok((x, y))
}

fn pointer_button(value: Option<&str>) -> Result<u8> {
    let raw = value.unwrap_or("1").trim();
    let button = raw
        .parse::<u8>()
        .map_err(|_| anyhow::anyhow!("mouse_click value must be a button number"))?;
    if !(1..=3).contains(&button) {
        return Err(anyhow::anyhow!("mouse_click button must be 1, 2, or 3"));
    }
    Ok(button)
}

fn parse_scroll_steps(value: Option<&str>) -> Result<i32> {
    let raw = value.unwrap_or("-3").trim();
    let steps = raw
        .parse::<i32>()
        .map_err(|_| anyhow::anyhow!("scroll value must be a signed step count"))?;
    if steps == 0 || !(-25..=25).contains(&steps) {
        return Err(anyhow::anyhow!(
            "scroll value must be between -25 and 25, excluding 0"
        ));
    }
    Ok(steps)
}

fn keyboard_action(action: &str, value: Option<&str>) -> Result<Value> {
    let raw = value.ok_or_else(|| anyhow::anyhow!("{action} requires --value"))?;
    match action {
        "send_key" => {
            let key = normalize_key_spec(raw)?;
            keyboard_receipt(
                action,
                &key,
                "xdotool",
                &["xdotool", "key", "--clearmodifiers", &key],
            )
        }
        "type_text" => {
            if raw.len() > 4000 {
                return Err(anyhow::anyhow!(
                    "type_text value must be 4000 bytes or fewer"
                ));
            }
            match keyboard_backend() {
                "wtype" => keyboard_receipt(action, raw, "wtype", &["wtype", raw]),
                "xdotool" => keyboard_receipt(
                    action,
                    raw,
                    "xdotool",
                    &[
                        "xdotool",
                        "type",
                        "--clearmodifiers",
                        "--delay",
                        "0",
                        "--",
                        raw,
                    ],
                ),
                _ => Ok(json!({
                    "ok": false,
                    "tool": "desktop_action",
                    "action": action,
                    "reason": "dependency_unavailable",
                    "error": "Keyboard text input requires wtype or xdotool.",
                })),
            }
        }
        _ => Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "unsupported_action",
            "error": format!("Unsupported keyboard action: {action}"),
        })),
    }
}

fn keyboard_receipt(action: &str, value: &str, backend: &str, argv: &[&str]) -> Result<Value> {
    let Some((program, _)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty keyboard command"));
    };
    if !command_exists(program) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "backend": backend,
            "reason": "dependency_unavailable",
            "error": format!("Required keyboard command `{program}` is not installed."),
        }));
    }
    let before = active_window()?;
    let output = run_capture_dynamic(argv)?;
    thread::sleep(Duration::from_millis(50));
    let after = active_window()?;
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "backend": backend,
        "value": keyboard_value_receipt(action, value),
        "before": before,
        "after": after,
        "verified": false,
        "verification": if output.status == 0 { "dispatched_unverified" } else { "failed" },
        "exit_code": output.status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

fn keyboard_value_receipt(action: &str, value: &str) -> Value {
    if action == "type_text" {
        json!({
            "byte_count": value.len(),
            "char_count": value.chars().count(),
            "content_returned": false,
        })
    } else {
        json!({
            "key": value,
            "content_returned": true,
        })
    }
}

fn normalize_key_spec(value: &str) -> Result<String> {
    let normalized = value.trim();
    if normalized.is_empty() || normalized.len() > 80 {
        return Err(anyhow::anyhow!(
            "send_key value must be a non-empty key spec"
        ));
    }
    if !normalized
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '+' | ':' | '.'))
    {
        return Err(anyhow::anyhow!(
            "send_key accepts only key names and modifiers, such as ctrl+l or Alt+Tab"
        ));
    }
    Ok(normalized.to_string())
}

fn resolve_applications(query: &str, limit: usize) -> Result<Vec<Value>> {
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

fn resolve_windows(query: &str, limit: usize) -> Result<Vec<Value>> {
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

fn match_score(query: &str, fields: &[&str]) -> i64 {
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

fn normalize_match_text(value: &str) -> String {
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

fn candidate_score(candidate: &Value) -> i64 {
    candidate.get("score").and_then(Value::as_i64).unwrap_or(0)
}

fn candidate_label(candidate: &Value) -> String {
    candidate
        .get("name")
        .or_else(|| candidate.get("title"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn clipboard_metadata() -> Result<Value> {
    let backend = clipboard_backend();
    if backend == "unavailable" {
        return Ok(json!({
            "ok": false,
            "backend": "unavailable",
            "reason": "backend_unavailable",
            "content_returned": false,
            "error": "Clipboard metadata requires wl-clipboard, xclip, xsel, or reachable Windows PowerShell under WSL.",
        }));
    }
    let types = clipboard_types(backend)?;
    let text = clipboard_text(backend)?;
    let readable = text.status == 0;
    Ok(json!({
        "ok": true,
        "backend": backend,
        "content_returned": false,
        "types": types,
        "text": {
            "available": readable,
            "byte_count": if readable { Some(text.stdout.len()) } else { None },
            "char_count": if readable { Some(text.stdout.chars().count()) } else { None },
            "line_count": if readable { Some(text.stdout.lines().count()) } else { None },
            "utf8": readable,
            "preview_redacted": true,
            "stderr": if readable { None } else { Some(text.stderr) },
        },
    }))
}

/// Open one stored attachment after an explicit click in the TUI.
pub(crate) fn desktop_open_user_file(path: &str) -> Result<Value> {
    let candidate = Path::new(path);
    if !candidate.is_file() {
        return Ok(json!({
            "ok": false,
            "reason": "attachment_unavailable",
            "error": "The attached file is no longer available in local storage.",
        }));
    }

    if is_wsl_runtime() && windows_host_powershell_available() {
        let converted = run_desktop_capture(&["wslpath", "-w", path])?;
        if converted.status != 0 || converted.stdout.is_empty() {
            return Ok(json!({
                "ok": false,
                "reason": "path_conversion_failed",
                "error": converted.stderr,
            }));
        }
        let output = ProcessCommand::new("powershell.exe")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Start-Process -FilePath $env:NYM_ATTACHMENT_PATH",
            ])
            .env("NYM_ATTACHMENT_PATH", &converted.stdout)
            .stdin(Stdio::null())
            .output()?;
        if !output.status.success() {
            return Ok(json!({
                "ok": false,
                "reason": "system_viewer_failed",
                "error": String::from_utf8_lossy(&output.stderr).trim(),
            }));
        }
        return Ok(json!({
            "ok": true,
            "backend": "windows_host",
            "path": path,
        }));
    }
    if command_exists("xdg-open") {
        return launch_desktop_target("open_attachment", path, true);
    }

    Ok(json!({
        "ok": false,
        "reason": "opener_unavailable",
        "error": "No desktop file opener is available. Install xdg-utils to open attachments.",
    }))
}

/// Open the desktop's file chooser after an explicit click in the TUI.
/// The returned path is never exposed to the model; it is immediately copied
/// into Agent's attachment store by the caller.
pub(crate) fn desktop_pick_file() -> Result<Value> {
    let output = if command_exists("zenity") {
        run_desktop_capture(&["zenity", "--file-selection", "--title=Add photos or files"])?
    } else if command_exists("kdialog") {
        run_desktop_capture(&[
            "kdialog",
            "--title",
            "Add photos or files",
            "--getopenfilename",
        ])?
    } else if is_wsl_runtime() && windows_host_powershell_available() {
        let script = "Add-Type -AssemblyName System.Windows.Forms; $dialog = New-Object System.Windows.Forms.OpenFileDialog; $dialog.Title = 'Add photos or files'; $dialog.Filter = 'All files (*.*)|*.*|Images (*.png;*.jpg;*.jpeg;*.webp;*.gif)|*.png;*.jpg;*.jpeg;*.webp;*.gif'; $dialog.Multiselect = $false; if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dialog.FileName) }";
        let selected = run_desktop_capture(&[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-Command",
            script,
        ])?;
        if selected.status != 0 {
            return Ok(json!({
                "ok": false,
                "reason": "file_picker_failed",
                "guidance": selected.stderr,
            }));
        }
        if selected.stdout.trim().is_empty() {
            return Ok(json!({"ok": false, "cancelled": true}));
        }
        let converted = run_desktop_capture(&["wslpath", "-u", selected.stdout.trim()])?;
        if converted.status != 0 || converted.stdout.trim().is_empty() {
            return Ok(json!({
                "ok": false,
                "reason": "file_picker_path_conversion_failed",
                "guidance": converted.stderr,
            }));
        }
        return Ok(json!({
            "ok": true,
            "path": converted.stdout.trim(),
            "backend": "windows_file_dialog",
        }));
    } else {
        return Ok(json!({
            "ok": false,
            "reason": "file_picker_unavailable",
            "guidance": "No native file picker is available. Install zenity or kdialog, or enter a path instead.",
        }));
    };
    if output.status != 0 || output.stdout.trim().is_empty() {
        return Ok(json!({"ok": false, "cancelled": true}));
    }
    Ok(json!({
        "ok": true,
        "path": output.stdout.trim(),
        "backend": if command_exists("zenity") { "zenity" } else { "kdialog" },
    }))
}

/// Return clipboard text only after an explicit user gesture in the terminal
/// UI.  Agent tools deliberately use `clipboard_metadata` instead, so an LLM
/// cannot silently exfiltrate clipboard contents.
pub(crate) fn desktop_clipboard_read_text() -> Result<Value> {
    let backend = clipboard_backend();
    if backend == "unavailable" {
        return Ok(json!({
            "ok": false,
            "reason": "backend_unavailable",
            "guidance": "Clipboard paste requires wl-clipboard, xclip, xsel, or reachable Windows PowerShell under WSL.",
        }));
    }
    let output = clipboard_text(backend)?;
    if output.status != 0 {
        return Ok(json!({
            "ok": false,
            "backend": backend,
            "error": output.stderr,
        }));
    }
    let max_bytes = std::env::var("AGENT_CLIPBOARD_MAX_TEXT_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1_048_576);
    if output.stdout.len() > max_bytes {
        return Ok(json!({
            "ok": false,
            "backend": backend,
            "reason": "clipboard_text_too_large",
            "guidance": format!("Clipboard text exceeds the configured {max_bytes}-byte limit."),
        }));
    }
    Ok(json!({
        "ok": true,
        "backend": backend,
        "text": output.stdout,
    }))
}

/// Materialize an image already present in the user's clipboard as a temporary
/// file.  Its bytes are streamed directly from the clipboard program to disk,
/// so the TUI does not retain a second full image allocation in memory.
pub(crate) fn desktop_clipboard_image_to_file() -> Result<Value> {
    let backend = clipboard_backend();
    if backend == "windows_host_powershell" {
        return windows_host_clipboard_image_to_file();
    }
    if !matches!(backend, "wl-clipboard" | "xclip") {
        return Ok(json!({
            "ok": false,
            "reason": "image_clipboard_backend_unavailable",
            "guidance": "Image paste requires wl-clipboard, xclip, or reachable Windows PowerShell under WSL. Use the paperclip to attach a saved image.",
        }));
    }
    let types = clipboard_types(backend)?;
    let mime = types
        .get("items")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .find(|item| matches!(*item, "image/png" | "image/jpeg" | "image/webp"));
    let Some(mime) = mime else {
        return Ok(json!({
            "ok": false,
            "reason": "clipboard_has_no_supported_image",
            "guidance": "The clipboard does not contain a PNG, JPEG, or WebP image. Use the paperclip to attach a file.",
        }));
    };
    let extension = match mime {
        "image/jpeg" => "jpg",
        "image/webp" => "webp",
        _ => "png",
    };
    let directory = std::env::temp_dir().join("agent-clipboard");
    fs::create_dir_all(&directory)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = directory.join(format!("clipboard-{timestamp}.{extension}"));
    let file = fs::File::create(&path)?;
    let mut command = ProcessCommand::new(if backend == "wl-clipboard" {
        "wl-paste"
    } else {
        "xclip"
    });
    if backend == "wl-clipboard" {
        command.args(["--type", mime]);
    } else {
        command.args(["-selection", "clipboard", "-t", mime, "-o"]);
    }
    let status = command
        .stdout(Stdio::from(file))
        .stderr(Stdio::null())
        .status()?;
    let size_bytes = fs::metadata(&path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let max_bytes = std::env::var("AGENT_MAX_ATTACHMENT_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(25 * 1024 * 1024);
    if !status.success() || size_bytes == 0 || size_bytes > max_bytes {
        let _ = fs::remove_file(&path);
        return Ok(json!({
            "ok": false,
            "reason": if size_bytes > max_bytes { "clipboard_image_too_large" } else { "clipboard_image_read_failed" },
            "guidance": format!("Clipboard image could not be imported within the configured {max_bytes}-byte attachment limit."),
        }));
    }
    Ok(json!({
        "ok": true,
        "path": path,
        "mime": mime,
        "size_bytes": size_bytes,
    }))
}

fn windows_host_clipboard_image_to_file() -> Result<Value> {
    let directory = std::env::temp_dir().join("agent-clipboard");
    fs::create_dir_all(&directory)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = directory.join(format!("clipboard-{timestamp}.png"));
    let windows_path = run_desktop_capture(&["wslpath", "-w", path.to_string_lossy().as_ref()])?;
    if windows_path.status != 0 || windows_path.stdout.trim().is_empty() {
        return Ok(json!({
            "ok": false,
            "reason": "clipboard_image_path_conversion_failed",
            "guidance": "Could not prepare a temporary attachment path for the Windows clipboard image.",
        }));
    }
    let escaped_path = windows_path.stdout.trim().replace('\'', "''");
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; \
         $image = Get-Clipboard -Format Image; if ($null -eq $image) {{ exit 2 }}; \
         try {{ $image.Save('{escaped_path}', [System.Drawing.Imaging.ImageFormat]::Png) }} \
         finally {{ $image.Dispose() }}"
    );
    let output = run_desktop_capture(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-STA",
        "-Command",
        &script,
    ])?;
    let size_bytes = fs::metadata(&path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    let max_bytes = std::env::var("AGENT_MAX_ATTACHMENT_BYTES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(25 * 1024 * 1024);
    if output.status != 0 || size_bytes == 0 || size_bytes > max_bytes {
        let _ = fs::remove_file(&path);
        return Ok(json!({
            "ok": false,
            "reason": if size_bytes > max_bytes { "clipboard_image_too_large" } else { "clipboard_image_read_failed" },
            "guidance": "The Windows clipboard does not contain a readable image, or it exceeds the attachment limit.",
        }));
    }
    Ok(json!({
        "ok": true,
        "path": path,
        "mime": "image/png",
        "size_bytes": size_bytes,
        "backend": "windows_host_powershell",
    }))
}

fn clipboard_write_action(action: &str, value: Option<&str>) -> Result<Value> {
    let text = value.ok_or_else(|| anyhow::anyhow!("{action} requires --value"))?;
    let backend = clipboard_backend();
    if backend == "unavailable" {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "dependency_unavailable",
            "error": "Clipboard write requires wl-copy, xclip, xsel, or reachable Windows PowerShell under WSL.",
        }));
    }
    let before = clipboard_metadata()?;
    let output = clipboard_write_backend(backend, text)?;
    thread::sleep(Duration::from_millis(75));
    let after = clipboard_metadata()?;
    let expected_bytes = text.len() as u64;
    let actual_bytes = after
        .get("text")
        .and_then(|item| item.get("byte_count"))
        .and_then(Value::as_u64);
    let verified = output.status == 0 && actual_bytes == Some(expected_bytes);
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "backend": backend,
        "value": {
            "byte_count": expected_bytes,
            "char_count": text.chars().count(),
            "content_returned": false,
        },
        "before": before,
        "after": after,
        "verified": verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "exit_code": output.status,
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

fn clipboard_types(backend: &str) -> Result<Value> {
    let output = match backend {
        "wl-clipboard" => run_capture_dynamic(&["wl-paste", "--list-types"])?,
        "xclip" => {
            run_capture_dynamic(&["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"])?
        }
        "xsel" => {
            return Ok(json!({
                "available": false,
                "backend": "xsel",
                "items": [],
                "reason": "targets_unavailable",
            }));
        }
        "windows_host_powershell" => run_capture_dynamic(&[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-Clipboard -Format Text -Raw | Out-Null; 'text/plain'",
        ])?,
        _ => {
            return Ok(json!({
                "available": false,
                "backend": "unavailable",
                "items": [],
            }));
        }
    };
    let items: Vec<&str> = output
        .stdout
        .lines()
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .collect();
    Ok(json!({
        "available": output.status == 0,
        "backend": backend,
        "items": items,
        "stderr": output.stderr,
    }))
}

fn clipboard_text(backend: &str) -> Result<DesktopCommandOutput> {
    match backend {
        "wl-clipboard" => run_desktop_capture(&["wl-paste", "--no-newline", "--type", "text"]),
        "xclip" => run_desktop_capture(&["xclip", "-selection", "clipboard", "-o"]),
        "xsel" => run_desktop_capture(&["xsel", "--clipboard", "--output"]),
        "windows_host_powershell" => run_desktop_capture(&[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-Clipboard -Raw",
        ]),
        _ => Ok(DesktopCommandOutput {
            status: -1,
            stdout: String::new(),
            stderr: String::from("Clipboard backend unavailable."),
        }),
    }
}

fn clipboard_write_backend(backend: &str, text: &str) -> Result<DesktopCommandOutput> {
    match backend {
        "wl-clipboard" => run_desktop_capture_with_stdin(&["wl-copy"], text),
        "xclip" => run_desktop_capture_with_stdin(&["xclip", "-selection", "clipboard"], text),
        "xsel" => run_desktop_capture_with_stdin(&["xsel", "--clipboard", "--input"], text),
        "windows_host_powershell" => run_desktop_capture_with_stdin(
            &[
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Set-Clipboard -Value ([Console]::In.ReadToEnd())",
            ],
            text,
        ),
        _ => Ok(DesktopCommandOutput {
            status: -1,
            stdout: String::new(),
            stderr: String::from("Clipboard backend unavailable."),
        }),
    }
}

pub(crate) fn desktop_clipboard_files(paths: &[PathBuf], operation: &str) -> Result<Value> {
    if paths.is_empty() || paths.len() > 32 {
        return Err(anyhow::anyhow!(
            "file clipboard requires between 1 and 32 paths"
        ));
    }
    if !matches!(operation, "copy" | "cut") {
        return Err(anyhow::anyhow!(
            "file clipboard operation must be copy or cut"
        ));
    }

    let mut canonical = Vec::with_capacity(paths.len());
    let mut seen = HashSet::new();
    for path in paths {
        let resolved = fs::canonicalize(path)
            .map_err(|error| anyhow::anyhow!("cannot use {}: {error}", path.display()))?;
        if seen.insert(resolved.clone()) {
            canonical.push(resolved);
        }
    }
    let backend = file_clipboard_backend();
    if backend == "unavailable" {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_clipboard_files",
            "operation": operation,
            "reason": "dependency_unavailable",
            "error": "File clipboard support requires wl-clipboard or xclip.",
        }));
    }

    let separator = if operation == "cut" { "\n" } else { "\r\n" };
    let mut uris = String::new();
    for path in &canonical {
        if !uris.is_empty() {
            uris.push_str(separator);
        }
        uris.push_str(&file_uri(path)?);
    }
    let (mime, payload) = if operation == "cut" {
        ("x-special/gnome-copied-files", format!("cut\n{uris}\n"))
    } else {
        ("text/uri-list", format!("{uris}\r\n"))
    };
    let output = match backend {
        "wl-clipboard" => run_desktop_capture_with_stdin(&["wl-copy", "--type", mime], &payload)?,
        "xclip" => run_desktop_capture_with_stdin(
            &["xclip", "-selection", "clipboard", "-t", mime, "-i"],
            &payload,
        )?,
        _ => unreachable!(),
    };
    thread::sleep(Duration::from_millis(75));
    let types = clipboard_types(backend)?;
    let advertised = types
        .get("items")
        .and_then(Value::as_array)
        .is_some_and(|items| items.iter().any(|item| item.as_str() == Some(mime)));
    let verified = output.status == 0 && advertised;
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_clipboard_files",
        "operation": operation,
        "backend": backend,
        "item_count": canonical.len(),
        "paths": canonical,
        "content_returned": false,
        "mime_type": mime,
        "advertised_types": types,
        "verified": verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "exit_code": output.status,
        "stderr": output.stderr,
    }))
}

fn file_clipboard_backend() -> &'static str {
    if command_exists("wl-copy")
        && command_exists("wl-paste")
        && env::var_os("WAYLAND_DISPLAY").is_some()
    {
        "wl-clipboard"
    } else if command_exists("xclip") && env::var_os("DISPLAY").is_some() {
        "xclip"
    } else {
        "unavailable"
    }
}

#[cfg(unix)]
fn file_uri(path: &Path) -> Result<String> {
    if !path.is_absolute() {
        return Err(anyhow::anyhow!("file clipboard paths must be absolute"));
    }
    let mut uri = String::from("file://");
    for byte in path.as_os_str().as_bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/') {
            uri.push(*byte as char);
        } else {
            use std::fmt::Write as _;
            write!(&mut uri, "%{byte:02X}")?;
        }
    }
    Ok(uri)
}

#[cfg(not(unix))]
fn file_uri(_path: &Path) -> Result<String> {
    Err(anyhow::anyhow!(
        "file clipboard is not supported on this platform"
    ))
}

#[derive(Debug)]
struct DesktopCommandOutput {
    status: i32,
    stdout: String,
    stderr: String,
}

fn run_desktop_capture(argv: &[&str]) -> Result<DesktopCommandOutput> {
    let output = run_capture_dynamic(argv)?;
    Ok(DesktopCommandOutput {
        status: output.status,
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn run_desktop_capture_with_stdin(argv: &[&str], input: &str) -> Result<DesktopCommandOutput> {
    let Some((program, arguments)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty clipboard command"));
    };
    let mut child = ProcessCommand::new(program)
        .args(arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(input.as_bytes())?;
    }
    let output = child.wait_with_output()?;
    Ok(DesktopCommandOutput {
        status: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    })
}

fn window_control_action(action: &str, target: &str) -> Result<Value> {
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

fn window_control_receipt<F>(action: &str, target: &str, argv: &[&str], verify: F) -> Result<Value>
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
    thread::sleep(Duration::from_millis(100));
    let after = window_state(target)?;
    let verified = output.status == 0 && verify(&before, &after);
    Ok(json!({
        "ok": output.status == 0,
        "tool": "desktop_action",
        "action": action,
        "target": target,
        "backend": program,
        "exit_code": output.status,
        "before": before,
        "after": after,
        "verified": verified,
        "verification": if output.status != 0 { "failed" } else if verified { "confirmed" } else { "not_confirmed" },
        "stdout": output.stdout,
        "stderr": output.stderr,
    }))
}

fn window_state(target: &str) -> Result<Value> {
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

fn active_window_id() -> Result<Option<String>> {
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

fn net_wm_states(target: &str) -> Result<Value> {
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

fn window_state_bool(state: &Value, key: &str) -> Option<bool> {
    state.get(key).and_then(Value::as_bool)
}

fn window_has_state(state: &Value, name: &str) -> bool {
    state
        .get("net_wm_state")
        .and_then(|value| value.get("states"))
        .and_then(Value::as_array)
        .is_some_and(|states| states.iter().any(|state| state.as_str() == Some(name)))
}

fn valid_window_id(value: &str) -> bool {
    normalize_window_id(value).is_some()
}

fn normalize_window_id(value: &str) -> Option<String> {
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

fn installed_applications(limit: usize) -> Result<Value> {
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

fn windows_host_registered_applications(limit: usize) -> Result<Vec<Value>> {
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

fn windows_host_registered_application_matches(query: &str, limit: usize) -> Result<Vec<Value>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let output = run_desktop_capture_with_stdin(
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

fn windows_host_registered_application_value(item: Value) -> Option<Value> {
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

fn windows_host_registered_application_matches_script(limit: usize) -> String {
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

fn windows_host_registered_applications_script(limit: usize) -> String {
    "Get-StartApps | Select-Object -First __LIMIT__ Name,AppID | ConvertTo-Json -Compress"
        .replace("__LIMIT__", &limit.to_string())
}

fn windows_host_start_menu_applications(limit: usize) -> Result<Vec<Value>> {
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

fn windows_host_start_menu_applications_script(limit: usize) -> String {
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

fn windows_shortcut_target(path: &str) -> Option<String> {
    if !valid_windows_shortcut_path(path) {
        return None;
    }
    Some(format!(
        "{WINDOWS_SHORTCUT_PREFIX}{}",
        hex_encode(path.as_bytes())
    ))
}

fn windows_app_target(app_id: &str) -> Option<String> {
    if !valid_windows_app_id(app_id) {
        return None;
    }
    Some(format!(
        "{WINDOWS_APP_PREFIX}{}",
        hex_encode(app_id.as_bytes())
    ))
}

fn decode_windows_app_target(target: &str) -> Result<Option<String>> {
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

fn valid_windows_app_id(app_id: &str) -> bool {
    !app_id.is_empty() && app_id.len() <= 512 && !app_id.chars().any(char::is_control)
}

fn decode_windows_shortcut_target(target: &str) -> Result<Option<String>> {
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

fn valid_windows_shortcut_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 1024
        && path.ends_with(".lnk")
        && !path.chars().any(char::is_control)
}

fn hex_encode(bytes: &[u8]) -> String {
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

fn hex_decode(value: &str) -> Result<Vec<u8>> {
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

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn desktop_application_dirs() -> Vec<PathBuf> {
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

fn collect_desktop_entries(
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

fn parse_desktop_entry(root: &Path, path: &Path) -> Result<Option<Value>> {
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

fn visible_windows(limit: usize) -> Result<Value> {
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

fn linux_windows(limit: usize) -> Result<Value> {
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

fn windows_host_windows(limit: usize) -> Result<Value> {
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

fn windows_host_window_list_script(limit: usize) -> String {
    r#"Get-Process | Where-Object {$_.MainWindowTitle -and $_.MainWindowHandle -ne 0} | Select-Object -First __LIMIT__ @{Name='Id';Expression={'0x{0:x}' -f $_.MainWindowHandle.ToInt64()}},@{Name='Pid';Expression={$_.Id}},ProcessName,MainWindowTitle,@{Name='Path';Expression={try {$_.Path} catch {$null}}} | ConvertTo-Json -Compress"#
        .replace("__LIMIT__", &limit.to_string())
}

fn active_window() -> Result<Value> {
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

fn linux_active_window() -> Result<Value> {
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

fn windows_host_active_window() -> Result<Value> {
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

fn windows_host_focus_window_receipt(action: &str, target: &str) -> Result<Value> {
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

fn windows_host_focus_window(target: &str) -> Result<DesktopCommandOutput> {
    let script = windows_host_focus_window_script(target)?;
    run_desktop_capture(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])
}

fn windows_host_focus_window_script(target: &str) -> Result<String> {
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

fn windows_host_terminate_process_receipt(action: &str, raw_pid: &str, pid: i32) -> Result<Value> {
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

fn windows_host_terminate_process(pid: i32) -> Result<DesktopCommandOutput> {
    let script = windows_host_terminate_process_script(pid)?;
    run_desktop_capture(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])
}

fn windows_host_terminate_process_script(pid: i32) -> Result<String> {
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

fn windows_host_process_state(pid: i32) -> Result<Value> {
    let script = windows_host_process_state_script(pid)?;
    let output = run_desktop_capture(&[
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

fn windows_host_process_state_script(pid: i32) -> Result<String> {
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

fn windows_host_close_window_receipt(action: &str, target: &str) -> Result<Value> {
    let before = window_state(target)?;
    let output = windows_host_close_window(target)?;
    thread::sleep(Duration::from_millis(250));
    let after = window_state(target)?;
    let after_closed = window_state_bool(&after, "visible") == Some(false);
    let verified = output.status == 0 && after_closed;
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

fn windows_host_close_window(target: &str) -> Result<DesktopCommandOutput> {
    let script = windows_host_close_window_script(target)?;
    run_desktop_capture(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        &script,
    ])
}

fn windows_host_close_window_script(target: &str) -> Result<String> {
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

fn json_values(value: Value) -> Vec<Value> {
    match value {
        Value::Array(items) => items,
        Value::Null => Vec::new(),
        item => vec![item],
    }
}

fn nth_field_start(line: &str, index: usize) -> Option<usize> {
    let mut in_field = false;
    let mut current = 0usize;
    for (offset, ch) in line.char_indices() {
        if ch.is_whitespace() {
            in_field = false;
            continue;
        }
        if !in_field {
            if current == index {
                return Some(offset);
            }
            current += 1;
            in_field = true;
        }
    }
    None
}

fn required_percent(action: &str, value: Option<&str>) -> Result<u8> {
    let raw =
        value.ok_or_else(|| anyhow::anyhow!("{action} requires --value between 0 and 100"))?;
    let percent = raw
        .parse::<u8>()
        .map_err(|_| anyhow::anyhow!("{action} value must be between 0 and 100"))?;
    if percent > 100 {
        return Err(anyhow::anyhow!("{action} value must be between 0 and 100"));
    }
    Ok(percent)
}

pub(crate) fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':' | '@'))
}

pub(crate) fn valid_path_token(value: &str) -> bool {
    value
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '/' | '-' | '_' | '.'))
}

pub(crate) fn valid_bluetooth_address(value: &str) -> bool {
    let mut count = 0;
    let valid = value.split(':').all(|part| {
        count += 1;
        part.len() == 2 && part.chars().all(|character| character.is_ascii_hexdigit())
    });
    valid && count == 6
}

fn audio_state(kind: &str) -> Value {
    let command = if kind == "mute" {
        "get-sink-mute"
    } else {
        "get-sink-volume"
    };
    run_capture_dynamic(&["pactl", command, "@DEFAULT_SINK@"])
        .map(|output| json!({ "available": true, "value": output.stdout }))
        .unwrap_or_else(|_| json!({ "available": false }))
}

fn brightness_state() -> Value {
    run_capture_dynamic(&["brightnessctl", "-m"])
        .map(|output| json!({ "available": true, "value": output.stdout }))
        .unwrap_or_else(|_| json!({ "available": false }))
}

fn bluetooth_connection_state(address: &str) -> Value {
    let connected = run_capture_dynamic(&["bluetoothctl", "info", address])
        .ok()
        .and_then(|output| bluetooth_info_value(&output.stdout, "Connected"))
        .as_deref()
        == Some("yes");
    json!({ "available": command_exists("bluetoothctl"), "connected": connected })
}

fn network_connection_state(interface: &str) -> Value {
    let path = PathBuf::from("/sys/class/net").join(interface);
    json!({
        "available": path.exists(),
        "operstate": read_trimmed(path.join("operstate")),
        "carrier": read_trimmed(path.join("carrier")).as_deref() == Some("1"),
    })
}

fn storage_mount_state(device: &str) -> Value {
    let mounted = fs::read_to_string("/proc/self/mountinfo")
        .map(|text| {
            text.lines().any(|line| {
                line.split_once(" - ")
                    .is_some_and(|(_, tail)| tail.split_whitespace().nth(1) == Some(device))
            })
        })
        .unwrap_or(false);
    json!({ "available": std::path::Path::new(device).exists(), "mounted": mounted })
}

fn process_state(pid: i32) -> Value {
    let path = PathBuf::from(format!("/proc/{pid}"));
    json!({
        "pid": pid,
        "running": path.exists(),
        "name": read_trimmed(path.join("comm")),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        accessibility_target_id, accessibility_tree_item, bounded_accessible_text,
        desktop_capabilities, desktop_observe, dialog_items, dialog_kind, hex_decode, hex_encode,
        parse_busctl_string, parse_pactl_mute, parse_pactl_volume_percent, parse_xrandr_displays,
        windows_host_volume_script,
    };
    use serde_json::{json, Value};

    #[test]
    fn hex_codec_roundtrips_every_byte_value() {
        let bytes = (u8::MIN..=u8::MAX).collect::<Vec<_>>();

        assert_eq!(hex_decode(&hex_encode(&bytes)).unwrap(), bytes);
    }

    #[test]
    fn desktop_capabilities_reports_runtime_actions() {
        let capabilities = desktop_capabilities().expect("desktop capabilities");

        assert_eq!(capabilities.get("ok").and_then(Value::as_bool), Some(true));
        assert_eq!(
            capabilities.get("tool").and_then(Value::as_str),
            Some("desktop_capabilities")
        );
        assert!(capabilities
            .get("actions")
            .and_then(Value::as_array)
            .is_some_and(|actions| actions
                .iter()
                .any(|action| action.get("action").and_then(Value::as_str) == Some("set_volume"))));
        assert!(capabilities
            .get("backends")
            .and_then(Value::as_object)
            .is_some());
        let actions = capabilities
            .get("actions")
            .and_then(Value::as_array)
            .expect("desktop actions");
        for action_name in ["launch_application", "close_window"] {
            let action = actions
                .iter()
                .find(|action| action.get("action").and_then(Value::as_str) == Some(action_name))
                .expect("direct desktop action");
            assert_eq!(action.get("safety").and_then(Value::as_str), Some("direct"));
        }
    }

    #[test]
    fn desktop_observe_reports_versioned_read_only_snapshot() {
        let snapshot = desktop_observe("active_window", 5).expect("desktop observation");

        assert_eq!(
            snapshot.get("tool").and_then(Value::as_str),
            Some("desktop_observe")
        );
        assert_eq!(
            snapshot.get("schema_version").and_then(Value::as_u64),
            Some(1)
        );
        assert_eq!(
            snapshot.get("scope").and_then(Value::as_str),
            Some("active_window")
        );
        assert!(snapshot
            .get("snapshot_id")
            .and_then(Value::as_str)
            .is_some());
        assert!(snapshot
            .get("backends")
            .and_then(Value::as_object)
            .is_some());
        assert!(snapshot
            .get("limitations")
            .and_then(Value::as_array)
            .is_some());
    }

    #[test]
    fn parses_atspi_busctl_string_reply() {
        assert_eq!(
            parse_busctl_string(r#"s "unix:path=/run/user/1000/at-spi/bus_0""#),
            Some("unix:path=/run/user/1000/at-spi/bus_0".to_string())
        );
        assert_eq!(parse_busctl_string("s \"\""), None);
        assert_eq!(parse_busctl_string("not a quoted reply"), None);
    }

    #[test]
    fn accessibility_tree_item_accepts_only_atspi_paths() {
        let item = accessibility_tree_item("/org/a11y/atspi/accessible/root")
            .expect("expected atspi item");

        assert_eq!(
            item.get("path").and_then(Value::as_str),
            Some("/org/a11y/atspi/accessible/root")
        );
        assert_eq!(
            item.get("kind").and_then(Value::as_str),
            Some("accessible_object")
        );
        assert!(accessibility_tree_item("├─/unrelated/path").is_none());
    }

    #[test]
    fn accessibility_target_ids_are_snapshot_bound_and_deterministic() {
        let first = accessibility_target_id("desktop-1", ":1.42", "/org/example/button/1");
        let repeated = accessibility_target_id("desktop-1", ":1.42", "/org/example/button/1");
        let newer = accessibility_target_id("desktop-2", ":1.42", "/org/example/button/1");

        assert_eq!(first, repeated);
        assert_ne!(first, newer);
        assert!(first.starts_with("ui-"));
    }

    #[test]
    fn accessibility_text_is_bounded_by_characters() {
        let text = format!("{}é", "a".repeat(300));
        let bounded = bounded_accessible_text(&text);

        assert_eq!(bounded.chars().count(), 256);
        assert!(bounded.is_char_boundary(bounded.len()));
    }

    #[test]
    fn classifies_protocol_dialog_roles_without_application_names() {
        assert_eq!(dialog_kind("dialog"), Some("dialog"));
        assert_eq!(dialog_kind("alert dialog"), Some("alert"));
        assert_eq!(dialog_kind("file chooser"), Some("file_picker"));
        assert_eq!(dialog_kind("push button"), None);
    }

    #[test]
    fn groups_dialog_controls_by_accessibility_parentage() {
        let items = vec![
            json!({"id": "root", "parent_id": null, "role": "application"}),
            json!({"id": "dialog", "parent_id": "root", "role": "file chooser"}),
            json!({"id": "panel", "parent_id": "dialog", "role": "panel"}),
            json!({"id": "name", "parent_id": "panel", "role": "text", "name": "Name"}),
            json!({"id": "save", "parent_id": "dialog", "role": "push button", "name": "Save"}),
            json!({"id": "outside", "parent_id": "root", "role": "push button"}),
        ];

        let dialogs = dialog_items(&items, 10);

        assert_eq!(dialogs.len(), 1);
        assert_eq!(
            dialogs[0].get("dialog_kind").and_then(Value::as_str),
            Some("file_picker")
        );
        let controls = dialogs[0]
            .get("controls")
            .and_then(Value::as_array)
            .unwrap();
        assert_eq!(controls.len(), 3);
        assert!(controls
            .iter()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("name")));
        assert!(!controls
            .iter()
            .any(|item| item.get("id").and_then(Value::as_str) == Some("outside")));
    }

    #[test]
    fn parses_xrandr_display_state_and_geometry() {
        let displays = parse_xrandr_displays(
            "eDP-1 connected primary 1920x1080+0+0 (normal left inverted right x axis y axis)\nHDMI-1 disconnected (normal left inverted right x axis y axis)\nDP-1 connected 2560x1440-2560+0 (normal left inverted right x axis y axis)",
        );

        assert_eq!(displays.len(), 3);
        assert_eq!(
            displays[0].get("name").and_then(Value::as_str),
            Some("eDP-1")
        );
        assert_eq!(
            displays[0].get("primary").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            displays[0]
                .pointer("/geometry/width")
                .and_then(Value::as_u64),
            Some(1920)
        );
        assert_eq!(
            displays[1].get("connected").and_then(Value::as_bool),
            Some(false)
        );
        assert!(displays[1].get("geometry").is_some_and(Value::is_null));
        assert_eq!(
            displays[2].pointer("/geometry/x").and_then(Value::as_i64),
            Some(-2560)
        );
    }

    #[test]
    fn parses_pactl_default_sink_state() {
        assert_eq!(
            parse_pactl_volume_percent(
                "Volume: front-left: 32768 / 50% / -18.06 dB, front-right: 32768 / 50% / -18.06 dB"
            ),
            Some(50)
        );
        assert_eq!(parse_pactl_mute("Mute: yes"), Some(true));
        assert_eq!(parse_pactl_mute("Mute: no"), Some(false));
        assert_eq!(parse_pactl_volume_percent("invalid"), None);
        assert_eq!(parse_pactl_mute("Mute: unknown"), None);
    }

    #[test]
    fn launch_verification_requires_real_evidence() {
        let before = json!({
            "process": null,
            "windows": {"ok": false, "count": 0, "ids": [], "titles": []},
        });
        let after = json!({
            "process": {"running": true},
            "windows": {"ok": false, "count": 0, "ids": [], "titles": []},
        });

        assert!(!super::launch_observation_changed(&before, &after, false));
        assert!(super::launch_observation_changed(&before, &after, true));
    }

    #[test]
    fn launch_verification_accepts_window_change() {
        let before = json!({
            "windows": {"ok": true, "count": 1, "ids": ["1"], "titles": ["Shell"]},
        });
        let after = json!({
            "windows": {"ok": true, "count": 2, "ids": ["1", "2"], "titles": ["Shell", "Browser"]},
        });

        assert!(super::launch_observation_changed(&before, &after, false));
    }

    #[test]
    fn window_ids_are_normalized_without_accepting_shell_shapes() {
        assert_eq!(
            super::normalize_window_id("0x03a00007").as_deref(),
            Some("0x3a00007")
        );
        assert_eq!(
            super::normalize_window_id("60817415").as_deref(),
            Some("0x3a00007")
        );
        assert!(super::normalize_window_id("0x03a00007;rm").is_none());
        assert!(super::normalize_window_id("").is_none());
    }

    #[test]
    fn window_state_verification_reads_net_wm_state_atoms() {
        let state = json!({
            "net_wm_state": {
                "states": [
                    "_NET_WM_STATE_MAXIMIZED_VERT",
                    "_NET_WM_STATE_MAXIMIZED_HORZ"
                ]
            }
        });

        assert!(super::window_has_state(
            &state,
            "_NET_WM_STATE_MAXIMIZED_VERT"
        ));
        assert!(!super::window_has_state(&state, "_NET_WM_STATE_HIDDEN"));
    }

    #[test]
    fn clipboard_metadata_never_returns_content() {
        let metadata = super::clipboard_metadata().expect("clipboard metadata");

        assert_eq!(
            metadata.get("content_returned").and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn clipboard_write_requires_value_before_backend_selection() {
        let error = super::clipboard_write_action("clipboard_write", None)
            .expect_err("missing clipboard value should fail");

        assert!(error.to_string().contains("requires --value"));
    }

    #[test]
    fn wsl_host_clipboard_wins_over_installed_linux_backends() {
        assert_eq!(
            super::select_clipboard_backend(true, true, true, true),
            "windows_host_powershell"
        );
        assert_eq!(
            super::select_clipboard_backend(false, true, true, true),
            "wl-clipboard"
        );
    }

    #[cfg(unix)]
    #[test]
    fn file_clipboard_uri_percent_encodes_path_bytes() {
        let uri = super::file_uri(std::path::Path::new("/tmp/a file#1.txt")).expect("file URI");

        assert_eq!(uri, "file:///tmp/a%20file%231.txt");
    }

    #[test]
    fn partial_download_detection_is_case_insensitive_and_bounded_to_suffixes() {
        assert!(super::is_partial_download_name("archive.zip.crdownload"));
        assert!(super::is_partial_download_name("video.PART"));
        assert!(!super::is_partial_download_name("part-notes.txt"));
        assert!(!super::is_partial_download_name("archive.zip"));
    }

    #[test]
    fn download_inventory_distinguishes_complete_and_partial_files() {
        let directory =
            std::env::temp_dir().join(format!("agent-download-test-{}", super::now_unix_millis()));
        std::fs::create_dir(&directory).expect("create download test directory");
        std::fs::write(directory.join("ready.zip"), b"ready").expect("write complete file");
        std::fs::write(directory.join("pending.zip.part"), b"pending").expect("write partial file");

        let inventory = super::download_inventory_at(&directory, 10).expect("download inventory");
        std::fs::remove_dir_all(&directory).expect("remove download test directory");

        assert_eq!(inventory.get("count").and_then(Value::as_u64), Some(2));
        assert_eq!(
            inventory.get("partial_count").and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    fn desktop_resolver_normalizes_and_scores_names() {
        let query = super::normalize_match_text("VS Code!");

        assert_eq!(query, "vs code");
        assert_eq!(super::match_score(&query, &["VS Code"]), 100);
        assert!(super::match_score(&query, &["Visual Studio Code"]) > 0);
        assert_eq!(super::match_score(&query, &["Terminal"]), 0);
    }

    #[test]
    fn desktop_resolver_preserves_all_query_terms() {
        let query = super::normalize_match_text("my Spark app");

        assert_eq!(query, "my spark app");
        assert_eq!(super::match_score(&query, &["Spark"]), 13);
        assert_eq!(
            super::match_score(
                &super::normalize_match_text("Spark Uninstaller"),
                &["Spark"]
            ),
            20
        );
    }

    #[test]
    fn windows_host_window_list_uses_real_window_handles() {
        let script = super::windows_host_window_list_script(10);

        assert!(script.contains("MainWindowHandle"));
        assert!(script.contains("'0x{0:x}'"));
        assert!(script.contains("Name='Pid'"));
    }

    #[test]
    fn windows_host_focus_script_restores_and_focuses_handle() {
        let script = super::windows_host_focus_window_script("0x3a00007").expect("focus script");

        assert!(script.contains("ShowWindowAsync($hwnd, 9)"));
        assert!(script.contains("SetForegroundWindow($hwnd)"));
        assert!(super::windows_host_focus_window_script("0x3a00007;rm").is_err());
    }

    #[test]
    fn launched_window_id_picks_new_window_after_launch() {
        let before = json!({"windows": {"ids": ["0x1", "0x2"]}});
        let after = json!({"windows": {"ids": ["0x1", "0x2", "0x3"]}});

        assert_eq!(
            super::launched_window_id(&before, &after).as_deref(),
            Some("0x3")
        );
        assert_eq!(super::launched_window_id(&after, &before), None);
    }

    #[test]
    fn windows_shortcut_focus_query_uses_shortcut_name() {
        assert_eq!(
            super::windows_shortcut_focus_query(
                r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Spark.lnk"
            ),
            "Spark"
        );
    }

    #[test]
    fn windows_host_close_script_requests_normal_window_close() {
        let script = super::windows_host_close_window_script("0x3a00007").expect("close script");

        assert!(script.contains("PostMessage($hwnd, 0x0010"));
        assert!(script.contains("IsWindow($hwnd)"));
        assert!(script.contains("$processId = 0"));
        assert!(script.contains("Stop-Process -Id $processId -Force"));
        assert!(!script.contains("$pid"));
        assert!(super::windows_host_close_window_script("0x3a00007;rm").is_err());
    }

    #[test]
    fn windows_host_process_scripts_do_not_overwrite_builtin_pid_variable() {
        let terminate =
            super::windows_host_terminate_process_script(16608).expect("terminate process script");
        let state = super::windows_host_process_state_script(16608).expect("process state script");

        assert!(terminate.contains("$processId = 16608"));
        assert!(terminate.contains("Stop-Process -Id $processId -Force"));
        assert!(!terminate.contains("$pid"));
        assert!(state.contains("$processId = 16608"));
        assert!(state.contains("Get-Process -Id $processId"));
        assert!(!state.contains("$pid"));
        assert!(super::windows_host_terminate_process_script(1).is_err());
        assert!(super::windows_host_process_state_script(0).is_err());
    }

    #[test]
    fn windows_shortcut_targets_are_identifier_safe_and_roundtrip() {
        let path = r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Spark.lnk";
        let target = super::windows_shortcut_target(path).expect("shortcut target");

        assert!(target.starts_with(super::WINDOWS_SHORTCUT_PREFIX));
        assert!(super::valid_identifier(&target));
        assert_eq!(
            super::decode_windows_shortcut_target(&target)
                .expect("decode")
                .as_deref(),
            Some(path)
        );
        assert!(super::windows_shortcut_target("C:\\Temp\\Spark.exe").is_none());
    }

    #[test]
    fn windows_shortcut_launch_script_uses_file_path() {
        let script = super::windows_host_start_shortcut_script();

        assert!(script.contains("Test-Path -LiteralPath"));
        assert!(script.contains("Start-Process -FilePath"));
        assert!(script.contains(".lnk"));
    }

    #[test]
    fn windows_registered_app_targets_are_identifier_safe_and_roundtrip() {
        let app_id = "Vitelglobal.Vitelglobal_v1ncde5y6f3mm!com.vitelglobal.vitelgloabalapp.winx";
        let target = super::windows_app_target(app_id).expect("app target");

        assert!(target.starts_with(super::WINDOWS_APP_PREFIX));
        assert!(super::valid_identifier(&target));
        assert_eq!(
            super::decode_windows_app_target(&target)
                .expect("decode")
                .as_deref(),
            Some(app_id)
        );
        assert!(super::windows_app_target("bad\nid").is_none());
    }

    #[test]
    fn windows_registered_app_launch_script_uses_apps_folder() {
        let script = super::windows_host_start_app_script();

        assert!(script.contains("shell:AppsFolder"));
        assert!(script.contains("Start-Process"));
        assert!(script.contains("Invalid app id"));
    }

    #[test]
    fn windows_registered_app_match_script_filters_before_limit() {
        let script = super::windows_host_registered_application_matches_script(10);

        assert!(script.contains("Where-Object"));
        assert!(script.contains("Contains($query)"));
        assert!(script.contains("Select-Object -First 10"));
    }

    #[test]
    fn keyboard_key_specs_reject_shell_shapes() {
        assert_eq!(super::normalize_key_spec("ctrl+l").unwrap(), "ctrl+l");
        assert_eq!(super::normalize_key_spec("Alt+Tab").unwrap(), "Alt+Tab");
        assert!(super::normalize_key_spec("ctrl+l;rm").is_err());
        assert!(super::normalize_key_spec("").is_err());
    }

    #[test]
    fn keyboard_type_receipt_redacts_text() {
        let receipt = super::keyboard_value_receipt("type_text", "secret typed text");

        assert_eq!(
            receipt.get("content_returned").and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(receipt.get("char_count").and_then(Value::as_u64), Some(17));
        assert!(receipt.get("secret typed text").is_none());
    }

    #[test]
    fn pointer_coordinates_and_scroll_are_bounded() {
        assert_eq!(super::parse_pointer_coordinates("10,20").unwrap(), (10, 20));
        assert!(super::parse_pointer_coordinates("10;rm,20").is_err());
        assert!(super::parse_pointer_coordinates("-1,20").is_err());
        assert_eq!(super::pointer_button(Some("3")).unwrap(), 3);
        assert!(super::pointer_button(Some("4")).is_err());
        assert_eq!(super::parse_scroll_steps(Some("-5")).unwrap(), -5);
        assert!(super::parse_scroll_steps(Some("0")).is_err());
        assert!(super::parse_scroll_steps(Some("26")).is_err());
    }

    #[test]
    fn windows_volume_script_uses_validated_scalar_and_verifies_result() {
        let script = windows_host_volume_script(0);

        assert!(script.contains("[AgentCoreAudio]::SetVolume(0)"));
        assert!(script.contains("after_percent"));
        assert!(script.contains("ConvertTo-Json -Compress"));
        assert!(script.contains("5CDF2C82-841E-4546-9722-0CF74078229A"));
    }

    #[test]
    fn windows_volume_script_converts_percent_to_scalar() {
        let script = windows_host_volume_script(75);

        assert!(script.contains("[AgentCoreAudio]::SetVolume(0.75)"));
    }
}
