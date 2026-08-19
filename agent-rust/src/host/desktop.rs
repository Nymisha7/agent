use super::{
    bluetooth::info_value as bluetooth_info_value,
    platform::{command_exists, is_wsl_runtime, user_downloads_directory},
    system::{
        read_trimmed, required_target, run_capture_dynamic, run_capture_with_stdin, CommandOutput,
    },
};
use anyhow::Result;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::OnceLock;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

mod accessibility;
mod applications;
mod audio;
mod clipboard;
mod displays;
mod downloads;
mod input;
mod resolver;
mod windows;

use accessibility::*;
use applications::*;
use audio::*;
use clipboard::{
    clipboard_backend, clipboard_metadata, clipboard_write_action, file_clipboard_backend,
};
pub(crate) use clipboard::{
    desktop_clipboard_files, desktop_clipboard_image_to_file, desktop_clipboard_read_text,
    desktop_open_user_file, desktop_pick_file,
};
#[cfg(test)]
use clipboard::{file_uri, select_clipboard_backend};
use displays::*;
use downloads::*;
use input::*;
use resolver::*;
use windows::*;

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
            action_capability("open_path", linux_supported && has_xdg_open, "xdg-open", "direct", Some("target_required")),
            action_capability("open_path_in_application", linux_supported, "application_command", "direct", Some("target_and_value_required")),
            action_capability("open_url", linux_supported && has_xdg_open, "xdg-open", "approval_required", Some("target_required")),
            action_capability("focus_window", linux_supported && (command_exists("wmctrl") || command_exists("xdotool") || (wsl && has_powershell_host)), window_control_backend(), "direct", Some("target_required")),
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
        "open_path_in_application" => open_path_in_application(
            action,
            required_target(action, target)?,
            value.ok_or_else(|| anyhow::anyhow!("{action} requires an application executable"))?,
        ),
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
mod tests;
