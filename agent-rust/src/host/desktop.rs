use super::{
    bluetooth_info_value, command_exists, is_wsl_runtime, read_trimmed, required_target,
    run_capture_dynamic,
};
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Stdio};
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn desktop_capabilities() -> Result<Value> {
    let runtime = desktop_runtime();
    let linux_supported = cfg!(target_os = "linux");
    let wsl = is_wsl_runtime();
    let has_pactl = command_exists("pactl");
    let has_powershell_host = windows_host_powershell_available();
    let has_gtk_launch = command_exists("gtk-launch");
    let has_xdg_open = command_exists("xdg-open");

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
                linux_supported && (command_exists("wmctrl") || command_exists("xdotool")),
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
                linux_supported && accessibility_backend() != "unavailable",
                accessibility_backend(),
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
            action_capability("terminate_process", linux_supported, "libc_signal", "approval_required", Some("target_required")),
            action_capability("launch_application", linux_supported && (has_gtk_launch || has_xdg_open), if has_gtk_launch { "gtk-launch" } else if has_xdg_open { "xdg-open" } else { "path_lookup" }, "approval_required", Some("target_required")),
            action_capability("open_path", linux_supported && has_xdg_open, "xdg-open", "approval_required", Some("target_required")),
            action_capability("open_url", linux_supported && has_xdg_open, "xdg-open", "approval_required", Some("target_required")),
            action_capability("focus_window", linux_supported && (command_exists("wmctrl") || command_exists("xdotool")), window_control_backend(), "approval_required", Some("target_required")),
            action_capability("minimize_window", linux_supported && command_exists("xdotool"), "xdotool", "approval_required", Some("target_required")),
            action_capability("maximize_window", linux_supported && command_exists("wmctrl"), "wmctrl", "approval_required", Some("target_required")),
            action_capability("restore_window", linux_supported && command_exists("wmctrl"), "wmctrl", "approval_required", Some("target_required")),
            action_capability("close_window", linux_supported && (command_exists("wmctrl") || command_exists("xdotool")), window_control_backend(), "approval_required", Some("target_required")),
            action_capability("clipboard_write", linux_supported && clipboard_backend() != "unavailable", clipboard_backend(), "approval_required", None),
            action_capability("send_key", linux_supported && command_exists("xdotool"), "xdotool", "approval_required", None),
            action_capability("type_text", linux_supported && keyboard_backend() != "unavailable", keyboard_backend(), "approval_required", None),
            action_capability("mouse_click", linux_supported && pointer_backend() != "unavailable", pointer_backend(), "approval_required", Some("target_required")),
            action_capability("scroll", linux_supported && pointer_backend() != "unavailable", pointer_backend(), "approval_required", None),
        ],
        "limitations": desktop_limitations(linux_supported, wsl),
    }))
}

pub(crate) fn desktop_observe(scope: &str, limit: usize) -> Result<Value> {
    let normalized_scope = match scope {
        "all" | "applications" | "windows" | "active_window" | "clipboard" | "ui_tree"
        | "displays" | "audio" => scope,
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
    let mut payload = json!({
        "ok": linux_supported,
        "tool": "desktop_observe",
        "schema_version": 1,
        "scope": normalized_scope,
        "runtime": desktop_runtime(),
        "platform": std::env::consts::OS,
        "wsl": wsl,
        "snapshot_id": format!("desktop-{}", now_unix_millis()),
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
                linux_supported && accessibility_backend() != "unavailable",
                accessibility_backend(),
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
        insert_payload_value(&mut payload, "ui_tree", accessibility_tree(bounded_limit)?);
    }
    if matches!(normalized_scope, "all" | "displays") {
        insert_payload_value(&mut payload, "displays", display_inventory(bounded_limit)?);
    }
    if normalized_scope == "audio" {
        insert_payload_value(&mut payload, "audio", audio_observation()?);
    }

    Ok(payload)
}

pub(crate) fn desktop_resolve(query: &str, kind: &str, limit: usize) -> Result<Value> {
    let normalized_query = normalize_match_text(query);
    if normalized_query.is_empty() {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_resolve",
            "reason": "query_required",
            "error": "desktop_resolve requires a non-empty query.",
        }));
    }
    let normalized_kind = match kind {
        "application" | "window" | "any" => kind,
        _ => "any",
    };
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
            if pid <= 1 || pid == std::process::id() as i32 {
                return Err(anyhow::anyhow!("Refusing to terminate protected PID {pid}"));
            }
            let before = process_state(pid);
            let status = unsafe { libc::kill(pid, libc::SIGTERM) };
            thread::sleep(Duration::from_millis(75));
            let after = process_state(pid);
            Ok(json!({
                "ok": status == 0,
                "tool": "desktop_action",
                "action": action,
                "target": raw_pid,
                "before": before,
                "after": after,
                "verified": status == 0 && after.get("running").and_then(Value::as_bool) == Some(false),
                "verification": if status == 0 && after.get("running").and_then(Value::as_bool) == Some(false) { "confirmed" } else { "not_confirmed" },
                "error": if status == 0 { None } else { Some(std::io::Error::last_os_error().to_string()) },
            }))
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
        _ => Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "reason": "unsupported_action",
            "error": format!("Unsupported desktop action: {action}"),
        })),
    }
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
}

fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
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
    if argv.first().is_some_and(|command| !command_exists(command)) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "value": value,
            "reason": "dependency_unavailable",
            "error": format!("Required desktop command `{}` is not installed.", argv[0]),
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
    }))
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
        .cloned()
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

fn clipboard_backend() -> &'static str {
    if command_exists("wl-copy") && command_exists("wl-paste") {
        "wl-clipboard"
    } else if command_exists("xclip") {
        "xclip"
    } else if command_exists("xsel") {
        "xsel"
    } else if windows_host_powershell_available() {
        "windows_host_powershell"
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
    if command_exists("busctl") && atspi_bus_address().is_ok() {
        "atspi_busctl"
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
    let all_items = parse_xrandr_displays(&output.stdout);
    let total = all_items.len();
    let items: Vec<Value> = all_items.into_iter().take(limit).collect();
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
    let fields: Vec<&str> = line.split_whitespace().collect();
    let status = *fields.get(1)?;
    if !matches!(status, "connected" | "disconnected") {
        return None;
    }
    let geometry = fields.iter().find_map(|field| parse_xrandr_geometry(field));
    Some(json!({
        "name": fields[0],
        "status": status,
        "connected": status == "connected",
        "primary": fields.contains(&"primary"),
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
    let stderr = [volume.stderr, mute.stderr, sink.stderr]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<String>>()
        .join("\n");
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

fn accessibility_tree(limit: usize) -> Result<Value> {
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
        "backend": "atspi_busctl",
        "content": "object_paths_only",
        "count": items.len(),
        "items": items,
        "truncated": output.stdout.lines().count() > limit,
        "stderr": output.stderr,
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
            let argv = vec![
                "xdotool".to_string(),
                "mousemove".to_string(),
                "--sync".to_string(),
                x.to_string(),
                y.to_string(),
                "click".to_string(),
                button.to_string(),
            ];
            let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
            pointer_receipt(action, json!({ "x": x, "y": y, "button": button }), &refs)
        }
        "scroll" => {
            let steps = parse_scroll_steps(value)?;
            let button = if steps < 0 { "5" } else { "4" };
            let repetitions = steps.unsigned_abs().min(25);
            let mut argv = vec!["xdotool".to_string()];
            for _ in 0..repetitions {
                argv.push("click".to_string());
                argv.push(button.to_string());
            }
            let refs: Vec<&str> = argv.iter().map(String::as_str).collect();
            pointer_receipt(
                action,
                json!({ "steps": steps, "button": button, "repetitions": repetitions }),
                &refs,
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

fn pointer_receipt(action: &str, value: Value, argv: &[&str]) -> Result<Value> {
    if argv.first().is_some_and(|command| !command_exists(command)) {
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
    if argv.first().is_some_and(|command| !command_exists(command)) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "backend": backend,
            "reason": "dependency_unavailable",
            "error": format!("Required keyboard command `{}` is not installed.", argv[0]),
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
            "byte_count": value.as_bytes().len(),
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
            let score = match_score(query, &[name, id, exec]);
            if score > 0 {
                candidates.push(json!({
                    "kind": "application",
                    "score": score,
                    "id": id,
                    "name": name,
                    "exec": exec,
                    "target": id,
                    "action": "launch_application",
                    "backend": apps.get("backend").cloned(),
                }));
            }
        }
    }
    candidates.sort_by(|left, right| candidate_score(right).cmp(&candidate_score(left)));
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
    candidates.sort_by(|left, right| candidate_score(right).cmp(&candidate_score(left)));
    candidates.truncate(limit);
    Ok(candidates)
}

fn match_score(query: &str, fields: &[&str]) -> i64 {
    let mut best = 0;
    let query_tokens: Vec<&str> = query.split_whitespace().collect();
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
            let matched = query_tokens
                .iter()
                .filter(|token| normalized.contains(**token))
                .count();
            if matched > 0 {
                (matched as i64 * 40) / query_tokens.len().max(1) as i64
            } else {
                0
            }
        };
        best = best.max(score);
    }
    best
}

fn normalize_match_text(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
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
            "byte_count": if readable { Some(text.stdout.as_bytes().len()) } else { None },
            "char_count": if readable { Some(text.stdout.chars().count()) } else { None },
            "line_count": if readable { Some(text.stdout.lines().count()) } else { None },
            "utf8": readable,
            "preview_redacted": true,
            "stderr": if readable { None } else { Some(text.stderr) },
        },
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
    let expected_bytes = text.as_bytes().len() as u64;
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
    if argv.is_empty() {
        return Err(anyhow::anyhow!("Cannot run an empty clipboard command"));
    }
    let mut child = ProcessCommand::new(argv[0])
        .args(&argv[1..])
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
            } else {
                window_control_receipt(
                    action,
                    target,
                    &["xdotool", "windowactivate", target],
                    |_, after| window_state_bool(after, "active") == Some(true),
                )
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
            } else {
                window_control_receipt(
                    action,
                    target,
                    &["xdotool", "windowclose", target],
                    |before, after| {
                        window_state_bool(before, "visible") == Some(true)
                            && window_state_bool(after, "visible") == Some(false)
                    },
                )
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
    if argv.first().is_some_and(|command| !command_exists(command)) {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_action",
            "action": action,
            "target": target,
            "reason": "dependency_unavailable",
            "error": format!("Required window command `{}` is not installed.", argv[0]),
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
        "backend": argv[0],
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
    Ok(json!({
        "ok": true,
        "backend": "freedesktop_desktop_entries",
        "count": apps.len(),
        "items": apps,
        "truncated": apps.len() >= limit,
    }))
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
    let mut entry_type = None;
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
            "Type" => entry_type = Some(value.to_string()),
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

    if entry_type.as_deref() != Some("Application") || name.is_none() || hidden || no_display {
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
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
        }
        let title_start = nth_field_start(line, 4).unwrap_or(line.len());
        items.push(json!({
            "id": fields[0],
            "desktop": fields[1],
            "pid": fields[2].parse::<u32>().ok(),
            "class": fields[3],
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
    let script = format!(
        "Get-Process | Where-Object {{$_.MainWindowTitle}} | Select-Object -First {} Id,ProcessName,MainWindowTitle,Path | ConvertTo-Json -Compress",
        limit
    );
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
                "pid": item.get("Id").cloned(),
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
$pid = 0
[void][AgentUser32]::GetWindowThreadProcessId($hwnd, [ref]$pid)
$process = if ($pid) { Get-Process -Id $pid -ErrorAction SilentlyContinue } else { $null }
@{ id = $hwnd.ToInt64(); pid = $pid; title = $builder.ToString(); process = $process.ProcessName; path = $process.Path } | ConvertTo-Json -Compress
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
    let parts: Vec<&str> = value.split(':').collect();
    parts.len() == 6
        && parts
            .iter()
            .all(|part| part.len() == 2 && part.chars().all(|ch| ch.is_ascii_hexdigit()))
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
        accessibility_tree_item, desktop_capabilities, desktop_observe, parse_busctl_string,
        parse_pactl_mute, parse_pactl_volume_percent, parse_xrandr_displays,
        windows_host_volume_script,
    };
    use serde_json::{json, Value};

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
    fn desktop_resolver_normalizes_and_scores_names() {
        let query = super::normalize_match_text("VS Code!");

        assert_eq!(query, "vs code");
        assert_eq!(super::match_score(&query, &["VS Code"]), 100);
        assert!(super::match_score(&query, &["Visual Studio Code"]) > 0);
        assert_eq!(super::match_score(&query, &["Terminal"]), 0);
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
