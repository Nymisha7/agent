use anyhow::Result;
use serde_json::{json, Value};
use std::ffi::OsStr;
use std::fs;
use std::process::Command as ProcessCommand;

mod desktop;
mod inventory;
pub(crate) use desktop::desktop_action;
pub(crate) use desktop::desktop_capabilities;
pub(crate) use desktop::desktop_clipboard_files;
pub(crate) use desktop::desktop_observe;
pub(crate) use desktop::desktop_open_user_file;
pub(crate) use desktop::desktop_pick_file;
pub(crate) use desktop::desktop_resolve;
pub(crate) use desktop::{desktop_clipboard_image_to_file, desktop_clipboard_read_text};
#[cfg(test)]
pub(crate) use desktop::{valid_bluetooth_address, valid_identifier, valid_path_token};
pub(crate) use inventory::connected_devices;
#[cfg(test)]
pub(crate) use inventory::windows_device_category;
use inventory::{bluetooth_info_value, is_wsl_runtime, read_trimmed};

pub(crate) fn system_info() -> Result<Value> {
    let hostname = fs::read_to_string("/etc/hostname")
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|_| String::from("unknown"));
    let uptime_seconds = read_uptime_seconds().unwrap_or(0.0);
    let meminfo = read_meminfo();
    let disk = root_disk_summary();
    let wsl = is_wsl_runtime();
    let runtime = if wsl { "wsl" } else { std::env::consts::OS };

    Ok(json!({
        "ok": true,
        "tool": "system_info",
        "runtime": runtime,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "hostname": hostname,
        "wsl": wsl,
        "cpu_count": std::thread::available_parallelism().map(|value| value.get()).unwrap_or(1),
        "uptime_seconds": uptime_seconds,
        "memory": meminfo,
        "disk": disk,
    }))
}

pub(crate) fn desktop_screenshot() -> Result<Value> {
    let directory = std::env::temp_dir().join("agent-screenshots");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("screenshot-{}.png", unix_millis()));
    let path_text = path.to_string_lossy().to_string();
    let (command, backend): (Vec<String>, &str) = if command_exists("grim") {
        (vec!["grim".to_owned(), path_text.clone()], "grim")
    } else if command_exists("gnome-screenshot") {
        (
            vec![
                "gnome-screenshot".to_owned(),
                "-f".to_owned(),
                path_text.clone(),
            ],
            "gnome-screenshot",
        )
    } else if command_exists("scrot") {
        (vec!["scrot".to_owned(), path_text.clone()], "scrot")
    } else if is_wsl_runtime() && command_exists("powershell.exe") {
        let windows_path = wsl_windows_path(&path)?;
        let quoted_path = windows_path.replace('\'', "''");
        let script = format!(
            "Add-Type -AssemblyName System.Windows.Forms; Add-Type -AssemblyName System.Drawing; \
             $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; \
             $bitmap = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height; \
             $graphics = [System.Drawing.Graphics]::FromImage($bitmap); \
             $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size); \
             $bitmap.Save('{quoted_path}', [System.Drawing.Imaging.ImageFormat]::Png); \
             $graphics.Dispose(); $bitmap.Dispose()"
        );
        (
            vec![
                "powershell.exe".to_owned(),
                "-NoProfile".to_owned(),
                "-NonInteractive".to_owned(),
                "-STA".to_owned(),
                "-Command".to_owned(),
                script,
            ],
            "powershell.exe",
        )
    } else {
        return Ok(json!({
            "ok": false,
            "tool": "desktop_screenshot",
            "reason": "screenshot_backend_unavailable",
            "guidance": "Install grim, gnome-screenshot, or scrot in the desktop runtime.",
        }));
    };
    let output = run_capture_dynamic(&command)?;
    let exists = path.is_file();
    Ok(json!({
        "ok": output.status == 0 && exists,
        "tool": "desktop_screenshot",
        "path": if exists { Some(path_text) } else { None },
        "backend": backend,
        "exit_code": output.status,
        "stderr": output.stderr,
        "verification": if output.status == 0 && exists { "confirmed" } else { "not_confirmed" },
    }))
}

fn wsl_windows_path(path: &std::path::Path) -> Result<String> {
    let path_text = path.to_string_lossy();
    let output = run_capture_dynamic(&["wslpath", "-w", path_text.as_ref()])?;
    if output.status != 0 || output.stdout.trim().is_empty() {
        return Err(anyhow::anyhow!(
            "Could not translate the screenshot path for Windows."
        ));
    }
    Ok(output.stdout.trim().to_string())
}

fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn command_exists(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let candidate = dir.join(name);
                candidate.is_file()
            })
        })
        .unwrap_or(false)
}

pub(crate) fn process_list(limit: usize, sort_by: &str) -> Result<Value> {
    let sort_flag = match sort_by {
        "memory" => "--sort=-%mem",
        _ => "--sort=-%cpu",
    };
    let command = ["ps", "-eo", "pid=,ppid=,comm=,%cpu=,%mem=,stat=", sort_flag];
    let output = run_capture(&command)?;
    let mut items = Vec::new();
    for line in output.stdout.lines() {
        let mut fields = line.split_whitespace();
        let Some(pid) = fields.next() else {
            continue;
        };
        let Some(ppid) = fields.next() else {
            continue;
        };
        let Some(command) = fields.next() else {
            continue;
        };
        let Some(cpu) = fields.next() else {
            continue;
        };
        let Some(memory) = fields.next() else {
            continue;
        };
        let Some(state) = fields.next() else {
            continue;
        };
        items.push(json!({
            "pid": pid.parse::<u32>().ok(),
            "ppid": ppid.parse::<u32>().ok(),
            "command": command,
            "cpu_percent": cpu.parse::<f64>().ok(),
            "memory_percent": memory.parse::<f64>().ok(),
            "state": state,
        }));
        if items.len() >= limit.max(1) {
            break;
        }
    }
    Ok(json!({
        "ok": true,
        "tool": "process_list",
        "sort_by": if sort_by == "memory" { "memory" } else { "cpu" },
        "count": items.len(),
        "processes": items,
    }))
}

pub(crate) fn run_system_command(
    command: &str,
    target: Option<&str>,
    limit: usize,
) -> Result<Value> {
    match command {
        "list_block_devices" => command_json(
            command,
            target,
            [
                "lsblk",
                "-J",
                "-o",
                "NAME,KNAME,TYPE,SIZE,MOUNTPOINTS,MODEL,TRAN",
            ],
        ),
        "list_network_interfaces" => command_json(command, target, ["ip", "-j", "addr"]),
        "list_listening_ports" => {
            let output = run_capture(&["ss", "-ltnp"])?;
            let lines: Vec<&str> = output.stdout.lines().take(limit.max(1)).collect();
            Ok(json!({
                "ok": output.status == 0,
                "tool": "run_system_command",
                "command": command,
                "target": target,
                "exit_code": output.status,
                "lines": lines,
                "stderr": output.stderr,
            }))
        }
        "service_status" => {
            let service = required_target(command, target)?;
            let active = run_capture(&["systemctl", "is-active", service])?;
            let enabled = run_capture(&["systemctl", "is-enabled", service])?;
            let stderr = format!("{}\n{}", active.stderr, enabled.stderr)
                .trim()
                .to_string();
            Ok(json!({
                "ok": true,
                "tool": "run_system_command",
                "command": command,
                "target": service,
                "active": active.stdout.trim(),
                "enabled": enabled.stdout.trim(),
                "active_exit_code": active.status,
                "enabled_exit_code": enabled.status,
                "stderr": stderr,
            }))
        }
        "start_service" | "stop_service" | "restart_service" => {
            let service = required_target(command, target)?;
            let action = match command {
                "start_service" => "start",
                "stop_service" => "stop",
                _ => "restart",
            };
            let output = run_capture(&["systemctl", action, service])?;
            Ok(json!({
                "ok": output.status == 0,
                "tool": "run_system_command",
                "command": command,
                "target": service,
                "exit_code": output.status,
                "stdout": output.stdout,
                "stderr": output.stderr,
            }))
        }
        _ => Ok(json!({
            "ok": false,
            "tool": "run_system_command",
            "command": command,
            "error": format!("Unsupported system command: {}", command),
        })),
    }
}

fn command_json<const N: usize>(
    command: &str,
    target: Option<&str>,
    argv: [&str; N],
) -> Result<Value> {
    let output = run_capture(&argv)?;
    let parsed =
        serde_json::from_str::<Value>(&output.stdout).unwrap_or_else(|_| json!(output.stdout));
    Ok(json!({
        "ok": output.status == 0,
        "tool": "run_system_command",
        "command": command,
        "target": target,
        "exit_code": output.status,
        "data": parsed,
        "stderr": output.stderr,
    }))
}

fn required_target<'a>(command: &str, target: Option<&'a str>) -> Result<&'a str> {
    target.ok_or_else(|| anyhow::anyhow!("{} requires --target", command))
}

#[derive(Debug)]
struct CommandOutput {
    status: i32,
    stdout: String,
    stderr: String,
}

fn run_capture<const N: usize>(argv: &[&str; N]) -> Result<CommandOutput> {
    run_capture_dynamic(argv)
}

fn run_capture_dynamic<T: AsRef<OsStr>>(argv: &[T]) -> Result<CommandOutput> {
    let Some((program, arguments)) = argv.split_first() else {
        return Err(anyhow::anyhow!("Cannot run an empty command"));
    };
    let mut command = ProcessCommand::new(program.as_ref());
    command.args(arguments.iter().map(AsRef::as_ref));
    let output = command.output()?;
    Ok(CommandOutput {
        status: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).trim().to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
    })
}

fn read_uptime_seconds() -> Option<f64> {
    let text = fs::read_to_string("/proc/uptime").ok()?;
    text.split_whitespace().next()?.parse::<f64>().ok()
}

fn read_meminfo() -> Value {
    let mut total_kib = None;
    let mut available_kib = None;
    if let Ok(text) = fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            if line.starts_with("MemTotal:") {
                total_kib = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|value| value.parse::<u64>().ok());
            }
            if line.starts_with("MemAvailable:") {
                available_kib = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|value| value.parse::<u64>().ok());
            }
        }
    }
    json!({
        "total_kib": total_kib,
        "available_kib": available_kib,
    })
}

fn root_disk_summary() -> Value {
    match run_capture(&["df", "-kP", "/"]) {
        Ok(output) => {
            let line = output.stdout.lines().nth(1).unwrap_or("");
            let mut fields = line.split_whitespace();
            if let (
                Some(filesystem),
                Some(size),
                Some(used),
                Some(available),
                Some(used_percent),
                Some(mountpoint),
            ) = (
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
                fields.next(),
            ) {
                json!({
                    "filesystem": filesystem,
                    "size_kib": size.parse::<u64>().ok(),
                    "used_kib": used.parse::<u64>().ok(),
                    "available_kib": available.parse::<u64>().ok(),
                    "used_percent": used_percent,
                    "mountpoint": mountpoint,
                })
            } else {
                json!({})
            }
        }
        Err(_) => json!({}),
    }
}
