use anyhow::Result;
use serde_json::{json, Value};
use std::fs;
use std::process::Command as ProcessCommand;

mod desktop;
mod inventory;
pub(crate) use desktop::desktop_action;
pub(crate) use desktop::desktop_capabilities;
pub(crate) use desktop::desktop_observe;
pub(crate) use desktop::desktop_resolve;
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
    let runtime = if is_wsl_runtime() {
        "wsl"
    } else {
        std::env::consts::OS
    };

    Ok(json!({
        "ok": true,
        "tool": "system_info",
        "runtime": runtime,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "hostname": hostname,
        "wsl": is_wsl_runtime(),
        "cpu_count": std::thread::available_parallelism().map(|value| value.get()).unwrap_or(1),
        "uptime_seconds": uptime_seconds,
        "memory": meminfo,
        "disk": disk,
    }))
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
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 6 {
            continue;
        }
        items.push(json!({
            "pid": fields[0].parse::<u32>().ok(),
            "ppid": fields[1].parse::<u32>().ok(),
            "command": fields[2],
            "cpu_percent": fields[3].parse::<f64>().ok(),
            "memory_percent": fields[4].parse::<f64>().ok(),
            "state": fields[5],
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

fn run_capture_dynamic(argv: &[&str]) -> Result<CommandOutput> {
    if argv.is_empty() {
        return Err(anyhow::anyhow!("Cannot run an empty command"));
    }
    let mut command = ProcessCommand::new(argv[0]);
    command.args(&argv[1..]);
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
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() >= 6 {
                json!({
                    "filesystem": fields[0],
                    "size_kib": fields[1].parse::<u64>().ok(),
                    "used_kib": fields[2].parse::<u64>().ok(),
                    "available_kib": fields[3].parse::<u64>().ok(),
                    "used_percent": fields[4],
                    "mountpoint": fields[5],
                })
            } else {
                json!({})
            }
        }
        Err(_) => json!({}),
    }
}
