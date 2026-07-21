use super::{command_exists, run_capture_dynamic};
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) fn connected_devices(scope: &str) -> Result<Value> {
    let mut categories = BTreeMap::from([
        (String::from("usb"), usb_devices()),
        (String::from("storage"), block_devices()),
        (String::from("network"), network_interfaces()),
        (String::from("input"), input_devices()),
        (String::from("bluetooth"), bluetooth_devices()),
        (String::from("audio"), audio_devices()),
        (String::from("display"), display_devices()),
        (String::from("camera"), camera_devices()),
        (String::from("printer"), printer_devices()),
        (String::from("power"), power_devices()),
    ]);

    let local_source = if is_wsl_runtime() {
        "wsl"
    } else {
        std::env::consts::OS
    };
    for records in categories.values_mut() {
        tag_device_source(records, local_source);
    }

    let windows_host = if is_wsl_runtime() {
        windows_host_devices().ok()
    } else {
        None
    };
    if let Some(host) = windows_host.as_ref() {
        for (category, records) in host {
            categories
                .entry(category.clone())
                .or_default()
                .extend(records.iter().cloned());
        }
    }

    let mut availability = base_device_availability();
    if windows_host.is_some() {
        for category in categories.keys() {
            availability.insert(
                category.clone(),
                json!({
                    "available": true,
                    "status": "available",
                    "source": "windows_pnp",
                }),
            );
        }
    }

    let requested_scope = normalize_category(scope);
    let normalized_scope = if requested_scope == "all" || categories.contains_key(&requested_scope)
    {
        requested_scope
    } else {
        String::from("all")
    };
    let counts = categories
        .iter()
        .map(|(category, records)| (category.clone(), json!(records.len())))
        .collect::<serde_json::Map<_, _>>();
    let category_payload = categories
        .iter()
        .filter(|(category, _)| {
            normalized_scope == "all" || category.as_str() == normalized_scope.as_str()
        })
        .map(|(category, records)| {
            let state = availability.get(category).cloned().unwrap_or_else(|| {
                json!({ "available": true, "status": "available", "source": "runtime_discovery" })
            });
            (
                category.clone(),
                json!({ "state": state, "records": records }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let devices = categories
        .iter()
        .filter(|(category, _)| normalized_scope == "all" || category.as_str() == normalized_scope)
        .flat_map(|(_, records)| records.iter().cloned())
        .collect::<Vec<_>>();

    let payload = json!({
        "ok": true,
        "tool": "connected_devices",
        "schema_version": 3,
        "scope": normalized_scope,
        "checked_at_unix_ms": unix_time_ms(),
        "visibility": if is_wsl_runtime() { "wsl_visible" } else { "current_os" },
        "runtime": if is_wsl_runtime() { "wsl" } else { std::env::consts::OS },
        "windows_host_bridge": if !is_wsl_runtime() { "not_applicable" } else if windows_host.is_some() { "available" } else { "unavailable" },
        "counts": counts,
        "availability": availability,
        "categories": category_payload,
        "devices": devices,
        "limitations": device_inventory_limitations(windows_host.is_some()),
    });

    Ok(payload)
}

fn base_device_availability() -> serde_json::Map<String, Value> {
    [
        ("usb", "/sys/bus/usb/devices", None),
        ("storage", "/sys/block", None),
        ("network", "/sys/class/net", None),
        ("input", "/sys/class/input", None),
        ("bluetooth", "/sys/class/bluetooth", Some("bluetoothctl")),
        ("audio", "/proc/asound", Some("pactl")),
        ("display", "/sys/class/drm", None),
        ("camera", "/sys/class/video4linux", None),
        ("printer", "/run/cups", Some("lpstat")),
        ("power", "/sys/class/power_supply", None),
    ]
    .into_iter()
    .map(|(category, path, command)| (category.to_string(), capability_availability(path, command)))
    .collect()
}

fn normalize_category(value: &str) -> String {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if normalized.is_empty() {
        String::from("all")
    } else {
        normalized
    }
}

fn device_inventory_limitations(windows_host_available: bool) -> Vec<String> {
    let mut limitations = Vec::new();
    if is_wsl_runtime() {
        if windows_host_available {
            limitations.push(String::from(
                "Windows-host Plug-and-Play records report presence, not guaranteed active connection state; WSL-visible records remain separately sourced.",
            ));
        } else {
            limitations.push(String::from(
                "Inventory is limited to devices exposed inside WSL; the Windows-host PowerShell bridge is unavailable.",
            ));
        }
    }
    if !command_exists("ip") {
        limitations.push(String::from(
            "The `ip` utility is unavailable, so network IP addresses may be omitted.",
        ));
    }
    if !command_exists("bluetoothctl") {
        limitations.push(String::from(
            "BlueZ bluetoothctl is unavailable, so paired, connected, and battery state cannot be inspected.",
        ));
    }
    if !command_exists("pactl") {
        limitations.push(String::from(
            "PulseAudio/PipeWire pactl is unavailable, so audio endpoint details may be omitted.",
        ));
    }
    if !command_exists("lpstat") {
        limitations.push(String::from(
            "CUPS lpstat is unavailable, so printer status cannot be inspected.",
        ));
    }
    limitations
}

fn tag_device_source(records: &mut [Value], source: &str) {
    for record in records {
        if let Some(object) = record.as_object_mut() {
            object.insert(String::from("source_runtime"), json!(source));
        }
    }
}

fn windows_host_devices() -> Result<HashMap<String, Vec<Value>>> {
    if !command_exists("powershell.exe") {
        return Err(anyhow::anyhow!("powershell.exe is unavailable"));
    }
    let script = concat!(
        "$ErrorActionPreference='Stop';",
        "Get-PnpDevice -PresentOnly | ",
        "Select-Object Class,FriendlyName,InstanceId,Status | ConvertTo-Json -Compress"
    );
    let output = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        script,
    ])?;
    if output.status != 0 {
        return Err(anyhow::anyhow!(
            "Windows device inventory failed: {}",
            output.stderr
        ));
    }
    if output.stdout.trim().is_empty() {
        return Ok(HashMap::new());
    }
    let parsed = serde_json::from_str::<Value>(&output.stdout)?;
    let records = match parsed {
        Value::Array(items) => items,
        Value::Object(_) => vec![parsed],
        _ => Vec::new(),
    };
    let mut grouped: HashMap<String, Vec<Value>> = HashMap::new();
    for record in records {
        let Some(class) = record.get("Class").and_then(Value::as_str) else {
            continue;
        };
        let category = windows_device_category(class);
        let id = record
            .get("InstanceId")
            .and_then(Value::as_str)
            .unwrap_or("windows-device");
        let name = record
            .get("FriendlyName")
            .and_then(Value::as_str)
            .unwrap_or(class);
        let native_status = record
            .get("Status")
            .and_then(Value::as_str)
            .unwrap_or("Unknown");
        grouped.entry(category.to_string()).or_default().push(json!({
            "id": id,
            "category": category,
            "status": if native_status.eq_ignore_ascii_case("OK") { "present" } else { "attention" },
            "name": name,
            "native_class": class,
            "native_status": native_status,
            "source_runtime": "windows_host",
        }));
    }
    Ok(grouped)
}

pub(crate) fn windows_device_category(class: &str) -> &'static str {
    match class.to_ascii_lowercase().as_str() {
        "usb" => "usb",
        "bluetooth" => "bluetooth",
        "diskdrive" | "cdrom" | "volume" => "storage",
        "net" => "network",
        "audioendpoint" | "media" => "audio",
        "monitor" | "display" => "display",
        "camera" | "image" => "camera",
        "printer" => "printer",
        "keyboard" | "mouse" | "hidclass" => "input",
        "battery" => "power",
        _ => "other",
    }
}

fn unix_time_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn capability_availability(path: &str, command: Option<&str>) -> Value {
    let path_available = std::path::Path::new(path).exists();
    let command_available = command.map(command_exists).unwrap_or(false);
    let available = path_available || command_available;
    json!({
        "available": available,
        "status": if available { "available" } else if cfg!(target_os = "linux") { "unavailable" } else { "unsupported_platform" },
        "source": if command_available { command } else if path_available { Some(path) } else { None },
    })
}

pub(super) fn is_wsl_runtime() -> bool {
    fs::read_to_string("/proc/version")
        .map(|text| text.to_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

fn usb_devices() -> Vec<Value> {
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/bus/usb/devices") {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.join("idVendor").exists() {
                continue;
            }
            let vendor = read_trimmed(path.join("idVendor"));
            let product_id = read_trimmed(path.join("idProduct"));
            let manufacturer = read_trimmed(path.join("manufacturer"));
            let product = read_trimmed(path.join("product"));
            let authorized = read_trimmed(path.join("authorized")).as_deref() != Some("0");
            items.push(json!({
                "id": entry.file_name().to_string_lossy().to_string(),
                "category": "usb",
                "status": if authorized { "connected" } else { "disabled" },
                "vendor_id": vendor,
                "product_id": product_id,
                "manufacturer": manufacturer,
                "product": product,
                "bus_number": read_trimmed(path.join("busnum")),
                "device_number": read_trimmed(path.join("devnum")),
                "speed_mbps": read_trimmed(path.join("speed")).and_then(|value| value.parse::<f64>().ok()),
                "authorized": authorized,
            }));
        }
    }
    items
}

fn block_devices() -> Vec<Value> {
    let mut items = Vec::new();
    let mounts = mounted_block_devices();
    if let Ok(entries) = fs::read_dir("/sys/block") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("loop") || name.starts_with("ram") {
                continue;
            }
            let path = entry.path();
            let sectors = read_trimmed(path.join("size"))
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            items.push(json!({
                "id": format!("/dev/{name}"),
                "category": "storage",
                "status": if mounts.contains_key(&name) { "mounted" } else { "available" },
                "name": name,
                "model": read_trimmed(path.join("device/model")),
                "removable": read_trimmed(path.join("removable")).as_deref() == Some("1"),
                "size_bytes": sectors.saturating_mul(512),
                "mount_points": mounts.get(&name).cloned().unwrap_or_default(),
            }));
        }
    }
    items
}

fn network_interfaces() -> Vec<Value> {
    let mut items = Vec::new();
    let addresses = network_addresses();
    if let Ok(entries) = fs::read_dir("/sys/class/net") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "lo" {
                continue;
            }
            let path = entry.path();
            let operstate =
                read_trimmed(path.join("operstate")).unwrap_or_else(|| String::from("unknown"));
            let carrier = read_trimmed(path.join("carrier")).as_deref() == Some("1");
            let interface_type = if path.join("wireless").exists() {
                "wifi"
            } else if name.starts_with("wl") {
                "wifi"
            } else if name.starts_with("en") || name.starts_with("eth") {
                "ethernet"
            } else {
                "other"
            };
            items.push(json!({
                "id": name,
                "category": "network",
                "status": if operstate == "up" || carrier { "connected" } else { "disconnected" },
                "name": name,
                "interface_type": interface_type,
                "operstate": operstate,
                "carrier": carrier,
                "mac_address": read_trimmed(path.join("address")),
                "mtu": read_trimmed(path.join("mtu")).and_then(|value| value.parse::<u64>().ok()),
                "addresses": addresses.get(&name).cloned().unwrap_or_default(),
                "ssid": if interface_type == "wifi" { wifi_ssid(&name) } else { None },
            }));
        }
    }
    items
}

fn input_devices() -> Vec<Value> {
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/input") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.starts_with("event") {
                continue;
            }
            let path = entry.path();
            items.push(json!({
                "id": name,
                "category": "input",
                "status": "connected",
                "name": name,
                "device_name": read_trimmed(path.join("device/name")),
            }));
        }
    }
    items
}

fn bluetooth_devices() -> Vec<Value> {
    if command_exists("bluetoothctl") {
        let mut items = Vec::new();
        if let Ok(output) = run_capture_dynamic(&["bluetoothctl", "devices"]) {
            for line in output.stdout.lines() {
                let mut parts = line.splitn(3, ' ');
                if parts.next() != Some("Device") {
                    continue;
                }
                let Some(address) = parts.next() else {
                    continue;
                };
                let name = parts.next().unwrap_or("Unknown Bluetooth device");
                let info = run_capture_dynamic(&["bluetoothctl", "info", address]).ok();
                let connected = info
                    .as_ref()
                    .and_then(|value| bluetooth_info_value(&value.stdout, "Connected"))
                    .as_deref()
                    == Some("yes");
                let paired = info
                    .as_ref()
                    .and_then(|value| bluetooth_info_value(&value.stdout, "Paired"))
                    .as_deref()
                    == Some("yes");
                let trusted = info
                    .as_ref()
                    .and_then(|value| bluetooth_info_value(&value.stdout, "Trusted"))
                    .as_deref()
                    == Some("yes");
                let battery_percent = info
                    .as_ref()
                    .and_then(|value| bluetooth_battery_percent(&value.stdout));
                items.push(json!({
                    "id": address,
                    "category": "bluetooth",
                    "status": if connected { "connected" } else if paired { "paired" } else { "discovered" },
                    "name": name,
                    "address": address,
                    "connected": connected,
                    "paired": paired,
                    "trusted": trusted,
                    "battery_percent": battery_percent,
                }));
            }
        }
        return items;
    }

    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/bluetooth") {
        for entry in entries.flatten() {
            items.push(json!({
                "id": entry.file_name().to_string_lossy().to_string(),
                "category": "bluetooth",
                "status": "adapter_visible",
                "name": entry.file_name().to_string_lossy().to_string(),
            }));
        }
    }
    items
}

fn audio_devices() -> Vec<Value> {
    if command_exists("pactl") {
        if let Ok(output) = run_capture_dynamic(&["pactl", "list", "short", "sinks"]) {
            return output
                .stdout
                .lines()
                .filter_map(|line| {
                    let fields: Vec<&str> = line.split('\t').collect();
                    if fields.len() < 2 {
                        return None;
                    }
                    Some(json!({
                        "id": fields[1],
                        "category": "audio",
                        "kind": "output",
                        "status": if fields.get(4).copied() == Some("SUSPENDED") { "idle" } else { "available" },
                        "name": fields[1],
                        "driver": fields.get(2),
                        "sample_spec": fields.get(3),
                    }))
                })
                .collect();
        }
    }

    let cards = fs::read_to_string("/proc/asound/cards").unwrap_or_default();
    cards
        .lines()
        .filter(|line| {
            line.trim_start()
                .chars()
                .next()
                .is_some_and(|ch| ch.is_ascii_digit())
        })
        .map(|line| {
            json!({
                "id": line.split_whitespace().next(),
                "category": "audio",
                "kind": "card",
                "status": "available",
                "name": line.trim(),
            })
        })
        .collect()
}

fn display_devices() -> Vec<Value> {
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/drm") {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.join("status").exists() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let status =
                read_trimmed(path.join("status")).unwrap_or_else(|| String::from("unknown"));
            let modes = fs::read_to_string(path.join("modes"))
                .unwrap_or_default()
                .lines()
                .map(str::to_string)
                .collect::<Vec<_>>();
            items.push(json!({
                "id": name,
                "category": "display",
                "status": status,
                "name": name,
                "active_mode": modes.first(),
                "modes": modes,
                "enabled": read_trimmed(path.join("enabled")),
            }));
        }
    }
    items
}

fn camera_devices() -> Vec<Value> {
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/video4linux") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            items.push(json!({
                "id": format!("/dev/{name}"),
                "category": "camera",
                "status": "available",
                "name": read_trimmed(entry.path().join("name")).unwrap_or_else(|| name.clone()),
                "device": format!("/dev/{name}"),
            }));
        }
    }
    items
}

fn printer_devices() -> Vec<Value> {
    if !command_exists("lpstat") {
        return Vec::new();
    }
    run_capture_dynamic(&["lpstat", "-p"])
        .map(|output| {
            output
                .stdout
                .lines()
                .filter_map(|line| {
                    let fields: Vec<&str> = line.split_whitespace().collect();
                    if fields.first().copied() != Some("printer") || fields.len() < 3 {
                        return None;
                    }
                    let name = fields[1];
                    Some(json!({
                        "id": name,
                        "category": "printer",
                        "status": if line.contains("disabled") { "disabled" } else { "available" },
                        "name": name,
                        "description": line,
                    }))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn power_devices() -> Vec<Value> {
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/power_supply") {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let kind = read_trimmed(path.join("type")).unwrap_or_else(|| String::from("unknown"));
            items.push(json!({
                "id": name,
                "category": "power",
                "kind": kind,
                "status": read_trimmed(path.join("status")).unwrap_or_else(|| String::from("available")),
                "name": read_trimmed(path.join("model_name")).unwrap_or_else(|| name.clone()),
                "capacity_percent": read_trimmed(path.join("capacity")).and_then(|value| value.parse::<u64>().ok()),
                "online": read_trimmed(path.join("online")).map(|value| value == "1"),
            }));
        }
    }
    items
}

fn mounted_block_devices() -> HashMap<String, Vec<String>> {
    let mut mounts: HashMap<String, Vec<String>> = HashMap::new();
    let text = fs::read_to_string("/proc/self/mountinfo").unwrap_or_default();
    for line in text.lines() {
        let Some((before, after)) = line.split_once(" - ") else {
            continue;
        };
        let fields: Vec<&str> = before.split_whitespace().collect();
        let after_fields: Vec<&str> = after.split_whitespace().collect();
        if fields.len() < 5 || after_fields.len() < 2 {
            continue;
        }
        let source = after_fields[1];
        let Some(device) = source.strip_prefix("/dev/") else {
            continue;
        };
        let base = device
            .trim_end_matches(|ch: char| ch.is_ascii_digit())
            .trim_end_matches('p');
        mounts
            .entry(base.to_string())
            .or_default()
            .push(fields[4].replace("\\040", " "));
    }
    mounts
}

fn network_addresses() -> HashMap<String, Vec<String>> {
    let mut addresses: HashMap<String, Vec<String>> = HashMap::new();
    let Ok(output) = run_capture_dynamic(&["ip", "-j", "addr"]) else {
        return addresses;
    };
    let Ok(Value::Array(interfaces)) = serde_json::from_str::<Value>(&output.stdout) else {
        return addresses;
    };
    for interface in interfaces {
        let Some(name) = interface.get("ifname").and_then(Value::as_str) else {
            continue;
        };
        let values = interface
            .get("addr_info")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|entry| {
                let local = entry.get("local")?.as_str()?;
                let prefix = entry.get("prefixlen").and_then(Value::as_u64)?;
                Some(format!("{local}/{prefix}"))
            })
            .collect::<Vec<_>>();
        addresses.insert(name.to_string(), values);
    }
    addresses
}

fn wifi_ssid(interface: &str) -> Option<String> {
    if command_exists("iwgetid") {
        return run_capture_dynamic(&["iwgetid", interface, "--raw"])
            .ok()
            .filter(|output| output.status == 0)
            .map(|output| output.stdout)
            .filter(|value| !value.is_empty());
    }
    None
}

pub(super) fn bluetooth_info_value(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (name, value) = line.trim().split_once(':')?;
        (name == key).then(|| value.trim().to_string())
    })
}

fn bluetooth_battery_percent(text: &str) -> Option<u64> {
    let raw = bluetooth_info_value(text, "Battery Percentage")?;
    raw.split_whitespace().find_map(|part| {
        part.trim_matches(|ch: char| !ch.is_ascii_digit())
            .parse::<u64>()
            .ok()
    })
}

pub(super) fn read_trimmed(path: PathBuf) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}
