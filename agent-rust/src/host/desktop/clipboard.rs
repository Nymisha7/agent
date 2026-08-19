use super::*;

pub(super) fn clipboard_backend() -> &'static str {
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

pub(super) fn select_clipboard_backend(
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

pub(super) fn clipboard_metadata() -> Result<Value> {
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
        let converted = run_capture_dynamic(&["wslpath", "-w", path])?;
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
        run_capture_dynamic(&["zenity", "--file-selection", "--title=Add photos or files"])?
    } else if command_exists("kdialog") {
        run_capture_dynamic(&[
            "kdialog",
            "--title",
            "Add photos or files",
            "--getopenfilename",
        ])?
    } else if is_wsl_runtime() && windows_host_powershell_available() {
        let script = "Add-Type -AssemblyName System.Windows.Forms; $dialog = New-Object System.Windows.Forms.OpenFileDialog; $dialog.Title = 'Add photos or files'; $dialog.Filter = 'All files (*.*)|*.*|Images (*.png;*.jpg;*.jpeg;*.webp;*.gif)|*.png;*.jpg;*.jpeg;*.webp;*.gif'; $dialog.Multiselect = $false; if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { [Console]::Out.Write($dialog.FileName) }";
        let selected = run_capture_dynamic(&[
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
        let converted = run_capture_dynamic(&["wslpath", "-u", selected.stdout.trim()])?;
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

pub(super) fn windows_host_clipboard_image_to_file() -> Result<Value> {
    let directory = std::env::temp_dir().join("agent-clipboard");
    fs::create_dir_all(&directory)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let path = directory.join(format!("clipboard-{timestamp}.png"));
    let windows_path = run_capture_dynamic(&["wslpath", "-w", path.to_string_lossy().as_ref()])?;
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
    let output = run_capture_dynamic(&[
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

pub(super) fn clipboard_write_action(action: &str, value: Option<&str>) -> Result<Value> {
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

pub(super) fn clipboard_types(backend: &str) -> Result<Value> {
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

pub(super) fn clipboard_text(backend: &str) -> Result<CommandOutput> {
    match backend {
        "wl-clipboard" => run_capture_dynamic(&["wl-paste", "--no-newline", "--type", "text"]),
        "xclip" => run_capture_dynamic(&["xclip", "-selection", "clipboard", "-o"]),
        "xsel" => run_capture_dynamic(&["xsel", "--clipboard", "--output"]),
        "windows_host_powershell" => run_capture_dynamic(&[
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-Clipboard -Raw",
        ]),
        _ => Ok(CommandOutput {
            status: -1,
            stdout: String::new(),
            stderr: String::from("Clipboard backend unavailable."),
        }),
    }
}

pub(super) fn clipboard_write_backend(backend: &str, text: &str) -> Result<CommandOutput> {
    match backend {
        "wl-clipboard" => run_capture_with_stdin(&["wl-copy"], text),
        "xclip" => run_capture_with_stdin(&["xclip", "-selection", "clipboard"], text),
        "xsel" => run_capture_with_stdin(&["xsel", "--clipboard", "--input"], text),
        "windows_host_powershell" => run_capture_with_stdin(
            &[
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Set-Clipboard -Value ([Console]::In.ReadToEnd())",
            ],
            text,
        ),
        _ => Ok(CommandOutput {
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
        "wl-clipboard" => run_capture_with_stdin(&["wl-copy", "--type", mime], &payload)?,
        "xclip" => run_capture_with_stdin(
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

pub(super) fn file_clipboard_backend() -> &'static str {
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
pub(super) fn file_uri(path: &Path) -> Result<String> {
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
pub(super) fn file_uri(_path: &Path) -> Result<String> {
    Err(anyhow::anyhow!(
        "file clipboard is not supported on this platform"
    ))
}
