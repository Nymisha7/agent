use super::system::run_capture_dynamic;
use std::{env, fs, path::PathBuf};

pub(super) fn command_exists(name: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let candidate = dir.join(name);
                candidate.is_file()
            })
        })
        .unwrap_or(false)
}

pub(super) fn is_wsl_runtime() -> bool {
    fs::read_to_string("/proc/version")
        .map(|text| text.to_lowercase().contains("microsoft"))
        .unwrap_or(false)
}

pub(super) fn user_downloads_directory() -> Option<PathBuf> {
    if let Some(path) = wsl_host_downloads_directory() {
        return Some(path);
    }

    let home = env::var_os("HOME").map(PathBuf::from)?;
    let config_home = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".config"));
    fs::read_to_string(config_home.join("user-dirs.dirs"))
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
        })
        .or_else(|| Some(home.join("Downloads")))
}

fn wsl_host_downloads_directory() -> Option<PathBuf> {
    if !is_wsl_runtime() || !command_exists("powershell.exe") || !command_exists("wslpath") {
        return None;
    }
    let script = concat!(
        "$folder=(New-Object -ComObject Shell.Application).NameSpace('shell:Downloads');",
        "if ($null -ne $folder) {",
        "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new();",
        "$folder.Self.Path}"
    );
    let windows = run_capture_dynamic(&[
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-STA",
        "-Command",
        script,
    ])
    .ok()?;
    if windows.status != 0 || windows.stdout.is_empty() {
        return None;
    }
    let linux = run_capture_dynamic(&["wslpath", "-u", windows.stdout.as_str()]).ok()?;
    (linux.status == 0 && !linux.stdout.is_empty()).then(|| PathBuf::from(linux.stdout))
}
