use std::fs;

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
