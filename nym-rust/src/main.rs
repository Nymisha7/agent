use anyhow::Result;
use clap::{Parser, Subcommand};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use nym_rust::{
    delete_path, edit_file, glob_files, grep_files, read_path, resolve_search_roots,
    resolve_target, search_files, system_search_roots, write_file, DeletePathOptions,
    EditFileOptions, FileSearchOptions, GlobKind, GlobOptions, GrepOptions, ReadLimits,
    ReadPathOptions, ResolveTargetOptions, SearchKind, SearchMode, SearchStrategy, TargetKind,
    WriteFileOptions,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Terminal;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(name = "nym-rust")]
#[command(about = " tools for Nym")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Search(SearchArgs),
    Glob(GlobArgs),
    Grep(GrepArgs),
    InspectTarget(InspectTargetArgs),
    Read(ReadArgs),
    WriteFile(WriteFileArgs),
    EditFile(EditFileArgs),
    DeletePath(DeletePathArgs),
    Locate(LocateArgs),
    SystemInfo(SystemInfoArgs),
    ConnectedDevices(ConnectedDevicesArgs),
    ProcessList(ProcessListArgs),
    RunSystemCommand(RunSystemCommandArgs),
    Tui(TuiArgs),
}

#[derive(Debug, Parser)]
struct SystemInfoArgs {}

#[derive(Debug, Parser)]
struct ConnectedDevicesArgs {
    #[arg(long, default_value = "all")]
    scope: String,
}

#[derive(Debug, Parser)]
struct ProcessListArgs {
    #[arg(long, default_value_t = 20)]
    limit: usize,
    #[arg(long, default_value = "cpu")]
    sort_by: String,
}

#[derive(Debug, Parser)]
struct RunSystemCommandArgs {
    command: String,
    #[arg(long)]
    target: Option<String>,
    #[arg(long, default_value_t = 50)]
    limit: usize,
}

#[derive(Debug, Parser)]
struct InspectTargetArgs {
    target: String,
    #[arg(long)]
    workspace_root: Option<PathBuf>,
    #[arg(long)]
    focus_path: Option<PathBuf>,
    #[arg(long, default_value = "any")]
    kind: String,
    #[arg(long, default_value_t = 120)]
    limit: usize,
    #[arg(long, default_value_t = 1)]
    offset: usize,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    system_fallback: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    contains_fallback: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    fuzzy_fallback: bool,
}

fn parse_target_kind(raw: &str) -> TargetKind {
    match raw {
        "file" => TargetKind::File,
        "directory" | "dir" | "folder" => TargetKind::Directory,
        _ => TargetKind::Any,
    }
}

#[derive(Debug, Parser)]
struct SearchArgs {
    /// Search query.
    query: String,

    /// Search root. Can be passed multiple times.
    #[arg(long = "root")]
    roots: Vec<PathBuf>,

    /// Maximum number of results.
    #[arg(long, default_value_t = 50)]
    limit: usize,

    /// Include hidden files and directories.
    #[arg(long)]
    hidden: bool,

    /// Follow symlinks.
    #[arg(long)]
    follow: bool,

    /// Search resource mode: interactive, balanced, aggressive, background.
    #[arg(long, default_value = "interactive")]
    mode: String,

    /// Match strategy: fuzzy-path, exact-name, contains-name.
    #[arg(long, default_value = "fuzzy-path")]
    strategy: String,

    /// Resource kind: any, file, directory.
    #[arg(long, default_value = "any")]
    kind: String,

    /// Include generated artifacts and dependency folders.
    #[arg(long)]
    include_generated: bool,
}

#[derive(Debug, Parser)]
struct GlobArgs {
    /// Glob pattern to match against file paths.
    pattern: String,

    /// Search root.
    #[arg(long)]
    root: Option<PathBuf>,

    /// Maximum number of results.
    #[arg(long, default_value_t = 100)]
    limit: usize,

    /// Include hidden files and directories.
    #[arg(long)]
    hidden: bool,

    /// Resource kind: any, file, directory.
    #[arg(long, default_value = "any")]
    kind: String,
}

#[derive(Debug, Parser)]
struct GrepArgs {
    /// Text or regex pattern to search for.
    pattern: String,

    /// Search root.
    #[arg(long)]
    root: Option<PathBuf>,

    /// Glob include filter, for example '*.rs' or '**/*.py'.
    #[arg(long)]
    include: Option<String>,

    /// Maximum number of matches.
    #[arg(long, default_value_t = 100)]
    limit: usize,

    /// Treat pattern as literal text instead of regex.
    #[arg(long)]
    literal_text: bool,

    /// Include hidden files and directories.
    #[arg(long)]
    hidden: bool,
}

#[derive(Debug, Parser)]
struct ReadArgs {
    /// File or directory path to read.
    path: PathBuf,

    /// Line number to start reading from.
    #[arg(long, default_value_t = 1)]
    offset: usize,

    /// Maximum number of lines to read.
    #[arg(long)]
    limit: Option<usize>,
}

#[derive(Debug, Parser)]
struct WriteFileArgs {
    /// File path to create or overwrite.
    path: PathBuf,

    /// Workspace root that the path must stay inside.
    #[arg(long)]
    workspace_root: PathBuf,

    /// File contents to write.
    #[arg(long)]
    content: String,

    /// Create parent directories if needed.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    create_dirs: bool,

    /// Overwrite an existing file.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    overwrite: bool,

    /// Preserve the existing line endings when overwriting.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    preserve_line_endings: bool,

    /// Optional expected sha256 hash to guard against concurrent edits.
    #[arg(long)]
    expected_sha256: Option<String>,
}

fn parse_mode(raw: &str) -> SearchMode {
    match raw {
        "interactive" => SearchMode::Interactive,
        "balanced" => SearchMode::Balanced,
        "aggressive" => SearchMode::Aggressive,
        "background" => SearchMode::Background,
        _ => SearchMode::Interactive,
    }
}

#[derive(Debug, Parser)]
struct EditFileArgs {
    /// File path to edit.
    path: PathBuf,

    /// Workspace root that the path must stay inside.
    #[arg(long)]
    workspace_root: PathBuf,

    /// Text to replace.
    #[arg(long)]
    old_text: String,

    /// Replacement text.
    #[arg(long)]
    new_text: String,

    /// Replace all occurrences rather than only the first.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    replace_all: bool,

    /// Optional expected sha256 hash to guard against concurrent edits.
    #[arg(long)]
    expected_sha256: Option<String>,
}

#[derive(Debug, Parser)]
struct DeletePathArgs {
    /// Path to delete.
    path: PathBuf,

    /// Workspace root that the path must stay inside.
    #[arg(long)]
    workspace_root: PathBuf,

    /// Recursively delete directories.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    recursive: bool,
}

#[derive(Debug, Parser)]
struct LocateArgs {
    /// Path/name query.
    query: String,

    /// Maximum number of results.
    #[arg(long, default_value_t = 50)]
    limit: usize,

    /// Include hidden files and directories.
    #[arg(long)]
    hidden: bool,

    /// Match strategy: fuzzy-path, exact-name, contains-name.
    #[arg(long, default_value = "exact-name")]
    strategy: String,

    /// Resource kind: any, file, directory.
    #[arg(long, default_value = "any")]
    kind: String,
}

#[derive(Debug, Parser, Clone)]
struct TuiArgs {
    #[arg(long)]
    python: String,

    #[arg(long)]
    repo_root: PathBuf,

    #[arg(long)]
    session_id: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeResponse {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    snapshot: Option<BridgeSnapshot>,
    #[serde(default)]
    completions: Option<BridgeCompletions>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeSnapshot {
    session: BridgeSession,
    #[serde(default)]
    approvals: Vec<BridgeApproval>,
    messages: Vec<BridgeMessage>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeSession {
    id: String,
    title: String,
    workspace_root: String,
    updated_at: String,
    provider: String,
    model: String,
    mode: String,
    configuration: String,
    cost_usd: f64,
    tokens: BridgeTokens,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeTokens {
    input: i64,
    output: i64,
    reasoning: i64,
    cache_read: i64,
    cache_write: i64,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeMessage {
    role: String,
    content: String,
    created_at: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeApproval {
    id: String,
    #[serde(default)]
    tool: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    requested_path: Option<String>,
    #[serde(default)]
    translated_path: Option<String>,
    #[serde(default)]
    resolved_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct BridgeCompletions {
    title: String,
    entries: Vec<BridgeCompletionEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeCompletionEntry {
    value: String,
    label: String,
    description: String,
    complete_to: String,
    execute: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeStreamFrame {
    kind: String,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    event: Option<BridgeEvent>,
    #[serde(default)]
    snapshot: Option<BridgeSnapshot>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeEvent {
    kind: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
    #[serde(default)]
    summary: Option<String>,
}

#[derive(Debug, Clone)]
struct ActivityLine {
    kind: String,
    text: String,
}

#[derive(Debug)]
enum AppEvent {
    StreamFrame(Result<BridgeStreamFrame>),
}

struct TuiApp {
    snapshot: BridgeSnapshot,
    input: String,
    status: String,
    scroll: u16,
    auto_follow: bool,
    submitting: bool,
    activity: Vec<ActivityLine>,
    palette: BridgeCompletions,
    palette_selected: usize,
    approval_selected: usize,
    current_tool: Option<String>,
    reasoning_text: String,
    streaming_text: String,
    running_prompt: Option<String>,
}

fn parse_strategy(raw: &str) -> SearchStrategy {
    match raw {
        "exact-name" => SearchStrategy::ExactName,
        "contains-name" => SearchStrategy::ContainsName,
        "fuzzy-path" => SearchStrategy::FuzzyPath,
        _ => SearchStrategy::FuzzyPath,
    }
}

fn parse_kind(raw: &str) -> SearchKind {
    match raw {
        "file" => SearchKind::File,
        "directory" => SearchKind::Directory,
        "any" => SearchKind::Any,
        _ => SearchKind::Any,
    }
}

fn parse_glob_kind(raw: &str) -> GlobKind {
    match raw {
        "file" => GlobKind::File,
        "directory" | "dir" | "folder" => GlobKind::Directory,
        "any" => GlobKind::Any,
        _ => GlobKind::Any,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Tui(args) => run_tui(args)?,
        Command::SystemInfo(_args) => {
            println!("{}", serde_json::to_string(&system_info()?)?);
        }
        Command::ConnectedDevices(args) => {
            println!(
                "{}",
                serde_json::to_string(&connected_devices(&args.scope)?)?
            );
        }
        Command::ProcessList(args) => {
            println!(
                "{}",
                serde_json::to_string(&process_list(args.limit, &args.sort_by)?)?
            );
        }
        Command::RunSystemCommand(args) => {
            println!(
                "{}",
                serde_json::to_string(&run_system_command(
                    &args.command,
                    args.target.as_deref(),
                    args.limit,
                )?)?
            );
        }
        Command::Search(args) => {
            let mut options = FileSearchOptions::new(args.query);

            options.roots = resolve_search_roots(args.roots)?;
            options.limit = NonZeroUsize::new(args.limit.max(1)).expect("limit is non-zero");
            options.include_hidden = args.hidden;
            options.follow_links = args.follow;
            options.search_mode = parse_mode(&args.mode);
            options.strategy = parse_strategy(&args.strategy);
            options.kind = parse_kind(&args.kind);
            options.include_generated = args.include_generated;

            let matches = search_files(options)?;
            println!("{}", serde_json::to_string(&matches)?);
        }

        Command::Glob(args) => {
            let root = match args.root {
                Some(root) => root,
                None => std::env::current_dir()?,
            };
            let result = glob_files(GlobOptions {
                pattern: args.pattern,
                root,
                limit: args.limit,
                include_hidden: args.hidden,
                kind: parse_glob_kind(&args.kind),
            })?;

            println!("{}", serde_json::to_string(&result)?);
        }

        Command::Grep(args) => {
            let root = match args.root {
                Some(root) => root,
                None => std::env::current_dir()?,
            };
            let result = grep_files(GrepOptions {
                pattern: args.pattern,
                root,
                include: args.include,
                limit: args.limit,
                literal_text: args.literal_text,
                include_hidden: args.hidden,
            })?;

            println!("{}", serde_json::to_string(&result)?);
        }

        Command::Read(args) => {
            let options = ReadPathOptions {
                path: args.path,
                offset: args.offset,
                limit: args.limit,
                limits: ReadLimits::default(),
            };

            let result = read_path(options)?;
            println!("{}", serde_json::to_string(&result)?);
        }

        Command::WriteFile(args) => {
            let result = write_file(WriteFileOptions {
                path: args.path,
                workspace_root: args.workspace_root,
                content: args.content,
                create_dirs: args.create_dirs,
                overwrite: args.overwrite,
                preserve_line_endings: args.preserve_line_endings,
                expected_sha256: args.expected_sha256,
            })?;
            println!("{}", serde_json::to_string(&result)?);
        }

        Command::EditFile(args) => {
            let result = edit_file(EditFileOptions {
                path: args.path,
                workspace_root: args.workspace_root,
                old_text: args.old_text,
                new_text: args.new_text,
                replace_all: args.replace_all,
                expected_sha256: args.expected_sha256,
            })?;
            println!("{}", serde_json::to_string(&result)?);
        }

        Command::DeletePath(args) => {
            let result = delete_path(DeletePathOptions {
                path: args.path,
                workspace_root: args.workspace_root,
                recursive: args.recursive,
            })?;
            println!("{}", serde_json::to_string(&result)?);
        }

        Command::Locate(args) => {
            let mut options = FileSearchOptions::new(args.query);

            options.roots = system_search_roots()?;
            options.limit = NonZeroUsize::new(args.limit.max(1)).expect("limit is non-zero");
            options.include_hidden = args.hidden;
            options.follow_links = false;
            options.search_mode = SearchMode::Interactive;
            options.strategy = parse_strategy(&args.strategy);
            options.kind = parse_kind(&args.kind);
            options.include_generated = false;

            let matches = search_files(options)?;
            println!("{}", serde_json::to_string(&matches)?);
        }

        Command::InspectTarget(args) => {
            let workspace_root = match args.workspace_root {
                Some(root) => root,
                None => std::env::current_dir()?,
            };

            let resolved = resolve_target(ResolveTargetOptions {
                raw_target: args.target,
                workspace_root,
                focus_path: args.focus_path,
                kind: parse_target_kind(&args.kind),
                limit: args.limit,
                allow_system_fallback: args.system_fallback,
                allow_contains_fallback: args.contains_fallback,
                allow_fuzzy_fallback: args.fuzzy_fallback,
            })?;

            println!("{}", serde_json::to_string(&resolved)?);
        }
    }

    Ok(())
}

fn system_info() -> Result<Value> {
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

fn connected_devices(scope: &str) -> Result<Value> {
    let normalized_scope = match scope {
        "usb" | "storage" | "network" | "input" | "bluetooth" | "all" => scope,
        _ => "all",
    };
    let usb = usb_devices();
    let storage = block_devices();
    let network = network_interfaces();
    let input = input_devices();
    let bluetooth = bluetooth_devices();

    let counts = json!({
        "usb": usb.len(),
        "storage": storage.len(),
        "network": network.len(),
        "input": input.len(),
        "bluetooth": bluetooth.len(),
    });

    let mut payload = json!({
        "ok": true,
        "tool": "connected_devices",
        "scope": normalized_scope,
        "visibility": if is_wsl_runtime() { "wsl_visible" } else { "current_os" },
        "counts": counts,
    });

    let details = match normalized_scope {
        "usb" => json!({ "usb": usb }),
        "storage" => json!({ "storage": storage }),
        "network" => json!({ "network": network }),
        "input" => json!({ "input": input }),
        "bluetooth" => json!({ "bluetooth": bluetooth }),
        _ => json!({
            "usb": usb,
            "storage": storage,
            "network": network,
            "input": input,
            "bluetooth": bluetooth,
        }),
    };

    if let Some(object) = payload.as_object_mut() {
        if let Some(extra) = details.as_object() {
            for (key, value) in extra {
                object.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(payload)
}

fn process_list(limit: usize, sort_by: &str) -> Result<Value> {
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

fn run_system_command(command: &str, target: Option<&str>, limit: usize) -> Result<Value> {
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

fn is_wsl_runtime() -> bool {
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
            items.push(json!({
                "id": entry.file_name().to_string_lossy().to_string(),
                "vendor_id": vendor,
                "product_id": product_id,
                "manufacturer": manufacturer,
                "product": product,
            }));
        }
    }
    items
}

fn block_devices() -> Vec<Value> {
    let mut items = Vec::new();
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
                "name": name,
                "model": read_trimmed(path.join("device/model")),
                "removable": read_trimmed(path.join("removable")).as_deref() == Some("1"),
                "size_bytes": sectors.saturating_mul(512),
            }));
        }
    }
    items
}

fn network_interfaces() -> Vec<Value> {
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/net") {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name == "lo" {
                continue;
            }
            let path = entry.path();
            items.push(json!({
                "name": name,
                "operstate": read_trimmed(path.join("operstate")),
                "mac_address": read_trimmed(path.join("address")),
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
                "name": name,
                "device_name": read_trimmed(path.join("device/name")),
            }));
        }
    }
    items
}

fn bluetooth_devices() -> Vec<Value> {
    let mut items = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/bluetooth") {
        for entry in entries.flatten() {
            items.push(json!({
                "name": entry.file_name().to_string_lossy().to_string(),
            }));
        }
    }
    items
}

fn read_trimmed(path: PathBuf) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn run_tui(args: TuiArgs) -> Result<()> {
    let initial = call_bridge(&args, "snapshot", None)?;
    let snapshot = initial
        .snapshot
        .ok_or_else(|| anyhow::anyhow!("Bridge did not return a snapshot."))?;

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_tui_loop(
        &mut terminal,
        args,
        TuiApp {
            snapshot,
            input: String::new(),
            status: String::from("Ready"),
            scroll: 0,
            auto_follow: true,
            submitting: false,
            activity: Vec::new(),
            palette: BridgeCompletions::default(),
            palette_selected: 0,
            approval_selected: 0,
            current_tool: None,
            reasoning_text: String::new(),
            streaming_text: String::new(),
            running_prompt: None,
        },
    );

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn run_tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    args: TuiArgs,
    mut app: TuiApp,
) -> Result<()> {
    let (tx, rx) = mpsc::channel::<AppEvent>();
    refresh_palette(&args, &mut app);

    loop {
        while let Ok(event) = rx.try_recv() {
            handle_app_event(&mut app, event);
        }

        terminal.draw(|frame| draw_app(frame, &app))?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Esc => break,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                apply_approval_action(&args, &mut app, "approve");
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                apply_approval_action(&args, &mut app, "deny");
            }
            KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if !app.snapshot.approvals.is_empty() {
                    let max_index = app.snapshot.approvals.len().saturating_sub(1);
                    app.approval_selected = (app.approval_selected + 1).min(max_index);
                }
            }
            KeyCode::Char('p') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if !app.snapshot.approvals.is_empty() {
                    app.approval_selected = app.approval_selected.saturating_sub(1);
                }
            }
            KeyCode::Backspace => {
                app.input.pop();
                refresh_palette(&args, &mut app);
            }
            KeyCode::Enter => {
                if !app.palette.entries.is_empty() && app.input.starts_with('/') {
                    let selected = app.palette.entries.get(app.palette_selected).cloned();
                    if let Some(entry) = selected {
                        if entry.execute {
                            app.input = entry.complete_to;
                        } else {
                            app.input = entry.complete_to;
                            refresh_palette(&args, &mut app);
                            continue;
                        }
                    }
                }
                let prompt = app.input.trim().to_string();
                if prompt.is_empty() || app.submitting {
                    continue;
                }
                if matches!(prompt.as_str(), "exit" | "quit" | "/exit" | "/quit" | "/q") {
                    break;
                }
                app.input.clear();
                app.submitting = true;
                app.auto_follow = true;
                app.status = format!("Running: {}", clip_status(&prompt, 72));
                app.running_prompt = Some(prompt.clone());
                app.activity.clear();
                app.reasoning_text.clear();
                app.streaming_text.clear();
                app.current_tool = None;
                app.palette = BridgeCompletions::default();
                app.palette_selected = 0;
                let tx_clone = tx.clone();
                let err_tx = tx.clone();
                let args_clone = args.clone();
                thread::spawn(move || {
                    let result = stream_bridge_submit(&args_clone, &prompt, tx_clone);
                    if let Err(err) = result {
                        let _ = err_tx.send(AppEvent::StreamFrame(Err(err)));
                    }
                });
            }
            KeyCode::Up => {
                if !app.palette.entries.is_empty() && app.input.starts_with('/') {
                    app.palette_selected = app.palette_selected.saturating_sub(1);
                } else {
                    app.auto_follow = false;
                    app.scroll = app.scroll.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if !app.palette.entries.is_empty() && app.input.starts_with('/') {
                    let max_index = app.palette.entries.len().saturating_sub(1);
                    app.palette_selected = (app.palette_selected + 1).min(max_index);
                } else {
                    app.scroll = app.scroll.saturating_add(1);
                }
            }
            KeyCode::PageUp => {
                app.auto_follow = false;
                app.scroll = app.scroll.saturating_sub(10);
            }
            KeyCode::PageDown => {
                app.scroll = app.scroll.saturating_add(10);
            }
            KeyCode::Home => {
                app.scroll = 0;
                app.auto_follow = false;
            }
            KeyCode::End => {
                app.auto_follow = true;
            }
            KeyCode::Tab => {
                if let Some(entry) = app.palette.entries.get(app.palette_selected) {
                    app.input = entry.complete_to.clone();
                    refresh_palette(&args, &mut app);
                }
            }
            KeyCode::Char(ch) => {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    app.input.push(ch);
                    refresh_palette(&args, &mut app);
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn draw_app(frame: &mut ratatui::Frame<'_>, app: &TuiApp) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(1),
            Constraint::Length(3),
        ])
        .split(area);
    let show_sidebar = area.width >= 92;
    let body = if show_sidebar {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(48), Constraint::Length(34)])
            .split(chunks[1])
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(0)])
            .split(chunks[1])
    };

    let session = &app.snapshot.session;
    let header_lines = vec![
        Line::from(vec![
            Span::styled(
                " nym ",
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(
                clip_status(&session.title, area.width.saturating_sub(48) as usize),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {}", clip_status(&session.workspace_root, 42)),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled(" session ", Style::default().fg(Color::DarkGray)),
            Span::styled(session.id.clone(), Style::default().fg(Color::Gray)),
            Span::styled("  model ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", session.provider, session.model),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("  mode ", Style::default().fg(Color::DarkGray)),
            Span::styled(session.mode.clone(), Style::default().fg(Color::Magenta)),
            Span::styled("  updated ", Style::default().fg(Color::DarkGray)),
            Span::styled(session.updated_at.clone(), Style::default().fg(Color::Gray)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(header_lines).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        ),
        chunks[0],
    );

    let transcript_lines = transcript_text(&app.snapshot, app);
    let transcript_height = body[0].height.saturating_sub(2) as usize;
    let max_scroll = transcript_lines
        .lines
        .len()
        .saturating_sub(transcript_height) as u16;
    let scroll = if app.auto_follow {
        max_scroll
    } else {
        app.scroll.min(max_scroll)
    };
    let transcript_title = if app.submitting {
        format!(" {} ", clip_status(&app.status, 42))
    } else {
        String::from(" transcript ")
    };
    let transcript = Paragraph::new(transcript_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(if app.submitting {
                    Color::Cyan
                } else {
                    Color::DarkGray
                }))
                .title(transcript_title),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(transcript, body[0]);

    if show_sidebar {
        draw_sidebar(frame, body[1], app);
    }

    let status = Line::from(vec![
        Span::styled(
            format!(" {} ", if app.submitting { "running" } else { "ready" }),
            Style::default()
                .fg(Color::Black)
                .bg(if app.submitting {
                    Color::Cyan
                } else {
                    Color::Green
                })
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            clip_status(&app.status, 54),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            format!(
                "  tokens {}  reasoning {}  cache {}  ${:.4}",
                format_number(session.tokens.input + session.tokens.output),
                format_number(session.tokens.reasoning),
                format_number(session.tokens.cache_read + session.tokens.cache_write),
                session.cost_usd,
            ),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            format!(
                "  {}",
                if app.snapshot.approvals.is_empty() {
                    "Enter submit  / commands  Tab complete  End follow  Esc exit"
                } else {
                    "Ctrl+A approve  Ctrl+D deny  Ctrl+N/P select"
                }
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(Paragraph::new(status), chunks[2]);

    let input_title = if app.submitting {
        " compose (busy) "
    } else if !app.palette.entries.is_empty() && app.input.starts_with('/') {
        " command "
    } else {
        " compose "
    };
    let input_text = Line::from(vec![
        Span::styled(
            "> ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(app.input.as_str()),
    ]);
    let input = Paragraph::new(input_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(if app.submitting {
                    Color::DarkGray
                } else {
                    Color::Cyan
                }))
                .title(input_title),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(input, chunks[3]);

    if !app.palette.entries.is_empty() && app.input.starts_with('/') {
        let popup_height = (app.palette.entries.len() as u16 + 2).min(11);
        let popup_y = chunks[3].y.saturating_sub(popup_height);
        let popup_area = Rect {
            x: chunks[3].x,
            y: popup_y,
            width: body[0].width.min(chunks[3].width).max(20),
            height: popup_height,
        };
        frame.render_widget(Clear, popup_area);
        let popup = Paragraph::new(palette_text(
            &app.palette,
            app.palette_selected,
            popup_area.width,
        ))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(format!(" {} ", app.palette.title)),
        )
        .wrap(Wrap { trim: false });
        frame.render_widget(popup, popup_area);
    }

    let cursor_x = chunks[3]
        .x
        .saturating_add(3)
        .saturating_add(app.input.chars().count() as u16);
    let cursor_y = chunks[3].y.saturating_add(1);
    frame.set_cursor_position((cursor_x.min(chunks[3].right().saturating_sub(2)), cursor_y));
}

fn draw_sidebar(frame: &mut ratatui::Frame<'_>, area: Rect, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Min(6),
        ])
        .split(area);

    let session = Paragraph::new(session_panel_text(app))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" session "),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(session, chunks[0]);

    let approvals_border = if app.snapshot.approvals.is_empty() {
        Color::DarkGray
    } else {
        Color::Red
    };
    let approvals = Paragraph::new(approval_panel_text(app))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(approvals_border))
                .title(" approvals "),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(approvals, chunks[1]);

    let activity = Paragraph::new(activity_panel_text(app))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(" activity "),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(activity, chunks[2]);
}

fn session_panel_text(app: &TuiApp) -> Text<'static> {
    let session = &app.snapshot.session;
    let lines = vec![
        metric_line("Source", &session.provider, Color::Cyan),
        metric_line("Model", &clip_status(&session.model, 20), Color::White),
        metric_line("Mode", &session.mode, Color::Magenta),
        metric_line(
            "Config",
            &clip_status(&session.configuration, 20),
            Color::Yellow,
        ),
        Line::from(""),
        metric_line(
            "Tokens",
            &format_number(session.tokens.input + session.tokens.output),
            Color::Green,
        ),
        metric_line("Cost", &format!("${:.4}", session.cost_usd), Color::Green),
    ];
    Text::from(lines)
}

fn approval_panel_text(app: &TuiApp) -> Text<'static> {
    if app.snapshot.approvals.is_empty() {
        return Text::from(vec![
            Line::from(vec![Span::styled(
                "No pending requests",
                Style::default().fg(Color::DarkGray),
            )]),
            Line::from(vec![Span::styled(
                "Ctrl+A approve  Ctrl+D deny",
                Style::default().fg(Color::DarkGray),
            )]),
        ]);
    }

    let mut lines = Vec::new();
    for (index, approval) in app.snapshot.approvals.iter().enumerate().take(3) {
        let selected = index == app.approval_selected;
        let path = approval
            .translated_path
            .as_deref()
            .or(approval.resolved_path.as_deref())
            .or(approval.requested_path.as_deref())
            .unwrap_or("target");
        let marker_style = if selected {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        lines.push(Line::from(vec![
            Span::styled(if selected { " > " } else { "   " }, marker_style),
            Span::styled(
                clip_status(&approval.tool, 18),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(clip_status(path, 28), Style::default().fg(Color::Gray)),
        ]));
        if !approval.reason.is_empty() {
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(
                    clip_status(&approval.reason, 28),
                    Style::default().fg(Color::Red),
                ),
            ]));
        }
    }
    lines.push(Line::from(vec![Span::styled(
        "Ctrl+N/P select",
        Style::default().fg(Color::DarkGray),
    )]));
    Text::from(lines)
}

fn activity_panel_text(app: &TuiApp) -> Text<'static> {
    let mut lines = Vec::new();
    if app.activity.is_empty()
        && app.reasoning_text.trim().is_empty()
        && app.streaming_text.trim().is_empty()
        && !app.submitting
    {
        lines.push(Line::from(vec![Span::styled(
            "Idle",
            Style::default().fg(Color::DarkGray),
        )]));
        return Text::from(lines);
    }

    for item in app.activity.iter().rev().take(12).rev() {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{} ", bullet_for_kind(&item.kind)),
                activity_style(&item.kind).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                clip_status(&item.text, 28),
                Style::default().fg(Color::Gray),
            ),
        ]));
    }
    if !app.streaming_text.trim().is_empty() {
        lines.push(Line::from(vec![
            Span::styled(
                ": ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                clip_status(&app.streaming_text, 28),
                Style::default().fg(Color::Yellow),
            ),
        ]));
    } else if !app.reasoning_text.trim().is_empty() {
        lines.push(Line::from(vec![
            Span::styled(
                "~ ",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                clip_status(&app.reasoning_text, 28),
                Style::default().fg(Color::Blue),
            ),
        ]));
    }
    Text::from(lines)
}

fn metric_line(label: &str, value: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{:<7}", label),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(value.to_string(), Style::default().fg(color)),
    ])
}

fn activity_style(kind: &str) -> Style {
    match kind {
        "thinking" => Style::default().fg(Color::Blue),
        "tool" => Style::default().fg(Color::Magenta),
        "guardrail" => Style::default().fg(Color::Red),
        "text" => Style::default().fg(Color::Yellow),
        _ => Style::default().fg(Color::White),
    }
}

fn format_number(value: i64) -> String {
    let raw = value.abs().to_string();
    let mut formatted = String::new();
    for (index, ch) in raw.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            formatted.push(',');
        }
        formatted.push(ch);
    }
    let mut result = formatted.chars().rev().collect::<String>();
    if value < 0 {
        result.insert(0, '-');
    }
    result
}

fn role_label(role: &str) -> (&'static str, Color) {
    match role {
        "user" => ("YOU", Color::Green),
        "assistant" => ("NYM", Color::Yellow),
        "tool" => ("TOOL", Color::Magenta),
        _ => ("SYS", Color::White),
    }
}

fn role_header(role: &str, detail: &str) -> Line<'static> {
    let (label, color) = role_label(role);
    Line::from(vec![
        Span::styled(
            format!(" {:<4} ", label),
            Style::default()
                .fg(Color::Black)
                .bg(color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(detail.to_string(), Style::default().fg(Color::DarkGray)),
    ])
}

fn message_body_line(text: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled("  | ", Style::default().fg(Color::DarkGray)),
        Span::raw(text.to_string()),
    ])
}

fn transcript_text(snapshot: &BridgeSnapshot, app: &TuiApp) -> Text<'static> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for message in &snapshot.messages {
        lines.push(role_header(&message.role, &message.created_at));
        for line in message.content.lines() {
            lines.push(message_body_line(line));
        }
        if message.content.is_empty() {
            lines.push(message_body_line(""));
        }
        lines.push(Line::from(""));
    }
    if let Some(prompt) = &app.running_prompt {
        lines.push(role_header("user", "in progress"));
        lines.push(message_body_line(prompt));
        lines.push(Line::from(""));
    }
    for item in &app.activity {
        lines.push(Line::from(vec![
            Span::styled(
                format!(" {:<9} ", item.kind.to_uppercase()),
                activity_style(&item.kind).add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(item.text.clone(), Style::default().fg(Color::Gray)),
        ]));
    }
    if !app.streaming_text.trim().is_empty() {
        lines.push(role_header("assistant", "drafting"));
        for line in app.streaming_text.lines() {
            lines.push(message_body_line(line));
        }
        lines.push(Line::from(""));
    } else if !app.reasoning_text.trim().is_empty() {
        lines.push(role_header("assistant", "thinking"));
        for line in app.reasoning_text.lines() {
            lines.push(message_body_line(line));
        }
        lines.push(Line::from(""));
    }
    if lines.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "No messages yet. Start with a prompt or type / for commands.",
            Style::default().fg(Color::DarkGray),
        )]));
    }
    Text::from(lines)
}

fn palette_text(palette: &BridgeCompletions, selected: usize, width: u16) -> Text<'static> {
    let mut lines = Vec::new();
    let label_width = if width > 64 { 24 } else { 16 };
    for (index, entry) in palette.entries.iter().enumerate() {
        let selected_style = if index == selected {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let _ = &entry.value;
        lines.push(Line::from(vec![
            Span::styled(
                if index == selected { " > " } else { "   " },
                selected_style,
            ),
            Span::styled(
                format!("{:<label_width$}", clip_status(&entry.label, label_width)),
                Style::default().fg(if index == selected {
                    Color::Cyan
                } else {
                    Color::White
                }),
            ),
            Span::styled(
                clip_status(
                    &entry.description,
                    width.saturating_sub(label_width as u16 + 8) as usize,
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }
    Text::from(lines)
}

fn bullet_for_kind(kind: &str) -> &'static str {
    match kind {
        "thinking" => "~",
        "tool" => "+",
        "guardrail" => "!",
        "text" => ":",
        _ => "-",
    }
}

fn apply_approval_action(args: &TuiArgs, app: &mut TuiApp, action: &str) {
    let Some(approval) = app.snapshot.approvals.get(app.approval_selected) else {
        return;
    };
    let request_id = approval.id.clone();
    let response = call_bridge_with_request_id(args, action, &request_id);
    match response {
        Ok(response) => {
            if let Some(snapshot) = response.snapshot {
                app.snapshot = snapshot;
                let max_index = app.snapshot.approvals.len().saturating_sub(1);
                app.approval_selected = app.approval_selected.min(max_index);
            }
            app.status = if action == "approve" {
                String::from("Approval granted")
            } else {
                String::from("Approval denied")
            };
        }
        Err(err) => {
            app.status = format!("Error: {}", clip_status(&err.to_string(), 96));
        }
    }
}

fn clip_status(value: &str, limit: usize) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.len() <= limit {
        return compact;
    }
    if limit <= 3 {
        return compact.chars().take(limit).collect();
    }
    let mut clipped = compact.chars().take(limit - 3).collect::<String>();
    clipped.push_str("...");
    clipped
}

fn refresh_palette(args: &TuiArgs, app: &mut TuiApp) {
    if !app.input.starts_with('/') {
        app.palette = BridgeCompletions::default();
        app.palette_selected = 0;
        return;
    }
    match call_bridge(args, "complete", Some(&app.input)) {
        Ok(response) => {
            app.palette = response.completions.unwrap_or_default();
            let max_index = app.palette.entries.len().saturating_sub(1);
            app.palette_selected = app.palette_selected.min(max_index);
        }
        Err(err) => {
            app.status = format!("Error: {}", clip_status(&err.to_string(), 96));
            app.palette = BridgeCompletions::default();
            app.palette_selected = 0;
        }
    }
}

fn handle_app_event(app: &mut TuiApp, event: AppEvent) {
    match event {
        AppEvent::StreamFrame(Ok(frame)) => match frame.kind.as_str() {
            "submitted" => {
                if let Some(prompt) = frame.prompt {
                    app.running_prompt = Some(prompt);
                }
                if let Some(snapshot) = frame.snapshot {
                    app.snapshot = snapshot;
                    let max_index = app.snapshot.approvals.len().saturating_sub(1);
                    app.approval_selected = app.approval_selected.min(max_index);
                }
                app.activity.push(ActivityLine {
                    kind: String::from("thinking"),
                    text: String::from("Thinking through the next step"),
                });
            }
            "stream_event" => {
                if let Some(snapshot) = frame.snapshot {
                    app.snapshot = snapshot;
                    let max_index = app.snapshot.approvals.len().saturating_sub(1);
                    app.approval_selected = app.approval_selected.min(max_index);
                }
                if let Some(event) = frame.event {
                    apply_bridge_event(app, event);
                }
            }
            "final" => {
                app.submitting = false;
                app.running_prompt = None;
                app.current_tool = None;
                app.reasoning_text.clear();
                app.streaming_text.clear();
                if let Some(snapshot) = frame.snapshot {
                    app.snapshot = snapshot;
                    let max_index = app.snapshot.approvals.len().saturating_sub(1);
                    app.approval_selected = app.approval_selected.min(max_index);
                }
                app.status =
                    if frame.error.is_some() || !frame.error.as_deref().unwrap_or("").is_empty() {
                        frame
                            .error
                            .as_deref()
                            .map(|text| format!("Error: {}", clip_status(text, 96)))
                            .unwrap_or_else(|| String::from("Ready"))
                    } else {
                        frame
                            .answer
                            .as_deref()
                            .map(|text| clip_status(text, 96))
                            .unwrap_or_else(|| String::from("Ready"))
                    };
                if !matches!(app.activity.last(), Some(ActivityLine { kind, .. }) if kind == "thinking")
                {
                    app.activity.push(ActivityLine {
                        kind: String::from("thinking"),
                        text: String::from("Completed"),
                    });
                }
            }
            _ => {}
        },
        AppEvent::StreamFrame(Err(err)) => {
            app.submitting = false;
            app.running_prompt = None;
            app.current_tool = None;
            app.reasoning_text.clear();
            app.status = format!("Error: {}", clip_status(&err.to_string(), 96));
        }
    }
}

fn apply_bridge_event(app: &mut TuiApp, event: BridgeEvent) {
    match event.kind.as_str() {
        "reasoning_delta" => {
            if let Some(delta) = event.delta {
                app.reasoning_text.push_str(&delta);
            }
        }
        "text_delta" => {
            if let Some(delta) = event.delta {
                if !delta.is_empty() {
                    app.reasoning_text.clear();
                }
                app.streaming_text.push_str(&delta);
            }
        }
        "tool_call_started" => {
            app.current_tool = event.name;
        }
        "tool_call_arguments_done" => {
            let tool = app
                .current_tool
                .take()
                .unwrap_or_else(|| String::from("tool"));
            let args = event.arguments.unwrap_or_default();
            app.activity.push(ActivityLine {
                kind: String::from("tool"),
                text: format!("{}({})", tool, clip_status(&args, 56)),
            });
        }
        "tool_result" | "approval_request" | "approval_decision" => {
            if let Some(summary) = event.summary {
                let kind = if event.kind.starts_with("approval") {
                    "guardrail"
                } else {
                    "tool"
                };
                app.activity.push(ActivityLine {
                    kind: kind.to_string(),
                    text: summary,
                });
            }
        }
        "response_completed" => {}
        _ => {}
    }
}

fn call_bridge(args: &TuiArgs, action: &str, prompt: Option<&str>) -> Result<BridgeResponse> {
    let mut command = ProcessCommand::new(&args.python);
    command
        .arg("-m")
        .arg("nym_agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .current_dir(&args.repo_root);
    if let Some(prompt) = prompt {
        command.arg("--bridge-prompt").arg(prompt);
    }

    let output = command.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if stdout.is_empty() {
        return Err(anyhow::anyhow!(
            "Bridge produced no JSON output. stderr: {}",
            stderr
        ));
    }

    let response: BridgeResponse = serde_json::from_str(&stdout).map_err(|err| {
        anyhow::anyhow!(
            "Could not parse bridge response: {}. stdout: {} stderr: {}",
            err,
            stdout,
            stderr
        )
    })?;

    if !response.ok {
        if let Some(error) = response.error.as_deref() {
            return Err(anyhow::anyhow!("Bridge error: {}", error));
        }
    }

    if !output.status.success() {
        if let Some(error) = response.error.as_deref() {
            return Err(anyhow::anyhow!("Bridge error: {}", error));
        }
        return Err(anyhow::anyhow!(
            "Bridge exited with {} and stderr: {}",
            output.status,
            stderr
        ));
    }

    Ok(response)
}

fn call_bridge_with_request_id(
    args: &TuiArgs,
    action: &str,
    request_id: &str,
) -> Result<BridgeResponse> {
    let mut command = ProcessCommand::new(&args.python);
    command
        .arg("-m")
        .arg("nym_agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .arg("--bridge-request-id")
        .arg(request_id)
        .current_dir(&args.repo_root);

    let output = command.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if stdout.is_empty() {
        return Err(anyhow::anyhow!(
            "Bridge produced no JSON output. stderr: {}",
            stderr
        ));
    }
    let response: BridgeResponse = serde_json::from_str(&stdout).map_err(|err| {
        anyhow::anyhow!(
            "Could not parse bridge response: {}. stdout: {} stderr: {}",
            err,
            stdout,
            stderr
        )
    })?;
    if !response.ok {
        if let Some(error) = response.error.as_deref() {
            return Err(anyhow::anyhow!("Bridge error: {}", error));
        }
    }
    if !output.status.success() {
        if let Some(error) = response.error.as_deref() {
            return Err(anyhow::anyhow!("Bridge error: {}", error));
        }
        return Err(anyhow::anyhow!(
            "Bridge exited with {} and stderr: {}",
            output.status,
            stderr
        ));
    }
    Ok(response)
}

fn stream_bridge_submit(args: &TuiArgs, prompt: &str, tx: mpsc::Sender<AppEvent>) -> Result<()> {
    let mut child = spawn_bridge(args, "stream-submit", Some(prompt))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("Bridge stdout was not captured."))?;
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<BridgeStreamFrame>(&line).map_err(|err| {
            anyhow::anyhow!(
                "Could not parse bridge stream frame: {} line: {}",
                err,
                line
            )
        })?;
        let _ = tx.send(AppEvent::StreamFrame(Ok(parsed)));
    }
    let status = child.wait()?;
    if !status.success() {
        let _ = tx.send(AppEvent::StreamFrame(Err(anyhow::anyhow!(
            "Bridge exited with {}",
            status
        ))));
    }
    Ok(())
}

fn spawn_bridge(args: &TuiArgs, action: &str, prompt: Option<&str>) -> Result<Child> {
    let mut command = ProcessCommand::new(&args.python);
    command
        .arg("-m")
        .arg("nym_agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .current_dir(&args.repo_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(prompt) = prompt {
        command.arg("--bridge-prompt").arg(prompt);
    }
    Ok(command.spawn()?)
}
