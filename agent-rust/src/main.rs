use anyhow::Result;
mod host;

use agent_rust::{
    delete_path, edit_file, glob_files, grep_files, read_path, resolve_search_roots,
    resolve_target, search_files, system_search_roots, write_file, DeletePathOptions,
    EditFileOptions, FileSearchOptions, GlobKind, GlobOptions, GrepOptions, ReadLimits,
    ReadPathOptions, ResolveTargetOptions, SearchKind, SearchMode, SearchStrategy, TargetKind,
    WriteFileOptions,
};
use clap::{Parser, Subcommand};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use host::{
    connected_devices, desktop_action, desktop_capabilities, desktop_clipboard_files,
    desktop_observe, desktop_resolve, process_list, run_system_command, system_info,
};
#[cfg(test)]
use host::{valid_bluetooth_address, valid_identifier, valid_path_token, windows_device_category};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Terminal;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::io::{self, BufRead, BufReader, Read};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader as AsyncBufReader};
use tokio::sync::{mpsc as async_mpsc, Semaphore};

#[derive(Debug, Parser)]
#[command(name = "agent-rust")]
#[command(about = " tools for Agent")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve(ServeArgs),
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
    DesktopCapabilities(DesktopCapabilitiesArgs),
    DesktopObserve(DesktopObserveArgs),
    DesktopResolve(DesktopResolveArgs),
    ProcessList(ProcessListArgs),
    RunSystemCommand(RunSystemCommandArgs),
    DesktopAction(DesktopActionArgs),
    DesktopClipboardFiles(DesktopClipboardFilesArgs),
    Tui(TuiArgs),
}

#[derive(Debug, Parser)]
struct ServeArgs {}

#[derive(Debug, Deserialize)]
struct WorkerRequest {
    id: Option<Value>,
    args: Vec<String>,
}

#[derive(Debug, Serialize)]
struct WorkerResponse {
    id: Option<Value>,
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Parser)]
struct SystemInfoArgs {}

#[derive(Debug, Parser)]
struct ConnectedDevicesArgs {
    #[arg(long, default_value = "all")]
    scope: String,
}

#[derive(Debug, Parser)]
struct DesktopCapabilitiesArgs {}

#[derive(Debug, Parser)]
struct DesktopObserveArgs {
    #[arg(long, default_value = "all")]
    scope: String,
    #[arg(long, default_value_t = 50)]
    limit: usize,
}

#[derive(Debug, Parser)]
struct DesktopResolveArgs {
    query: String,
    #[arg(long, default_value = "any")]
    kind: String,
    #[arg(long, default_value_t = 10)]
    limit: usize,
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
struct DesktopActionArgs {
    action: String,
    #[arg(long)]
    target: Option<String>,
    #[arg(long)]
    value: Option<String>,
    #[arg(long, hide = true)]
    backend_bus: Option<String>,
    #[arg(long, hide = true)]
    backend_path: Option<String>,
}

#[derive(Debug, Parser)]
struct DesktopClipboardFilesArgs {
    #[arg(long = "path", required = true)]
    paths: Vec<PathBuf>,
    #[arg(long, default_value = "copy")]
    operation: String,
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

    /// Include ignored artifacts and dependency folders.
    #[arg(long)]
    include_generated: bool,

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

    /// API keys entered in the masked TUI prompt. They are forwarded to bridge
    /// children, which save them in the user's private Agent credential store.
    #[arg(skip)]
    api_keys: Arc<Mutex<HashMap<String, String>>>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeResponse {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    snapshot: Option<BridgeSnapshot>,
    #[serde(default)]
    completions: Option<BridgeCompletions>,
    #[serde(default)]
    gateway: Option<BridgeGatewaySnapshot>,
    #[serde(default)]
    command_result: Option<BridgeCommandResult>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeCommandResult {
    code: BridgeCommandCode,
    #[serde(default)]
    setup_required: bool,
    #[serde(default)]
    error: bool,
    #[serde(default)]
    secret_provider: Option<String>,
    #[serde(default)]
    next_command: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BridgeCommandCode {
    Ok,
    ApiKeyRequired,
    CredentialsRequired,
    ModelNotInstalled,
    RuntimeUnavailable,
    RuntimeNotInstalled,
    InstallConfirmationRequired,
    InstallFailed,
    InstallUnverified,
    InstallNotReady,
    ManualSetupRequired,
    Unavailable,
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeGatewaySnapshot {
    generated_at: String,
    overview: BridgeGatewayOverview,
    #[serde(default)]
    routes: Vec<BridgeGatewayRoute>,
    #[serde(default)]
    bindings: Vec<BridgeGatewayBinding>,
    #[serde(default)]
    channels: Vec<BridgeGatewayChannel>,
    #[serde(default)]
    sessions: Vec<BridgeGatewaySession>,
    #[serde(default)]
    methods: Vec<BridgeGatewayMethod>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeGatewayOverview {
    control_plane: String,
    state: String,
    started_at: String,
    session_store: String,
    default_agent: String,
    default_scope: String,
    bindings: usize,
    #[serde(default)]
    channels: Vec<String>,
    method_count: usize,
    #[serde(default)]
    config_sources: Vec<String>,
    execution_model: String,
    active_session: String,
    active_agent: String,
    #[serde(default)]
    active_route: Option<String>,
    workspace_root: String,
    tool_policy: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeGatewayRoute {
    route_key: String,
    session_id: String,
    agent_id: String,
    scope: String,
    channel: String,
    account_id: String,
    #[serde(default)]
    peer_kind: Option<String>,
    #[serde(default)]
    peer_id: Option<String>,
    #[serde(default)]
    sender_id: Option<String>,
    #[serde(default)]
    guild_id: Option<String>,
    #[serde(default)]
    team_id: Option<String>,
    updated_at: String,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeGatewayBinding {
    agent_id: String,
    channel: String,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    account_id: Option<String>,
    #[serde(default)]
    peer_kind: Option<String>,
    #[serde(default)]
    peer_id: Option<String>,
    #[serde(default)]
    guild_id: Option<String>,
    #[serde(default)]
    team_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeGatewayChannel {
    channel: String,
    account_id: String,
    state: String,
    generation: usize,
    consecutive_failures: usize,
    #[serde(default)]
    last_error: Option<String>,
    #[serde(default)]
    last_heartbeat: Option<String>,
    #[serde(default)]
    retry_at: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeGatewaySession {
    id: String,
    title: String,
    workspace_root: String,
    agent_id: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    updated_at: String,
    #[serde(default)]
    last_prompt: Option<String>,
    routes: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeGatewayMethod {
    name: String,
    owner: String,
    #[serde(default)]
    scopes: Vec<String>,
    requires_ready: bool,
    control_write: bool,
}

#[derive(Debug, Clone)]
struct GatewayViewState {
    snapshot: BridgeGatewaySnapshot,
    tab: usize,
    scroll: u16,
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
    provider: String,
    model: String,
    mode: String,
    #[serde(default = "provider_controlled_reasoning")]
    reasoning_effort: String,
    configuration: String,
    #[serde(default = "ready_configuration_state")]
    configuration_state: String,
    cost_usd: f64,
    tokens: BridgeTokens,
}

fn provider_controlled_reasoning() -> String {
    String::from("provider controlled")
}

fn ready_configuration_state() -> String {
    String::from("ready")
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
    display_path: Option<String>,
    #[serde(default)]
    translated_path: Option<String>,
    #[serde(default)]
    resolved_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct BridgeCompletions {
    title: String,
    #[serde(default)]
    selected_index: Option<usize>,
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
    #[serde(default)]
    command_result: Option<BridgeCommandResult>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct BridgeEvent {
    kind: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    run_id: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    total: Option<usize>,
    #[serde(default)]
    completed: Option<usize>,
    #[serde(default)]
    failed: Option<usize>,
    #[serde(default)]
    work_file: Option<String>,
    #[serde(default)]
    owned_paths: Vec<String>,
    #[serde(default)]
    changed_count: Option<usize>,
    #[serde(default)]
    tasks: Vec<BridgeSubagentTask>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeSubagentTask {
    id: String,
    task: String,
    #[serde(default)]
    owns: Vec<String>,
}

#[derive(Debug, Clone)]
struct ActivityLine {
    kind: String,
    text: String,
}

#[derive(Debug, Clone)]
struct SubagentTaskState {
    id: String,
    description: String,
    status: String,
    summary: String,
    owned_paths: Vec<String>,
    changed_count: usize,
}

#[derive(Debug, Clone)]
struct SubagentRunState {
    run_id: String,
    total: usize,
    completed: usize,
    failed: usize,
    status: String,
    work_file: Option<String>,
    tasks: Vec<SubagentTaskState>,
}

#[derive(Debug, Clone)]
struct UiNotice {
    title: String,
    text: String,
    error: bool,
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
    cancel_requested: bool,
    cancel_signal: Arc<AtomicBool>,
    active_bridge: Arc<Mutex<Option<Child>>>,
    queued_prompts: VecDeque<String>,
    activity: Vec<ActivityLine>,
    subagent_run: Option<SubagentRunState>,
    palette: BridgeCompletions,
    palette_selected: usize,
    approval_selected: usize,
    current_tool: Option<String>,
    reasoning_text: String,
    streaming_text: String,
    running_prompt: Option<String>,
    secret_provider: Option<String>,
    secret_input: String,
    notices: Vec<UiNotice>,
    setup_required: bool,
    gateway_view: Option<GatewayViewState>,
}

const GATEWAY_TABS: [&str; 6] = [
    "Overview", "Routes", "Bindings", "Channels", "Sessions", "Methods",
];

const PALETTE_PAGE_SIZE: usize = 12;

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

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Serve(_args) => run_worker().await?,
        Command::Tui(args) => run_tui(args)?,
        command => {
            println!("{}", serde_json::to_string(&run_command(command)?)?);
        }
    }

    Ok(())
}

async fn run_worker() -> Result<()> {
    let mut lines = AsyncBufReader::new(tokio::io::stdin()).lines();
    let concurrency = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);
    let permits = Arc::new(Semaphore::new(concurrency));
    let (response_tx, mut response_rx) = async_mpsc::channel::<WorkerResponse>(concurrency * 2);
    let writer = tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(response) = response_rx.recv().await {
            let mut line = serde_json::to_vec(&response)?;
            line.push(b'\n');
            stdout.write_all(&line).await?;
            stdout.flush().await?;
        }
        Result::<()>::Ok(())
    });

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<WorkerRequest>(&line) {
            Ok(request) => request,
            Err(error) => {
                response_tx
                    .send(WorkerResponse {
                        id: None,
                        ok: false,
                        result: None,
                        error: Some(error.to_string()),
                    })
                    .await
                    .map_err(|_| anyhow::anyhow!("worker response channel closed"))?;
                continue;
            }
        };
        let response_tx = response_tx.clone();
        let permit = Arc::clone(&permits).acquire_owned().await?;
        tokio::spawn(async move {
            let request_id = request.id.clone();
            let response =
                match tokio::task::spawn_blocking(move || run_worker_request(request)).await {
                    Ok(response) => response,
                    Err(error) => WorkerResponse {
                        id: request_id,
                        ok: false,
                        result: None,
                        error: Some(error.to_string()),
                    },
                };
            let _permit = permit;
            let _ = response_tx.send(response).await;
        });
    }

    drop(response_tx);
    writer.await??;
    Ok(())
}

#[cfg(test)]
mod worker_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_request_runs_on_tokio_blocking_pool() {
        let response = tokio::task::spawn_blocking(|| {
            run_worker_request(WorkerRequest {
                id: Some(json!(7)),
                args: vec![String::from("system-info")],
            })
        })
        .await
        .expect("blocking worker task");

        assert!(response.ok);
        assert_eq!(response.id, Some(json!(7)));
        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }
}

fn run_worker_request(request: WorkerRequest) -> WorkerResponse {
    let mut argv = Vec::with_capacity(request.args.len() + 1);
    argv.push("agent-rust".to_string());
    argv.extend(request.args);

    let result = Cli::try_parse_from(argv)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .and_then(|cli| match cli.command {
            Command::Serve(_) | Command::Tui(_) => Err(anyhow::anyhow!(
                "command is not supported by the JSON worker"
            )),
            command => run_command(command),
        });

    match result {
        Ok(value) => WorkerResponse {
            id: request.id,
            ok: true,
            result: Some(value),
            error: None,
        },
        Err(error) => WorkerResponse {
            id: request.id,
            ok: false,
            result: None,
            error: Some(error.to_string()),
        },
    }
}

fn run_command(command: Command) -> Result<Value> {
    match command {
        Command::Serve(_) | Command::Tui(_) => Err(anyhow::anyhow!(
            "command is not supported by JSON command dispatch"
        )),
        Command::SystemInfo(_args) => Ok(serde_json::to_value(system_info()?)?),
        Command::ConnectedDevices(args) => {
            Ok(serde_json::to_value(connected_devices(&args.scope)?)?)
        }
        Command::DesktopCapabilities(_args) => Ok(serde_json::to_value(desktop_capabilities()?)?),
        Command::DesktopObserve(args) => Ok(serde_json::to_value(desktop_observe(
            &args.scope,
            args.limit,
        )?)?),
        Command::DesktopResolve(args) => Ok(serde_json::to_value(desktop_resolve(
            &args.query,
            &args.kind,
            args.limit,
        )?)?),
        Command::ProcessList(args) => Ok(serde_json::to_value(process_list(
            args.limit,
            &args.sort_by,
        )?)?),
        Command::RunSystemCommand(args) => Ok(serde_json::to_value(run_system_command(
            &args.command,
            args.target.as_deref(),
            args.limit,
        )?)?),
        Command::DesktopAction(args) => Ok(serde_json::to_value(desktop_action(
            &args.action,
            args.target.as_deref(),
            args.value.as_deref(),
            args.backend_bus.as_deref(),
            args.backend_path.as_deref(),
        )?)?),
        Command::DesktopClipboardFiles(args) => Ok(serde_json::to_value(desktop_clipboard_files(
            &args.paths,
            &args.operation,
        )?)?),
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

            Ok(serde_json::to_value(search_files(options)?)?)
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
                include_generated: args.include_generated,
                kind: parse_glob_kind(&args.kind),
            })?;

            Ok(serde_json::to_value(result)?)
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

            Ok(serde_json::to_value(result)?)
        }

        Command::Read(args) => {
            let options = ReadPathOptions {
                path: args.path,
                offset: args.offset,
                limit: args.limit,
                limits: ReadLimits::default(),
            };

            Ok(serde_json::to_value(read_path(options)?)?)
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
            Ok(serde_json::to_value(result)?)
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
            Ok(serde_json::to_value(result)?)
        }

        Command::DeletePath(args) => {
            let result = delete_path(DeletePathOptions {
                path: args.path,
                workspace_root: args.workspace_root,
                recursive: args.recursive,
            })?;
            Ok(serde_json::to_value(result)?)
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

            Ok(serde_json::to_value(search_files(options)?)?)
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

            Ok(serde_json::to_value(resolved)?)
        }
    }
}

fn run_tui(args: TuiArgs) -> Result<()> {
    let initial = call_bridge(&args, "snapshot", None)?;
    let snapshot = initial
        .snapshot
        .ok_or_else(|| anyhow::anyhow!("Bridge did not return a snapshot."))?;
    let initial_needs_setup = snapshot.session.configuration_state != "ready";
    let initial_status = if initial_needs_setup {
        format!(
            "{} needs setup",
            provider_display_name(&snapshot.session.provider)
        )
    } else {
        String::from("Ready")
    };
    let initial_notices = if initial_needs_setup {
        vec![provider_setup_notice(
            &snapshot.session.provider,
            &snapshot.session.configuration,
            &snapshot.session.configuration_state,
        )]
    } else {
        Vec::new()
    };

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = run_tui_loop(
        &mut terminal,
        args,
        TuiApp {
            snapshot,
            input: String::new(),
            status: initial_status,
            scroll: 0,
            auto_follow: true,
            submitting: false,
            cancel_requested: false,
            cancel_signal: Arc::new(AtomicBool::new(false)),
            active_bridge: Arc::new(Mutex::new(None)),
            queued_prompts: VecDeque::new(),
            activity: Vec::new(),
            subagent_run: None,
            palette: BridgeCompletions::default(),
            palette_selected: 0,
            approval_selected: 0,
            current_tool: None,
            reasoning_text: String::new(),
            streaming_text: String::new(),
            running_prompt: None,
            secret_provider: None,
            secret_input: String::new(),
            notices: initial_notices,
            setup_required: initial_needs_setup,
            gateway_view: None,
        },
    );

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        DisableMouseCapture,
        LeaveAlternateScreen
    )?;
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

        if !app.submitting
            && !bridge_process_active(&app)
            && app.secret_provider.is_none()
            && app.snapshot.approvals.is_empty()
        {
            if let Some(prompt) = app.queued_prompts.pop_front() {
                start_prompt_submission(&args, &tx, &mut app, prompt);
            }
        }

        terminal.draw(|frame| draw_app(frame, &app))?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }

        let key = match event::read()? {
            Event::Key(key) => key,
            Event::Mouse(mouse) => {
                match mouse.kind {
                    MouseEventKind::ScrollUp => scroll_active_view(&mut app, false),
                    MouseEventKind::ScrollDown => scroll_active_view(&mut app, true),
                    _ => {}
                }
                continue;
            }
            _ => continue,
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        if app.gateway_view.is_some() && app.snapshot.approvals.is_empty() {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    app.gateway_view = None;
                    app.status = String::from("Ready");
                }
                KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
                    if let Some(view) = app.gateway_view.as_mut() {
                        view.tab = (view.tab + 1) % GATEWAY_TABS.len();
                        view.scroll = 0;
                    }
                }
                KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => {
                    if let Some(view) = app.gateway_view.as_mut() {
                        view.tab = (view.tab + GATEWAY_TABS.len() - 1) % GATEWAY_TABS.len();
                        view.scroll = 0;
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if let Some(view) = app.gateway_view.as_mut() {
                        view.scroll = view.scroll.saturating_add(1);
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if let Some(view) = app.gateway_view.as_mut() {
                        view.scroll = view.scroll.saturating_sub(1);
                    }
                }
                KeyCode::PageDown => {
                    if let Some(view) = app.gateway_view.as_mut() {
                        view.scroll = view.scroll.saturating_add(10);
                    }
                }
                KeyCode::PageUp => {
                    if let Some(view) = app.gateway_view.as_mut() {
                        view.scroll = view.scroll.saturating_sub(10);
                    }
                }
                KeyCode::Home => {
                    if let Some(view) = app.gateway_view.as_mut() {
                        view.scroll = 0;
                    }
                }
                KeyCode::Char('r') => open_gateway_view(&args, &mut app),
                _ => {}
            }
            continue;
        }

        match key.code {
            KeyCode::Esc => {
                if app.secret_provider.is_some() {
                    app.secret_provider = None;
                    app.secret_input.clear();
                    app.status = String::from("API key entry cancelled");
                    continue;
                }
                if !app.snapshot.approvals.is_empty() {
                    apply_approval_action(&args, &mut app, "deny");
                    continue;
                }
                if palette_is_open(&app) || !app.input.is_empty() {
                    app.input.clear();
                    app.palette = BridgeCompletions::default();
                    app.palette_selected = 0;
                    continue;
                }
                if app.submitting {
                    request_active_stop(&mut app);
                    continue;
                }
                break;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                if app.submitting {
                    request_active_stop(&mut app);
                } else {
                    break;
                }
            }
            KeyCode::Char('a') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                apply_approval_action(&args, &mut app, "approve");
            }
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                apply_approval_action(&args, &mut app, "deny");
            }
            KeyCode::Char('y' | 'Y')
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !app.snapshot.approvals.is_empty() =>
            {
                apply_approval_action(&args, &mut app, "approve");
            }
            KeyCode::Char('n' | 'N')
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !app.snapshot.approvals.is_empty() =>
            {
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
                if app.secret_provider.is_some() {
                    app.secret_input.pop();
                } else {
                    app.input.pop();
                    refresh_palette(&args, &mut app);
                }
            }
            KeyCode::Enter => {
                if let Some(provider) = app.secret_provider.take() {
                    let api_key = app.secret_input.trim().to_string();
                    app.secret_input.clear();
                    if api_key.is_empty() {
                        app.status = String::from("API key entry cancelled");
                        continue;
                    }
                    let Some(env_name) = provider_api_key_env(&provider) else {
                        app.status = format!("No API-key setup is available for {provider}");
                        continue;
                    };
                    if let Ok(mut api_keys) = args.api_keys.lock() {
                        api_keys.insert(env_name.to_string(), api_key);
                    } else {
                        app.status = String::from("Could not hold the API key in memory");
                        continue;
                    }

                    let prompt = format!("/apikey {provider}");
                    app.submitting = true;
                    app.cancel_signal.store(false, Ordering::SeqCst);
                    app.status = format!("Configuring {}", provider_display_name(&provider));
                    app.running_prompt = None;
                    app.activity.clear();
                    app.subagent_run = None;
                    app.reasoning_text.clear();
                    app.streaming_text.clear();
                    app.current_tool = None;
                    let tx_clone = tx.clone();
                    let err_tx = tx.clone();
                    let args_clone = args.clone();
                    let active_bridge = Arc::clone(&app.active_bridge);
                    let cancel_signal = Arc::clone(&app.cancel_signal);
                    thread::spawn(move || {
                        let result = stream_bridge_submit(
                            &args_clone,
                            &prompt,
                            tx_clone,
                            active_bridge,
                            cancel_signal,
                        );
                        if let Err(err) = result {
                            let _ = err_tx.send(AppEvent::StreamFrame(Err(err)));
                        }
                    });
                    continue;
                }
                if !app.snapshot.approvals.is_empty() {
                    apply_approval_action(&args, &mut app, "approve");
                    continue;
                }
                if !app.palette.entries.is_empty() && app.input.starts_with('/') {
                    app.palette_selected =
                        closest_selectable_palette_index(&app.palette, app.palette_selected)
                            .unwrap_or(0);
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
                if prompt.is_empty() {
                    continue;
                }
                if app.submitting || bridge_process_active(&app) {
                    app.input.clear();
                    let queued_count = enqueue_prompt(&mut app.queued_prompts, prompt);
                    app.palette = BridgeCompletions::default();
                    app.palette_selected = 0;
                    app.status = format!("Queued {} prompt(s)", queued_count);
                    continue;
                }
                if matches!(prompt.as_str(), "/exit" | "/quit" | "/q") {
                    break;
                }
                if prompt == "/gateway" {
                    app.input.clear();
                    app.palette = BridgeCompletions::default();
                    app.palette_selected = 0;
                    open_gateway_view(&args, &mut app);
                    continue;
                }
                start_prompt_submission(&args, &tx, &mut app, prompt);
            }
            KeyCode::Up => {
                if palette_is_open(&app) {
                    app.palette_selected =
                        previous_palette_index(&app.palette, app.palette_selected);
                } else {
                    app.auto_follow = false;
                    app.scroll = app.scroll.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if palette_is_open(&app) {
                    app.palette_selected = next_palette_index(&app.palette, app.palette_selected);
                } else {
                    app.scroll = app.scroll.saturating_add(1);
                }
            }
            KeyCode::PageUp => {
                if palette_is_open(&app) {
                    app.palette_selected = move_palette_index(
                        &app.palette,
                        app.palette_selected,
                        -(PALETTE_PAGE_SIZE as isize),
                    );
                } else {
                    app.auto_follow = false;
                    app.scroll = app.scroll.saturating_sub(10);
                }
            }
            KeyCode::PageDown => {
                if palette_is_open(&app) {
                    app.palette_selected = move_palette_index(
                        &app.palette,
                        app.palette_selected,
                        PALETTE_PAGE_SIZE as isize,
                    );
                } else {
                    app.scroll = app.scroll.saturating_add(10);
                }
            }
            KeyCode::Home => {
                if palette_is_open(&app) {
                    app.palette_selected =
                        first_selectable_palette_index(&app.palette).unwrap_or(0);
                } else {
                    app.scroll = 0;
                    app.auto_follow = false;
                }
            }
            KeyCode::End => {
                if palette_is_open(&app) {
                    app.palette_selected = last_selectable_palette_index(&app.palette).unwrap_or(0);
                } else {
                    app.auto_follow = true;
                }
            }
            KeyCode::Tab => {
                if app.secret_provider.is_some() {
                    continue;
                }
                app.palette_selected =
                    closest_selectable_palette_index(&app.palette, app.palette_selected)
                        .unwrap_or(0);
                if let Some(entry) = app.palette.entries.get(app.palette_selected) {
                    app.input = entry.complete_to.clone();
                    refresh_palette(&args, &mut app);
                }
            }
            KeyCode::Char(ch) => {
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT {
                    if app.secret_provider.is_some() {
                        app.secret_input.push(ch);
                    } else {
                        app.input.push(ch);
                        refresh_palette(&args, &mut app);
                    }
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn open_gateway_view(args: &TuiArgs, app: &mut TuiApp) {
    match call_bridge(args, "gateway", None) {
        Ok(response) => {
            if let Some(snapshot) = response.gateway {
                let tab = app.gateway_view.as_ref().map(|view| view.tab).unwrap_or(0);
                app.gateway_view = Some(GatewayViewState {
                    snapshot,
                    tab,
                    scroll: 0,
                });
                app.status = String::from("Gateway control plane");
            } else {
                app.status = String::from("Gateway bridge returned no control snapshot");
            }
        }
        Err(err) => {
            app.status = format!("Gateway unavailable: {}", clip_status(&err.to_string(), 72));
        }
    }
}

fn enqueue_prompt(queue: &mut VecDeque<String>, prompt: String) -> usize {
    queue.push_back(prompt);
    queue.len()
}

fn request_active_stop(app: &mut TuiApp) {
    app.cancel_requested = true;
    app.cancel_signal.store(true, Ordering::SeqCst);
    app.status = String::from("Stopping current task...");
    if let Ok(mut active) = app.active_bridge.lock() {
        if let Some(child) = active.as_mut() {
            let _ = interrupt_process_tree(child);
        }
    }
}

#[cfg(unix)]
fn interrupt_process_tree(child: &mut Child) -> io::Result<()> {
    let result = unsafe { libc::kill(-(child.id() as i32), libc::SIGINT) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(windows)]
fn interrupt_process_tree(child: &mut Child) -> io::Result<()> {
    let status = ProcessCommand::new("taskkill")
        .args(["/PID", &child.id().to_string(), "/T", "/F"])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        child.kill()
    }
}

fn bridge_process_active(app: &TuiApp) -> bool {
    app.active_bridge
        .lock()
        .map(|active| active.is_some())
        .unwrap_or(false)
}

fn start_prompt_submission(
    args: &TuiArgs,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
    prompt: String,
) {
    app.input.clear();
    app.submitting = true;
    app.cancel_requested = false;
    app.cancel_signal.store(false, Ordering::SeqCst);
    app.auto_follow = true;
    app.status = format!("Running: {}", clip_status(&prompt, 72));
    app.running_prompt = Some(prompt.clone());
    app.activity.clear();
    app.subagent_run = None;
    app.reasoning_text.clear();
    app.streaming_text.clear();
    app.current_tool = None;
    app.palette = BridgeCompletions::default();
    app.palette_selected = 0;
    let tx_clone = tx.clone();
    let err_tx = tx.clone();
    let args_clone = args.clone();
    let active_bridge = Arc::clone(&app.active_bridge);
    let cancel_signal = Arc::clone(&app.cancel_signal);
    thread::spawn(move || {
        let result =
            stream_bridge_submit(&args_clone, &prompt, tx_clone, active_bridge, cancel_signal);
        if let Err(err) = result {
            let _ = err_tx.send(AppEvent::StreamFrame(Err(err)));
        }
    });
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
                " AGENT ",
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
        ]),
        Line::from(vec![
            Span::styled(" workspace ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                clip_status(&session.workspace_root, 34),
                Style::default().fg(Color::Gray),
            ),
            Span::styled("  session ", Style::default().fg(Color::DarkGray)),
            Span::styled(session.id.clone(), Style::default().fg(Color::Gray)),
            Span::styled("  model ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", session.provider, session.model),
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("  mode ", Style::default().fg(Color::DarkGray)),
            Span::styled(session.mode.clone(), Style::default().fg(Color::Magenta)),
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

    let transcript_lines = transcript_text(&app.snapshot, app, !show_sidebar);
    let max_scroll = transcript_max_scroll(
        &transcript_lines,
        body[0].width.saturating_sub(2),
        body[0].height.saturating_sub(2),
    );
    let scroll = if app.auto_follow {
        max_scroll
    } else {
        app.scroll.min(max_scroll)
    };
    let transcript_title = if app.submitting {
        format!(" {} ", clip_status(&app.status, 42))
    } else {
        String::from(" conversation ")
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

    let (state_label, state_color) = ui_state_badge(app);
    let status = Line::from(vec![
        Span::styled(
            format!(" {state_label} "),
            Style::default()
                .fg(Color::Black)
                .bg(state_color)
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
                footer_help_text(
                    app.secret_provider.is_some(),
                    !app.snapshot.approvals.is_empty(),
                    app.submitting,
                )
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(Paragraph::new(status), chunks[2]);

    let input_title = if let Some(provider) = app.secret_provider.as_deref() {
        format!(
            " {} API key (kept in memory) ",
            provider_display_name(provider)
        )
    } else if install_command_is_confirmed(&app.input) {
        String::from(" confirm local install · Enter starts · Esc cancels ")
    } else if app.submitting {
        String::from(" message (agent working · Enter queues) ")
    } else if !app.palette.entries.is_empty() && app.input.starts_with('/') {
        String::from(" command ")
    } else {
        String::from(" message ")
    };
    let visible_input = if app.secret_provider.is_some() {
        "•".repeat(app.secret_input.chars().count())
    } else {
        app.input.clone()
    };
    let input_text = Line::from(vec![
        Span::styled(
            "> ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(visible_input.as_str()),
    ]);
    let input = Paragraph::new(input_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(if app.submitting {
                    Color::DarkGray
                } else if app.secret_provider.is_some() {
                    Color::Yellow
                } else {
                    Color::Cyan
                }))
                .title(input_title),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(input, chunks[3]);

    if palette_is_open(app) {
        let max_popup_height = chunks[3].y.saturating_sub(area.y).clamp(3, 20);
        let popup_height = (app.palette.entries.len() as u16 + 2).min(max_popup_height);
        let visible_count = popup_height.saturating_sub(2) as usize;
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
            visible_count,
            popup_area.width,
        ))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(format!(
                    " {} ",
                    palette_title(&app.palette, app.palette_selected, visible_count)
                )),
        )
        .wrap(Wrap { trim: false });
        frame.render_widget(popup, popup_area);
    }

    if let Some(view) = app.gateway_view.as_ref() {
        draw_gateway_dialog(frame, area, view);
    }

    if !app.snapshot.approvals.is_empty() {
        draw_approval_dialog(frame, area, app);
    }

    let cursor_x = chunks[3]
        .x
        .saturating_add(3)
        .saturating_add(visible_input.chars().count() as u16);
    let cursor_y = chunks[3].y.saturating_add(1);
    if app.snapshot.approvals.is_empty() && app.gateway_view.is_none() {
        frame.set_cursor_position((cursor_x.min(chunks[3].right().saturating_sub(2)), cursor_y));
    }
}

fn footer_help_text(auth_active: bool, approval_active: bool, submitting: bool) -> &'static str {
    if auth_active {
        "Enter save key  Esc cancel  input hidden"
    } else if approval_active {
        "Enter/Y approve  N/Esc deny  Ctrl+N/P select"
    } else if submitting {
        "Esc/Ctrl+C stop  Enter queue  End follow"
    } else {
        "Enter send  / commands  Tab complete  End follow  Esc close"
    }
}

fn draw_approval_dialog(frame: &mut ratatui::Frame<'_>, area: Rect, app: &TuiApp) {
    let Some(approval) = app.snapshot.approvals.get(app.approval_selected) else {
        return;
    };
    let width = area.width.saturating_sub(4).min(72).max(1);
    let height = area.height.saturating_sub(4).min(11).max(1);
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    };
    let target = approval_target_text(approval, "requested action");
    let lines = vec![
        Line::from(vec![
            Span::styled("Tool    ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                approval.tool.clone(),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Action  ", Style::default().fg(Color::DarkGray)),
            Span::styled(target.to_string(), Style::default().fg(Color::White)),
        ]),
        Line::from(""),
        Line::from(vec![Span::styled(
            "Enter or Y  Approve",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::from(vec![Span::styled(
            "N or Esc    Deny",
            Style::default().fg(Color::Red),
        )]),
        Line::from(vec![Span::styled(
            "The agent remains paused until you decide.",
            Style::default().fg(Color::Gray),
        )]),
    ];
    frame.render_widget(Clear, popup_area);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow))
                    .title(" Approval required "),
            )
            .wrap(Wrap { trim: false }),
        popup_area,
    );
}

fn draw_gateway_dialog(frame: &mut ratatui::Frame<'_>, area: Rect, view: &GatewayViewState) {
    let width = area.width.saturating_sub(4).min(112).max(1);
    let height = area.height.saturating_sub(2).min(36).max(1);
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan))
        .title(" Agent Gateway ");
    let inner = block.inner(popup_area);
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .split(inner);

    frame.render_widget(Clear, popup_area);
    frame.render_widget(block, popup_area);
    frame.render_widget(Paragraph::new(gateway_tabs_line(view.tab)), sections[0]);
    frame.render_widget(
        Paragraph::new(gateway_tab_text(view))
            .wrap(Wrap { trim: false })
            .scroll((view.scroll, 0)),
        sections[1],
    );
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("Tab/←/→", Style::default().fg(Color::Cyan)),
            Span::styled(" switch  ", Style::default().fg(Color::DarkGray)),
            Span::styled("↑/↓ PgUp/PgDn", Style::default().fg(Color::Cyan)),
            Span::styled(" scroll  ", Style::default().fg(Color::DarkGray)),
            Span::styled("R", Style::default().fg(Color::Cyan)),
            Span::styled(" refresh  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc/Q", Style::default().fg(Color::Cyan)),
            Span::styled(" close", Style::default().fg(Color::DarkGray)),
        ])),
        sections[2],
    );
}

fn gateway_tabs_line(selected: usize) -> Line<'static> {
    let mut spans = Vec::new();
    for (index, label) in GATEWAY_TABS.iter().enumerate() {
        let style = if index == selected {
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        spans.push(Span::styled(format!(" {label} "), style));
        spans.push(Span::raw(" "));
    }
    Line::from(spans)
}

fn gateway_tab_text(view: &GatewayViewState) -> Text<'static> {
    match view.tab {
        1 => gateway_routes_text(&view.snapshot.routes),
        2 => gateway_bindings_text(
            &view.snapshot.bindings,
            &view.snapshot.overview.default_scope,
        ),
        3 => gateway_channels_text(&view.snapshot.channels),
        4 => gateway_sessions_text(&view.snapshot.sessions),
        5 => gateway_methods_text(&view.snapshot.methods),
        _ => gateway_overview_text(&view.snapshot),
    }
}

fn gateway_overview_text(snapshot: &BridgeGatewaySnapshot) -> Text<'static> {
    let overview = &snapshot.overview;
    let route = overview
        .active_route
        .as_deref()
        .unwrap_or("direct CLI session");
    let config = if overview.config_sources.is_empty() {
        String::from("built-in defaults")
    } else {
        overview.config_sources.join(", ")
    };
    Text::from(vec![
        gateway_field("State", &overview.state, Color::Green),
        gateway_field("Runtime", &overview.control_plane, Color::Cyan),
        gateway_field("Workspace", &overview.workspace_root, Color::White),
        gateway_field("Session", &overview.active_session, Color::Yellow),
        gateway_field("Route", route, Color::Magenta),
        gateway_field("Agent", &overview.active_agent, Color::Yellow),
        gateway_field("Default agent", &overview.default_agent, Color::Gray),
        gateway_field("Default scope", &overview.default_scope, Color::Gray),
        gateway_field("Tool policy", &overview.tool_policy, Color::White),
        gateway_field("Session store", &overview.session_store, Color::Gray),
        gateway_field("Bindings", &overview.bindings.to_string(), Color::Gray),
        gateway_field("Channels", &overview.channels.join(", "), Color::Gray),
        gateway_field(
            "RPC methods",
            &overview.method_count.to_string(),
            Color::Gray,
        ),
        gateway_field("Config", &config, Color::Gray),
        Line::from(""),
        gateway_field("Execution", &overview.execution_model, Color::White),
        gateway_field("Started", &overview.started_at, Color::DarkGray),
        gateway_field("Snapshot", &snapshot.generated_at, Color::DarkGray),
    ])
}

fn gateway_routes_text(routes: &[BridgeGatewayRoute]) -> Text<'static> {
    if routes.is_empty() {
        return gateway_empty_text("No durable channel routes yet.");
    }
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Durable routes",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {} total", routes.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ])];
    for route in routes {
        let peer = gateway_route_identity(route);
        lines.extend([
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    format!("{}/{}", route.channel, route.account_id),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {} · {}", route.agent_id, route.scope),
                    Style::default().fg(Color::Gray),
                ),
            ]),
            gateway_field("Identity", &peer, Color::White),
            gateway_field("Session", &route.session_id, Color::Cyan),
            gateway_field("Route key", &route.route_key, Color::DarkGray),
            gateway_field("Updated", &route.updated_at, Color::DarkGray),
        ]);
    }
    Text::from(lines)
}

fn gateway_bindings_text(bindings: &[BridgeGatewayBinding], default_scope: &str) -> Text<'static> {
    if bindings.is_empty() {
        return gateway_empty_text(
            "No explicit bindings. Inbound messages use the default agent and session scope.",
        );
    }
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Routing bindings",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {} configured", bindings.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ])];
    for (index, binding) in bindings.iter().enumerate() {
        let account = binding.account_id.as_deref().unwrap_or("default");
        let scope = binding.scope.as_deref().unwrap_or(default_scope);
        let matcher = gateway_binding_matcher(binding);
        lines.extend([
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    format!("{}. {}", index + 1, binding.channel),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(format!(" / {account}"), Style::default().fg(Color::Gray)),
            ]),
            gateway_field("Agent", &binding.agent_id, Color::Cyan),
            gateway_field("Scope", scope, Color::Magenta),
            gateway_field("Match", &matcher, Color::White),
        ]);
    }
    Text::from(lines)
}

fn gateway_channels_text(channels: &[BridgeGatewayChannel]) -> Text<'static> {
    if channels.is_empty() {
        return gateway_empty_text("No channel adapters are registered.");
    }
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Channel accounts",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {} registered", channels.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ])];
    for channel in channels {
        let state_color = match channel.state.as_str() {
            "running" => Color::Green,
            "backoff" | "starting" => Color::Yellow,
            "crash_loop" => Color::Red,
            _ => Color::Gray,
        };
        lines.extend([
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    format!("{}/{}", channel.channel, channel.account_id),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {}", channel.state),
                    Style::default().fg(state_color),
                ),
            ]),
            gateway_field("Generation", &channel.generation.to_string(), Color::Gray),
            gateway_field(
                "Failures",
                &channel.consecutive_failures.to_string(),
                Color::Gray,
            ),
            gateway_field(
                "Heartbeat",
                channel.last_heartbeat.as_deref().unwrap_or("not running"),
                Color::DarkGray,
            ),
        ]);
        if let Some(retry_at) = channel.retry_at.as_deref() {
            lines.push(gateway_field("Retry", retry_at, Color::Yellow));
        }
        if let Some(error) = channel.last_error.as_deref() {
            lines.push(gateway_field("Error", error, Color::Red));
        }
    }
    Text::from(lines)
}

fn gateway_sessions_text(sessions: &[BridgeGatewaySession]) -> Text<'static> {
    if sessions.is_empty() {
        return gateway_empty_text("No sessions are stored.");
    }
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "Recent sessions",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {} shown", sessions.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ])];
    for session in sessions {
        let model = match (&session.provider, &session.model) {
            (Some(provider), Some(model)) => format!("{provider}/{model}"),
            (Some(provider), None) => provider.clone(),
            _ => String::from("provider default"),
        };
        lines.extend([
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    session.id.clone(),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {}", session.title),
                    Style::default().fg(Color::White),
                ),
            ]),
            gateway_field("Agent", &session.agent_id, Color::Cyan),
            gateway_field("Model", &model, Color::Magenta),
            gateway_field("Routes", &session.routes.to_string(), Color::Gray),
            gateway_field("Workspace", &session.workspace_root, Color::Gray),
            gateway_field("Updated", &session.updated_at, Color::DarkGray),
        ]);
        if let Some(prompt) = session.last_prompt.as_deref() {
            lines.push(gateway_field("Last prompt", prompt, Color::White));
        }
    }
    Text::from(lines)
}

fn gateway_methods_text(methods: &[BridgeGatewayMethod]) -> Text<'static> {
    if methods.is_empty() {
        return gateway_empty_text("No gateway RPC methods are registered.");
    }
    let mut lines = vec![Line::from(vec![
        Span::styled(
            "RPC method registry",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("  {} methods", methods.len()),
            Style::default().fg(Color::DarkGray),
        ),
    ])];
    for method in methods {
        let access = if method.control_write {
            "control write"
        } else {
            "read only"
        };
        lines.extend([
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    method.name.clone(),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  {} · {access}", method.owner),
                    Style::default().fg(Color::Gray),
                ),
            ]),
            gateway_field("Scopes", &method.scopes.join(", "), Color::Cyan),
            gateway_field(
                "Ready gate",
                if method.requires_ready {
                    "required"
                } else {
                    "startup-safe"
                },
                Color::Gray,
            ),
        ]);
    }
    Text::from(lines)
}

fn gateway_field(label: &str, value: &str, color: Color) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<14}"), Style::default().fg(Color::DarkGray)),
        Span::styled(value.to_string(), Style::default().fg(color)),
    ])
}

fn gateway_empty_text(message: &str) -> Text<'static> {
    Text::from(vec![
        Line::from(""),
        Line::from(vec![Span::styled(
            message.to_string(),
            Style::default().fg(Color::Gray),
        )]),
    ])
}

fn gateway_route_identity(route: &BridgeGatewayRoute) -> String {
    let mut parts = Vec::new();
    if let (Some(kind), Some(peer)) = (&route.peer_kind, &route.peer_id) {
        parts.push(format!("{kind}:{peer}"));
    }
    if let Some(sender) = &route.sender_id {
        parts.push(format!("sender:{sender}"));
    }
    if let Some(guild) = &route.guild_id {
        parts.push(format!("guild:{guild}"));
    }
    if let Some(team) = &route.team_id {
        parts.push(format!("team:{team}"));
    }
    if parts.is_empty() {
        String::from("shared/default")
    } else {
        parts.join(" · ")
    }
}

fn gateway_binding_matcher(binding: &BridgeGatewayBinding) -> String {
    if let (Some(kind), Some(peer)) = (&binding.peer_kind, &binding.peer_id) {
        return format!("peer {kind}:{peer}");
    }
    if let Some(guild) = &binding.guild_id {
        return format!("guild {guild}");
    }
    if let Some(team) = &binding.team_id {
        return format!("team {team}");
    }
    String::from("channel/account")
}

fn transcript_max_scroll(lines: &Text<'_>, width: u16, height: u16) -> u16 {
    let wrapped = Paragraph::new(lines.clone()).wrap(Wrap { trim: false });
    wrapped
        .line_count(width)
        .saturating_sub(height as usize)
        .min(u16::MAX as usize) as u16
}

fn draw_sidebar(frame: &mut ratatui::Frame<'_>, area: Rect, app: &TuiApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
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
    let configuration_ready = session.configuration_state == "ready";
    let lines = vec![
        metric_line("Source", &session.provider, Color::Cyan),
        metric_line("Model", &clip_status(&session.model, 20), Color::White),
        metric_line("Mode", &session.mode, Color::Magenta),
        metric_line("Reason", &session.reasoning_effort, Color::Blue),
        metric_line(
            "Config",
            configuration_summary(session),
            if configuration_ready {
                Color::Green
            } else {
                Color::Yellow
            },
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
        let path = approval_target_text(approval, "target");
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
            Span::styled(clip_status(&path, 28), Style::default().fg(Color::Gray)),
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
        "Enter/Y approve · N/Esc deny",
        Style::default().fg(Color::Gray),
    )]));
    if app.snapshot.approvals.len() > 1 {
        lines.push(Line::from(vec![Span::styled(
            "Ctrl+N/P select",
            Style::default().fg(Color::DarkGray),
        )]));
    }
    Text::from(lines)
}

fn approval_target_text(approval: &BridgeApproval, fallback: &str) -> String {
    let raw = approval
        .display_path
        .as_deref()
        .or(approval.translated_path.as_deref())
        .or(approval.resolved_path.as_deref())
        .or(approval.requested_path.as_deref())
        .unwrap_or(fallback);
    sanitize_approval_target(raw).unwrap_or_else(|| fallback.to_string())
}

fn sanitize_approval_target(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.len() >= 3 && parts[0] == "desktop" {
        let action = parts[1];
        let target = parts[2];
        if target.starts_with("windows-app:") || target.starts_with("windows-shortcut:") {
            if let Some(label) = parts.last().copied().filter(|label| !label.contains(':')) {
                return Some(format!("desktop {action} {label}"));
            }
            return Some(format!("desktop {action} selected app"));
        }
        if matches!(
            action,
            "focus_window"
                | "close_window"
                | "minimize_window"
                | "maximize_window"
                | "restore_window"
        ) && normalize_window_id_for_display(target).is_some()
        {
            return Some(format!("desktop {action} selected window"));
        }
    }
    Some(trimmed.to_string())
}

fn normalize_window_id_for_display(value: &str) -> Option<u64> {
    let trimmed = value.trim();
    if let Some(hex) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        u64::from_str_radix(hex, 16).ok()
    } else {
        trimmed.parse::<u64>().ok()
    }
}

fn activity_panel_text(app: &TuiApp) -> Text<'static> {
    let mut lines = Vec::new();
    if !app.submitting {
        if app.activity.is_empty() && app.subagent_run.is_none() {
            lines.push(Line::from(vec![Span::styled(
                "Idle",
                Style::default().fg(Color::DarkGray),
            )]));
            return Text::from(lines);
        }
        if let Some(run) = app.subagent_run.as_ref() {
            append_subagent_run_lines(&mut lines, run, false);
        }
        for item in app.activity.iter().rev().take(10).rev() {
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
        return Text::from(lines);
    }

    if let Some(run) = app.subagent_run.as_ref() {
        append_subagent_run_lines(&mut lines, run, true);
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
                "· ",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Reasoning", Style::default().fg(Color::Blue)),
        ]));
    }
    Text::from(lines)
}

fn append_subagent_run_lines(lines: &mut Vec<Line<'static>>, run: &SubagentRunState, live: bool) {
    let title = if live {
        format!("Parallel agents  {}/{} complete", run.completed, run.total)
    } else {
        format!(
            "Recent parallel run  {}/{} complete",
            run.completed, run.total
        )
    };
    let title_color = if run.failed > 0 {
        Color::Yellow
    } else {
        Color::Cyan
    };
    lines.push(Line::from(vec![Span::styled(
        title,
        Style::default()
            .fg(title_color)
            .add_modifier(Modifier::BOLD),
    )]));

    for task in run.tasks.iter().take(8) {
        let (symbol, color) = match task.status.as_str() {
            "complete" => ("✓", Color::Green),
            "blocked" => ("!", Color::Yellow),
            "failed" => ("×", Color::Red),
            "running" => ("●", Color::Cyan),
            _ => ("○", Color::DarkGray),
        };
        let detail = match task.status.as_str() {
            "running" if !task.summary.trim().is_empty() => task.summary.clone(),
            "queued" if !task.owned_paths.is_empty() => {
                format!("owns {}", task.owned_paths.join(", "))
            }
            "complete" if task.changed_count > 0 => {
                format!("{} file(s) changed", task.changed_count)
            }
            "failed" | "blocked" if !task.summary.trim().is_empty() => task.summary.clone(),
            _ if !task.description.trim().is_empty() => task.description.clone(),
            _ => task.summary.clone(),
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{symbol} "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(clip_status(&task.id, 18), Style::default().fg(Color::White)),
            Span::styled(
                format!(" · {}", clip_status(&task.status, 10)),
                Style::default().fg(color),
            ),
            Span::styled(
                format!(" — {}", clip_status(&detail, 38)),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }
    if run.tasks.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            clip_status(&run.run_id, 48),
            Style::default().fg(Color::DarkGray),
        )]));
    }
    if let Some(work_file) = run.work_file.as_deref() {
        lines.push(Line::from(vec![
            Span::styled("log ", Style::default().fg(Color::DarkGray)),
            Span::styled(clip_status(work_file, 48), Style::default().fg(Color::Blue)),
        ]));
    }
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
        "install" => Style::default().fg(Color::Cyan),
        "tool" => Style::default().fg(Color::Magenta),
        "subagent" => Style::default().fg(Color::Cyan),
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
        "assistant" => ("AGENT", Color::Yellow),
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
    let mut spans = vec![Span::styled("  | ", Style::default().fg(Color::DarkGray))];
    let trimmed = text.trim_start();
    if matches!(trimmed, "---" | "***" | "___") {
        spans.push(Span::styled(
            "────────────────────────",
            Style::default().fg(Color::DarkGray),
        ));
        return Line::from(spans);
    }

    let heading = ["### ", "## ", "# "]
        .iter()
        .find_map(|prefix| trimmed.strip_prefix(prefix));
    if let Some(content) = heading {
        spans.extend(markdown_inline_spans(
            content,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        return Line::from(spans);
    }

    if let Some(content) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
    {
        spans.push(Span::styled(
            "• ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        spans.extend(markdown_inline_spans(content, Style::default()));
        return Line::from(spans);
    }

    spans.extend(markdown_inline_spans(text, Style::default()));
    Line::from(spans)
}

fn markdown_inline_spans(text: &str, base_style: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut remaining = text;
    while !remaining.is_empty() {
        let bold_at = remaining.find("**");
        let code_at = remaining.find('`');
        let marker = match (bold_at, code_at) {
            (Some(bold), Some(code)) if bold <= code => Some((bold, "**")),
            (Some(_), Some(code)) => Some((code, "`")),
            (Some(bold), None) => Some((bold, "**")),
            (None, Some(code)) => Some((code, "`")),
            (None, None) => None,
        };
        let Some((start, delimiter)) = marker else {
            spans.push(Span::styled(remaining.to_string(), base_style));
            break;
        };
        if start > 0 {
            spans.push(Span::styled(remaining[..start].to_string(), base_style));
        }
        let after_start = start + delimiter.len();
        let after_marker = &remaining[after_start..];
        let Some(end) = after_marker.find(delimiter) else {
            spans.push(Span::styled(remaining[start..].to_string(), base_style));
            break;
        };
        let content = &after_marker[..end];
        let style = if delimiter == "**" {
            base_style.add_modifier(Modifier::BOLD)
        } else {
            base_style.fg(Color::Cyan).bg(Color::Rgb(28, 28, 28))
        };
        spans.push(Span::styled(content.to_string(), style));
        remaining = &after_marker[end + delimiter.len()..];
    }
    spans
}

fn transcript_text(
    snapshot: &BridgeSnapshot,
    app: &TuiApp,
    show_inline_activity: bool,
) -> Text<'static> {
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
    for notice in &app.notices {
        let color = if notice.error {
            Color::Red
        } else {
            Color::Cyan
        };
        lines.push(Line::from(vec![
            Span::styled(
                if notice.error { " ERROR " } else { " SETUP " },
                Style::default()
                    .fg(Color::Black)
                    .bg(color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(notice.title.clone(), Style::default().fg(color)),
        ]));
        for line in notice.text.lines() {
            lines.push(message_body_line(line));
        }
        lines.push(Line::from(""));
    }
    if let Some(prompt) = &app.running_prompt {
        let prompt_is_persisted = snapshot.messages.last().is_some_and(|message| {
            message.role == "user" && message.content.trim() == prompt.trim()
        });
        if !prompt_is_persisted {
            lines.push(role_header("user", "in progress"));
            lines.push(message_body_line(prompt));
            lines.push(Line::from(""));
        }
    }
    if app.submitting && show_inline_activity {
        const INLINE_ACTIVITY_LIMIT: usize = 8;
        let hidden = app.activity.len().saturating_sub(INLINE_ACTIVITY_LIMIT);
        if hidden > 0 {
            lines.push(Line::from(vec![
                Span::styled(" ACTIVITY  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("… {hidden} earlier steps hidden"),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));
        }
        for item in app.activity.iter().skip(hidden) {
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {:<9} ", item.kind.to_uppercase()),
                    activity_style(&item.kind).add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(item.text.clone(), Style::default().fg(Color::Gray)),
            ]));
        }
    }
    if app.submitting && !app.streaming_text.trim().is_empty() {
        lines.push(role_header("assistant", "drafting"));
        for line in app.streaming_text.lines() {
            lines.push(message_body_line(line));
        }
        lines.push(Line::from(""));
    } else if app.submitting && show_inline_activity && !app.reasoning_text.trim().is_empty() {
        lines.push(Line::from(vec![
            Span::styled("  · ", Style::default().fg(Color::Blue)),
            Span::styled("Reasoning", Style::default().fg(Color::DarkGray)),
        ]));
        lines.push(Line::from(""));
    }
    if lines.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "What would you like to build? Type / for commands.",
            Style::default().fg(Color::DarkGray),
        )]));
    }
    Text::from(lines)
}

fn palette_is_open(app: &TuiApp) -> bool {
    app.secret_provider.is_none()
        && app.snapshot.approvals.is_empty()
        && !app.palette.entries.is_empty()
        && app.input.starts_with('/')
}

fn palette_visible_start(total: usize, selected: usize, visible_count: usize) -> usize {
    if total <= visible_count || visible_count == 0 {
        return 0;
    }
    let selected = selected.min(total.saturating_sub(1));
    let max_start = total.saturating_sub(visible_count);
    selected
        .saturating_sub(visible_count.saturating_sub(1))
        .min(max_start)
}

fn first_selectable_palette_index(palette: &BridgeCompletions) -> Option<usize> {
    palette.entries.iter().position(palette_entry_selectable)
}

fn last_selectable_palette_index(palette: &BridgeCompletions) -> Option<usize> {
    palette.entries.iter().rposition(palette_entry_selectable)
}

fn closest_selectable_palette_index(palette: &BridgeCompletions, selected: usize) -> Option<usize> {
    if palette.entries.is_empty() {
        return None;
    }
    let selected = selected.min(palette.entries.len().saturating_sub(1));
    if palette
        .entries
        .get(selected)
        .is_some_and(palette_entry_selectable)
    {
        return Some(selected);
    }
    (selected + 1..palette.entries.len())
        .find(|index| palette_entry_selectable(&palette.entries[*index]))
        .or_else(|| {
            (0..selected)
                .rev()
                .find(|index| palette_entry_selectable(&palette.entries[*index]))
        })
}

fn next_palette_index(palette: &BridgeCompletions, selected: usize) -> usize {
    if palette.entries.is_empty() {
        return 0;
    }
    let selected = selected.min(palette.entries.len().saturating_sub(1));
    (selected + 1..palette.entries.len())
        .find(|index| palette_entry_selectable(&palette.entries[*index]))
        .unwrap_or(selected)
}

fn previous_palette_index(palette: &BridgeCompletions, selected: usize) -> usize {
    if palette.entries.is_empty() {
        return 0;
    }
    let selected = selected.min(palette.entries.len().saturating_sub(1));
    (0..selected)
        .rev()
        .find(|index| palette_entry_selectable(&palette.entries[*index]))
        .unwrap_or(selected)
}

fn move_palette_index(palette: &BridgeCompletions, selected: usize, delta: isize) -> usize {
    let mut index = closest_selectable_palette_index(palette, selected).unwrap_or(0);
    let steps = delta.unsigned_abs();
    for _ in 0..steps {
        let next = if delta >= 0 {
            next_palette_index(palette, index)
        } else {
            previous_palette_index(palette, index)
        };
        if next == index {
            break;
        }
        index = next;
    }
    index
}

fn palette_entry_selectable(entry: &BridgeCompletionEntry) -> bool {
    !entry.value.starts_with("section:")
}

fn palette_title(palette: &BridgeCompletions, selected: usize, visible_count: usize) -> String {
    let total = palette.entries.len();
    if total == 0 {
        return palette.title.clone();
    }
    let selected = selected.min(total.saturating_sub(1));
    let start = palette_visible_start(total, selected, visible_count.max(1));
    let end = (start + visible_count.max(1)).min(total);
    format!(
        "{} · {}-{}/{} · ↑↓ PgUp/PgDn wheel",
        palette.title,
        start + 1,
        end,
        total,
    )
}

fn scroll_active_view(app: &mut TuiApp, down: bool) {
    if let Some(view) = app.gateway_view.as_mut() {
        view.scroll = if down {
            view.scroll.saturating_add(3)
        } else {
            view.scroll.saturating_sub(3)
        };
        return;
    }
    if palette_is_open(app) {
        app.palette_selected = if down {
            move_palette_index(&app.palette, app.palette_selected, 3)
        } else {
            move_palette_index(&app.palette, app.palette_selected, -3)
        };
        return;
    }
    if down {
        app.scroll = app.scroll.saturating_add(3);
    } else {
        app.auto_follow = false;
        app.scroll = app.scroll.saturating_sub(3);
    }
}

fn palette_text(
    palette: &BridgeCompletions,
    selected: usize,
    visible_count: usize,
    width: u16,
) -> Text<'static> {
    let mut lines = Vec::new();
    let label_width = if width > 64 { 24 } else { 16 };
    let total = palette.entries.len();
    let selected = selected.min(total.saturating_sub(1));
    let visible_count = visible_count.max(1);
    let start = palette_visible_start(total, selected, visible_count);
    let end = (start + visible_count).min(total);
    for (index, entry) in palette.entries[start..end].iter().enumerate() {
        let index = start + index;
        if !palette_entry_selectable(entry) {
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(
                    clip_status(&entry.label, width.saturating_sub(4) as usize),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            continue;
        }
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
        "install" => "↓",
        "tool" => "+",
        "subagent" => "↳",
        "guardrail" => "!",
        "text" => ":",
        _ => "-",
    }
}

fn tool_activity_label(tool: &str) -> String {
    match tool {
        "read_path" => String::from("Reading files"),
        "list_path" => String::from("Listing files"),
        "path_status" => String::from("Checking path status"),
        "inspect_tree" => String::from("Exploring the workspace"),
        "inspect_target" => String::from("Inspecting a target"),
        "glob" => String::from("Finding files"),
        "grep" => String::from("Searching code"),
        "language_server" => String::from("Checking code intelligence"),
        "write_file" => String::from("Writing a file"),
        "edit_file" => String::from("Editing a file"),
        "delete_path" => String::from("Deleting a path"),
        "run_system_command" => String::from("Running a command"),
        "system_info" => String::from("Inspecting the system"),
        "connected_devices" => String::from("Checking connected devices"),
        "desktop_capabilities" => String::from("Checking desktop capabilities"),
        "desktop_observe" => String::from("Observing the desktop"),
        "desktop_resolve" => String::from("Resolving desktop target"),
        "process_list" => String::from("Checking running processes"),
        "desktop_action" => String::from("Performing a desktop action"),
        "parallel_subagents" => String::from("Spawning independent parallel agents"),
        "load_skill" => String::from("Loading a skill"),
        "finish_task" => String::from("Preparing the response"),
        _ => format!("Running {tool}"),
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
            if !response.ok {
                app.status = format!(
                    "Approval failed: {}",
                    clip_status(
                        response
                            .error
                            .as_deref()
                            .unwrap_or("bridge rejected decision"),
                        80
                    )
                );
                return;
            }
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
            app.palette_selected = app
                .palette
                .selected_index
                .unwrap_or(app.palette_selected)
                .min(app.palette.entries.len().saturating_sub(1));
            app.palette_selected =
                closest_selectable_palette_index(&app.palette, app.palette_selected).unwrap_or(0);
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
                    let install_preview = prompt.trim_start().starts_with("/install ")
                        && !install_command_is_confirmed(&prompt);
                    let installing = install_command_is_confirmed(&prompt);
                    app.running_prompt = Some(prompt);
                    if installing {
                        app.status = String::from("Starting local model installation");
                    } else if install_preview {
                        app.status = String::from("Preparing local model install preview");
                    }
                }
                if let Some(snapshot) = frame.snapshot {
                    app.snapshot = snapshot;
                    let max_index = app.snapshot.approvals.len().saturating_sub(1);
                    app.approval_selected = app.approval_selected.min(max_index);
                }
                app.activity.push(ActivityLine {
                    kind: if app
                        .running_prompt
                        .as_deref()
                        .is_some_and(|prompt| prompt.trim_start().starts_with("/install "))
                    {
                        String::from("install")
                    } else {
                        String::from("thinking")
                    },
                    text: if app
                        .running_prompt
                        .as_deref()
                        .is_some_and(install_command_is_confirmed)
                    {
                        String::from("Starting local download")
                    } else if app
                        .running_prompt
                        .as_deref()
                        .is_some_and(|prompt| prompt.trim_start().starts_with("/install "))
                    {
                        String::from("Reviewing model size and runtime requirements")
                    } else {
                        String::from("Waiting for model response")
                    },
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
                let completed_prompt = app.running_prompt.clone();
                let command_result = frame.command_result.as_ref();
                let next_command = command_result.and_then(|result| result.next_command.clone());
                let key_prompt_provider =
                    command_result.and_then(|result| result.secret_provider.clone());
                app.submitting = false;
                if let Some(snapshot) = frame.snapshot {
                    app.snapshot = snapshot;
                    let max_index = app.snapshot.approvals.len().saturating_sub(1);
                    app.approval_selected = app.approval_selected.min(max_index);
                }
                ensure_final_frame_messages_visible(
                    app,
                    completed_prompt.as_deref(),
                    frame.answer.as_deref(),
                );
                clear_live_turn_display(app);
                if app.cancel_requested {
                    app.cancel_requested = false;
                    app.cancel_signal.store(false, Ordering::SeqCst);
                    app.status = String::from("Stopped — Agent is ready for another message");
                    return;
                }
                let error = frame.error.as_deref().filter(|text| !text.is_empty());
                if let Some(error) = error {
                    push_notice(app, "Request failed", error, true);
                    app.status = String::from("Error — details shown in conversation");
                } else {
                    let command_needs_setup =
                        command_result.is_some_and(|result| result.setup_required);
                    app.setup_required =
                        app.snapshot.session.configuration_state != "ready" || command_needs_setup;
                    if app.snapshot.session.configuration_state == "ready" && !command_needs_setup {
                        app.notices.clear();
                    }
                    app.status = command_result
                        .map(command_result_status)
                        .unwrap_or_else(|| String::from("Ready"));
                }
                if let Some(command) = next_command {
                    app.input = command;
                    app.palette = BridgeCompletions::default();
                    app.palette_selected = 0;
                }
                if let Some(provider) = key_prompt_provider {
                    app.secret_provider = Some(provider.clone());
                    app.secret_input.clear();
                    app.status = format!("{} needs an API key", provider_display_name(&provider));
                }
            }
            _ => {}
        },
        AppEvent::StreamFrame(Err(err)) => {
            app.submitting = false;
            clear_live_turn_display(app);
            if app.cancel_requested {
                app.cancel_requested = false;
                app.cancel_signal.store(false, Ordering::SeqCst);
                app.status = String::from("Stopped — Agent is ready for another message");
                return;
            }
            push_notice(app, "Bridge failed", &err.to_string(), true);
            app.status = String::from("Error — details shown in conversation");
        }
    }
}

fn clear_live_turn_display(app: &mut TuiApp) {
    app.running_prompt = None;
    app.current_tool = None;
    app.reasoning_text.clear();
    app.streaming_text.clear();
    app.activity.clear();
}

fn ensure_final_frame_messages_visible(
    app: &mut TuiApp,
    completed_prompt: Option<&str>,
    answer: Option<&str>,
) {
    if let Some(prompt) = completed_prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if !snapshot_contains_message(&app.snapshot, "user", prompt) {
            app.snapshot.messages.push(BridgeMessage {
                role: String::from("user"),
                content: prompt.to_string(),
                created_at: String::from("just now"),
            });
        }
    }

    if let Some(answer) = answer.map(str::trim).filter(|value| !value.is_empty()) {
        if !snapshot_contains_message(&app.snapshot, "assistant", answer) {
            app.snapshot.messages.push(BridgeMessage {
                role: String::from("assistant"),
                content: answer.to_string(),
                created_at: String::from("just now"),
            });
        }
    }
}

fn snapshot_contains_message(snapshot: &BridgeSnapshot, role: &str, content: &str) -> bool {
    snapshot
        .messages
        .iter()
        .any(|message| message.role == role && message.content.trim() == content.trim())
}

fn apply_bridge_event(app: &mut TuiApp, event: BridgeEvent) {
    match event.kind.as_str() {
        "reasoning_delta" | "reasoning_started" => {
            // Raw chain-of-thought is private. Keep only a visible activity
            // indicator and render tool/results as the inspectable trace.
            app.reasoning_text = format!("Reasoning · {}", app.snapshot.session.reasoning_effort);
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
            app.activity.push(ActivityLine {
                kind: String::from("tool"),
                text: tool_activity_label(&tool),
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
        "subagent_run_started"
        | "subagent_task_started"
        | "subagent_task_progress"
        | "subagent_task_completed"
        | "subagent_run_completed" => {
            apply_subagent_lifecycle_event(app, &event);
        }
        "install_progress" => {
            if let Some(summary) = event.summary {
                app.status = clip_status(&summary, 72);
                if let Some(last) = app
                    .activity
                    .last_mut()
                    .filter(|item| item.kind == "install")
                {
                    last.text = summary;
                } else {
                    app.activity.push(ActivityLine {
                        kind: String::from("install"),
                        text: summary,
                    });
                }
            }
        }
        "response_completed" => {}
        _ => {}
    }
}

fn apply_subagent_lifecycle_event(app: &mut TuiApp, event: &BridgeEvent) {
    let run_id = event
        .run_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("parallel-current");
    let has_different_run = app
        .subagent_run
        .as_ref()
        .is_some_and(|run| run.run_id != run_id);
    if has_different_run && event.kind != "subagent_run_started" {
        return;
    }
    let needs_run = app.subagent_run.is_none() || has_different_run;
    if needs_run {
        app.subagent_run = Some(SubagentRunState {
            run_id: run_id.to_string(),
            total: event.total.unwrap_or(event.tasks.len()),
            completed: 0,
            failed: 0,
            status: String::from("running"),
            work_file: event.work_file.clone(),
            tasks: Vec::new(),
        });
    }

    let Some(run) = app.subagent_run.as_mut() else {
        return;
    };
    if let Some(total) = event.total {
        run.total = total;
    }
    if let Some(work_file) = event.work_file.as_ref() {
        run.work_file = Some(work_file.clone());
    }

    match event.kind.as_str() {
        "subagent_run_started" => {
            run.status = String::from("running");
            run.completed = 0;
            run.failed = 0;
            run.tasks = event
                .tasks
                .iter()
                .map(|task| SubagentTaskState {
                    id: task.id.clone(),
                    description: task.task.clone(),
                    status: String::from("queued"),
                    summary: String::new(),
                    owned_paths: task.owns.clone(),
                    changed_count: 0,
                })
                .collect();
        }
        "subagent_task_started" | "subagent_task_progress" | "subagent_task_completed" => {
            let Some(task_id) = event
                .task_id
                .as_deref()
                .filter(|value| !value.trim().is_empty())
            else {
                return;
            };
            let index = run
                .tasks
                .iter()
                .position(|task| task.id == task_id)
                .unwrap_or_else(|| {
                    run.tasks.push(SubagentTaskState {
                        id: task_id.to_string(),
                        description: String::new(),
                        status: String::from("queued"),
                        summary: String::new(),
                        owned_paths: Vec::new(),
                        changed_count: 0,
                    });
                    run.tasks.len() - 1
                });
            let task = &mut run.tasks[index];
            if event.kind == "subagent_task_started" || event.kind == "subagent_task_progress" {
                task.status = String::from("running");
            } else {
                task.status = event
                    .status
                    .clone()
                    .unwrap_or_else(|| String::from("complete"));
            }
            if let Some(summary) = event.summary.as_ref() {
                task.summary = summary.clone();
            }
            if !event.owned_paths.is_empty() {
                task.owned_paths = event.owned_paths.clone();
            }
            if let Some(changed_count) = event.changed_count {
                task.changed_count = changed_count;
            }
            run.completed = run
                .tasks
                .iter()
                .filter(|task| task.status == "complete")
                .count();
            run.failed = run
                .tasks
                .iter()
                .filter(|task| task.status == "failed")
                .count();
            run.total = run.total.max(run.tasks.len());
        }
        "subagent_run_completed" => {
            run.status = if event.failed.unwrap_or(0) > 0 {
                String::from("incomplete")
            } else {
                String::from("complete")
            };
            run.completed = event.completed.unwrap_or(run.completed);
            run.failed = event.failed.unwrap_or(run.failed);
        }
        _ => {}
    }

    if let Some(summary) = event.summary.as_ref() {
        app.status = clip_status(summary, 72);
    } else {
        app.status = format!("Parallel agents: {}/{} complete", run.completed, run.total);
    }
}

fn provider_api_key_env(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("OPENAI_API_KEY"),
        "anthropic" => Some("ANTHROPIC_API_KEY"),
        "gemini" => Some("GOOGLE_API_KEY"),
        "groq" => Some("GROQ_API_KEY"),
        "openrouter" => Some("OPENROUTER_API_KEY"),
        "azure" => Some("AZURE_OPENAI_API_KEY"),
        "deepseek" => Some("DEEPSEEK_API_KEY"),
        "glm" => Some("GLM_API_KEY"),
        "openai-compatible" => Some("AGENT_OPENAI_COMPAT_API_KEY"),
        _ => None,
    }
}

fn provider_display_name(provider: &str) -> &'static str {
    match provider {
        "openai" => "OpenAI",
        "anthropic" => "Anthropic",
        "gemini" => "Google Gemini",
        "groq" => "Groq",
        "openrouter" => "OpenRouter",
        "azure" => "Azure OpenAI",
        "bedrock" => "AWS Bedrock",
        "vertexai" => "Vertex AI",
        "copilot" => "GitHub Copilot",
        "openai-compatible" => "OpenAI-compatible",
        "ollama" => "Ollama",
        "lmstudio" => "LM Studio",
        "llamacpp" => "llama.cpp",
        "vllm" => "vLLM",
        "localai" => "LocalAI",
        "deepseek" => "DeepSeek",
        "glm" => "GLM",
        _ => "Provider",
    }
}

fn provider_setup_url(provider: &str) -> Option<&'static str> {
    match provider {
        "openai" => Some("https://platform.openai.com/api-keys"),
        "anthropic" => Some("https://console.anthropic.com/settings/keys"),
        "gemini" => Some("https://aistudio.google.com/app/apikey"),
        "groq" => Some("https://console.groq.com/keys"),
        "openrouter" => Some("https://openrouter.ai/settings/keys"),
        "azure" => Some("https://ai.azure.com"),
        "bedrock" => Some("https://console.aws.amazon.com/bedrock/home"),
        "vertexai" => Some("https://console.cloud.google.com/vertex-ai"),
        "copilot" => Some("https://github.com/login/device"),
        "deepseek" => Some("https://platform.deepseek.com/api_keys"),
        "glm" => Some("https://bigmodel.cn/usercenter/proj-mgmt/apikeys"),
        _ => None,
    }
}

fn provider_setup_notice(
    provider: &str,
    configuration: &str,
    configuration_state: &str,
) -> UiNotice {
    let mut text = configuration.to_string();
    let needs_api_key = provider != "openai-compatible"
        && provider_api_key_env(provider).is_some()
        && configuration_state == "api_key_required";
    if needs_api_key {
        text.push_str(&format!(
            "\nRun /apikey {provider} to enter it in a masked prompt."
        ));
    } else if provider == "openai-compatible" {
        text.push_str("\nConfigure AGENT_OPENAI_COMPAT_BASE_URL, then restart Agent.");
    } else {
        text.push_str("\nComplete the provider setup, then return to Agent.");
    }
    if let Some(url) = provider_setup_url(provider) {
        text.push_str("\nAccount/API keys: ");
        text.push_str(url);
    }
    UiNotice {
        title: format!("{} setup", provider_display_name(provider)),
        text,
        error: false,
    }
}

fn push_notice(app: &mut TuiApp, title: &str, text: &str, error: bool) {
    const MAX_NOTICES: usize = 20;
    if app.notices.len() >= MAX_NOTICES {
        app.notices.remove(0);
    }
    app.notices.push(UiNotice {
        title: title.to_string(),
        text: text.to_string(),
        error,
    });
}

fn ui_state_badge(app: &TuiApp) -> (&'static str, Color) {
    if app.submitting {
        return ("running", Color::Cyan);
    }
    if app.secret_provider.is_some() {
        return ("setup", Color::Yellow);
    }
    if app.status.starts_with("Error") || app.status.starts_with("Could not") {
        return ("error", Color::Red);
    }
    if app.setup_required || app.snapshot.session.configuration_state != "ready" {
        return ("setup", Color::Yellow);
    }
    ("ready", Color::Green)
}

fn configuration_needs_api_key(session: &BridgeSession) -> bool {
    session.configuration_state == "api_key_required"
}

fn configuration_summary(session: &BridgeSession) -> &'static str {
    if session.configuration_state == "ready" {
        "Ready"
    } else if configuration_needs_api_key(session) {
        "API key required"
    } else {
        "Setup required"
    }
}

fn command_result_status(result: &BridgeCommandResult) -> String {
    match result.code {
        BridgeCommandCode::InstallConfirmationRequired => {
            String::from("Review the model size above, then press Enter to install")
        }
        BridgeCommandCode::ModelNotInstalled => {
            String::from("Model not installed — use /install when available")
        }
        BridgeCommandCode::RuntimeUnavailable | BridgeCommandCode::RuntimeNotInstalled => {
            String::from("Local runtime unavailable — setup details shown above")
        }
        BridgeCommandCode::InstallFailed | BridgeCommandCode::InstallUnverified => {
            String::from("Error — local model installation failed")
        }
        BridgeCommandCode::InstallNotReady | BridgeCommandCode::ManualSetupRequired => {
            String::from("Local model setup incomplete — details shown above")
        }
        BridgeCommandCode::ApiKeyRequired => String::from("API key required"),
        BridgeCommandCode::CredentialsRequired => String::from("Provider credentials required"),
        BridgeCommandCode::Unavailable => String::from("Requested model is unavailable"),
        BridgeCommandCode::Ok | BridgeCommandCode::Unknown => {
            if result.error {
                String::from("Error — details shown in conversation")
            } else {
                String::from("Ready")
            }
        }
    }
}

fn install_command_is_confirmed(command: &str) -> bool {
    let parts = command.split_whitespace().collect::<Vec<_>>();
    parts.len() == 4 && parts[0] == "/install" && parts[3] == "--yes"
}

fn apply_bridge_credentials(command: &mut ProcessCommand, args: &TuiArgs) {
    if let Ok(api_keys) = args.api_keys.lock() {
        for (env_name, api_key) in api_keys.iter() {
            command.env(env_name, api_key);
        }
    }
}

fn call_bridge(args: &TuiArgs, action: &str, prompt: Option<&str>) -> Result<BridgeResponse> {
    let mut command = ProcessCommand::new(&args.python);
    command
        .arg("-m")
        .arg("agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .current_dir(&args.repo_root);
    apply_bridge_credentials(&mut command, args);
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
        .arg("agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .arg("--bridge-request-id")
        .arg(request_id)
        .current_dir(&args.repo_root);
    apply_bridge_credentials(&mut command, args);

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

fn stream_bridge_submit(
    args: &TuiArgs,
    prompt: &str,
    tx: mpsc::Sender<AppEvent>,
    active_bridge: Arc<Mutex<Option<Child>>>,
    cancel_signal: Arc<AtomicBool>,
) -> Result<()> {
    let (stdout, stderr) = {
        let mut active = active_bridge
            .lock()
            .map_err(|_| anyhow::anyhow!("Could not claim the bridge process slot."))?;
        if active.is_some() {
            return Err(anyhow::anyhow!(
                "A bridge turn is already active; wait for it to finish before starting another."
            ));
        }
        let mut child = spawn_bridge(args, "stream-submit", Some(prompt))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Bridge stdout was not captured."))?;
        let stderr = child.stderr.take();
        *active = Some(child);
        if cancel_signal.load(Ordering::SeqCst) {
            if let Some(child) = active.as_mut() {
                let _ = interrupt_process_tree(child);
            }
        }
        (stdout, stderr)
    };
    let stderr_reader = stderr.map(|mut pipe| {
        thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            text
        })
    });
    let reader = BufReader::new(stdout);
    let mut saw_final_frame = false;
    let mut stream_error: Option<anyhow::Error> = None;
    for line_result in reader.lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(err) => {
                stream_error = Some(err.into());
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let parsed = match parse_bridge_stream_line(&line) {
            Ok(parsed) => parsed,
            Err(err) => {
                stream_error = Some(err);
                break;
            }
        };
        if parsed.kind == "final" {
            saw_final_frame = true;
        }
        let _ = tx.send(AppEvent::StreamFrame(Ok(parsed)));
    }
    if stream_error.is_some() {
        if let Ok(mut active) = active_bridge.lock() {
            if let Some(child) = active.as_mut() {
                let _ = interrupt_process_tree(child);
            }
        }
    }
    let status = {
        let mut active = active_bridge
            .lock()
            .map_err(|_| anyhow::anyhow!("Could not finish the active bridge process."))?;
        let mut child = active
            .take()
            .ok_or_else(|| anyhow::anyhow!("Active bridge process was lost."))?;
        drop(active);
        child.wait()?
    };
    let stderr = stderr_reader
        .and_then(|handle| handle.join().ok())
        .unwrap_or_default();
    if let Some(error) = stream_error {
        return Err(error);
    }
    if !saw_final_frame {
        let detail = stderr.trim();
        let message = if status.success() && detail.is_empty() {
            String::from("Bridge ended without a final turn-completion frame")
        } else if detail.is_empty() {
            format!("Bridge exited with {}", status)
        } else {
            format!(
                "Bridge exited with {}: {}",
                status,
                clip_status(detail, 160)
            )
        };
        let _ = tx.send(AppEvent::StreamFrame(Err(anyhow::anyhow!(message))));
    }
    Ok(())
}

fn parse_bridge_stream_line(line: &str) -> Result<BridgeStreamFrame> {
    match serde_json::from_str::<BridgeStreamFrame>(line) {
        Ok(parsed) => Ok(parsed),
        Err(frame_err) => match serde_json::from_str::<BridgeResponse>(line) {
            Ok(response) => Ok(bridge_response_to_final_frame(response)),
            Err(response_err) => Err(anyhow::anyhow!(
                "Could not parse bridge stream frame: {}. fallback response parse failed: {}. line: {}",
                frame_err,
                response_err,
                line
            )),
        },
    }
}

fn bridge_response_to_final_frame(response: BridgeResponse) -> BridgeStreamFrame {
    BridgeStreamFrame {
        kind: String::from("final"),
        prompt: None,
        answer: response.answer,
        error: if response.ok {
            None
        } else {
            response
                .error
                .or_else(|| Some(String::from("Bridge returned ok=false.")))
        },
        event: None,
        snapshot: response.snapshot,
        command_result: response.command_result,
    }
}

#[cfg(test)]
mod tui_lifecycle_tests {
    use super::*;

    #[test]
    fn busy_prompt_is_retained_for_the_next_turn() {
        let mut queue = VecDeque::new();

        assert_eq!(enqueue_prompt(&mut queue, String::from("project/")), 1);
        assert_eq!(queue.pop_front().as_deref(), Some("project/"));
    }

    #[cfg(unix)]
    #[test]
    fn active_process_group_can_be_stopped_without_exiting_tui() {
        use std::os::unix::process::CommandExt;

        let mut command = ProcessCommand::new("sh");
        command.args(["-c", "sleep 30"]).process_group(0);
        let mut child = command.spawn().expect("spawn cancellable task");

        interrupt_process_tree(&mut child).expect("interrupt process group");
        let status = child.wait().expect("wait for interrupted task");

        assert!(!status.success());
    }

    #[cfg(unix)]
    #[test]
    fn second_stream_bridge_is_rejected_while_one_is_active() {
        let child = ProcessCommand::new("sh")
            .args(["-c", "sleep 30"])
            .spawn()
            .expect("spawn active bridge placeholder");
        let active_bridge = Arc::new(Mutex::new(Some(child)));
        let args = TuiArgs {
            python: String::from("python3"),
            repo_root: PathBuf::from("."),
            session_id: String::from("session-1"),
            api_keys: Arc::new(Mutex::new(HashMap::new())),
        };
        let (tx, _rx) = mpsc::channel();

        let result = stream_bridge_submit(
            &args,
            "second prompt",
            tx,
            Arc::clone(&active_bridge),
            Arc::new(AtomicBool::new(false)),
        );

        assert!(result
            .expect_err("second bridge must be rejected")
            .to_string()
            .contains("already active"));
        let mut child = active_bridge
            .lock()
            .expect("active bridge lock")
            .take()
            .expect("active bridge child");
        child.kill().expect("stop placeholder bridge");
        child.wait().expect("reap placeholder bridge");
    }
}

#[cfg(test)]
mod host_capability_tests {
    use super::*;

    #[test]
    fn device_inventory_has_versioned_availability_and_status_records() {
        let inventory = connected_devices("network").expect("device inventory");

        assert_eq!(
            inventory.get("schema_version").and_then(Value::as_u64),
            Some(3)
        );
        assert!(inventory
            .get("availability")
            .and_then(Value::as_object)
            .is_some());
        let categories = inventory
            .get("categories")
            .and_then(Value::as_object)
            .expect("dynamic categories");
        assert_eq!(categories.len(), 1);
        assert!(categories.contains_key("network"));
        let network = categories["network"]["records"]
            .as_array()
            .expect("network records");
        assert!(network.iter().all(|record| record.get("status").is_some()));
    }

    #[test]
    fn desktop_target_validators_reject_command_injection_shapes() {
        assert!(valid_identifier("org.example.App"));
        assert!(!valid_identifier("app;shutdown"));
        assert!(valid_bluetooth_address("AA:BB:CC:DD:EE:FF"));
        assert!(!valid_bluetooth_address("AA:BB:CC;rm -rf"));
        assert!(valid_path_token("/dev/sdb1"));
        assert!(!valid_path_token("/dev/sdb1;touch"));
    }

    #[test]
    fn windows_device_classes_map_to_normalized_categories() {
        assert_eq!(windows_device_category("Bluetooth"), "bluetooth");
        assert_eq!(windows_device_category("Net"), "network");
        assert_eq!(windows_device_category("AudioEndpoint"), "audio");
        assert_eq!(windows_device_category("HIDClass"), "input");
        assert_eq!(windows_device_category("SoftwareComponent"), "other");
    }
}

fn spawn_bridge(args: &TuiArgs, action: &str, prompt: Option<&str>) -> Result<Child> {
    let mut command = ProcessCommand::new(&args.python);
    command
        .arg("-m")
        .arg("agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .current_dir(&args.repo_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    apply_bridge_credentials(&mut command, args);
    if let Some(prompt) = prompt {
        command.arg("--bridge-prompt").arg(prompt);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    Ok(command.spawn()?)
}

#[cfg(test)]
mod tui_tests {
    use super::*;

    fn anthropic_session() -> BridgeSession {
        BridgeSession {
            id: String::from("session-1"),
            title: String::from("test"),
            workspace_root: String::from("/workspace"),
            provider: String::from("anthropic"),
            model: String::from("claude-sonnet-4.5"),
            mode: String::from("hosted"),
            reasoning_effort: String::from("provider controlled"),
            configuration: String::from("Anthropic is not configured. Set ANTHROPIC_API_KEY."),
            configuration_state: String::from("api_key_required"),
            cost_usd: 0.0,
            tokens: BridgeTokens {
                input: 0,
                output: 0,
                reasoning: 0,
                cache_read: 0,
                cache_write: 0,
            },
        }
    }

    fn test_app() -> TuiApp {
        let mut session = anthropic_session();
        session.provider = String::from("openai");
        session.model = String::from("gpt-test");
        session.configuration = String::from("ready");
        session.configuration_state = String::from("ready");
        TuiApp {
            snapshot: BridgeSnapshot {
                session,
                approvals: Vec::new(),
                messages: Vec::new(),
            },
            input: String::new(),
            status: String::from("Ready"),
            scroll: 0,
            auto_follow: true,
            submitting: false,
            cancel_requested: false,
            cancel_signal: Arc::new(AtomicBool::new(false)),
            active_bridge: Arc::new(Mutex::new(None)),
            queued_prompts: VecDeque::new(),
            activity: Vec::new(),
            subagent_run: None,
            palette: BridgeCompletions::default(),
            palette_selected: 0,
            approval_selected: 0,
            current_tool: None,
            reasoning_text: String::new(),
            streaming_text: String::new(),
            running_prompt: None,
            secret_provider: None,
            secret_input: String::new(),
            notices: Vec::new(),
            setup_required: false,
            gateway_view: None,
        }
    }

    fn test_palette_entry(label: &str, execute: bool) -> BridgeCompletionEntry {
        BridgeCompletionEntry {
            value: label.to_string(),
            label: label.to_string(),
            description: String::new(),
            complete_to: format!("/{label}"),
            execute,
        }
    }

    fn gateway_snapshot() -> BridgeGatewaySnapshot {
        serde_json::from_value(json!({
            "generated_at": "now",
            "overview": {
                "control_plane": "local-python",
                "state": "ready",
                "started_at": "now",
                "session_store": "/tmp/agent.sqlite3",
                "default_agent": "main",
                "default_scope": "per-sender",
                "bindings": 1,
                "channels": ["tui"],
                "method_count": 1,
                "config_sources": ["/workspace/.agent/config.json"],
                "execution_model": "one active parent turn",
                "active_session": "session-1",
                "active_agent": "main",
                "active_route": "agent-route-v1:test",
                "workspace_root": "/workspace",
                "tool_policy": "all parent tools"
            },
            "routes": [{
                "route_key": "agent-route-v1:test",
                "session_id": "session-1",
                "agent_id": "main",
                "scope": "per-sender",
                "channel": "tui",
                "account_id": "default",
                "sender_id": "local-user",
                "updated_at": "now"
            }],
            "bindings": [{
                "agent_id": "main",
                "channel": "tui",
                "scope": "per-sender",
                "account_id": "*"
            }],
            "channels": [{
                "channel": "tui",
                "account_id": "default",
                "state": "registered",
                "generation": 0,
                "consecutive_failures": 0
            }],
            "sessions": [{
                "id": "session-1",
                "title": "Gateway test",
                "workspace_root": "/workspace",
                "agent_id": "main",
                "provider": "openai",
                "model": "gpt-test",
                "updated_at": "now",
                "routes": 1
            }],
            "methods": [{
                "name": "gateway.status",
                "owner": "core",
                "scopes": ["gateway.read"],
                "requires_ready": false,
                "control_write": false
            }]
        }))
        .expect("gateway snapshot")
    }

    fn rendered_text(text: &Text<'_>) -> String {
        text.lines
            .iter()
            .flat_map(|line| line.spans.iter())
            .map(|span| span.content.as_ref())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parallel_subagent_events_are_visible_in_activity_panel() {
        let mut app = test_app();
        app.submitting = true;

        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_run_started"),
                summary: Some(String::from(
                    "Spawned 2 parallel subagents · log: .agent/parallel-work.md",
                )),
                run_id: Some(String::from("parallel-test")),
                total: Some(2),
                work_file: Some(String::from(".agent/parallel-work.md")),
                tasks: vec![
                    BridgeSubagentTask {
                        id: String::from("architecture"),
                        task: String::from("inspect architecture"),
                        owns: vec![String::from("frontend")],
                    },
                    BridgeSubagentTask {
                        id: String::from("tests"),
                        task: String::from("inspect tests"),
                        owns: vec![String::from("tests")],
                    },
                ],
                ..BridgeEvent::default()
            },
        );
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_task_started"),
                summary: Some(String::from(
                    "architecture · running — inspect architecture",
                )),
                run_id: Some(String::from("parallel-test")),
                task_id: Some(String::from("architecture")),
                ..BridgeEvent::default()
            },
        );

        let rendered = rendered_text(&activity_panel_text(&app));
        assert!(rendered.contains("Parallel agents"));
        assert!(rendered.contains("0/2 complete"));
        assert!(rendered.contains("architecture"));
        assert!(rendered.contains("running"));
        assert!(rendered.contains("owns tests"));
        assert!(rendered.contains(".agent/parallel-work.md"));
        assert_eq!(app.status, "architecture · running — inspect architecture");
    }

    #[test]
    fn parallel_subagent_completion_fields_update_structured_ui_state() {
        let mut app = test_app();
        app.submitting = true;
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_run_started"),
                run_id: Some(String::from("parallel-test")),
                total: Some(2),
                tasks: vec![
                    BridgeSubagentTask {
                        id: String::from("python"),
                        task: String::from("inspect Python"),
                        owns: vec![String::from("agent")],
                    },
                    BridgeSubagentTask {
                        id: String::from("rust"),
                        task: String::from("inspect Rust"),
                        owns: vec![String::from("agent-rust")],
                    },
                ],
                ..BridgeEvent::default()
            },
        );
        for (task_id, status) in [("python", "complete"), ("rust", "failed")] {
            apply_bridge_event(
                &mut app,
                BridgeEvent {
                    kind: String::from("subagent_task_completed"),
                    run_id: Some(String::from("parallel-test")),
                    task_id: Some(task_id.to_string()),
                    status: Some(status.to_string()),
                    summary: Some(format!("{task_id} · {status}")),
                    ..BridgeEvent::default()
                },
            );
        }
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_run_completed"),
                run_id: Some(String::from("parallel-test")),
                total: Some(2),
                completed: Some(1),
                failed: Some(1),
                ..BridgeEvent::default()
            },
        );

        let run = app.subagent_run.as_ref().expect("structured run state");
        assert_eq!(run.completed, 1);
        assert_eq!(run.failed, 1);
        assert_eq!(run.status, "incomplete");
        let rendered = rendered_text(&activity_panel_text(&app));
        let compact = rendered.replace('\n', "");
        assert!(rendered.contains("1/2 complete"));
        assert!(compact.contains("python · complete"));
        assert!(compact.contains("rust · failed"));
    }

    #[test]
    fn parallel_subagent_progress_tracks_ownership_and_changes() {
        let mut app = test_app();
        app.submitting = true;
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_run_started"),
                run_id: Some(String::from("parallel-progress")),
                total: Some(2),
                tasks: vec![
                    BridgeSubagentTask {
                        id: String::from("client"),
                        task: String::from("implement client"),
                        owns: vec![String::from("tasker/client")],
                    },
                    BridgeSubagentTask {
                        id: String::from("server"),
                        task: String::from("implement server"),
                        owns: vec![String::from("tasker/server")],
                    },
                ],
                ..BridgeEvent::default()
            },
        );
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_task_progress"),
                run_id: Some(String::from("parallel-progress")),
                task_id: Some(String::from("client")),
                status: Some(String::from("running")),
                owned_paths: vec![String::from("tasker/client")],
                changed_count: Some(2),
                summary: Some(String::from("write_file · observed tasker/client/app.ts")),
                ..BridgeEvent::default()
            },
        );

        let task = &app.subagent_run.as_ref().unwrap().tasks[0];
        assert_eq!(task.status, "running");
        assert_eq!(task.owned_paths, vec![String::from("tasker/client")]);
        assert_eq!(task.changed_count, 2);
        assert!(rendered_text(&activity_panel_text(&app)).contains("write_file"));
    }

    #[test]
    fn stale_subagent_event_cannot_replace_the_active_run() {
        let mut app = test_app();
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_run_started"),
                run_id: Some(String::from("parallel-current")),
                total: Some(2),
                ..BridgeEvent::default()
            },
        );
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("subagent_task_completed"),
                run_id: Some(String::from("parallel-stale")),
                task_id: Some(String::from("stale-task")),
                status: Some(String::from("complete")),
                ..BridgeEvent::default()
            },
        );

        let run = app.subagent_run.as_ref().unwrap();
        assert_eq!(run.run_id, "parallel-current");
        assert!(run.tasks.is_empty());
    }

    #[test]
    fn conversation_markdown_markers_are_rendered_as_terminal_styles() {
        let heading = rendered_text(&Text::from(message_body_line("### Result")));
        let inline = rendered_text(&Text::from(message_body_line(
            "This is **important** and `planner.py` is code.",
        )));
        let bullet = rendered_text(&Text::from(message_body_line("- one item")));
        let compact_bullet = bullet.replace('\n', "");

        assert!(heading.contains("Result"));
        assert!(!heading.contains("###"));
        assert!(inline.contains("important"));
        assert!(inline.contains("planner.py"));
        assert!(!inline.contains("**"));
        assert!(!inline.contains('`'));
        assert!(compact_bullet.contains("• one item"));
    }

    #[test]
    fn completed_turn_retains_recent_parallel_activity() {
        let mut app = test_app();
        app.submitting = true;
        app.activity.push(ActivityLine {
            kind: String::from("tool"),
            text: String::from("ordinary tool trace"),
        });
        app.subagent_run = Some(SubagentRunState {
            run_id: String::from("parallel-test"),
            total: 2,
            completed: 2,
            failed: 0,
            status: String::from("complete"),
            work_file: Some(String::from(".agent/parallel-work.md")),
            tasks: vec![
                SubagentTaskState {
                    id: String::from("architecture"),
                    description: String::from("inspect architecture"),
                    status: String::from("complete"),
                    summary: String::new(),
                    owned_paths: vec![String::from("frontend")],
                    changed_count: 1,
                },
                SubagentTaskState {
                    id: String::from("tests"),
                    description: String::from("inspect tests"),
                    status: String::from("complete"),
                    summary: String::new(),
                    owned_paths: vec![String::from("tests")],
                    changed_count: 2,
                },
            ],
        });

        clear_live_turn_display(&mut app);
        app.submitting = false;

        let rendered = rendered_text(&activity_panel_text(&app));
        assert!(!rendered.contains("ordinary tool trace"));
        assert!(rendered.contains("Recent parallel run"));
        assert!(rendered.contains("2/2 complete"));
        assert!(rendered.contains("architecture"));
    }

    #[test]
    fn completed_turn_replaces_live_tool_trace_with_persisted_messages() {
        let mut app = test_app();
        app.submitting = true;
        app.running_prompt = Some(String::from("inspect the project"));
        app.reasoning_text = String::from("Reasoning");
        app.streaming_text = String::from("draft answer");
        app.activity.push(ActivityLine {
            kind: String::from("tool"),
            text: String::from("read_path read a private implementation file"),
        });
        let mut snapshot = app.snapshot.clone();
        snapshot.messages = vec![
            BridgeMessage {
                role: String::from("user"),
                content: String::from("inspect the project"),
                created_at: String::from("now"),
            },
            BridgeMessage {
                role: String::from("assistant"),
                content: String::from("Here is the result."),
                created_at: String::from("now"),
            },
        ];

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(BridgeStreamFrame {
                kind: String::from("final"),
                prompt: None,
                answer: Some(String::from("Here is the result.")),
                error: None,
                event: None,
                snapshot: Some(snapshot),
                command_result: None,
            })),
        );

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));
        assert!(!app.submitting);
        assert!(app.activity.is_empty());
        assert!(app.reasoning_text.is_empty());
        assert!(app.streaming_text.is_empty());
        assert_eq!(app.status, "Ready");
        assert!(rendered.contains("Here is the result."));
        assert!(!rendered.contains("read_path read"));
        assert!(!rendered.contains("draft answer"));
    }

    #[test]
    fn successful_configured_turn_clears_stale_error_notice() {
        let mut app = test_app();
        push_notice(
            &mut app,
            "Request failed",
            "OpenAI is not configured. Set OPENAI_API_KEY.",
            true,
        );
        app.running_prompt = Some(String::from("hello"));

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(BridgeStreamFrame {
                kind: String::from("final"),
                prompt: None,
                answer: Some(String::from("Hi!")),
                error: None,
                event: None,
                snapshot: None,
                command_result: None,
            })),
        );

        assert!(app.notices.is_empty());
        assert_eq!(app.status, "Ready");
    }

    #[test]
    fn final_frame_without_fresh_snapshot_still_renders_prompt_and_answer() {
        let mut app = test_app();
        app.submitting = true;
        app.running_prompt = Some(String::from("project/"));
        app.streaming_text = String::from("Tell me the exact folder name or path.");

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(BridgeStreamFrame {
                kind: String::from("final"),
                prompt: None,
                answer: Some(String::from("Tell me the exact folder name or path.")),
                error: None,
                event: None,
                snapshot: None,
                command_result: None,
            })),
        );

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));
        assert!(!app.submitting);
        assert_eq!(app.status, "Ready");
        assert!(rendered.contains("project/"));
        assert!(rendered.contains("Tell me the exact folder name or path."));
        assert!(!rendered.contains("drafting"));
    }

    #[test]
    fn non_stream_bridge_error_json_is_treated_as_final_frame() {
        let frame =
            parse_bridge_stream_line(r#"{"ok":false,"error":"Bridge exited before startup"}"#)
                .expect("fallback final frame");

        assert_eq!(frame.kind, "final");
        assert_eq!(frame.error.as_deref(), Some("Bridge exited before startup"));
    }

    #[test]
    fn non_stream_bridge_success_json_preserves_answer() {
        let frame = parse_bridge_stream_line(r#"{"ok":true,"answer":"local command answer"}"#)
            .expect("fallback final frame");

        assert_eq!(frame.kind, "final");
        assert_eq!(frame.answer.as_deref(), Some("local command answer"));
        assert!(frame.error.is_none());
    }

    #[test]
    fn wide_layout_keeps_live_tool_trace_in_activity_panel_only() {
        let mut app = test_app();
        app.submitting = true;
        app.running_prompt = Some(String::from("inspect the project"));
        app.activity.push(ActivityLine {
            kind: String::from("tool"),
            text: String::from("Reading files"),
        });

        let wide_transcript = rendered_text(&transcript_text(&app.snapshot, &app, false));
        let narrow_transcript = rendered_text(&transcript_text(&app.snapshot, &app, true));
        let activity_panel = rendered_text(&activity_panel_text(&app));

        assert!(!wide_transcript.contains("Reading files"));
        assert!(narrow_transcript.contains("Reading files"));
        assert!(activity_panel.contains("Reading files"));
    }

    #[test]
    fn gateway_overlay_renders_every_control_plane_tab() {
        let snapshot = gateway_snapshot();
        let expected = [
            "agent-route-v1:test",
            "agent-route-v1:test",
            "Routing bindings",
            "Channel accounts",
            "Gateway test",
            "gateway.status",
        ];

        for (tab, expected_text) in expected.iter().enumerate() {
            let view = GatewayViewState {
                snapshot: snapshot.clone(),
                tab,
                scroll: 0,
            };
            let rendered = rendered_text(&gateway_tab_text(&view));
            assert!(
                rendered.contains(expected_text),
                "tab {tab} should contain {expected_text}: {rendered}"
            );
        }
    }

    #[test]
    fn typed_command_result_carries_exact_secret_provider() {
        let result = BridgeCommandResult {
            code: BridgeCommandCode::ApiKeyRequired,
            setup_required: true,
            error: false,
            secret_provider: Some(String::from("openai")),
            next_command: None,
        };

        assert_eq!(result.secret_provider.as_deref(), Some("openai"));
    }

    #[test]
    fn compatible_provider_endpoint_requirement_is_not_an_api_key_requirement() {
        let mut session = anthropic_session();
        session.provider = String::from("openai-compatible");
        session.configuration = String::from(
            "OpenAI-compatible provider is not configured. Set AGENT_OPENAI_COMPAT_BASE_URL.",
        );
        session.configuration_state = String::from("endpoint_required");
        assert!(!configuration_needs_api_key(&session));
    }

    #[test]
    fn model_palette_window_follows_selection_below_first_page() {
        assert_eq!(palette_visible_start(30, 0, 8), 0);
        assert_eq!(palette_visible_start(30, 12, 8), 5);
        assert_eq!(palette_visible_start(30, 29, 8), 22);
    }

    #[test]
    fn model_palette_title_shows_visible_range_and_navigation() {
        let mut palette = BridgeCompletions {
            title: String::from("Models"),
            selected_index: None,
            entries: Vec::new(),
        };
        for index in 0..30 {
            palette.entries.push(BridgeCompletionEntry {
                value: format!("model-{index}"),
                label: format!("model-{index}"),
                description: String::new(),
                complete_to: format!("/model provider model-{index}"),
                execute: true,
            });
        }

        let title = palette_title(&palette, 12, 8);

        assert!(title.contains("Models · 6-13/30"));
        assert!(title.contains("PgUp/PgDn"));
        assert!(title.contains("wheel"));
    }

    #[test]
    fn palette_selection_skips_section_rows() {
        let palette = BridgeCompletions {
            title: String::from("Models"),
            selected_index: Some(0),
            entries: vec![
                test_palette_entry("section:ready", false),
                test_palette_entry("first", true),
                test_palette_entry("second", true),
                test_palette_entry("section:needs-setup", false),
                test_palette_entry("third", true),
            ],
        };

        assert_eq!(closest_selectable_palette_index(&palette, 0), Some(1));
        assert_eq!(previous_palette_index(&palette, 4), 2);
        assert_eq!(next_palette_index(&palette, 2), 4);
        assert_eq!(move_palette_index(&palette, 0, 3), 4);
        assert_eq!(last_selectable_palette_index(&palette), Some(4));
    }

    #[test]
    fn palette_selection_keeps_submenu_commands_selectable() {
        let palette = BridgeCompletions {
            title: String::from("Commands"),
            selected_index: Some(0),
            entries: vec![
                test_palette_entry("/model", false),
                test_palette_entry("/install", false),
                test_palette_entry("/reasoning", false),
                test_palette_entry("/tools", true),
            ],
        };

        assert_eq!(closest_selectable_palette_index(&palette, 0), Some(0));
        assert_eq!(next_palette_index(&palette, 0), 1);
        assert_eq!(next_palette_index(&palette, 1), 2);
        assert_eq!(next_palette_index(&palette, 2), 3);
        assert_eq!(previous_palette_index(&palette, 3), 2);
    }

    #[test]
    fn mouse_wheel_moves_model_picker_selection() {
        let mut app = test_app();
        app.input = String::from("/model ");
        app.palette.title = String::from("Models");
        for index in 0..10 {
            app.palette.entries.push(BridgeCompletionEntry {
                value: format!("model-{index}"),
                label: format!("model-{index}"),
                description: String::new(),
                complete_to: format!("/model provider model-{index}"),
                execute: true,
            });
        }

        scroll_active_view(&mut app, true);
        assert_eq!(app.palette_selected, 3);
        scroll_active_view(&mut app, false);
        assert_eq!(app.palette_selected, 0);
    }

    #[test]
    fn approval_controls_take_priority_while_agent_is_waiting() {
        assert_eq!(
            footer_help_text(false, true, true),
            "Enter/Y approve  N/Esc deny  Ctrl+N/P select"
        );
    }

    #[test]
    fn approval_target_sanitizes_encoded_windows_app_ids() {
        assert_eq!(
            sanitize_approval_target("desktop launch_application windows-app:abcdef Vitelglobal")
                .as_deref(),
            Some("desktop launch_application Vitelglobal")
        );
        assert_eq!(
            sanitize_approval_target("desktop launch_application windows-app:abcdef").as_deref(),
            Some("desktop launch_application selected app")
        );
    }

    #[test]
    fn approval_target_hides_raw_window_ids() {
        assert_eq!(
            sanitize_approval_target("desktop close_window 0x800e8").as_deref(),
            Some("desktop close_window selected window")
        );
    }

    #[test]
    fn transcript_scroll_counts_visual_wrapped_lines() {
        let long_device_row = "connected-device ".repeat(30);
        let text = Text::from(vec![
            Line::from(long_device_row),
            Line::from("Capabilities response below the device inventory"),
        ]);

        let max_scroll = transcript_max_scroll(&text, 24, 4);

        assert!(max_scroll > 0);
        assert!(max_scroll as usize > text.lines.len().saturating_sub(4));
    }

    #[test]
    fn typed_local_install_result_keeps_ui_in_setup_state() {
        let result = BridgeCommandResult {
            code: BridgeCommandCode::InstallUnverified,
            setup_required: true,
            error: true,
            secret_provider: None,
            next_command: None,
        };

        assert!(result.setup_required);
        assert_eq!(
            command_result_status(&result),
            "Error — local model installation failed"
        );
    }

    #[test]
    fn install_preview_uses_typed_next_command() {
        let result = BridgeCommandResult {
            code: BridgeCommandCode::InstallConfirmationRequired,
            setup_required: false,
            error: false,
            secret_provider: None,
            next_command: Some(String::from("/install ollama qwen3 --yes")),
        };

        assert_eq!(
            result.next_command.as_deref(),
            Some("/install ollama qwen3 --yes")
        );
    }

    #[test]
    fn unconfirmed_install_is_presented_as_preview_not_download() {
        let mut app = test_app();
        app.submitting = true;

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(BridgeStreamFrame {
                kind: String::from("submitted"),
                prompt: Some(String::from("/install ollama qwen3")),
                answer: None,
                error: None,
                event: None,
                snapshot: None,
                command_result: None,
            })),
        );

        assert_eq!(app.status, "Preparing local model install preview");
        assert_eq!(
            app.activity.last().map(|item| item.text.as_str()),
            Some("Reviewing model size and runtime requirements")
        );
    }

    #[test]
    fn completed_install_preview_is_ready_for_one_key_confirmation() {
        let mut app = test_app();
        app.submitting = true;
        app.running_prompt = Some(String::from("/install ollama qwen3"));
        let answer = "Local model install preview\nDownload: ~5.2 GB\n\
                      Confirm download: /install ollama qwen3 --yes\n\
                      Nothing has been downloaded yet.";

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(BridgeStreamFrame {
                kind: String::from("final"),
                prompt: None,
                answer: Some(answer.to_string()),
                error: None,
                event: None,
                snapshot: None,
                command_result: Some(BridgeCommandResult {
                    code: BridgeCommandCode::InstallConfirmationRequired,
                    setup_required: false,
                    error: false,
                    secret_provider: None,
                    next_command: Some(String::from("/install ollama qwen3 --yes")),
                }),
            })),
        );

        assert!(!app.submitting);
        assert_eq!(app.input, "/install ollama qwen3 --yes");
        assert!(app.status.contains("press Enter to install"));
    }
}
