use anyhow::Result;
mod host;
mod math_render;
mod session_store;

use agent_rust::{
    compact_whitespace, delete_path, edit_file, glob_files, grep_files, inspect_tree, read_path,
    resolve_search_roots, resolve_target, search_files, system_search_roots, write_file,
    DeletePathOptions, EditFileOptions, FileSearchOptions, GlobKind, GlobOptions, GrepOptions,
    InspectTreeOptions, ReadLimits, ReadPathOptions, ResolveTargetOptions, SearchKind, SearchMode,
    SearchStrategy, TargetKind, WriteFileOptions,
};
use clap::{CommandFactory, FromArgMatches, Parser, Subcommand};
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event, KeyCode, KeyEventKind, KeyModifiers, MouseEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use host::{
    connected_devices, desktop_action, desktop_capabilities, desktop_clipboard_files,
    desktop_clipboard_image_to_file, desktop_clipboard_read_text, desktop_observe,
    desktop_open_user_file, desktop_pick_file, desktop_resolve, desktop_screenshot, process_list,
    run_system_command, system_info,
};
#[cfg(test)]
use host::{valid_bluetooth_address, valid_identifier, valid_path_token, windows_device_category};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, WidgetRef, Wrap};
use ratatui::Terminal;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::json;
use serde_json::Value;
use session_store::{SessionStoreCall, SessionStoreRegistry};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, Read};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader as AsyncBufReader};
use tokio::process::{Child as AsyncChild, Command as AsyncProcessCommand};
use tokio::runtime::Handle as RuntimeHandle;
use tokio::sync::{mpsc as async_mpsc, watch, Semaphore};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

#[derive(Debug, Parser)]
#[command(name = "agent-rust")]
#[command(about = " tools for Agent")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
#[allow(clippy::enum_variant_names)] // Renaming changes the public command contract.
enum Command {
    Serve(ServeArgs),
    Search(SearchArgs),
    Glob(GlobArgs),
    Grep(GrepArgs),
    InspectTarget(InspectTargetArgs),
    InspectTree(InspectTreeArgs),
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
    DesktopScreenshot(DesktopScreenshotArgs),
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
    #[serde(skip_serializing_if = "Option::is_none")]
    error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    retryable: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Value>,
}

impl WorkerResponse {
    fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            id,
            ok: true,
            result: Some(result),
            error: None,
            error_code: None,
            retryable: None,
            details: None,
        }
    }

    fn failure(id: Option<Value>, error: impl ToString) -> Self {
        Self {
            id,
            ok: false,
            result: None,
            error: Some(error.to_string()),
            error_code: None,
            retryable: None,
            details: None,
        }
    }

    fn store_failure(id: Option<Value>, error: session_store::StoreError) -> Self {
        let session_store::StoreError {
            code,
            message,
            retryable,
            details,
        } = error;
        let error_code = serde_json::to_value(code)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned));
        Self {
            id,
            ok: false,
            result: None,
            error: Some(message),
            error_code,
            retryable: Some(retryable),
            details,
        }
    }
}

thread_local! {
    static WORKER_CLI: RefCell<clap::Command> = RefCell::new(Cli::command());
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
struct DesktopScreenshotArgs {}

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

#[derive(Debug, Parser)]
struct InspectTreeArgs {
    path: PathBuf,
    #[arg(long)]
    workspace_root: PathBuf,
    #[arg(long, default_value_t = 25)]
    max_files: usize,
    #[arg(long, default_value_t = 1_000)]
    max_entries: usize,
    #[arg(long, default_value_t = 12_000)]
    max_bytes_per_file: usize,
    #[arg(long, default_value_t = 80_000)]
    max_total_bytes: usize,
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

    #[arg(long = "paste-key", hide = true)]
    paste_keys: Vec<String>,

    #[arg(long = "copy-key", hide = true)]
    copy_keys: Vec<String>,

    #[arg(long, hide = true)]
    mouse_capture: bool,

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
    #[serde(default)]
    transient: bool,
    #[serde(default)]
    pending_action: Option<BridgePendingAction>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
struct BridgePendingAction {
    kind: BridgePendingActionKind,
    label: String,
    command: String,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum BridgePendingActionKind {
    InstallModel,
    #[serde(other)]
    Unknown,
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
    Incompatible,
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
    #[serde(default = "default_agent_name")]
    agent_name: String,
    #[serde(default)]
    approvals: Vec<BridgeApproval>,
    #[serde(default)]
    voice: BridgeVoice,
    messages: Vec<BridgeMessage>,
}

fn default_agent_name() -> String {
    String::from("Agent")
}

#[derive(Debug, Clone, Deserialize, Default)]
struct BridgeVoice {
    #[serde(default)]
    input_ready: bool,
    #[serde(default)]
    input_reason: Option<String>,
    #[serde(default)]
    input_secret_provider: Option<String>,
    #[serde(default)]
    tts_ready: bool,
    #[serde(default)]
    auto_speak: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeSession {
    id: String,
    title: String,
    workspace_root: String,
    provider: String,
    model: String,
    mode: String,
    configuration: String,
    #[serde(default = "ready_configuration_state")]
    configuration_state: String,
    #[serde(default = "default_true")]
    model_selected: bool,
    #[serde(default, rename = "context_limit")]
    _context_limit: Option<i64>,
    #[serde(default)]
    pending_attachments: Vec<BridgeAttachment>,
    #[serde(rename = "tokens")]
    tokens: BridgeTokens,
    #[serde(default)]
    cost_usd: f64,
    #[serde(default)]
    costs: BridgeCosts,
}

fn ready_configuration_state() -> String {
    String::from("ready")
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeTokens {
    #[serde(rename = "input")]
    input: i64,
    #[serde(rename = "output")]
    output: i64,
    #[serde(default, rename = "reasoning")]
    reasoning: i64,
    #[serde(default, rename = "cache_read")]
    cache_read: i64,
    #[serde(default, rename = "cache_write")]
    cache_write: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct BridgeCosts {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    cached_input: f64,
    #[serde(default)]
    cache_write: f64,
    #[serde(default)]
    output: f64,
}

impl BridgeCosts {
    fn input_total(&self) -> f64 {
        self.input + self.cached_input + self.cache_write
    }

    fn total(&self) -> f64 {
        self.input_total() + self.output
    }
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeMessage {
    role: String,
    content: String,
    created_at: String,
    #[serde(default)]
    attachments: Vec<BridgeAttachment>,
}

#[derive(Debug, Clone, Deserialize)]
struct BridgeAttachment {
    #[serde(default)]
    id: Option<String>,
    filename: String,
    mime: String,
    size_bytes: i64,
    #[serde(default)]
    storage_path: Option<String>,
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

#[derive(Debug, Clone, Deserialize)]
struct BridgeVoiceFrame {
    kind: String,
    #[serde(default)]
    delta: Option<String>,
    #[serde(default)]
    transcript: Option<String>,
    #[serde(default)]
    error: Option<String>,
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

#[derive(Debug, Clone)]
struct UiSetupSurface {
    title: String,
    text: String,
    error: bool,
    pending_action: Option<BridgePendingAction>,
}

#[derive(Debug, Clone)]
struct AttachmentHitArea {
    area: Rect,
    remove_area: Option<Rect>,
    attachment_id: Option<String>,
    filename: String,
    mime: String,
    size_bytes: i64,
    storage_path: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct PaletteHitArea {
    area: Rect,
    index: usize,
}

#[derive(Debug, Clone)]
struct AttachmentPreview {
    filename: String,
    mime: String,
    size_bytes: i64,
    storage_path: String,
    text: Option<String>,
    truncated: bool,
    scroll: u16,
}

#[derive(Debug)]
enum AppEvent {
    StreamFrame(Result<Box<BridgeStreamFrame>>),
    VoiceFrame {
        session_id: u64,
        result: Result<BridgeVoiceFrame>,
    },
    AttachmentMutation {
        operation: AttachmentMutation,
        result: Box<Result<BridgeResponse>>,
    },
}

#[derive(Debug, Clone)]
enum AttachmentMutation {
    Add { filename: String },
    Remove { filename: String },
}

struct TranscriptCache {
    show_inline_activity: bool,
    area: Rect,
    auto_follow: bool,
    requested_scroll: u16,
    paragraph: Paragraph<'static>,
    attachment_hit_areas: Vec<AttachmentHitArea>,
    selection_lines: Vec<String>,
}

struct TuiApp {
    snapshot: BridgeSnapshot,
    input: String,
    attachment_path_mode: bool,
    attachment_button_area: Option<Rect>,
    mic_button_area: Option<Rect>,
    cost_button_area: Option<Rect>,
    pending_action_area: Option<Rect>,
    attachment_hit_areas: Vec<AttachmentHitArea>,
    palette_hit_areas: Vec<PaletteHitArea>,
    attachment_preview: Option<AttachmentPreview>,
    status: String,
    scroll: u16,
    auto_follow: bool,
    submitting: bool,
    cancel_requested: bool,
    cancel_signal: Arc<AtomicBool>,
    active_bridge: Arc<Mutex<Option<AsyncChild>>>,
    queued_prompts: VecDeque<String>,
    activity: Vec<ActivityLine>,
    subagent_run: Option<SubagentRunState>,
    palette: BridgeCompletions,
    palette_source: Option<Arc<str>>,
    palette_selected: usize,
    transcript_cache: Option<TranscriptCache>,
    approval_selected: usize,
    current_tool: Option<String>,
    reasoning_text: String,
    streaming_text: String,
    running_prompt: Option<String>,
    secret_provider: Option<String>,
    secret_input: String,
    notices: VecDeque<UiNotice>,
    setup_required: bool,
    setup_surface: Option<UiSetupSurface>,
    gateway_view: Option<GatewayViewState>,
    show_cost_details: bool,
    voice_recording: bool,
    voice_session_id: u64,
    voice_cancel_signal: Option<Arc<AtomicBool>>,
    active_voice_process: Arc<Mutex<Option<u32>>>,
    voice_input_prefix: Option<String>,
    voice_partial: String,
    paste_keys: Vec<PasteKey>,
    copy_keys: Vec<PasteKey>,
    mouse_capture: bool,
    transcript_drag_start: Option<SelectionPoint>,
    transcript_selection: Option<TranscriptSelection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SelectionPoint {
    row: usize,
    col: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TranscriptSelection {
    start: SelectionPoint,
    end: SelectionPoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CtrlCAction {
    CopySelection,
    StopTask,
    Exit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PasteKey {
    code: PasteKeyCode,
    modifiers: KeyModifiers,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PasteKeyCode {
    Char(char),
    Insert,
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
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    #[cfg(feature = "tokio-console")]
    {
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry()
            .with(console_subscriber::spawn())
            .init();
    }

    let cli = Cli::parse();

    match cli.command {
        Command::Serve(_args) => run_worker().await?,
        Command::Tui(args) => run_tui(args).await?,
        command => {
            println!("{}", serde_json::to_string(&run_command(command)?)?);
        }
    }

    Ok(())
}

#[cfg_attr(
    feature = "tokio-console",
    tracing::instrument(name = "worker.run", skip_all)
)]
async fn run_worker() -> Result<()> {
    let mut lines = AsyncBufReader::new(tokio::io::stdin()).lines();
    let session_stores = Arc::new(SessionStoreRegistry::new());
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
        let raw_request = match serde_json::from_str::<Value>(&line) {
            Ok(request) => request,
            Err(error) => {
                response_tx
                    .send(WorkerResponse::failure(None, error))
                    .await
                    .map_err(|_| anyhow::anyhow!("worker response channel closed"))?;
                continue;
            }
        };
        let request_id = raw_request.get("id").cloned();
        if raw_request.get("service").and_then(Value::as_str) == Some("session_store") {
            let call = match serde_json::from_value::<SessionStoreCall>(raw_request) {
                Ok(call) => call,
                Err(error) => {
                    response_tx
                        .send(WorkerResponse::failure(request_id, error))
                        .await
                        .map_err(|_| anyhow::anyhow!("worker response channel closed"))?;
                    continue;
                }
            };
            let response_tx = response_tx.clone();
            let permit = Arc::clone(&permits).acquire_owned().await?;
            let session_stores = Arc::clone(&session_stores);
            tokio::spawn(async move {
                let response = match session_stores.handle(call).await {
                    Ok(result) => WorkerResponse::success(request_id, result),
                    Err(error) => WorkerResponse::store_failure(request_id, error),
                };
                let _permit = permit;
                let _ = response_tx.send(response).await;
            });
            continue;
        }
        let request = match serde_json::from_value::<WorkerRequest>(raw_request) {
            Ok(request) => request,
            Err(error) => {
                response_tx
                    .send(WorkerResponse::failure(request_id, error))
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
                    Err(error) => WorkerResponse::failure(request_id, error),
                };
            let _permit = permit;
            let _ = response_tx.send(response).await;
        });
    }

    drop(response_tx);
    writer.await??;
    session_stores.close().await;
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

    #[test]
    fn worker_parser_recovers_after_invalid_request() {
        let invalid = run_worker_request(WorkerRequest {
            id: None,
            args: vec![String::from("not-a-command")],
        });
        let valid = run_worker_request(WorkerRequest {
            id: None,
            args: vec![String::from("system-info")],
        });

        assert!(!invalid.ok);
        assert!(valid.ok);
    }
}

#[cfg_attr(
    feature = "tokio-console",
    tracing::instrument(name = "worker.request", skip_all)
)]
fn run_worker_request(request: WorkerRequest) -> WorkerResponse {
    let mut argv = Vec::with_capacity(request.args.len() + 1);
    argv.push("agent-rust".to_string());
    argv.extend(request.args);

    let result = WORKER_CLI
        .with_borrow_mut(|command| {
            let mut matches = command.try_get_matches_from_mut(argv)?;
            Cli::from_arg_matches_mut(&mut matches)
        })
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .and_then(|cli| match cli.command {
            Command::Serve(_) | Command::Tui(_) => Err(anyhow::anyhow!(
                "command is not supported by the JSON worker"
            )),
            command => run_command(command),
        });

    match result {
        Ok(value) => WorkerResponse::success(request.id, value),
        Err(error) => WorkerResponse::failure(request.id, error),
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
        Command::DesktopScreenshot(_) => Ok(serde_json::to_value(desktop_screenshot()?)?),
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
        Command::InspectTree(args) => {
            Ok(serde_json::to_value(inspect_tree(InspectTreeOptions {
                root: args.path,
                workspace_root: args.workspace_root,
                max_files: args.max_files.clamp(1, 300),
                max_entries: args.max_entries.clamp(10, 5_000),
                max_bytes_per_file: args.max_bytes_per_file.clamp(1_000, 200_000),
                max_total_bytes: args.max_total_bytes.clamp(10_000, 800_000),
            })?)?)
        }
    }
}

async fn run_tui(args: TuiArgs) -> Result<()> {
    let runtime = RuntimeHandle::current();
    tokio::task::block_in_place(move || run_tui_blocking(Arc::new(args), runtime))
}

fn run_tui_blocking(args: Arc<TuiArgs>, runtime: RuntimeHandle) -> Result<()> {
    let initial = call_bridge(args.as_ref(), "snapshot", None)?;
    let snapshot = initial
        .snapshot
        .ok_or_else(|| anyhow::anyhow!("Bridge did not return a snapshot."))?;
    let initial_needs_setup = snapshot.session.configuration_state != "ready";
    let initial_secret_provider = None;
    let initial_status = if !snapshot.session.model_selected {
        String::from("Choose a model to start")
    } else if initial_needs_setup {
        format!(
            "{} needs setup",
            provider_display_name(&snapshot.session.provider)
        )
    } else {
        String::from("Ready")
    };
    let initial_setup_surface = session_setup_surface(&snapshot.session);

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;
    if args.mouse_capture {
        execute!(stdout, EnableMouseCapture)?;
    }
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let paste_keys = parse_paste_keys(&args.paste_keys);
    let copy_keys = parse_paste_keys(&args.copy_keys);
    let mouse_capture = args.mouse_capture;

    let result = run_tui_loop(
        &mut terminal,
        args,
        runtime,
        TuiApp {
            snapshot,
            input: String::new(),
            attachment_path_mode: false,
            attachment_button_area: None,
            mic_button_area: None,
            cost_button_area: None,
            pending_action_area: None,
            attachment_hit_areas: Vec::new(),
            palette_hit_areas: Vec::new(),
            attachment_preview: None,
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
            palette_source: None,
            palette_selected: 0,
            transcript_cache: None,
            approval_selected: 0,
            current_tool: None,
            reasoning_text: String::new(),
            streaming_text: String::new(),
            running_prompt: None,
            secret_provider: initial_secret_provider,
            secret_input: String::new(),
            notices: VecDeque::new(),
            setup_required: initial_needs_setup,
            setup_surface: initial_setup_surface,
            gateway_view: None,
            show_cost_details: false,
            voice_recording: false,
            voice_session_id: 0,
            voice_cancel_signal: None,
            active_voice_process: Arc::new(Mutex::new(None)),
            voice_input_prefix: None,
            voice_partial: String::new(),
            paste_keys,
            copy_keys,
            mouse_capture,
            transcript_drag_start: None,
            transcript_selection: None,
        },
    );

    disable_raw_mode()?;
    if mouse_capture {
        execute!(terminal.backend_mut(), DisableMouseCapture)?;
    }
    execute!(
        terminal.backend_mut(),
        DisableBracketedPaste,
        LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;
    result
}

fn run_tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    args: Arc<TuiArgs>,
    runtime: RuntimeHandle,
    mut app: TuiApp,
) -> Result<()> {
    let (tx, rx) = mpsc::channel::<AppEvent>();
    let (palette_tx, palette_rx) = watch::channel(None::<Arc<str>>);
    let (palette_result_tx, palette_result_rx) =
        mpsc::channel::<(Arc<str>, Result<BridgeResponse>)>();
    spawn_palette_worker(&runtime, Arc::clone(&args), palette_rx, palette_result_tx);
    request_palette_refresh(&palette_tx, &mut app);
    let mut needs_redraw = true;

    loop {
        while let Ok(event) = rx.try_recv() {
            handle_app_event(&mut app, event);
            needs_redraw = true;
        }
        while let Ok((prompt, result)) = palette_result_rx.try_recv() {
            apply_palette_result(&mut app, &prompt, result);
            needs_redraw = true;
        }

        if !app.submitting
            && !bridge_process_active(&app)
            && app.secret_provider.is_none()
            && app.snapshot.approvals.is_empty()
        {
            if let Some(prompt) = app.queued_prompts.pop_front() {
                start_prompt_submission(&runtime, &args, &tx, &mut app, prompt);
                needs_redraw = true;
            }
        }

        if needs_redraw {
            terminal.draw(|frame| draw_app(frame, &mut app))?;
            needs_redraw = false;
        }

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }

        let key = match event::read()? {
            Event::Key(key) => key,
            Event::Mouse(mouse) => {
                match mouse.kind {
                    MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if !app.submitting
                            && app.pending_action_area.is_some_and(|area| {
                                mouse_in_rect(mouse.column, mouse.row, area)
                            }) =>
                    {
                        submit_pending_action(&runtime, &args, &tx, &mut app);
                    }
                    MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if app.snapshot.approvals.is_empty()
                            && app.cost_button_area.is_some_and(|area| {
                                mouse_in_rect(mouse.column, mouse.row, area)
                            }) =>
                    {
                        app.show_cost_details = !app.show_cost_details;
                    }
                    _ if app.show_cost_details => {}
                    MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if app.secret_provider.is_none()
                            && app.snapshot.approvals.is_empty()
                            && app.attachment_preview.is_none()
                            && app.gateway_view.is_none()
                            && palette_index_for_mouse(&app, mouse.column, mouse.row).is_some() =>
                    {
                        let index = palette_index_for_mouse(&app, mouse.column, mouse.row)
                            .expect("palette click guard resolved an entry");
                        activate_palette_mouse_entry(
                            &runtime,
                            &args,
                            &tx,
                            &palette_tx,
                            &mut app,
                            index,
                        );
                    }
                    MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if app.snapshot.approvals.is_empty()
                            && app.attachment_preview.is_none()
                            && app.gateway_view.is_none()
                            && transcript_point_for_mouse(&app, mouse.column, mouse.row)
                                .is_some()
                            && !app.attachment_hit_areas.iter().any(|target| {
                                mouse_in_rect(mouse.column, mouse.row, target.area)
                            }) =>
                    {
                        start_transcript_selection(&mut app, mouse.column, mouse.row);
                    }
                    MouseEventKind::Drag(crossterm::event::MouseButton::Left)
                        if app.transcript_drag_start.is_some() =>
                    {
                        update_transcript_selection(&mut app, mouse.column, mouse.row);
                    }
                    MouseEventKind::Up(crossterm::event::MouseButton::Left)
                        if app.transcript_drag_start.is_some() =>
                    {
                        update_transcript_selection(&mut app, mouse.column, mouse.row);
                        finish_transcript_selection(&mut app);
                    }
                    MouseEventKind::ScrollUp => scroll_active_view(&mut app, false),
                    MouseEventKind::ScrollDown => scroll_active_view(&mut app, true),
                    MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if app.secret_provider.is_none()
                            && app.snapshot.approvals.is_empty()
                            && app.attachment_preview.is_none()
                            && app.attachment_hit_areas.iter().any(|target| {
                                mouse_in_rect(mouse.column, mouse.row, target.area)
                            }) =>
                    {
                        activate_clicked_attachment(
                            &runtime,
                            &args,
                            &tx,
                            &mut app,
                            mouse.column,
                            mouse.row,
                        );
                    }
                    MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if app.secret_provider.is_none()
                            && app.snapshot.approvals.is_empty()
                            && app.attachment_preview.is_none()
                            && app.attachment_button_area.is_some_and(|area| {
                                mouse_in_rect(mouse.column, mouse.row, area)
                            }) =>
                    {
                        open_attachment_picker(&runtime, &args, &tx, &mut app);
                    }
                    MouseEventKind::Down(crossterm::event::MouseButton::Left)
                        if app.secret_provider.is_none()
                            && app.snapshot.approvals.is_empty()
                            && app.attachment_preview.is_none()
                            && app.mic_button_area.is_some_and(|area| {
                                mouse_in_rect(mouse.column, mouse.row, area)
                            }) =>
                    {
                        start_voice_recording(&runtime, &args, &tx, &mut app);
                    }
                    _ => {}
                }
                needs_redraw = true;
                continue;
            }
            Event::Paste(text) => {
                insert_composer_paste(&mut app, &text);
                needs_redraw = true;
                continue;
            }
            Event::Resize(_, _) => {
                needs_redraw = true;
                continue;
            }
            _ => continue,
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        needs_redraw = true;

        if app.show_cost_details && app.snapshot.approvals.is_empty() {
            if matches!(key.code, KeyCode::Esc | KeyCode::Char('q') | KeyCode::Enter) {
                app.show_cost_details = false;
                app.status = String::from("Ready");
            }
            continue;
        }

        if app.attachment_preview.is_some() && app.snapshot.approvals.is_empty() {
            match key.code {
                KeyCode::Esc | KeyCode::Char('q') => {
                    app.attachment_preview = None;
                    app.status = String::from("Ready");
                }
                KeyCode::Enter | KeyCode::Char('o') => open_preview_in_system_viewer(&mut app),
                KeyCode::Down | KeyCode::Char('j') => {
                    if let Some(preview) = app.attachment_preview.as_mut() {
                        preview.scroll = preview.scroll.saturating_add(1);
                    }
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    if let Some(preview) = app.attachment_preview.as_mut() {
                        preview.scroll = preview.scroll.saturating_sub(1);
                    }
                }
                KeyCode::PageDown => {
                    if let Some(preview) = app.attachment_preview.as_mut() {
                        preview.scroll = preview.scroll.saturating_add(10);
                    }
                }
                KeyCode::PageUp => {
                    if let Some(preview) = app.attachment_preview.as_mut() {
                        preview.scroll = preview.scroll.saturating_sub(10);
                    }
                }
                KeyCode::Home => {
                    if let Some(preview) = app.attachment_preview.as_mut() {
                        preview.scroll = 0;
                    }
                }
                _ => {}
            }
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
                KeyCode::Char('r') => open_gateway_view(args.as_ref(), &mut app),
                _ => {}
            }
            continue;
        }

        match key.code {
            KeyCode::Esc => {
                if app
                    .setup_surface
                    .as_ref()
                    .is_some_and(|surface| surface.pending_action.is_some())
                {
                    app.setup_surface = session_setup_surface(&app.snapshot.session);
                    app.pending_action_area = None;
                    app.status = String::from("Local model installation cancelled");
                    continue;
                }
                if app.secret_provider.is_some() {
                    app.secret_provider = None;
                    app.secret_input.clear();
                    app.status = String::from("API key entry cancelled");
                    continue;
                }
                if app.attachment_path_mode {
                    app.attachment_path_mode = false;
                    app.input.clear();
                    app.status = String::from("Attachment cancelled");
                    continue;
                }
                if !app.snapshot.approvals.is_empty() {
                    apply_approval_action(&args, &mut app, "deny");
                    continue;
                }
                if app.voice_recording {
                    stop_voice_recording(&mut app);
                    continue;
                }
                if palette_is_open(&app) || !app.input.is_empty() {
                    app.input.clear();
                    clear_palette(&mut app);
                    continue;
                }
                if app.submitting {
                    request_active_stop(&mut app);
                    continue;
                }
                break;
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                match ctrl_c_action(&app) {
                    CtrlCAction::CopySelection => {
                        copy_transcript_selection_to_clipboard(&mut app);
                    }
                    CtrlCAction::StopTask => request_active_stop(&mut app),
                    CtrlCAction::Exit => break,
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
            KeyCode::F(4) if app.secret_provider.is_none() && app.snapshot.approvals.is_empty() => {
                open_attachment_picker(&runtime, &args, &tx, &mut app);
            }
            KeyCode::F(5)
                if app.secret_provider.is_none()
                    && app.snapshot.approvals.is_empty()
                    && !app.submitting =>
            {
                start_voice_recording(&runtime, &args, &tx, &mut app);
            }
            KeyCode::Char('o')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && app.secret_provider.is_none()
                    && app.snapshot.approvals.is_empty() =>
            {
                open_attachment_picker(&runtime, &args, &tx, &mut app);
            }
            code if is_clipboard_shortcut(&code, key.modifiers, &app.paste_keys) => {
                paste_clipboard_into_composer(&runtime, &args, &tx, &mut app);
            }
            code if is_clipboard_shortcut(&code, key.modifiers, &app.copy_keys) => {
                copy_latest_response_to_clipboard(&mut app);
            }
            KeyCode::Backspace => {
                if app.secret_provider.is_some() {
                    app.secret_input.pop();
                } else if app.input.is_empty()
                    && !app.submitting
                    && !bridge_process_active(&app)
                    && !app.attachment_path_mode
                    && !app.snapshot.session.pending_attachments.is_empty()
                {
                    remove_last_pending_attachment(&runtime, &args, &tx, &mut app);
                } else {
                    app.input.pop();
                    if !app.attachment_path_mode {
                        request_palette_refresh(&palette_tx, &mut app);
                    }
                }
            }
            KeyCode::Enter => {
                if app
                    .setup_surface
                    .as_ref()
                    .is_some_and(|surface| surface.pending_action.is_some())
                    && !app.submitting
                {
                    submit_pending_action(&runtime, &args, &tx, &mut app);
                    continue;
                }
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
                    invalidate_transcript(&mut app);
                    let tx_clone = tx.clone();
                    let err_tx = tx.clone();
                    let args_clone = args.clone();
                    let active_bridge = Arc::clone(&app.active_bridge);
                    let cancel_signal = Arc::clone(&app.cancel_signal);
                    runtime.spawn(async move {
                        let result = stream_bridge_submit(
                            args_clone.as_ref(),
                            &prompt,
                            tx_clone,
                            active_bridge,
                            cancel_signal,
                        )
                        .await;
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
                if app.attachment_path_mode {
                    let path = app.input.trim().to_string();
                    if path.is_empty() {
                        app.status = String::from("Attachment cancelled — no path entered");
                        app.attachment_path_mode = false;
                        continue;
                    }
                    app.attachment_path_mode = false;
                    app.input.clear();
                    submit_attachment_path(&runtime, &args, &tx, &mut app, path);
                    continue;
                }
                if palette_is_open(&app) {
                    let query = palette_context(&app.input)
                        .map(|context| context.query)
                        .unwrap_or_default();
                    app.palette_selected =
                        closest_selectable_palette_index(&app.palette, query, app.palette_selected)
                            .unwrap_or(0);
                    let selected = app.palette.entries.get(app.palette_selected).cloned();
                    if let Some(entry) = selected {
                        if entry.execute {
                            app.input = entry.complete_to;
                        } else {
                            app.input = entry.complete_to;
                            request_palette_refresh(&palette_tx, &mut app);
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
                    clear_palette(&mut app);
                    app.status = format!("Queued {} prompt(s)", queued_count);
                    continue;
                }
                if matches!(prompt.as_str(), "/exit" | "/quit" | "/q") {
                    break;
                }
                if prompt == "/gateway" {
                    app.input.clear();
                    clear_palette(&mut app);
                    open_gateway_view(args.as_ref(), &mut app);
                    continue;
                }
                if !prompt.starts_with('/') {
                    if !app.snapshot.session.model_selected {
                        app.setup_required = true;
                        app.setup_surface = session_setup_surface(&app.snapshot.session);
                        app.status = String::from("Choose a model before sending a message");
                        continue;
                    }
                    if let Some(provider) = required_text_api_key_provider(&app.snapshot.session) {
                        app.secret_provider = Some(provider.clone());
                        app.secret_input.clear();
                        app.status = api_key_prompt_status(&provider);
                        continue;
                    }
                }
                start_prompt_submission(&runtime, &args, &tx, &mut app, prompt);
            }
            KeyCode::Up => {
                if palette_is_open(&app) {
                    let query = palette_context(&app.input)
                        .map(|context| context.query)
                        .unwrap_or_default();
                    app.palette_selected =
                        previous_palette_index(&app.palette, query, app.palette_selected);
                } else {
                    app.auto_follow = false;
                    app.scroll = app.scroll.saturating_sub(1);
                }
            }
            KeyCode::Down => {
                if palette_is_open(&app) {
                    let query = palette_context(&app.input)
                        .map(|context| context.query)
                        .unwrap_or_default();
                    app.palette_selected =
                        next_palette_index(&app.palette, query, app.palette_selected);
                } else {
                    app.scroll = app.scroll.saturating_add(1);
                }
            }
            KeyCode::PageUp => {
                if palette_is_open(&app) {
                    let query = palette_context(&app.input)
                        .map(|context| context.query)
                        .unwrap_or_default();
                    app.palette_selected = move_palette_index(
                        &app.palette,
                        query,
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
                    let query = palette_context(&app.input)
                        .map(|context| context.query)
                        .unwrap_or_default();
                    app.palette_selected = move_palette_index(
                        &app.palette,
                        query,
                        app.palette_selected,
                        PALETTE_PAGE_SIZE as isize,
                    );
                } else {
                    app.scroll = app.scroll.saturating_add(10);
                }
            }
            KeyCode::Home => {
                if palette_is_open(&app) {
                    let query = palette_context(&app.input)
                        .map(|context| context.query)
                        .unwrap_or_default();
                    app.palette_selected =
                        first_selectable_palette_index(&app.palette, query).unwrap_or(0);
                } else {
                    app.scroll = 0;
                    app.auto_follow = false;
                }
            }
            KeyCode::End => {
                if palette_is_open(&app) {
                    let query = palette_context(&app.input)
                        .map(|context| context.query)
                        .unwrap_or_default();
                    app.palette_selected =
                        last_selectable_palette_index(&app.palette, query).unwrap_or(0);
                } else {
                    app.auto_follow = true;
                }
            }
            KeyCode::Tab => {
                if app.secret_provider.is_some() {
                    continue;
                }
                let query = palette_context(&app.input)
                    .map(|context| context.query)
                    .unwrap_or_default();
                app.palette_selected =
                    closest_selectable_palette_index(&app.palette, query, app.palette_selected)
                        .unwrap_or(0);
                if let Some(entry) = app.palette.entries.get(app.palette_selected) {
                    app.input = entry.complete_to.clone();
                    request_palette_refresh(&palette_tx, &mut app);
                }
            }
            KeyCode::Char(ch)
                if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT =>
            {
                if app.secret_provider.is_some() {
                    app.secret_input.push(ch);
                } else {
                    app.input.push(ch);
                    if !app.attachment_path_mode {
                        request_palette_refresh(&palette_tx, &mut app);
                    }
                }
            }
            _ => {}
        }
    }

    if app.submitting || bridge_process_active(&app) {
        request_active_stop(&mut app);
    }
    if app.voice_recording || voice_process_active(&app) {
        cancel_voice_transport(&mut app);
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

fn mouse_in_rect(column: u16, row: u16, area: Rect) -> bool {
    column >= area.x && column < area.right() && row >= area.y && row < area.bottom()
}

fn palette_index_for_mouse(app: &TuiApp, column: u16, row: u16) -> Option<usize> {
    app.palette_hit_areas
        .iter()
        .find(|target| mouse_in_rect(column, row, target.area))
        .map(|target| target.index)
}

fn activate_palette_mouse_entry(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    palette_tx: &watch::Sender<Option<Arc<str>>>,
    app: &mut TuiApp,
    index: usize,
) {
    let Some(entry) = app.palette.entries.get(index).cloned() else {
        return;
    };
    if !palette_entry_selectable(&entry) {
        return;
    }
    app.palette_selected = index;
    app.input = entry.complete_to;
    if !entry.execute {
        request_palette_refresh(palette_tx, app);
        return;
    }

    let prompt = app.input.trim().to_string();
    if app.submitting || bridge_process_active(app) {
        app.input.clear();
        let queued_count = enqueue_prompt(&mut app.queued_prompts, prompt);
        clear_palette(app);
        app.status = format!("Queued {queued_count} prompt(s)");
        return;
    }
    start_prompt_submission(runtime, args, tx, app, prompt);
}

fn transcript_point_for_mouse(app: &TuiApp, column: u16, row: u16) -> Option<SelectionPoint> {
    let cache = app.transcript_cache.as_ref()?;
    let inner_x = cache.area.x.saturating_add(1);
    let inner_y = cache.area.y.saturating_add(1);
    let inner_width = cache.area.width.saturating_sub(2);
    let inner_height = cache.area.height.saturating_sub(2);
    if column < inner_x
        || row < inner_y
        || column >= inner_x + inner_width
        || row >= inner_y + inner_height
    {
        return None;
    }
    Some(SelectionPoint {
        row: row.saturating_sub(inner_y) as usize,
        col: column.saturating_sub(inner_x) as usize,
    })
}

fn start_transcript_selection(app: &mut TuiApp, column: u16, row: u16) {
    let Some(point) = transcript_point_for_mouse(app, column, row) else {
        return;
    };
    app.transcript_drag_start = Some(point);
    app.transcript_selection = Some(TranscriptSelection {
        start: point,
        end: point,
    });
    app.status = String::from("Selecting transcript text...");
}

fn update_transcript_selection(app: &mut TuiApp, column: u16, row: u16) {
    let Some(start) = app.transcript_drag_start else {
        return;
    };
    let Some(end) = transcript_point_for_mouse(app, column, row) else {
        return;
    };
    app.transcript_selection = Some(TranscriptSelection { start, end });
}

fn finish_transcript_selection(app: &mut TuiApp) {
    app.transcript_drag_start = None;
    let selected = app
        .transcript_selection
        .and_then(|selection| selected_transcript_text(app, selection));
    if selected.is_some_and(|text| !text.trim().is_empty()) {
        app.status = String::from("Text selected · Ctrl+C copy");
    } else {
        app.transcript_selection = None;
        app.status = String::from("No transcript text selected");
    }
}

fn ctrl_c_action(app: &TuiApp) -> CtrlCAction {
    let has_selection = app
        .transcript_selection
        .and_then(|selection| selected_transcript_text(app, selection))
        .is_some_and(|text| !text.trim().is_empty());
    if has_selection {
        CtrlCAction::CopySelection
    } else if app.submitting {
        CtrlCAction::StopTask
    } else {
        CtrlCAction::Exit
    }
}

fn copy_transcript_selection_to_clipboard(app: &mut TuiApp) {
    let Some(text) = take_transcript_selection(app) else {
        app.status = String::from("No transcript text selected");
        return;
    };
    copy_text_to_clipboard(app, &text, "Copied selected transcript text to clipboard");
}

fn take_transcript_selection(app: &mut TuiApp) -> Option<String> {
    let selected = app
        .transcript_selection
        .and_then(|selection| selected_transcript_text(app, selection));
    app.transcript_drag_start = None;
    app.transcript_selection = None;
    selected.filter(|text| !text.trim().is_empty())
}

fn selected_transcript_text(app: &TuiApp, selection: TranscriptSelection) -> Option<String> {
    if selection.start == selection.end {
        return None;
    }
    let lines = &app.transcript_cache.as_ref()?.selection_lines;
    if lines.is_empty() {
        return None;
    }
    let (start, end) = ordered_transcript_selection(selection);
    let start_row = start.row.min(lines.len().saturating_sub(1));
    let end_row = end.row.min(lines.len().saturating_sub(1));
    Some(
        (start_row..=end_row)
            .map(|row| {
                let line = &lines[row];
                let first = if row == start_row { start.col } else { 0 };
                let last = if row == end_row {
                    end.col.saturating_add(1)
                } else {
                    line.chars().count()
                };
                line.chars()
                    .skip(first)
                    .take(last.saturating_sub(first))
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn ordered_transcript_selection(
    selection: TranscriptSelection,
) -> (SelectionPoint, SelectionPoint) {
    if (selection.start.row, selection.start.col) <= (selection.end.row, selection.end.col) {
        (selection.start, selection.end)
    } else {
        (selection.end, selection.start)
    }
}

fn insert_composer_paste(app: &mut TuiApp, text: &str) {
    if text.is_empty() {
        return;
    }
    if app.secret_provider.is_some() {
        app.secret_input.push_str(text);
        app.status = String::from("API key pasted into the protected field");
    } else {
        app.input.push_str(text);
        app.status = String::from("Pasted text — review and press Enter to send");
    }
}

fn parse_paste_keys(raw_keys: &[String]) -> Vec<PasteKey> {
    raw_keys
        .iter()
        .filter_map(|raw| parse_paste_key(raw))
        .collect()
}

fn parse_paste_key(raw: &str) -> Option<PasteKey> {
    let mut modifiers = KeyModifiers::empty();
    let mut key_name = None;
    for part in raw.split('+') {
        let token = part.trim().to_ascii_lowercase();
        match token.as_str() {
            "ctrl" | "control" => modifiers.insert(KeyModifiers::CONTROL),
            "shift" => modifiers.insert(KeyModifiers::SHIFT),
            "alt" | "option" | "meta" => modifiers.insert(KeyModifiers::ALT),
            "super" | "cmd" | "command" => modifiers.insert(KeyModifiers::SUPER),
            "" => {}
            _ => key_name = Some(token),
        }
    }
    let code = match key_name?.as_str() {
        "insert" | "ins" => PasteKeyCode::Insert,
        value if value.chars().count() == 1 => {
            PasteKeyCode::Char(value.chars().next()?.to_ascii_lowercase())
        }
        _ => return None,
    };
    (!modifiers.is_empty()).then_some(PasteKey { code, modifiers })
}

fn is_clipboard_shortcut(code: &KeyCode, modifiers: KeyModifiers, bindings: &[PasteKey]) -> bool {
    bindings.iter().any(|binding| {
        modifiers.contains(binding.modifiers)
            && match (&binding.code, code) {
                (PasteKeyCode::Char(expected), KeyCode::Char(actual)) => {
                    actual.eq_ignore_ascii_case(expected)
                }
                (PasteKeyCode::Insert, KeyCode::Insert) => true,
                _ => false,
            }
    })
}

fn start_voice_recording(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
) {
    if app.voice_recording {
        stop_voice_recording(app);
        return;
    }
    if voice_process_active(app) {
        app.status = String::from("Stopping voice input...");
        return;
    }
    if !app.snapshot.voice.input_ready {
        if let Some(provider) = app.snapshot.voice.input_secret_provider.clone() {
            if let Some(notice) = voice_setup_notice(&app.snapshot.voice) {
                app.notices.retain(|item| item.title != "Voice setup");
                push_notice(app, &notice.title, &notice.text, notice.error);
            }
            app.secret_provider = Some(provider);
            app.secret_input.clear();
            app.status = api_key_prompt_status("voice");
            return;
        }
        app.status = app
            .snapshot
            .voice
            .input_reason
            .clone()
            .unwrap_or_else(|| String::from("Voice input is unavailable"));
        return;
    }
    app.voice_recording = true;
    app.voice_session_id = app.voice_session_id.wrapping_add(1);
    let session_id = app.voice_session_id;
    let cancel_signal = Arc::new(AtomicBool::new(false));
    app.voice_cancel_signal = Some(Arc::clone(&cancel_signal));
    let mut prefix = app.input.clone();
    if prefix
        .chars()
        .last()
        .is_some_and(|character| !character.is_whitespace())
    {
        prefix.push(' ');
    }
    app.voice_input_prefix = Some(prefix);
    app.voice_partial.clear();
    app.status = String::from("Listening and transcribing... select stop to keep text");
    let tx = tx.clone();
    let args = Arc::clone(args);
    let active_voice_process = Arc::clone(&app.active_voice_process);
    runtime.spawn(async move {
        if let Err(err) = stream_voice_record(
            args.as_ref(),
            tx.clone(),
            session_id,
            Arc::clone(&cancel_signal),
            active_voice_process,
        )
        .await
        {
            if !cancel_signal.load(Ordering::SeqCst) {
                let _ = tx.send(AppEvent::VoiceFrame {
                    session_id,
                    result: Err(err),
                });
            }
        }
    });
}

fn stop_voice_recording(app: &mut TuiApp) {
    cancel_voice_transport(app);
    clear_voice_recording_state(app);
    app.status = String::from("Voice input stopped — transcript kept in message");
    invalidate_transcript(app);
}

fn cancel_voice_transport(app: &mut TuiApp) {
    if let Some(signal) = app.voice_cancel_signal.take() {
        signal.store(true, Ordering::SeqCst);
    }
    if let Ok(active) = app.active_voice_process.lock() {
        if let Some(pid) = *active {
            let _ = interrupt_process_tree_pid(pid);
        }
    }
}

fn voice_process_active(app: &TuiApp) -> bool {
    app.active_voice_process
        .lock()
        .map(|active| active.is_some())
        .unwrap_or(false)
}

fn submit_attachment_path(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
    path: String,
) {
    if app.submitting || bridge_process_active(app) {
        app.status = String::from("Finish the current task before adding a file");
        return;
    }
    let filename = attachment_display_name(&path).into_owned();
    start_attachment_mutation(
        runtime,
        args,
        tx,
        app,
        "attachment-add",
        path,
        AttachmentMutation::Add {
            filename: filename.clone(),
        },
    );
    app.status = format!("Adding {filename}...");
}

fn attachment_display_name(path: &str) -> Cow<'_, str> {
    PathBuf::from(path)
        .file_name()
        .map(|name| Cow::Owned(name.to_string_lossy().into_owned()))
        .unwrap_or_else(|| Cow::Borrowed(path))
}

const ATTACHMENT_PREVIEW_BYTES: u64 = 256 * 1024;

fn open_clicked_attachment(app: &mut TuiApp, column: u16, row: u16) {
    let Some(target) = app
        .attachment_hit_areas
        .iter()
        .find(|target| mouse_in_rect(column, row, target.area))
        .cloned()
    else {
        return;
    };
    let Some(path) = target.storage_path.as_deref() else {
        app.status = format!("{} cannot be opened from this session", target.filename);
        return;
    };
    match load_attachment_preview(&target, path) {
        Ok(preview) => {
            app.status = format!("Previewing {}", target.filename);
            app.attachment_preview = Some(preview);
        }
        Err(err) => {
            app.status = format!("{}: {}", target.filename, clip_status(&err.to_string(), 72));
        }
    }
}

fn activate_clicked_attachment(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
    column: u16,
    row: u16,
) {
    let target = app
        .attachment_hit_areas
        .iter()
        .find(|target| mouse_in_rect(column, row, target.area))
        .cloned();
    let Some(target) = target else {
        return;
    };
    if target
        .remove_area
        .is_some_and(|area| mouse_in_rect(column, row, area))
    {
        if let Some(attachment_id) = target.attachment_id.as_deref() {
            submit_attachment_removal(
                runtime,
                args,
                tx,
                app,
                attachment_id.to_string(),
                target.filename,
            );
        }
        return;
    }
    open_clicked_attachment(app, column, row);
}

fn remove_last_pending_attachment(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
) {
    let attachment = app.snapshot.session.pending_attachments.last().cloned();
    if let Some(attachment) = attachment {
        if let Some(attachment_id) = attachment.id {
            submit_attachment_removal(runtime, args, tx, app, attachment_id, attachment.filename);
        }
    }
}

fn submit_attachment_removal(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
    attachment_id: String,
    filename: String,
) {
    if app.submitting || bridge_process_active(app) {
        app.status = String::from("Finish the current task before removing a file");
        return;
    }
    start_attachment_mutation(
        runtime,
        args,
        tx,
        app,
        "attachment-remove",
        attachment_id,
        AttachmentMutation::Remove { filename },
    );
    app.status = String::from("Removing attachment...");
}

fn start_attachment_mutation(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
    action: &'static str,
    value: String,
    operation: AttachmentMutation,
) {
    app.submitting = true;
    let tx = tx.clone();
    let args = Arc::clone(args);
    runtime.spawn(async move {
        let result = call_bridge_async(args.as_ref(), action, Some(&value)).await;
        let _ = tx.send(AppEvent::AttachmentMutation {
            operation,
            result: Box::new(result),
        });
    });
}

fn load_attachment_preview(target: &AttachmentHitArea, path: &str) -> Result<AttachmentPreview> {
    let text_like = target.mime.starts_with("text/")
        || matches!(
            target.mime.as_str(),
            "application/json" | "application/xml" | "application/javascript"
        );
    let (text, truncated) = if text_like {
        let mut bytes = Vec::new();
        File::open(path)?
            .take(ATTACHMENT_PREVIEW_BYTES + 1)
            .read_to_end(&mut bytes)?;
        let truncated = bytes.len() as u64 > ATTACHMENT_PREVIEW_BYTES;
        if truncated {
            bytes.truncate(ATTACHMENT_PREVIEW_BYTES as usize);
        }
        (
            Some(String::from_utf8_lossy(&bytes).into_owned()),
            truncated,
        )
    } else {
        if !std::path::Path::new(path).is_file() {
            return Err(anyhow::anyhow!(
                "The stored attachment is no longer available"
            ));
        }
        (None, false)
    };
    Ok(AttachmentPreview {
        filename: target.filename.clone(),
        mime: target.mime.clone(),
        size_bytes: target.size_bytes,
        storage_path: path.to_string(),
        text,
        truncated,
        scroll: 0,
    })
}

fn open_preview_in_system_viewer(app: &mut TuiApp) {
    let Some(preview) = app.attachment_preview.as_ref() else {
        return;
    };
    match desktop_open_user_file(&preview.storage_path) {
        Ok(value) if value.get("ok").and_then(Value::as_bool) == Some(true) => {
            app.status = format!("Sent {} to the system viewer", preview.filename);
        }
        Ok(value) => {
            let error = value
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("The system viewer could not open this attachment");
            app.status = format!("{}: {}", preview.filename, clip_status(error, 72));
        }
        Err(err) => {
            app.status = format!(
                "{}: {}",
                preview.filename,
                clip_status(&err.to_string(), 72)
            );
        }
    }
}

fn open_attachment_picker(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
) {
    if app.submitting || bridge_process_active(app) {
        app.status = String::from("Finish the current task before adding a file");
        return;
    }
    match desktop_pick_file() {
        Ok(value) if value.get("ok").and_then(Value::as_bool) == Some(true) => {
            if let Some(path) = value.get("path").and_then(Value::as_str) {
                submit_attachment_path(runtime, args, tx, app, path.to_string());
                return;
            }
            app.status = String::from("The file picker did not return a file");
        }
        Ok(value) if value.get("cancelled").and_then(Value::as_bool) == Some(true) => {
            app.status = String::from("File selection cancelled");
        }
        Ok(value) => {
            app.attachment_path_mode = true;
            app.input.clear();
            clear_palette(app);
            let guidance = value
                .get("guidance")
                .and_then(Value::as_str)
                .unwrap_or("No native picker is available");
            app.status = format!("{guidance} Enter a file path instead");
        }
        Err(err) => {
            app.attachment_path_mode = true;
            app.input.clear();
            clear_palette(app);
            app.status = format!(
                "File picker unavailable: {}. Enter a file path instead",
                clip_status(&err.to_string(), 64)
            );
        }
    }
}

fn paste_clipboard_into_composer(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
) {
    if !app.snapshot.approvals.is_empty() {
        return;
    }
    if app.secret_provider.is_some() {
        if !paste_clipboard_text(app) {
            app.status = String::from("Clipboard does not contain an API key as text");
        }
        return;
    }
    if app.submitting || bridge_process_active(app) {
        if !paste_clipboard_text(app) {
            app.status = String::from("Finish the current task before attaching a clipboard image");
        }
        return;
    }
    let image_result = desktop_clipboard_image_to_file();
    match &image_result {
        Ok(value) if value.get("ok").and_then(Value::as_bool) == Some(true) => {
            if let Some(path) = value.get("path").and_then(Value::as_str) {
                submit_attachment_path(runtime, args, tx, app, path.to_string());
                return;
            }
            app.status = String::from("Clipboard image could not be prepared for attachment");
            return;
        }
        Ok(_) => {}
        Err(_) => {}
    }
    if paste_clipboard_text(app) {
        return;
    }
    match image_result {
        Ok(value) => {
            app.status = value
                .get("guidance")
                .or_else(|| value.get("error"))
                .and_then(Value::as_str)
                .unwrap_or("Clipboard has no supported text or image to paste")
                .to_string();
        }
        Err(err) => {
            app.status = format!(
                "Clipboard paste unavailable: {}",
                clip_status(&err.to_string(), 72)
            );
        }
    }
}

fn paste_clipboard_text(app: &mut TuiApp) -> bool {
    desktop_clipboard_read_text().is_ok_and(|value| insert_clipboard_text(app, &value))
}

fn insert_clipboard_text(app: &mut TuiApp, value: &Value) -> bool {
    let Some(text) = value
        .get("text")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    else {
        return false;
    };

    let secret_input = app.secret_provider.is_some();
    insert_composer_paste(app, text);
    if !secret_input {
        app.status = String::from("Pasted text from the clipboard");
    }
    true
}

fn copy_latest_response_to_clipboard(app: &mut TuiApp) {
    let Some(text) = app
        .streaming_text
        .trim()
        .is_empty()
        .then(|| {
            app.snapshot
                .messages
                .iter()
                .rev()
                .find(|message| message.role == "assistant" && !message.content.trim().is_empty())
                .map(|message| message.content.trim().to_string())
        })
        .flatten()
        .or_else(|| {
            (!app.streaming_text.trim().is_empty()).then(|| app.streaming_text.trim().to_string())
        })
    else {
        app.status = String::from("No assistant response to copy yet");
        return;
    };
    copy_text_to_clipboard(app, &text, "Copied latest response to clipboard");
}

fn copy_text_to_clipboard(app: &mut TuiApp, text: &str, success_status: &str) {
    match desktop_action("clipboard_write", None, Some(text), None, None) {
        Ok(value) if value.get("ok").and_then(Value::as_bool) == Some(true) => {
            app.status = String::from(success_status);
        }
        Ok(value) => {
            let guidance = value
                .get("guidance")
                .or_else(|| value.get("error"))
                .and_then(Value::as_str)
                .unwrap_or("Clipboard copy failed");
            app.status = format!("Copy failed: {}", clip_status(guidance, 72));
        }
        Err(error) => app.status = format!("Copy failed: {}", clip_status(&error.to_string(), 72)),
    }
}

fn request_active_stop(app: &mut TuiApp) {
    app.cancel_requested = true;
    app.cancel_signal.store(true, Ordering::SeqCst);
    app.status = String::from("Stopping current task...");
    invalidate_transcript(app);
    if let Ok(active) = app.active_bridge.lock() {
        if let Some(child) = active.as_ref() {
            if let Some(pid) = child.id() {
                let _ = interrupt_process_tree_pid(pid);
            }
        }
    }
}

#[cfg(unix)]
fn interrupt_process_tree_pid(pid: u32) -> io::Result<()> {
    let result = unsafe { libc::kill(-(pid as i32), libc::SIGINT) };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(windows)]
fn interrupt_process_tree_pid(pid: u32) -> io::Result<()> {
    let status = ProcessCommand::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other("Could not stop the bridge process tree"))
    }
}

fn bridge_process_active(app: &TuiApp) -> bool {
    app.active_bridge
        .lock()
        .map(|active| active.is_some())
        .unwrap_or(false)
}

fn start_prompt_submission(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
    prompt: String,
) {
    if app.voice_recording || voice_process_active(app) {
        cancel_voice_transport(app);
    }
    clear_voice_recording_state(app);
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
    clear_palette(app);
    invalidate_transcript(app);
    let tx_clone = tx.clone();
    let err_tx = tx.clone();
    let args_clone = Arc::clone(args);
    let active_bridge = Arc::clone(&app.active_bridge);
    let cancel_signal = Arc::clone(&app.cancel_signal);
    runtime.spawn(async move {
        let result = stream_bridge_submit(
            args_clone.as_ref(),
            &prompt,
            tx_clone,
            active_bridge,
            cancel_signal,
        )
        .await;
        if let Err(err) = result {
            let _ = err_tx.send(AppEvent::StreamFrame(Err(err)));
        }
    });
}

fn submit_pending_action(
    runtime: &RuntimeHandle,
    args: &Arc<TuiArgs>,
    tx: &mpsc::Sender<AppEvent>,
    app: &mut TuiApp,
) {
    let Some(action) = app
        .setup_surface
        .as_mut()
        .and_then(|surface| surface.pending_action.take())
    else {
        return;
    };
    if action.kind == BridgePendingActionKind::Unknown {
        app.status = String::from("This action is not supported by this version of Nym");
        return;
    }
    app.setup_surface = None;
    app.pending_action_area = None;
    start_prompt_submission(runtime, args, tx, app, action.command);
}

fn draw_app(frame: &mut ratatui::Frame<'_>, app: &mut TuiApp) {
    app.attachment_hit_areas.clear();
    app.palette_hit_areas.clear();
    app.pending_action_area = None;
    app.cost_button_area = None;
    let area = frame.area();
    let setup_height = setup_surface_height(app.setup_surface.as_ref(), area.width);
    let [header_area, content_area, setup_area, status_area, composer_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(setup_height),
            Constraint::Length(1),
            Constraint::Length(4),
        ])
        .areas(area);
    let show_sidebar = should_show_sidebar(area.width, app);
    let [conversation_area, sidebar_area] = if show_sidebar {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(48), Constraint::Length(34)])
            .areas(content_area)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(0)])
            .areas(content_area)
    };

    render_transcript(frame, conversation_area, app, true);
    render_setup_surface(frame, setup_area, app);

    let session = &app.snapshot.session;
    let agent_badge = agent_label(&app.snapshot.agent_name);
    let header_lines = vec![
        Line::from(vec![
            Span::styled(
                format!(" {agent_badge} "),
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
            Span::styled(session.id.as_str(), Style::default().fg(Color::Gray)),
            Span::styled("  model ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if session.model_selected {
                    format!("{}/{}", session.provider, session.model)
                } else {
                    String::from("not selected")
                },
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("  mode ", Style::default().fg(Color::DarkGray)),
            Span::styled(session.mode.as_str(), Style::default().fg(Color::Magenta)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(header_lines).block(
            Block::default()
                .borders(Borders::BOTTOM)
                .border_style(Style::default().fg(Color::DarkGray)),
        ),
        header_area,
    );

    if show_sidebar {
        draw_sidebar(frame, sidebar_area, app);
    }

    let (state_label, state_color) = ui_state_badge(app);
    let cost_text = footer_cost_text(app.snapshot.session.cost_usd);
    let cost_width = cost_text.len() as u16;
    let cost_gap = if area.width > cost_width + 2 {
        cost_width + 1
    } else {
        0
    };
    let left_status_area = Rect {
        x: status_area.x,
        y: status_area.y,
        width: status_area.width.saturating_sub(cost_gap),
        height: status_area.height,
    };
    let status_limit = left_status_area.width.saturating_sub(40) as usize;
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
            clip_status(&app.status, status_limit.max(16)),
            Style::default().fg(Color::White),
        ),
        Span::styled(
            format!(
                "  {}",
                footer_help_text(
                    app.secret_provider.is_some(),
                    !app.snapshot.approvals.is_empty(),
                    app.submitting,
                    app.mouse_capture,
                )
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    frame.render_widget(Paragraph::new(status), left_status_area);
    if cost_gap > 0 {
        let cost_area = Rect {
            x: status_area.x + status_area.width.saturating_sub(cost_width),
            y: status_area.y,
            width: cost_width,
            height: status_area.height,
        };
        app.cost_button_area = Some(cost_area);
        frame.render_widget(
            Paragraph::new(cost_text).style(
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::UNDERLINED),
            ),
            cost_area,
        );
    }

    let input_title = if let Some(provider) = app.secret_provider.as_deref() {
        format!(
            " {} API key · hidden input ",
            provider_display_name(provider)
        )
    } else if app.attachment_path_mode {
        String::from(" attach file path · Enter adds it · Esc cancels ")
    } else if install_command_is_confirmed(&app.input) {
        String::from(" confirm local install · Enter starts · Esc cancels ")
    } else if app.submitting {
        String::from(" message (agent working · Enter queues) ")
    } else if !app.palette.entries.is_empty() && app.input.starts_with('/') {
        String::from(" Commands ")
    } else {
        let count = app.snapshot.session.pending_attachments.len();
        if count == 0 {
            String::from(" Message ")
        } else {
            format!(
                " Message · {count} {} ",
                if count == 1 { "file" } else { "files" }
            )
        }
    };
    let controls_enabled = app.secret_provider.is_none()
        && app.snapshot.approvals.is_empty()
        && !app.attachment_path_mode
        && !app.submitting;
    let mic_enabled = controls_enabled && app.snapshot.voice.input_ready;
    let composer_inner = Rect {
        x: composer_area.x.saturating_add(1),
        y: composer_area.y.saturating_add(1),
        width: composer_area.width.saturating_sub(2),
        height: composer_area.height.saturating_sub(2),
    };
    let [attachment_area, mic_area, input_area] = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Min(12),
        ])
        .areas(composer_inner);
    app.attachment_button_area = Some(attachment_area);
    app.mic_button_area = Some(mic_area);

    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(if app.submitting {
                Color::DarkGray
            } else if app.secret_provider.is_some() {
                Color::Yellow
            } else if app.attachment_path_mode {
                Color::Cyan
            } else {
                Color::Rgb(62, 68, 78)
            }))
            .title(input_title),
        composer_area,
    );
    let visible_input = if app.secret_provider.is_some() {
        Cow::Owned("•".repeat(app.secret_input.chars().count()))
    } else {
        Cow::Borrowed(app.input.as_str())
    };
    let show_attachment_chips =
        !app.snapshot.session.pending_attachments.is_empty() && app.secret_provider.is_none();
    let mut input_lines = Vec::with_capacity(if show_attachment_chips { 2 } else { 1 });
    if show_attachment_chips {
        let (line, hit_areas) = attachment_chip_line(
            &app.snapshot.session.pending_attachments,
            input_area.x,
            input_area.y,
            input_area.width,
        );
        app.attachment_hit_areas.extend(hit_areas);
        input_lines.push(line);
    }
    input_lines.push(Line::from(vec![
        Span::styled(
            if app.attachment_path_mode {
                "path: "
            } else {
                "› "
            },
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(visible_input.as_ref()),
    ]));
    let input = Paragraph::new(input_lines).wrap(Wrap { trim: false });
    frame.render_widget(input, input_area);

    frame.render_widget(
        Paragraph::new(attachment_button_text(controls_enabled))
            .alignment(ratatui::layout::Alignment::Center),
        attachment_area,
    );
    frame.render_widget(
        Paragraph::new(mic_button_text(mic_enabled, app.voice_recording))
            .alignment(ratatui::layout::Alignment::Center),
        mic_area,
    );
    if palette_is_open(app) {
        let query = palette_context(&app.input)
            .map(|context| context.query)
            .unwrap_or_default();
        let palette_lines = visible_palette_entries(&app.palette, query)
            .map(|(_, entry)| palette_entry_height(entry))
            .sum::<usize>();
        let max_popup_height = input_area.y.saturating_sub(area.y).clamp(3, 20);
        let desired_popup_height = palette_lines.saturating_add(2).min(u16::MAX as usize) as u16;
        let popup_height = desired_popup_height.min(max_popup_height);
        let line_budget = popup_height.saturating_sub(2) as usize;
        let popup_y = input_area.y.saturating_sub(popup_height);
        let popup_area = Rect {
            x: input_area.x,
            y: popup_y,
            width: conversation_area.width.min(input_area.width).max(20),
            height: popup_height,
        };
        app.palette_hit_areas = palette_hit_areas(
            &app.palette,
            query,
            app.palette_selected,
            line_budget,
            popup_area,
        );
        frame.render_widget(Clear, popup_area);
        let popup = Paragraph::new(palette_text(
            &app.palette,
            query,
            app.palette_selected,
            line_budget,
            popup_area.width,
        ))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Magenta))
                .title(format!(
                    " {} ",
                    palette_title(&app.palette, query, app.palette_selected, line_budget)
                )),
        )
        .wrap(Wrap { trim: false });
        frame.render_widget(popup, popup_area);
    }

    if let Some(view) = app.gateway_view.as_ref() {
        draw_gateway_dialog(frame, area, view);
    }

    if let Some(preview) = app.attachment_preview.as_ref() {
        draw_attachment_preview(frame, area, preview);
    }

    if app.show_cost_details {
        draw_cost_dialog(frame, area, &app.snapshot.session);
    }

    if !app.snapshot.approvals.is_empty() {
        draw_approval_dialog(frame, area, app);
    }

    let cursor_x = input_area
        .x
        .saturating_add(2)
        .saturating_add(visible_input.chars().count() as u16);
    let cursor_y = input_area
        .y
        .saturating_add(u16::from(show_attachment_chips));
    if app.snapshot.approvals.is_empty()
        && app.gateway_view.is_none()
        && app.attachment_preview.is_none()
        && !app.show_cost_details
    {
        frame.set_cursor_position((cursor_x.min(input_area.right().saturating_sub(2)), cursor_y));
    }
}

fn setup_surface_height(surface: Option<&UiSetupSurface>, width: u16) -> u16 {
    let Some(surface) = surface else {
        return 0;
    };
    let text_height = Paragraph::new(surface.text.as_str())
        .wrap(Wrap { trim: false })
        .line_count(width.saturating_sub(2).max(1))
        .min(u16::MAX as usize) as u16;
    let action_height = u16::from(surface.pending_action.is_some());
    text_height
        .saturating_add(action_height)
        .saturating_add(2)
        .clamp(3, 8)
}

fn render_setup_surface(frame: &mut ratatui::Frame<'_>, area: Rect, app: &mut TuiApp) {
    let Some(surface) = app.setup_surface.as_ref() else {
        return;
    };
    let color = if surface.error {
        Color::Red
    } else {
        Color::Yellow
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color))
        .title(format!(" {} ", surface.title));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let [text_area, action_area] = if surface.pending_action.is_some() && inner.height > 1 {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .areas(inner)
    } else {
        [inner, Rect::default()]
    };
    frame.render_widget(
        Paragraph::new(surface.text.as_str())
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false }),
        text_area,
    );
    if let Some(action) = surface.pending_action.as_ref() {
        app.pending_action_area = Some(action_area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(
                    format!(" [ {} ] ", action.label),
                    Style::default()
                        .fg(Color::Black)
                        .bg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "  Enter confirms · Esc cancels",
                    Style::default().fg(Color::Gray),
                ),
            ])),
            action_area,
        );
    }
}

fn attachment_chip_line(
    attachments: &[BridgeAttachment],
    x: u16,
    y: u16,
    width: u16,
) -> (Line<'static>, Vec<AttachmentHitArea>) {
    let mut spans = Vec::new();
    let mut hit_areas = Vec::new();
    let mut used = 0u16;
    for attachment in attachments {
        let separator = if used == 0 { "" } else { "  " };
        let separator_width = separator.chars().count() as u16;
        let available = width.saturating_sub(used).saturating_sub(separator_width);
        if available < 7 {
            break;
        }
        let filename_limit = available.saturating_sub(6).min(28) as usize;
        let filename = clip_status(&attachment.filename, filename_limit).into_owned();
        let chip = format!(" ▤ {filename} × ");
        let chip_width = chip.chars().count() as u16;
        if !separator.is_empty() {
            spans.push(Span::raw(separator));
        }
        let chip_x = x.saturating_add(used).saturating_add(separator_width);
        spans.push(Span::styled(
            chip,
            Style::default()
                .fg(Color::Rgb(205, 220, 232))
                .bg(Color::Rgb(35, 40, 48)),
        ));
        hit_areas.push(AttachmentHitArea {
            area: Rect::new(chip_x, y, chip_width, 1),
            remove_area: Some(Rect::new(
                chip_x.saturating_add(chip_width.saturating_sub(3)),
                y,
                2,
                1,
            )),
            attachment_id: attachment.id.clone(),
            filename: attachment.filename.clone(),
            mime: attachment.mime.clone(),
            size_bytes: attachment.size_bytes,
            storage_path: attachment.storage_path.clone(),
        });
        used = used
            .saturating_add(separator_width)
            .saturating_add(chip_width);
    }
    (Line::from(spans), hit_areas)
}

fn attachment_button_text(enabled: bool) -> Text<'static> {
    let icon = if enabled {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    Text::from(Line::from(Span::styled(
        "+",
        Style::default().fg(icon).add_modifier(Modifier::BOLD),
    )))
}

fn mic_button_text(enabled: bool, recording: bool) -> Text<'static> {
    let color = if recording {
        Color::Red
    } else if enabled {
        Color::Cyan
    } else {
        Color::DarkGray
    };
    Text::from(Line::from(Span::styled(
        if recording { "■" } else { "🎙" },
        Style::default().fg(color).add_modifier(Modifier::BOLD),
    )))
}

fn footer_help_text(
    auth_active: bool,
    approval_active: bool,
    submitting: bool,
    mouse_capture: bool,
) -> &'static str {
    if auth_active {
        "Enter save key  Esc cancel  input hidden"
    } else if approval_active {
        "Enter/Y approve  N/Esc deny  Ctrl+N/P select"
    } else if submitting {
        "Esc/Ctrl+C stop  Enter queue  End follow"
    } else if mouse_capture {
        "Enter send  +/Ctrl+O/F4 attach  mic/F5 speak  drag select  Ctrl+C copy  /help commands"
    } else {
        "Enter send  +/Ctrl+O/F4 attach  mic/F5 speak  Ctrl+V paste  Alt+C copy  /help commands"
    }
}

fn draw_approval_dialog(frame: &mut ratatui::Frame<'_>, area: Rect, app: &TuiApp) {
    let Some(approval) = app.snapshot.approvals.get(app.approval_selected) else {
        return;
    };
    let width = area.width.saturating_sub(4).clamp(1, 72);
    let height = area.height.saturating_sub(4).clamp(1, 11);
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

fn draw_cost_dialog(frame: &mut ratatui::Frame<'_>, area: Rect, session: &BridgeSession) {
    let width = area.width.saturating_sub(4).clamp(1, 62);
    let unclassified = (session.cost_usd - session.costs.total()).max(0.0);
    let extra_rows = usize::from(unclassified > 1e-12);
    let height = (13 + extra_rows as u16).min(area.height.saturating_sub(2).max(1));
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    };
    let tokens = &session.tokens;
    let uncached_input = tokens
        .input
        .saturating_sub(tokens.cache_read.max(0))
        .saturating_sub(tokens.cache_write.max(0))
        .max(0);
    let mut lines = vec![
        cost_detail_line("Total", None, session.cost_usd, true),
        Line::from(""),
        cost_detail_line(
            "Input",
            Some(tokens.input),
            session.costs.input_total(),
            true,
        ),
        cost_detail_line(
            "  Uncached",
            Some(uncached_input),
            session.costs.input,
            false,
        ),
        cost_detail_line(
            "  Cached",
            Some(tokens.cache_read),
            session.costs.cached_input,
            false,
        ),
        cost_detail_line(
            "  Cache write",
            Some(tokens.cache_write),
            session.costs.cache_write,
            false,
        ),
        cost_detail_line("Output", Some(tokens.output), session.costs.output, true),
        Line::from(vec![
            Span::styled("  Reasoning", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!(
                    "  {} tokens (included in output)",
                    format_token_count(tokens.reasoning)
                ),
                Style::default().fg(Color::Gray),
            ),
        ]),
    ];
    if unclassified > 1e-12 {
        lines.push(cost_detail_line("Legacy/other", None, unclassified, false));
    }
    lines.extend([
        Line::from(""),
        Line::from(vec![
            Span::styled("Provider  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}/{}", session.provider, session.model),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(Span::styled(
            "Enter or Esc closes",
            Style::default().fg(Color::DarkGray),
        )),
    ]);

    frame.render_widget(Clear, popup_area);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Green))
                    .title(" Session token cost "),
            )
            .wrap(Wrap { trim: false }),
        popup_area,
    );
}

fn cost_detail_line(
    label: &str,
    tokens: Option<i64>,
    cost_usd: f64,
    emphasized: bool,
) -> Line<'static> {
    let modifier = if emphasized {
        Modifier::BOLD
    } else {
        Modifier::empty()
    };
    let token_text = tokens
        .map(|value| format!("{} tokens", format_token_count(value)))
        .unwrap_or_default();
    Line::from(vec![
        Span::styled(
            format!("{label:<13}"),
            Style::default().fg(Color::White).add_modifier(modifier),
        ),
        Span::styled(
            format!("{token_text:>16}"),
            Style::default().fg(Color::Gray),
        ),
        Span::styled(
            format!("  {:>9}", format_cost_usd(cost_usd)),
            Style::default().fg(Color::Green).add_modifier(modifier),
        ),
    ])
}

fn format_token_count(value: i64) -> String {
    let digits = value.max(0).to_string();
    let mut grouped = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(character);
    }
    grouped
}

fn draw_gateway_dialog(frame: &mut ratatui::Frame<'_>, area: Rect, view: &GatewayViewState) {
    let width = area.width.saturating_sub(4).clamp(1, 112);
    let height = area.height.saturating_sub(2).clamp(1, 36);
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
    let [tabs_area, content_area, help_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .areas(inner);

    frame.render_widget(Clear, popup_area);
    frame.render_widget(block, popup_area);
    frame.render_widget(Paragraph::new(gateway_tabs_line(view.tab)), tabs_area);
    frame.render_widget(
        Paragraph::new(gateway_tab_text(view))
            .wrap(Wrap { trim: false })
            .scroll((view.scroll, 0)),
        content_area,
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
        help_area,
    );
}

fn draw_attachment_preview(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    preview: &AttachmentPreview,
) {
    let width = area.width.saturating_sub(6).clamp(1, 100);
    let height = area.height.saturating_sub(4).clamp(1, 32);
    let popup_area = Rect {
        x: area.x + area.width.saturating_sub(width) / 2,
        y: area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(82, 92, 105)))
        .title(format!(
            " Preview · {} ",
            clip_status(&preview.filename, 64)
        ));
    let inner = block.inner(popup_area);
    let [metadata_area, preview_area, help_area] = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Min(2),
            Constraint::Length(1),
        ])
        .areas(inner);

    frame.render_widget(Clear, popup_area);
    frame.render_widget(block, popup_area);
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(preview.mime.clone(), Style::default().fg(Color::Cyan)),
            Span::styled("  ·  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                human_file_size(preview.size_bytes),
                Style::default().fg(Color::Gray),
            ),
            if preview.truncated {
                Span::styled("  ·  preview truncated", Style::default().fg(Color::Yellow))
            } else {
                Span::raw("")
            },
        ])),
        metadata_area,
    );

    if let Some(text) = preview.text.as_deref() {
        frame.render_widget(
            Paragraph::new(text)
                .wrap(Wrap { trim: false })
                .scroll((preview.scroll, 0)),
            preview_area,
        );
    } else {
        let kind = if preview.mime.starts_with("image/") {
            "Image"
        } else if preview.mime == "application/pdf" {
            "PDF"
        } else {
            "Binary file"
        };
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(Span::styled(
                    format!("{kind} ready"),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
                Line::from(""),
                Line::from(Span::styled(
                    "Open this file in its system viewer to see the full preview.",
                    Style::default().fg(Color::Gray),
                )),
            ])
            .wrap(Wrap { trim: false }),
            preview_area,
        );
    }

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("↑/↓ PgUp/PgDn", Style::default().fg(Color::Cyan)),
            Span::styled(" scroll  ", Style::default().fg(Color::DarkGray)),
            Span::styled("O/Enter", Style::default().fg(Color::Cyan)),
            Span::styled(" system viewer  ", Style::default().fg(Color::DarkGray)),
            Span::styled("Esc/Q", Style::default().fg(Color::Cyan)),
            Span::styled(" close", Style::default().fg(Color::DarkGray)),
        ])),
        help_area,
    );
}

fn human_file_size(size_bytes: i64) -> String {
    let size = size_bytes.max(0) as f64;
    if size < 1024.0 {
        return format!("{} B", size as u64);
    }
    if size < 1024.0 * 1024.0 {
        return format!("{:.1} KiB", size / 1024.0);
    }
    format!("{:.1} MiB", size / (1024.0 * 1024.0))
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

#[cfg(test)]
fn transcript_max_scroll(lines: &Text<'_>, width: u16, height: u16) -> u16 {
    let wrapped = Paragraph::new(lines.clone()).wrap(Wrap { trim: false });
    wrapped
        .line_count(width)
        .saturating_sub(height as usize)
        .min(u16::MAX as usize) as u16
}

fn invalidate_transcript(app: &mut TuiApp) {
    app.transcript_cache = None;
}

fn render_transcript(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &mut TuiApp,
    show_inline_activity: bool,
) {
    let cache_is_current = app.transcript_cache.as_ref().is_some_and(|cache| {
        cache.show_inline_activity == show_inline_activity
            && cache.area == area
            && cache.auto_follow == app.auto_follow
            && cache.requested_scroll == app.scroll
    });

    if !cache_is_current {
        let lines = transcript_text(&app.snapshot, app, show_inline_activity);
        let mut paragraph = Paragraph::new(lines.clone()).wrap(Wrap { trim: false });
        let max_scroll = paragraph
            .line_count(area.width.saturating_sub(2))
            .saturating_sub(area.height.saturating_sub(2) as usize)
            .min(u16::MAX as usize) as u16;
        let scroll = if app.auto_follow {
            max_scroll
        } else {
            app.scroll.min(max_scroll)
        };
        let title = if app.submitting {
            format!(" {} ", clip_status(&app.status, 42))
        } else {
            String::from(" conversation ")
        };
        paragraph = paragraph
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(if app.submitting {
                        Color::Cyan
                    } else {
                        Color::DarkGray
                    }))
                    .title(title),
            )
            .scroll((scroll, 0));
        let attachment_hit_areas =
            transcript_attachment_hit_areas(&lines, &app.snapshot, area, scroll);
        let selection_lines = transcript_visible_plain_lines(&lines, area, scroll);
        app.transcript_cache = Some(TranscriptCache {
            show_inline_activity,
            area,
            auto_follow: app.auto_follow,
            requested_scroll: app.scroll,
            paragraph,
            attachment_hit_areas,
            selection_lines,
        });
    }

    if let Some(cache) = app.transcript_cache.as_ref() {
        cache.paragraph.render_ref(area, frame.buffer_mut());
        if let Some(selection) = app.transcript_selection {
            render_transcript_selection(
                frame.buffer_mut(),
                cache.area,
                &cache.selection_lines,
                selection,
            );
        }
        app.attachment_hit_areas
            .extend(cache.attachment_hit_areas.iter().cloned());
    }
}

fn render_transcript_selection(
    buffer: &mut ratatui::buffer::Buffer,
    area: Rect,
    lines: &[String],
    selection: TranscriptSelection,
) {
    if selection.start == selection.end || lines.is_empty() {
        return;
    }
    let inner_x = area.x.saturating_add(1);
    let inner_y = area.y.saturating_add(1);
    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2) as usize;
    if inner_width == 0 || inner_height == 0 {
        return;
    }
    let (start, end) = ordered_transcript_selection(selection);
    let start_row = start.row.min(lines.len().saturating_sub(1));
    let end_row = end
        .row
        .min(lines.len().saturating_sub(1))
        .min(inner_height.saturating_sub(1));
    for (row, line) in lines
        .iter()
        .enumerate()
        .take(end_row.saturating_add(1))
        .skip(start_row)
    {
        let line_width = line.chars().count().min(inner_width);
        let first = if row == start_row { start.col } else { 0 }.min(line_width);
        let last = if row == end_row {
            end.col.saturating_add(1)
        } else {
            line_width
        }
        .min(line_width);
        for col in first..last {
            if let Some(cell) = buffer.cell_mut((
                inner_x.saturating_add(col as u16),
                inner_y.saturating_add(row as u16),
            )) {
                cell.set_bg(Color::Rgb(52, 78, 110));
                cell.set_fg(Color::White);
            }
        }
    }
}

fn transcript_visible_plain_lines(lines: &Text<'_>, area: Rect, scroll: u16) -> Vec<String> {
    let width = area.width.saturating_sub(2) as usize;
    let height = area.height.saturating_sub(2) as usize;
    if width == 0 || height == 0 {
        return Vec::new();
    }
    lines
        .lines
        .iter()
        .flat_map(|line| {
            let plain = line
                .spans
                .iter()
                .map(|span| span.content.as_ref())
                .collect::<String>();
            if plain.is_empty() {
                return vec![String::new()];
            }
            plain
                .chars()
                .collect::<Vec<_>>()
                .chunks(width.max(1))
                .map(|chunk| chunk.iter().collect::<String>())
                .collect()
        })
        .skip(scroll as usize)
        .take(height)
        .collect()
}

fn transcript_attachment_hit_areas(
    lines: &Text<'_>,
    snapshot: &BridgeSnapshot,
    area: Rect,
    scroll: u16,
) -> Vec<AttachmentHitArea> {
    let width = area.width.saturating_sub(2);
    let height = area.height.saturating_sub(2);
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut targets = Vec::new();
    let mut line_index = 0usize;
    for message in &snapshot.messages {
        line_index += 1;
        line_index += message.content.lines().count().max(1);
        for attachment in &message.attachments {
            targets.push((line_index, attachment));
            line_index += 1;
        }
        line_index += 1;
    }

    let mut hit_areas = Vec::with_capacity(targets.len());
    let mut visual_row = 0usize;
    let mut target_index = 0usize;
    for (index, line) in lines.lines.iter().enumerate() {
        let row_count = Paragraph::new(Text::from(line.clone()))
            .wrap(Wrap { trim: false })
            .line_count(width)
            .max(1);
        while let Some((target_line, attachment)) = targets.get(target_index) {
            if *target_line != index {
                break;
            }
            let visible_top = scroll as usize;
            let visible_bottom = visible_top + height as usize;
            let target_bottom = visual_row + row_count;
            if target_bottom > visible_top && visual_row < visible_bottom {
                let clipped_start = visual_row.max(visible_top);
                let clipped_end = target_bottom.min(visible_bottom);
                let visible_start = clipped_start - visible_top;
                hit_areas.push(AttachmentHitArea {
                    area: Rect::new(
                        area.x.saturating_add(1),
                        area.y.saturating_add(1 + visible_start as u16),
                        width,
                        (clipped_end - clipped_start) as u16,
                    ),
                    remove_area: None,
                    attachment_id: None,
                    filename: attachment.filename.clone(),
                    mime: attachment.mime.clone(),
                    size_bytes: attachment.size_bytes,
                    storage_path: attachment.storage_path.clone(),
                });
            }
            target_index += 1;
        }
        visual_row += row_count;
        if target_index >= targets.len() {
            break;
        }
    }
    hit_areas
}

fn draw_sidebar(frame: &mut ratatui::Frame<'_>, area: Rect, app: &TuiApp) {
    if !app.snapshot.approvals.is_empty() {
        let approvals = Paragraph::new(approval_panel_text(app))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Yellow))
                    .title(" Action needed "),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(approvals, area);
    }
}

fn should_show_sidebar(width: u16, app: &TuiApp) -> bool {
    width >= 92 && !app.snapshot.approvals.is_empty()
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
                clip_status(&approval.tool, 18).into_owned(),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(
                clip_status(&path, 28).into_owned(),
                Style::default().fg(Color::Gray),
            ),
        ]));
        if !approval.reason.is_empty() {
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(
                    clip_status(&approval.reason, 28).into_owned(),
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
    let mut parts = trimmed.split_whitespace();
    if parts.next() == Some("desktop") {
        let (Some(action), Some(target)) = (parts.next(), parts.next()) else {
            return Some(trimmed.to_string());
        };
        if target.starts_with("windows-app:") || target.starts_with("windows-shortcut:") {
            if let Some(label) = parts.next_back().filter(|label| !label.contains(':')) {
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

fn append_subagent_run_lines(lines: &mut Vec<Line<'static>>, run: &SubagentRunState, _live: bool) {
    let running = run
        .tasks
        .iter()
        .filter(|task| task.status == "running")
        .count();
    let running_label = if running == 1 { "agent" } else { "agents" };
    let title = match run.status.as_str() {
        "complete" => format!("✓ {} agents finished", run.total),
        "incomplete" => format!(
            "! {}/{} agents finished · {} failed",
            run.completed, run.total, run.failed
        ),
        _ if run.completed > 0 => {
            format!(
                "● {running} {running_label} running · {} done",
                run.completed
            )
        }
        _ => format!("● {running} {running_label} running"),
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

    let visible_tasks = run.tasks.iter().take(8);
    let visible_count = visible_tasks.len();
    for (index, task) in visible_tasks.enumerate() {
        let (symbol, color) = match task.status.as_str() {
            "complete" => ("✓", Color::Green),
            "blocked" => ("!", Color::Yellow),
            "failed" => ("×", Color::Red),
            "running" => ("●", Color::Cyan),
            _ => ("○", Color::DarkGray),
        };
        let detail = match task.status.as_str() {
            "running" | "queued" if !task.description.trim().is_empty() => task.description.clone(),
            "complete" if task.changed_count > 0 => {
                format!("{} file(s) changed", task.changed_count)
            }
            "failed" | "blocked" if !task.summary.trim().is_empty() => task.summary.clone(),
            _ if !task.description.trim().is_empty() => task.description.clone(),
            _ => task.summary.clone(),
        };
        let branch = if index + 1 == visible_count {
            "└─"
        } else {
            "├─"
        };
        let state = match task.status.as_str() {
            "complete" => "Done",
            "blocked" => "Blocked",
            "failed" => "Failed",
            "running" => "Running",
            _ => "Queued",
        };
        lines.push(Line::from(vec![
            Span::styled(
                format!("{branch} {symbol} "),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                clip_status(&task.id, 18).into_owned(),
                Style::default().fg(Color::White),
            ),
            Span::styled(format!("  {state}"), Style::default().fg(color)),
            Span::styled(
                format!(" · {}", clip_status(&detail, 42)),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }
    if run.tasks.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            clip_status(&run.run_id, 48).into_owned(),
            Style::default().fg(Color::DarkGray),
        )]));
    }
    if let Some(work_file) = run.work_file.as_deref() {
        lines.push(Line::from(vec![
            Span::styled("↳ shared log  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                clip_status(work_file, 48).into_owned(),
                Style::default().fg(Color::Blue),
            ),
        ]));
    }
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

fn footer_cost_text(value: f64) -> String {
    format!("cost {}", format_cost_usd(value))
}

fn format_cost_usd(value: f64) -> String {
    let value = value.max(0.0);
    if value == 0.0 {
        String::from("$0")
    } else if value < 0.01 {
        format!("${value:.4}")
    } else {
        format!("${value:.2}")
    }
}

fn agent_label(agent_name: &str) -> String {
    let name = clip_status(agent_name, 12).into_owned();
    if name.trim().is_empty() || name.eq_ignore_ascii_case("agent") {
        String::from("AGENT")
    } else {
        name
    }
}

fn role_label(role: &str, agent_name: &str) -> (String, Color) {
    match role {
        "user" => (String::from("YOU"), Color::Green),
        "assistant" => (agent_label(agent_name), Color::Yellow),
        "tool" => (String::from("TOOL"), Color::Magenta),
        _ => (String::from("SYS"), Color::White),
    }
}

fn role_header(role: &str, detail: &str, agent_name: &str) -> Line<'static> {
    let (label, color) = role_label(role, agent_name);
    Line::from(vec![
        Span::styled(
            format!(" {label} "),
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
        lines.push(role_header(
            &message.role,
            &message.created_at,
            &snapshot.agent_name,
        ));
        for line in math_render::message_lines(&message.content) {
            lines.push(message_body_line(&line));
        }
        if message.content.is_empty() {
            lines.push(message_body_line(""));
        }
        for attachment in &message.attachments {
            lines.push(Line::from(Span::styled(
                format!(
                    "  Attachment: {} ({}, {} bytes)",
                    attachment.filename, attachment.mime, attachment.size_bytes
                ),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::UNDERLINED),
            )));
        }
        lines.push(Line::from(""));
    }
    if app.voice_recording {
        lines.push(role_header("user", "listening", &snapshot.agent_name));
        let partial = app.voice_partial.trim();
        lines.push(message_body_line(if partial.is_empty() {
            "Listening..."
        } else {
            partial
        }));
        lines.push(Line::from(""));
    }
    if let Some(prompt) = &app.running_prompt {
        if is_attachment_management_bridge_prompt(prompt) {
            return Text::from(lines);
        }
        let prompt_is_persisted = snapshot.messages.last().is_some_and(|message| {
            message.role == "user" && message.content.trim() == prompt.trim()
        });
        if !prompt_is_persisted {
            lines.push(role_header("user", "in progress", &snapshot.agent_name));
            lines.push(message_body_line(prompt));
            lines.push(Line::from(""));
        }
    }
    if show_inline_activity && !app.reasoning_text.trim().is_empty() {
        let summary = clean_reasoning_summary(&app.reasoning_text);
        if !summary.is_empty() {
            for (index, line) in summary.lines().take(4).enumerate() {
                lines.push(Line::from(vec![
                    Span::styled(
                        if index == 0 { "• " } else { "  " },
                        Style::default().fg(Color::Blue),
                    ),
                    Span::styled(
                        line.to_string(),
                        Style::default().fg(if index == 0 {
                            Color::Blue
                        } else {
                            Color::DarkGray
                        }),
                    ),
                ]));
            }
        }
    }
    if show_inline_activity {
        if let Some(run) = app.subagent_run.as_ref() {
            append_subagent_run_lines(&mut lines, run, app.submitting);
        }
        const WORK_TRACE_LIMIT: usize = 6;
        let visible_activity = app
            .activity
            .iter()
            .filter(|item| item.kind != "thinking")
            .count();
        for item in app
            .activity
            .iter()
            .filter(|item| item.kind != "thinking")
            .skip(visible_activity.saturating_sub(WORK_TRACE_LIMIT))
        {
            lines.push(Line::from(vec![
                Span::styled("• ", activity_style(&item.kind)),
                Span::styled(item.text.clone(), Style::default().fg(Color::Gray)),
            ]));
        }
        if app.subagent_run.is_some() || app.activity.iter().any(|item| item.kind != "thinking") {
            lines.push(Line::from(""));
        }
    }
    if app.submitting && !app.streaming_text.trim().is_empty() {
        lines.push(role_header("assistant", "drafting", &snapshot.agent_name));
        for line in app.streaming_text.lines() {
            lines.push(message_body_line(line));
        }
        lines.push(Line::from(""));
    }
    // Notices belong after the active conversation turn. Keeping them here
    // prevents setup or error text from splitting the prompt from its live
    // reasoning/tool progress.
    for notice in &app.notices {
        let color = if notice.error {
            Color::Red
        } else {
            Color::Cyan
        };
        let badge = if notice.error { " ERROR " } else { " SETUP " };
        lines.push(Line::from(vec![
            Span::styled(
                badge,
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
    if lines.is_empty() {
        lines.push(Line::from(vec![Span::styled(
            "What would you like to build? Type / for commands.",
            Style::default().fg(Color::DarkGray),
        )]));
    }
    Text::from(lines)
}

fn clean_reasoning_summary(value: &str) -> String {
    let mut cleaned = String::with_capacity(value.len());
    for line in value.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed == "<!-- -->" {
            continue;
        }
        if !cleaned.is_empty() {
            cleaned.push('\n');
        }
        cleaned.push_str(line);
    }
    cleaned
}

#[derive(Clone, Copy)]
enum PaletteSource<'a> {
    Root(&'a str),
    Command(&'a str),
}

impl PaletteSource<'_> {
    fn matches(self, stored: &str) -> bool {
        match self {
            Self::Root(marker) => stored == marker,
            Self::Command(command) => stored
                .strip_suffix(' ')
                .is_some_and(|stored_command| stored_command.eq_ignore_ascii_case(command)),
        }
    }

    fn into_shared(self) -> Arc<str> {
        match self {
            Self::Root(marker) => Arc::from(marker),
            Self::Command(command) => Arc::from(format!("{command} ")),
        }
    }
}

struct PaletteContext<'a> {
    source: PaletteSource<'a>,
    query: &'a str,
}

fn palette_context(input: &str) -> Option<PaletteContext<'_>> {
    if !input.starts_with('/') {
        return None;
    }
    let trimmed = input.trim_end();
    let mut words = trimmed.split_whitespace();
    let command = words.next().unwrap_or(&input[..1]);
    let query = words.next();
    if words.next().is_some() {
        return None;
    }
    let has_argument_slot = input[command.len()..].chars().any(char::is_whitespace);
    if has_argument_slot && command.eq_ignore_ascii_case("/name") {
        return None;
    }
    Some(if has_argument_slot {
        PaletteContext {
            source: PaletteSource::Command(command),
            query: query.unwrap_or_default(),
        }
    } else {
        PaletteContext {
            source: PaletteSource::Root(&input[..1]),
            query: command,
        }
    })
}

fn palette_entry_matches(entry: &BridgeCompletionEntry, query: &str) -> bool {
    query.is_empty()
        || starts_with_ignore_ascii_case(&entry.label, query)
        || starts_with_ignore_ascii_case(&entry.value, query)
}

fn starts_with_ignore_ascii_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|start| start.eq_ignore_ascii_case(prefix))
}

fn palette_has_matches(palette: &BridgeCompletions, query: &str) -> bool {
    palette
        .entries
        .iter()
        .filter(|entry| palette_entry_selectable(entry))
        .any(|entry| palette_entry_matches(entry, query))
}

fn palette_entry_visible(
    palette: &BridgeCompletions,
    index: usize,
    entry: &BridgeCompletionEntry,
    query: &str,
    has_matches: bool,
) -> bool {
    if !has_matches {
        return true;
    }
    if palette_entry_selectable(entry) {
        return palette_entry_matches(entry, query);
    }
    palette
        .entries
        .iter()
        .skip(index + 1)
        .take_while(|candidate| palette_entry_selectable(candidate))
        .any(|candidate| palette_entry_matches(candidate, query))
}

fn visible_palette_entries<'a>(
    palette: &'a BridgeCompletions,
    query: &'a str,
) -> impl Iterator<Item = (usize, &'a BridgeCompletionEntry)> + 'a {
    let has_matches = palette_has_matches(palette, query);
    palette
        .entries
        .iter()
        .enumerate()
        .filter(move |(index, entry)| {
            palette_entry_visible(palette, *index, entry, query, has_matches)
        })
}

fn visible_palette_indices<'a>(
    palette: &'a BridgeCompletions,
    query: &'a str,
) -> impl Iterator<Item = usize> + 'a {
    visible_palette_entries(palette, query).map(|(index, _)| index)
}

fn palette_is_open(app: &TuiApp) -> bool {
    app.secret_provider.is_none()
        && app.snapshot.approvals.is_empty()
        && palette_context(&app.input).is_some_and(|context| {
            visible_palette_indices(&app.palette, context.query)
                .next()
                .is_some()
        })
}

fn first_selectable_palette_index(palette: &BridgeCompletions, query: &str) -> Option<usize> {
    visible_palette_entries(palette, query)
        .find(|(_, entry)| palette_entry_selectable(entry))
        .map(|(index, _)| index)
}

fn last_selectable_palette_index(palette: &BridgeCompletions, query: &str) -> Option<usize> {
    visible_palette_entries(palette, query)
        .filter(|(_, entry)| palette_entry_selectable(entry))
        .map(|(index, _)| index)
        .last()
}

fn closest_selectable_palette_index(
    palette: &BridgeCompletions,
    query: &str,
    selected: usize,
) -> Option<usize> {
    visible_palette_entries(palette, query)
        .find(|(index, entry)| *index >= selected && palette_entry_selectable(entry))
        .map(|(index, _)| index)
        .or_else(|| {
            visible_palette_entries(palette, query)
                .filter(|(_, entry)| palette_entry_selectable(entry))
                .map(|(index, _)| index)
                .last()
        })
}

fn next_palette_index(palette: &BridgeCompletions, query: &str, selected: usize) -> usize {
    visible_palette_entries(palette, query)
        .find(|(index, entry)| *index > selected && palette_entry_selectable(entry))
        .map(|(index, _)| index)
        .unwrap_or(selected)
}

fn previous_palette_index(palette: &BridgeCompletions, query: &str, selected: usize) -> usize {
    visible_palette_entries(palette, query)
        .filter(|(index, entry)| *index < selected && palette_entry_selectable(entry))
        .map(|(index, _)| index)
        .last()
        .unwrap_or(selected)
}

fn move_palette_index(
    palette: &BridgeCompletions,
    query: &str,
    selected: usize,
    delta: isize,
) -> usize {
    let mut index = closest_selectable_palette_index(palette, query, selected).unwrap_or(0);
    let steps = delta.unsigned_abs();
    for _ in 0..steps {
        let next = if delta >= 0 {
            next_palette_index(palette, query, index)
        } else {
            previous_palette_index(palette, query, index)
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

fn palette_entry_height(entry: &BridgeCompletionEntry) -> usize {
    1 + usize::from(is_model_palette_entry(entry) && !entry.description.is_empty())
}

fn palette_hit_areas(
    palette: &BridgeCompletions,
    query: &str,
    selected: usize,
    line_budget: usize,
    popup_area: Rect,
) -> Vec<PaletteHitArea> {
    let (start, end, _) = palette_visible_window(palette, query, selected, line_budget);
    let mut row = popup_area.y.saturating_add(1);
    let mut remaining_lines = line_budget.max(1);
    let mut targets = Vec::new();
    for (index, entry) in visible_palette_entries(palette, query)
        .skip(start)
        .take(end.saturating_sub(start))
    {
        if remaining_lines == 0 {
            break;
        }
        let height = palette_entry_height(entry).min(remaining_lines) as u16;
        if palette_entry_selectable(entry) {
            targets.push(PaletteHitArea {
                area: Rect::new(
                    popup_area.x.saturating_add(1),
                    row,
                    popup_area.width.saturating_sub(2),
                    height,
                ),
                index,
            });
        }
        row = row.saturating_add(height);
        remaining_lines = remaining_lines.saturating_sub(height as usize);
    }
    targets
}

fn palette_visible_window(
    palette: &BridgeCompletions,
    query: &str,
    selected: usize,
    line_budget: usize,
) -> (usize, usize, usize) {
    let entries = visible_palette_entries(palette, query).collect::<Vec<_>>();
    let total = entries.len();
    if total == 0 {
        return (0, 0, 0);
    }

    let selected_position = entries
        .iter()
        .position(|(index, _)| *index == selected)
        .unwrap_or(0);
    let line_budget = line_budget.max(1);
    let mut start = selected_position;
    let mut used_lines = 0usize;
    for position in (0..=selected_position).rev() {
        let height = palette_entry_height(entries[position].1);
        if used_lines > 0 && used_lines.saturating_add(height) > line_budget {
            break;
        }
        start = position;
        used_lines = used_lines.saturating_add(height);
    }

    let mut end = start;
    used_lines = 0;
    for (position, (_, entry)) in entries.iter().enumerate().skip(start) {
        let height = palette_entry_height(entry);
        if used_lines > 0 && used_lines.saturating_add(height) > line_budget {
            break;
        }
        end = position + 1;
        used_lines = used_lines.saturating_add(height);
    }
    (start, end, total)
}

fn palette_title(
    palette: &BridgeCompletions,
    query: &str,
    selected: usize,
    line_budget: usize,
) -> String {
    let (start, end, total) = palette_visible_window(palette, query, selected, line_budget);
    if total == 0 {
        return palette.title.clone();
    }
    format!(
        "{} · {}-{}/{} · ↑↓ PgUp/PgDn wheel",
        palette.title,
        start + 1,
        end,
        total,
    )
}

fn scroll_active_view(app: &mut TuiApp, down: bool) {
    if let Some(preview) = app.attachment_preview.as_mut() {
        preview.scroll = if down {
            preview.scroll.saturating_add(3)
        } else {
            preview.scroll.saturating_sub(3)
        };
        return;
    }
    if let Some(view) = app.gateway_view.as_mut() {
        view.scroll = if down {
            view.scroll.saturating_add(3)
        } else {
            view.scroll.saturating_sub(3)
        };
        return;
    }
    if palette_is_open(app) {
        let query = palette_context(&app.input)
            .map(|context| context.query)
            .unwrap_or_default();
        app.palette_selected = if down {
            move_palette_index(&app.palette, query, app.palette_selected, 3)
        } else {
            move_palette_index(&app.palette, query, app.palette_selected, -3)
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
    query: &str,
    selected: usize,
    line_budget: usize,
    width: u16,
) -> Text<'static> {
    let mut lines = Vec::new();
    let label_width = if width > 64 { 24 } else { 16 };
    let line_budget = line_budget.max(1);
    let (start, end, _) = palette_visible_window(palette, query, selected, line_budget);
    let mut remaining_lines = line_budget;
    for (index, entry) in visible_palette_entries(palette, query)
        .skip(start)
        .take(end.saturating_sub(start))
    {
        if remaining_lines == 0 {
            break;
        }
        if !palette_entry_selectable(entry) {
            lines.push(Line::from(vec![
                Span::raw("   "),
                Span::styled(
                    clip_status(&entry.label, width.saturating_sub(4) as usize).into_owned(),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                ),
            ]));
            remaining_lines = remaining_lines.saturating_sub(1);
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
        if is_model_palette_entry(entry) {
            lines.push(Line::from(vec![
                Span::styled(
                    if index == selected { " > " } else { "   " },
                    selected_style,
                ),
                Span::styled(
                    if palette.selected_index == Some(index) {
                        "✓ "
                    } else {
                        "  "
                    },
                    Style::default().fg(Color::Green),
                ),
                Span::styled(
                    entry.label.clone(),
                    Style::default().fg(if index == selected {
                        Color::Cyan
                    } else {
                        Color::White
                    }),
                ),
            ]));
            remaining_lines = remaining_lines.saturating_sub(1);
            if remaining_lines > 0 && !entry.description.is_empty() {
                lines.push(Line::from(vec![
                    Span::raw("       "),
                    Span::styled(
                        entry.description.clone(),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
                remaining_lines = remaining_lines.saturating_sub(1);
            }
            continue;
        }
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
                )
                .into_owned(),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        remaining_lines = remaining_lines.saturating_sub(1);
    }
    Text::from(lines)
}

fn is_model_palette_entry(entry: &BridgeCompletionEntry) -> bool {
    entry.execute && entry.complete_to.starts_with("/model ")
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
    invalidate_transcript(app);
}

fn clip_status(value: &str, limit: usize) -> Cow<'_, str> {
    let compact = compact_whitespace(value);
    if compact.chars().count() <= limit {
        return compact;
    }
    if limit <= 3 {
        return Cow::Owned(compact.chars().take(limit).collect());
    }
    let mut clipped = compact.chars().take(limit - 3).collect::<String>();
    clipped.push_str("...");
    Cow::Owned(clipped)
}

fn spawn_palette_worker(
    runtime: &RuntimeHandle,
    args: Arc<TuiArgs>,
    mut requests: watch::Receiver<Option<Arc<str>>>,
    results: mpsc::Sender<(Arc<str>, Result<BridgeResponse>)>,
) {
    runtime.spawn(async move {
        while requests.changed().await.is_ok() {
            let request = match requests.borrow_and_update().clone() {
                Some(request) => request,
                None => continue,
            };
            let result = call_bridge_async(args.as_ref(), "complete", Some(request.as_ref())).await;
            if results.send((request, result)).is_err() {
                return;
            }
        }
    });
}

fn clear_palette(app: &mut TuiApp) {
    app.palette = BridgeCompletions::default();
    app.palette_source = None;
    app.palette_selected = 0;
}

fn request_palette_refresh(requests: &watch::Sender<Option<Arc<str>>>, app: &mut TuiApp) {
    let Some(context) = palette_context(&app.input) else {
        clear_palette(app);
        return;
    };
    if app
        .palette_source
        .as_deref()
        .is_some_and(|source| context.source.matches(source))
    {
        return;
    }

    let source = context.source.into_shared();
    clear_palette(app);
    app.palette_source = Some(Arc::clone(&source));
    if requests.is_closed() {
        app.palette_source = None;
        app.status = String::from("Command completion worker is unavailable");
        return;
    }
    requests.send_replace(Some(source));
}

fn apply_palette_result(app: &mut TuiApp, prompt: &str, result: Result<BridgeResponse>) {
    if !palette_context(&app.input).is_some_and(|context| context.source.matches(prompt)) {
        return;
    }

    match result {
        Ok(response) => {
            app.palette = response.completions.unwrap_or_default();
            app.palette_selected = app
                .palette
                .selected_index
                .unwrap_or(app.palette_selected)
                .min(app.palette.entries.len().saturating_sub(1));
            let query = palette_context(&app.input)
                .map(|context| context.query)
                .unwrap_or_default();
            app.palette_selected =
                closest_selectable_palette_index(&app.palette, query, app.palette_selected)
                    .unwrap_or(0);
        }
        Err(err) => {
            app.status = format!("Error: {}", clip_status(&err.to_string(), 96));
            clear_palette(app);
        }
    }
}

fn handle_app_event(app: &mut TuiApp, event: AppEvent) {
    invalidate_transcript(app);
    match event {
        AppEvent::StreamFrame(Ok(frame)) => match frame.kind.as_str() {
            "submitted" => {
                app.notices
                    .retain(|notice| notice.title != "Request failed");
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
                if app
                    .running_prompt
                    .as_deref()
                    .is_some_and(|prompt| prompt.trim_start().starts_with("/install "))
                {
                    app.activity.push(ActivityLine {
                        kind: String::from("install"),
                        text: if app
                            .running_prompt
                            .as_deref()
                            .is_some_and(install_command_is_confirmed)
                        {
                            String::from("Starting local download")
                        } else {
                            String::from("Reviewing model size and runtime requirements")
                        },
                    });
                }
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
                let transient = command_result.is_some_and(|result| result.transient);
                let next_command = command_result
                    .filter(|result| !result.transient)
                    .and_then(|result| result.next_command.clone());
                let key_prompt_provider =
                    command_result.and_then(|result| result.secret_provider.clone());
                app.submitting = false;
                if let Some(snapshot) = frame.snapshot {
                    app.snapshot = snapshot;
                    let max_index = app.snapshot.approvals.len().saturating_sub(1);
                    app.approval_selected = app.approval_selected.min(max_index);
                }
                if !transient {
                    ensure_final_frame_messages_visible(
                        app,
                        completed_prompt.as_deref(),
                        frame.answer.as_deref(),
                    );
                }
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
                    if transient {
                        let result = command_result.expect("transient command has a result");
                        app.setup_surface = Some(UiSetupSurface {
                            title: command_result_surface_title(result).to_string(),
                            text: frame.answer.clone().unwrap_or_default(),
                            error: result.error,
                            pending_action: result.pending_action.clone(),
                        });
                    } else if app.snapshot.session.configuration_state == "ready"
                        && !command_needs_setup
                    {
                        app.setup_surface = None;
                        app.notices
                            .retain(|notice| notice.title == "Update available");
                    } else {
                        app.setup_surface = session_setup_surface(&app.snapshot.session);
                    }
                    app.status = if completed_prompt
                        .as_deref()
                        .is_some_and(is_attachment_bridge_prompt)
                    {
                        app.snapshot
                            .session
                            .pending_attachments
                            .last()
                            .map(|attachment| {
                                format!("Attached {} · click the file to open", attachment.filename)
                            })
                            .unwrap_or_else(|| String::from("Attachment was not added"))
                    } else if completed_prompt
                        .as_deref()
                        .is_some_and(is_attachment_detach_bridge_prompt)
                    {
                        frame
                            .answer
                            .clone()
                            .unwrap_or_else(|| String::from("Attachment removed"))
                    } else {
                        command_result
                            .map(command_result_status)
                            .unwrap_or_else(|| String::from("Ready"))
                    };
                }
                if let Some(command) = next_command {
                    app.input = command;
                    clear_palette(app);
                }
                if let Some(provider) = key_prompt_provider {
                    app.secret_provider = Some(provider.clone());
                    app.secret_input.clear();
                    app.status = api_key_prompt_status(&provider);
                }
            }
            _ => {}
        },
        AppEvent::VoiceFrame {
            session_id,
            result: Ok(frame),
        } if app.voice_recording && session_id == app.voice_session_id => match frame.kind.as_str()
        {
            "delta" => {
                if let Some(delta) = frame.delta {
                    app.voice_partial.push_str(&delta);
                    replace_voice_composer_text(app, &app.voice_partial.clone());
                    app.status =
                        String::from("Listening and transcribing... select stop to keep text");
                }
            }
            "final" => {
                let transcript = frame
                    .transcript
                    .unwrap_or_else(|| app.voice_partial.clone());
                let transcript = transcript.trim();
                if transcript.is_empty() {
                    app.status = String::from("No speech was detected");
                } else {
                    replace_voice_composer_text(app, transcript);
                    let prompt = app.input.trim().to_string();
                    app.input.clear();
                    enqueue_prompt(&mut app.queued_prompts, prompt);
                    app.status = String::from("Sending voice message...");
                }
                clear_voice_recording_state(app);
            }
            "error" => {
                let error = frame
                    .error
                    .unwrap_or_else(|| String::from("Voice input failed"));
                clear_voice_recording_state(app);
                push_notice(app, "Voice input", &error, true);
                app.status = String::from("Voice input unavailable");
            }
            _ => {}
        },
        AppEvent::VoiceFrame { result: Ok(_), .. } => {}
        AppEvent::VoiceFrame {
            session_id,
            result: Err(err),
        } if session_id == app.voice_session_id => {
            clear_voice_recording_state(app);
            push_notice(app, "Voice input", &err.to_string(), true);
            app.status = String::from("Voice input unavailable");
        }
        AppEvent::VoiceFrame { result: Err(_), .. } => {}
        AppEvent::AttachmentMutation { operation, result } => {
            app.submitting = false;
            match *result {
                Ok(response) => {
                    if let Some(snapshot) = response.snapshot {
                        app.snapshot = snapshot;
                    }
                    app.status = match operation {
                        AttachmentMutation::Add { filename } => {
                            format!("Attached {filename}")
                        }
                        AttachmentMutation::Remove { filename } => {
                            format!("Removed {filename}")
                        }
                    };
                }
                Err(err) => {
                    push_notice(app, "Attachment", &err.to_string(), true);
                    app.status = String::from("Attachment update failed");
                }
            }
        }
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

fn replace_voice_composer_text(app: &mut TuiApp, transcript: &str) {
    if let Some(prefix) = app.voice_input_prefix.as_deref() {
        app.input.clear();
        app.input.push_str(prefix);
        app.input.push_str(transcript);
    }
}

fn clear_voice_recording_state(app: &mut TuiApp) {
    app.voice_recording = false;
    app.voice_cancel_signal = None;
    app.voice_input_prefix = None;
    app.voice_partial.clear();
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
        if !is_attachment_management_bridge_prompt(prompt)
            && !snapshot_contains_message(&app.snapshot, "user", prompt)
        {
            app.snapshot.messages.push(BridgeMessage {
                role: String::from("user"),
                content: prompt.to_string(),
                created_at: String::from("just now"),
                attachments: Vec::new(),
            });
        }
    }

    if !completed_prompt.is_some_and(is_attachment_management_bridge_prompt) {
        if let Some(answer) = answer.map(str::trim).filter(|value| !value.is_empty()) {
            if !snapshot_contains_message(&app.snapshot, "assistant", answer) {
                app.snapshot.messages.push(BridgeMessage {
                    role: String::from("assistant"),
                    content: answer.to_string(),
                    created_at: String::from("just now"),
                    attachments: Vec::new(),
                });
            }
        }
    }
}

fn is_attachment_bridge_prompt(prompt: &str) -> bool {
    prompt.split_whitespace().next() == Some("/__nym_attach")
}

fn is_attachment_detach_bridge_prompt(prompt: &str) -> bool {
    prompt.split_whitespace().next() == Some("/__nym_detach")
}

fn is_attachment_management_bridge_prompt(prompt: &str) -> bool {
    is_attachment_bridge_prompt(prompt) || is_attachment_detach_bridge_prompt(prompt)
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
            // Raw chain-of-thought remains private. A provider-supplied
            // reasoning summary can replace this neutral state below.
            if app.reasoning_text.is_empty() {
                app.reasoning_text = String::from("Thinking…");
            }
        }
        "reasoning_summary_delta" => {
            if let Some(delta) = event.delta {
                if app.reasoning_text == "Thinking…" {
                    app.reasoning_text.clear();
                }
                app.reasoning_text.push_str(&delta);
            }
        }
        "text_delta" => {
            if let Some(delta) = event.delta {
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
        "tool_result" => {
            if let Some(summary) = event.summary {
                if let Some(last) = app.activity.last_mut().filter(|item| item.kind == "tool") {
                    last.text = summary;
                } else {
                    app.activity.push(ActivityLine {
                        kind: String::from("tool"),
                        text: summary,
                    });
                }
            }
        }
        "approval_request" | "approval_decision" => {
            if let Some(summary) = event.summary {
                app.activity.push(ActivityLine {
                    kind: String::from("guardrail"),
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
                app.status = clip_status(&summary, 72).into_owned();
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
        app.status = clip_status(summary, 72).into_owned();
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
        "voice" => Some("AGENT_VOICE_API_KEY"),
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
        "voice" => "Voice",
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
        "voice" => Some("https://platform.openai.com/api-keys"),
        _ => None,
    }
}

fn provider_setup_surface(
    provider: &str,
    configuration: &str,
    configuration_state: &str,
) -> UiSetupSurface {
    let mut text = configuration.to_string();
    if configuration_state == "api_key_required" {
        text.push_str("\nPaste your own API key in the masked field below.");
    } else {
        text.push_str("\nComplete the provider setup, then return to Agent.");
    }
    if let Some(url) = provider_setup_url(provider) {
        text.push_str("\nAccount/API keys: ");
        text.push_str(url);
    }
    UiSetupSurface {
        title: format!("{} setup", provider_display_name(provider)),
        text,
        error: false,
        pending_action: None,
    }
}

fn session_setup_surface(session: &BridgeSession) -> Option<UiSetupSurface> {
    if !session.model_selected || session.configuration_state == "model_required" {
        return Some(UiSetupSurface {
            title: String::from("Choose a model"),
            text: String::from("Open /model and select a hosted or local model to start."),
            error: false,
            pending_action: None,
        });
    }
    (session.configuration_state != "ready").then(|| {
        provider_setup_surface(
            &session.provider,
            &session.configuration,
            &session.configuration_state,
        )
    })
}

fn required_text_api_key_provider(session: &BridgeSession) -> Option<String> {
    (session.configuration_state == "api_key_required").then(|| session.provider.clone())
}

fn api_key_prompt_status(provider: &str) -> String {
    format!(
        "Paste your {} API key. Input is hidden.",
        provider_display_name(provider)
    )
}

fn voice_setup_notice(voice: &BridgeVoice) -> Option<UiNotice> {
    if voice.input_ready || voice.input_secret_provider.is_none() {
        return None;
    }
    let mut text = voice
        .input_reason
        .clone()
        .unwrap_or_else(|| String::from("Voice needs an API key."));
    text.push_str(
        "\nSelect the microphone, then paste your own API key in the masked field below.",
    );
    Some(UiNotice {
        title: String::from("Voice setup"),
        text,
        error: false,
    })
}

fn push_notice(app: &mut TuiApp, title: &str, text: &str, error: bool) {
    const MAX_NOTICES: usize = 20;
    if app.notices.len() >= MAX_NOTICES {
        app.notices.pop_front();
    }
    app.notices.push_back(UiNotice {
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
    if app
        .setup_surface
        .as_ref()
        .is_some_and(|surface| surface.pending_action.is_some())
    {
        return ("confirm", Color::Yellow);
    }
    if app.status.starts_with("Error") || app.status.starts_with("Could not") {
        return ("error", Color::Red);
    }
    if app.setup_required || app.snapshot.session.configuration_state != "ready" {
        return ("setup", Color::Yellow);
    }
    ("ready", Color::Green)
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
        BridgeCommandCode::Incompatible => {
            String::from("Model is incompatible with native agent tool use")
        }
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

fn command_result_surface_title(result: &BridgeCommandResult) -> &'static str {
    match result.code {
        BridgeCommandCode::InstallConfirmationRequired => "Install local model",
        BridgeCommandCode::RuntimeNotInstalled | BridgeCommandCode::RuntimeUnavailable => {
            "Local runtime required"
        }
        BridgeCommandCode::InstallFailed | BridgeCommandCode::InstallUnverified => {
            "Local model install failed"
        }
        BridgeCommandCode::Ok => "Local model ready",
        _ => "Setup required",
    }
}

fn install_command_is_confirmed(command: &str) -> bool {
    let mut parts = command.split_whitespace();
    parts.next() == Some("/install")
        && parts.next().is_some()
        && parts.next().is_some()
        && parts.next() == Some("--yes")
        && parts.next().is_none()
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

    parse_bridge_output(command.output()?)
}

fn parse_bridge_output(output: std::process::Output) -> Result<BridgeResponse> {
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

async fn call_bridge_async(
    args: &TuiArgs,
    action: &str,
    prompt: Option<&str>,
) -> Result<BridgeResponse> {
    let mut command = AsyncProcessCommand::new(&args.python);
    command
        .arg("-m")
        .arg("agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .current_dir(&args.repo_root)
        .kill_on_drop(true);
    if let Ok(api_keys) = args.api_keys.lock() {
        for (env_name, api_key) in api_keys.iter() {
            command.env(env_name, api_key);
        }
    }
    if let Some(prompt) = prompt {
        command.arg("--bridge-prompt").arg(prompt);
    }

    parse_bridge_output(command.output().await?)
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

    parse_bridge_output(command.output()?)
}

async fn stream_voice_record(
    args: &TuiArgs,
    tx: mpsc::Sender<AppEvent>,
    session_id: u64,
    cancel_signal: Arc<AtomicBool>,
    active_voice_process: Arc<Mutex<Option<u32>>>,
) -> Result<()> {
    if cancel_signal.load(Ordering::SeqCst) {
        return Ok(());
    }
    let (mut child, pid, stdout, stderr) = {
        let mut active = active_voice_process
            .lock()
            .map_err(|_| anyhow::anyhow!("Could not claim the voice process slot."))?;
        if cancel_signal.load(Ordering::SeqCst) {
            return Ok(());
        }
        if active.is_some() {
            return Err(anyhow::anyhow!("Voice input is still stopping."));
        }
        let mut child = spawn_bridge_async(args, "voice-stream", None)?;
        let pid = child
            .id()
            .ok_or_else(|| anyhow::anyhow!("Voice bridge process id is unavailable."))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Voice bridge stdout was not captured."))?;
        let stderr = child.stderr.take();
        *active = Some(pid);
        (child, pid, stdout, stderr)
    };
    let stderr_reader = stderr.map(|mut pipe| {
        tokio::spawn(async move {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text).await;
            text
        })
    });
    let mut reader = AsyncBufReader::new(stdout).lines();
    let mut saw_terminal_frame = false;
    let stream_result: Result<()> = async {
        while let Some(line) = reader.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }
            let frame: BridgeVoiceFrame = serde_json::from_str(&line).map_err(|err| {
                anyhow::anyhow!(
                    "Could not parse voice stream frame: {}. line: {}",
                    err,
                    line
                )
            })?;
            if matches!(frame.kind.as_str(), "final" | "error") {
                saw_terminal_frame = true;
            }
            if tx
                .send(AppEvent::VoiceFrame {
                    session_id,
                    result: Ok(frame),
                })
                .is_err()
            {
                return Err(anyhow::anyhow!("UI event receiver closed"));
            }
        }
        Ok(())
    }
    .await;
    if stream_result.is_err() && !cancel_signal.load(Ordering::SeqCst) {
        let _ = interrupt_process_tree_pid(pid);
    }
    let status_result = child.wait().await;
    let stderr = match stderr_reader {
        Some(handle) => handle.await.unwrap_or_default(),
        None => String::new(),
    };
    if let Ok(mut active) = active_voice_process.lock() {
        if *active == Some(pid) {
            *active = None;
        }
    }
    if cancel_signal.load(Ordering::SeqCst) {
        return Ok(());
    }
    stream_result?;
    let status = status_result?;
    if !saw_terminal_frame {
        let detail = stderr.trim();
        if detail.is_empty() {
            return Err(anyhow::anyhow!("Voice bridge exited with {}", status));
        }
        return Err(anyhow::anyhow!(
            "Voice bridge exited with {}: {}",
            status,
            clip_status(detail, 160)
        ));
    }
    Ok(())
}

async fn stream_bridge_submit(
    args: &TuiArgs,
    prompt: &str,
    tx: mpsc::Sender<AppEvent>,
    active_bridge: Arc<Mutex<Option<AsyncChild>>>,
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
        let mut child = spawn_bridge_async(args, "stream-submit", Some(prompt))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Bridge stdout was not captured."))?;
        let stderr = child.stderr.take();
        *active = Some(child);
        if cancel_signal.load(Ordering::SeqCst) {
            if let Some(child) = active.as_mut() {
                if let Some(pid) = child.id() {
                    let _ = interrupt_process_tree_pid(pid);
                }
            }
        }
        (stdout, stderr)
    };
    let stderr_reader = stderr.map(|mut pipe| {
        tokio::spawn(async move {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text).await;
            text
        })
    });
    let mut reader = AsyncBufReader::new(stdout).lines();
    let mut saw_final_frame = false;
    let mut stream_error: Option<anyhow::Error> = None;
    loop {
        let line = match reader.next_line().await {
            Ok(Some(line)) => line,
            Ok(None) => break,
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
        let speech = if parsed.kind == "final"
            && parsed
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.voice.auto_speak && snapshot.voice.tts_ready)
        {
            parsed.answer.clone().filter(|text| !text.trim().is_empty())
        } else {
            None
        };
        if tx
            .send(AppEvent::StreamFrame(Ok(Box::new(parsed))))
            .is_err()
        {
            stream_error = Some(anyhow::anyhow!("UI event receiver closed"));
            break;
        }
        if let Some(text) = speech {
            let voice_args = args.clone();
            tokio::spawn(async move {
                let _ = call_bridge_async(&voice_args, "voice-speak", Some(&text)).await;
            });
        }
    }
    if stream_error.is_some() {
        if let Ok(active) = active_bridge.lock() {
            if let Some(child) = active.as_ref() {
                if let Some(pid) = child.id() {
                    let _ = interrupt_process_tree_pid(pid);
                }
            }
        }
    }
    let mut child = {
        let mut active = active_bridge
            .lock()
            .map_err(|_| anyhow::anyhow!("Could not finish the active bridge process."))?;
        active
            .take()
            .ok_or_else(|| anyhow::anyhow!("Active bridge process was lost."))?
    };
    let status = child.wait().await?;
    let stderr = match stderr_reader {
        Some(handle) => handle.await.unwrap_or_default(),
        None => String::new(),
    };
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

        interrupt_process_tree_pid(child.id()).expect("interrupt process group");
        let status = child.wait().expect("wait for interrupted task");

        assert!(!status.success());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn second_stream_bridge_is_rejected_while_one_is_active() {
        let child = AsyncProcessCommand::new("sh")
            .args(["-c", "sleep 30"])
            .spawn()
            .expect("spawn active bridge placeholder");
        let active_bridge = Arc::new(Mutex::new(Some(child)));
        let args = TuiArgs {
            python: String::from("python3"),
            repo_root: PathBuf::from("."),
            session_id: String::from("session-1"),
            paste_keys: Vec::new(),
            copy_keys: Vec::new(),
            mouse_capture: false,
            api_keys: Arc::new(Mutex::new(HashMap::new())),
        };
        let (tx, _rx) = mpsc::channel();

        let result = stream_bridge_submit(
            &args,
            "second prompt",
            tx,
            Arc::clone(&active_bridge),
            Arc::new(AtomicBool::new(false)),
        )
        .await;

        assert!(result
            .expect_err("second bridge must be rejected")
            .to_string()
            .contains("already active"));
        let mut child = active_bridge
            .lock()
            .expect("active bridge lock")
            .take()
            .expect("active bridge child");
        child.start_kill().expect("stop placeholder bridge");
        child.wait().await.expect("reap placeholder bridge");
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

fn spawn_bridge_async(args: &TuiArgs, action: &str, prompt: Option<&str>) -> Result<AsyncChild> {
    let mut command = AsyncProcessCommand::new(&args.python);
    command
        .arg("-m")
        .arg("agent.main")
        .arg("--tui-bridge")
        .arg(action)
        .arg("--bridge-session-id")
        .arg(&args.session_id)
        .current_dir(&args.repo_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    if let Ok(api_keys) = args.api_keys.lock() {
        for (env_name, api_key) in api_keys.iter() {
            command.env(env_name, api_key);
        }
    }
    if let Some(prompt) = prompt {
        command.arg("--bridge-prompt").arg(prompt);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.as_std_mut().process_group(0);
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
            configuration: String::from("Anthropic is not configured. Set ANTHROPIC_API_KEY."),
            configuration_state: String::from("api_key_required"),
            model_selected: true,
            _context_limit: Some(128_000),
            pending_attachments: Vec::new(),
            tokens: BridgeTokens {
                input: 0,
                output: 0,
                reasoning: 0,
                cache_read: 0,
                cache_write: 0,
            },
            cost_usd: 0.0,
            costs: BridgeCosts::default(),
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
                agent_name: String::from("Agent"),
                approvals: Vec::new(),
                voice: BridgeVoice::default(),
                messages: Vec::new(),
            },
            input: String::new(),
            attachment_path_mode: false,
            attachment_button_area: None,
            mic_button_area: None,
            cost_button_area: None,
            pending_action_area: None,
            attachment_hit_areas: Vec::new(),
            palette_hit_areas: Vec::new(),
            attachment_preview: None,
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
            palette_source: None,
            palette_selected: 0,
            transcript_cache: None,
            approval_selected: 0,
            current_tool: None,
            reasoning_text: String::new(),
            streaming_text: String::new(),
            running_prompt: None,
            secret_provider: None,
            secret_input: String::new(),
            notices: VecDeque::new(),
            setup_required: false,
            setup_surface: None,
            gateway_view: None,
            show_cost_details: false,
            voice_recording: false,
            voice_session_id: 0,
            voice_cancel_signal: None,
            active_voice_process: Arc::new(Mutex::new(None)),
            voice_input_prefix: None,
            voice_partial: String::new(),
            paste_keys: parse_paste_keys(&[
                String::from("ctrl+v"),
                String::from("ctrl+shift+v"),
                String::from("shift+insert"),
                String::from("alt+v"),
            ]),
            copy_keys: parse_paste_keys(&[String::from("alt+c"), String::from("ctrl+y")]),
            mouse_capture: false,
            transcript_drag_start: None,
            transcript_selection: None,
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
    fn parallel_subagent_events_are_visible_in_conversation_trace() {
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

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));
        assert!(rendered.contains("1 agent running"));
        assert!(rendered.contains("architecture"));
        assert!(rendered.contains("Running"));
        assert!(rendered.contains("inspect tests"));
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
        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));
        let compact = rendered.replace('\n', "");
        assert!(rendered.contains("1/2 agents finished"));
        assert!(compact.contains("python  Done"));
        assert!(compact.contains("rust  Failed"));
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
        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));
        assert!(rendered.contains("implement client"));
        assert!(!rendered.contains("write_file"));
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
    fn provider_reasoning_summary_is_inline_but_raw_reasoning_is_not() {
        let mut app = test_app();
        app.submitting = true;
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("reasoning_started"),
                ..BridgeEvent::default()
            },
        );
        apply_bridge_event(
            &mut app,
            BridgeEvent {
                kind: String::from("reasoning_summary_delta"),
                delta: Some(String::from("Inspecting the affected files")),
                ..BridgeEvent::default()
            },
        );

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));

        assert!(rendered.contains("Inspecting the affected files"));
        assert!(!rendered.contains("Reasoning ·"));
        assert!(!rendered.contains("ACTIVITY"));
    }

    #[test]
    fn reasoning_summary_filters_empty_protocol_placeholders() {
        assert_eq!(
            clean_reasoning_summary("**Checking**\n<!-- -->\n\nReading tests"),
            "**Checking**\nReading tests"
        );
    }

    #[test]
    fn transcript_renders_delimited_math_as_terminal_text() {
        let mut app = test_app();
        app.snapshot.messages.push(BridgeMessage {
            role: String::from("assistant"),
            content: String::from(concat!(
                "The standard formula is:\n\n",
                "\\[\n",
                "\\text{Attention}(Q, K, V) = ",
                "\\text{softmax}\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V\n",
                "\\]"
            )),
            created_at: String::from("now"),
            attachments: Vec::new(),
        });

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));

        assert!(rendered.contains("Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V"));
        assert!(!rendered.contains(r"\text"));
        assert!(!rendered.contains(r"\["));
        assert!(!rendered.contains("Ô"));
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

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));
        assert!(!rendered.contains("ordinary tool trace"));
        assert!(rendered.contains("✓ 2 agents finished"));
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
                attachments: Vec::new(),
            },
            BridgeMessage {
                role: String::from("assistant"),
                content: String::from("Here is the result."),
                created_at: String::from("now"),
                attachments: Vec::new(),
            },
        ];

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(Box::new(BridgeStreamFrame {
                kind: String::from("final"),
                prompt: None,
                answer: Some(String::from("Here is the result.")),
                error: None,
                event: None,
                snapshot: Some(snapshot),
                command_result: None,
            }))),
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
    fn assistant_headers_use_configured_agent_name() {
        let mut app = test_app();
        app.snapshot.agent_name = String::from("Nymi");
        app.snapshot.messages.push(BridgeMessage {
            role: String::from("assistant"),
            content: String::from("Ready."),
            created_at: String::from("now"),
            attachments: Vec::new(),
        });

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));

        assert!(rendered.contains("Nymi"));
        assert!(!rendered.contains("AGENT"));
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
            AppEvent::StreamFrame(Ok(Box::new(BridgeStreamFrame {
                kind: String::from("final"),
                prompt: None,
                answer: Some(String::from("Hi!")),
                error: None,
                event: None,
                snapshot: None,
                command_result: None,
            }))),
        );

        assert!(app.notices.is_empty());
        assert_eq!(app.status, "Ready");
    }

    #[test]
    fn submitting_a_new_request_clears_the_previous_request_error() {
        let mut app = test_app();
        push_notice(
            &mut app,
            "Request failed",
            "The previously selected local model is unavailable.",
            true,
        );

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(Box::new(BridgeStreamFrame {
                kind: String::from("submitted"),
                prompt: Some(String::from("/model ollama qwen2.5:0.5b")),
                answer: None,
                error: None,
                event: None,
                snapshot: None,
                command_result: None,
            }))),
        );

        assert!(app.notices.is_empty());
    }

    #[test]
    fn final_frame_without_fresh_snapshot_still_renders_prompt_and_answer() {
        let mut app = test_app();
        app.submitting = true;
        app.running_prompt = Some(String::from("project/"));
        app.streaming_text = String::from("Tell me the exact folder name or path.");

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(Box::new(BridgeStreamFrame {
                kind: String::from("final"),
                prompt: None,
                answer: Some(String::from("Tell me the exact folder name or path.")),
                error: None,
                event: None,
                snapshot: None,
                command_result: None,
            }))),
        );

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));
        assert!(!app.submitting);
        assert_eq!(app.status, "Ready");
        assert!(rendered.contains("project/"));
        assert!(rendered.contains("Tell me the exact folder name or path."));
        assert!(!rendered.contains("drafting"));
    }

    #[test]
    fn attachment_management_commands_stay_out_of_the_transcript() {
        let mut app = test_app();

        ensure_final_frame_messages_visible(
            &mut app,
            Some("/__nym_detach attachment-1"),
            Some("Removed attachment: notes.txt."),
        );

        assert!(app.snapshot.messages.is_empty());
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
    fn useful_live_trace_is_inline_at_every_layout_width() {
        let mut app = test_app();
        app.submitting = true;
        app.running_prompt = Some(String::from("inspect the project"));
        app.reasoning_text = String::from("Checking the relevant modules");
        app.streaming_text = String::from("I found the issue.");
        app.activity.push(ActivityLine {
            kind: String::from("tool"),
            text: String::from("Reading files"),
        });
        push_notice(&mut app, "Old setup notice", "No longer blocking", false);

        let wide_transcript = rendered_text(&transcript_text(&app.snapshot, &app, true));
        let narrow_transcript = rendered_text(&transcript_text(&app.snapshot, &app, true));
        assert!(wide_transcript.contains("Reading files"));
        assert!(narrow_transcript.contains("Reading files"));

        for rendered in [&wide_transcript, &narrow_transcript] {
            let prompt = rendered.find("inspect the project").expect("prompt");
            let reasoning = rendered
                .find("Checking the relevant modules")
                .expect("reasoning summary");
            let tool = rendered.find("Reading files").expect("tool progress");
            let answer = rendered.find("I found the issue.").expect("draft answer");
            let notice = rendered.find("Old setup notice").expect("notice");
            assert!(prompt < reasoning);
            assert!(reasoning < tool);
            assert!(tool < answer);
            assert!(answer < notice);
        }
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
            transient: false,
            pending_action: None,
        };

        assert_eq!(result.secret_provider.as_deref(), Some("openai"));
    }

    #[test]
    fn clipboard_text_is_pasted_into_masked_api_key_input() {
        let mut app = test_app();
        app.secret_provider = Some(String::from("openai"));
        let clipboard = json!({"ok": true, "text": "sk-test-secret"});

        assert!(insert_clipboard_text(&mut app, &clipboard));
        assert_eq!(app.secret_input, "sk-test-secret");
        assert!(app.input.is_empty());
        assert_eq!(app.status, "API key pasted into the protected field");
        assert!(!app.status.contains("sk-test-secret"));
    }

    #[test]
    fn clipboard_text_still_uses_the_normal_composer_without_secret_prompt() {
        let mut app = test_app();
        let clipboard = json!({"ok": true, "text": "ordinary text"});

        assert!(insert_clipboard_text(&mut app, &clipboard));
        assert_eq!(app.input, "ordinary text");
        assert!(app.secret_input.is_empty());
        assert_eq!(app.status, "Pasted text from the clipboard");
    }

    #[test]
    fn voice_provider_uses_masked_api_key_prompt() {
        assert_eq!(provider_api_key_env("voice"), Some("AGENT_VOICE_API_KEY"));
        assert_eq!(provider_display_name("voice"), "Voice");
    }

    #[test]
    fn missing_text_api_key_selects_masked_setup_provider() {
        let session = anthropic_session();

        assert_eq!(
            required_text_api_key_provider(&session).as_deref(),
            Some("anthropic")
        );
    }

    #[test]
    fn non_api_key_provider_state_does_not_select_masked_setup_provider() {
        let mut session = anthropic_session();
        session.configuration_state = String::from("credentials_required");
        assert_eq!(required_text_api_key_provider(&session), None);

        session.configuration_state = String::from("ready");
        assert_eq!(required_text_api_key_provider(&session), None);
    }

    #[test]
    fn missing_voice_key_produces_actionable_setup_notice() {
        let voice = BridgeVoice {
            input_ready: false,
            input_reason: Some(String::from("Voice needs an API key.")),
            input_secret_provider: Some(String::from("voice")),
            tts_ready: false,
            auto_speak: false,
        };

        let notice = voice_setup_notice(&voice).expect("voice setup notice");
        assert_eq!(notice.title, "Voice setup");
        assert!(notice.text.contains("Select the microphone"));
        assert!(notice.text.contains("masked field"));
        assert!(!notice.error);
    }

    #[test]
    fn ready_voice_does_not_produce_setup_notice() {
        let voice = BridgeVoice {
            input_ready: true,
            input_reason: None,
            input_secret_provider: None,
            tts_ready: false,
            auto_speak: false,
        };

        assert!(voice_setup_notice(&voice).is_none());
    }

    #[test]
    fn voice_deltas_update_composer_and_final_transcript_submits_automatically() {
        let mut app = test_app();
        app.input = String::from("Ask Nym ");
        app.voice_recording = true;
        app.voice_input_prefix = Some(app.input.clone());

        handle_app_event(
            &mut app,
            AppEvent::VoiceFrame {
                session_id: 0,
                result: Ok(BridgeVoiceFrame {
                    kind: String::from("delta"),
                    delta: Some(String::from("helo")),
                    transcript: None,
                    error: None,
                }),
            },
        );
        assert_eq!(app.input, "Ask Nym helo");
        assert!(app.voice_recording);

        handle_app_event(
            &mut app,
            AppEvent::VoiceFrame {
                session_id: 0,
                result: Ok(BridgeVoiceFrame {
                    kind: String::from("final"),
                    delta: None,
                    transcript: Some(String::from("hello")),
                    error: None,
                }),
            },
        );
        assert!(app.input.is_empty());
        assert!(!app.voice_recording);
        assert_eq!(
            app.queued_prompts.front().map(String::as_str),
            Some("Ask Nym hello")
        );
        assert_eq!(app.status, "Sending voice message...");
    }

    #[test]
    fn stopping_voice_keeps_partial_transcript_without_submitting_it() {
        let mut app = test_app();
        let cancel_signal = Arc::new(AtomicBool::new(false));
        app.input = String::from("partially transcribed message");
        app.voice_recording = true;
        app.voice_input_prefix = Some(String::new());
        app.voice_partial = app.input.clone();
        app.voice_cancel_signal = Some(Arc::clone(&cancel_signal));

        stop_voice_recording(&mut app);

        assert!(!app.voice_recording);
        assert!(cancel_signal.load(Ordering::SeqCst));
        assert_eq!(app.input, "partially transcribed message");
        assert!(app.queued_prompts.is_empty());
        assert_eq!(
            app.status,
            "Voice input stopped — transcript kept in message"
        );
    }

    #[test]
    fn stopped_voice_session_cannot_write_into_a_new_recording() {
        let mut app = test_app();
        app.voice_recording = true;
        app.voice_session_id = 2;
        app.voice_input_prefix = Some(String::new());

        handle_app_event(
            &mut app,
            AppEvent::VoiceFrame {
                session_id: 1,
                result: Ok(BridgeVoiceFrame {
                    kind: String::from("delta"),
                    delta: Some(String::from("stale transcript")),
                    transcript: None,
                    error: None,
                }),
            },
        );

        assert!(app.input.is_empty());
        assert!(app.voice_partial.is_empty());
        assert!(app.voice_recording);
    }

    #[test]
    fn voice_partial_is_visible_as_an_in_progress_chat_message() {
        let mut app = test_app();
        app.voice_recording = true;
        app.voice_partial = String::from("hello from the microphone");

        let rendered = rendered_text(&transcript_text(&app.snapshot, &app, true));

        assert!(rendered.contains("YOU"));
        assert!(rendered.contains("listening"));
        assert!(rendered.contains("hello from the microphone"));
    }

    #[test]
    fn transcript_drag_selection_has_visible_background() {
        let area = Rect::new(2, 3, 10, 5);
        let mut buffer = ratatui::buffer::Buffer::empty(Rect::new(0, 0, 20, 10));
        let lines = vec![String::from("abcdef"), String::from("ghij")];
        render_transcript_selection(
            &mut buffer,
            area,
            &lines,
            TranscriptSelection {
                start: SelectionPoint { row: 1, col: 2 },
                end: SelectionPoint { row: 0, col: 1 },
            },
        );

        let selected = Color::Rgb(52, 78, 110);
        assert_eq!(
            buffer.cell((4, 4)).expect("selected first row").bg,
            selected
        );
        assert_eq!(
            buffer.cell((3, 5)).expect("selected second row").bg,
            selected
        );
        assert_eq!(buffer.cell((5, 5)).expect("selected end cell").bg, selected);
        assert_eq!(
            buffer.cell((3, 4)).expect("unselected cell").bg,
            Color::Reset
        );
        assert_eq!(
            buffer.cell((6, 5)).expect("cell after selection").bg,
            Color::Reset
        );
    }

    #[test]
    fn transcript_selection_remains_active_after_mouse_release() {
        let mut app = test_app();
        let start = SelectionPoint { row: 0, col: 1 };
        app.transcript_cache = Some(TranscriptCache {
            show_inline_activity: false,
            area: Rect::new(0, 0, 20, 5),
            auto_follow: true,
            requested_scroll: 0,
            paragraph: Paragraph::new("abcdef"),
            attachment_hit_areas: Vec::new(),
            selection_lines: vec![String::from("abcdef")],
        });
        app.transcript_drag_start = Some(start);
        app.transcript_selection = Some(TranscriptSelection {
            start,
            end: SelectionPoint { row: 0, col: 3 },
        });

        finish_transcript_selection(&mut app);

        assert!(app.transcript_drag_start.is_none());
        assert!(app.transcript_selection.is_some());
        assert_eq!(app.status, "Text selected · Ctrl+C copy");

        assert_eq!(take_transcript_selection(&mut app).as_deref(), Some("bcd"));
        assert!(app.transcript_selection.is_none());
    }

    #[test]
    fn ctrl_c_prioritizes_selection_then_active_task_then_exit() {
        let mut app = test_app();
        app.transcript_cache = Some(TranscriptCache {
            show_inline_activity: false,
            area: Rect::new(0, 0, 20, 5),
            auto_follow: true,
            requested_scroll: 0,
            paragraph: Paragraph::new("abcdef"),
            attachment_hit_areas: Vec::new(),
            selection_lines: vec![String::from("abcdef")],
        });
        app.transcript_selection = Some(TranscriptSelection {
            start: SelectionPoint { row: 0, col: 1 },
            end: SelectionPoint { row: 0, col: 3 },
        });
        app.submitting = true;

        assert_eq!(ctrl_c_action(&app), CtrlCAction::CopySelection);
        app.transcript_selection = None;
        assert_eq!(ctrl_c_action(&app), CtrlCAction::StopTask);
        app.submitting = false;
        assert_eq!(ctrl_c_action(&app), CtrlCAction::Exit);
    }

    #[test]
    fn late_voice_final_does_not_restore_a_submitted_transcript() {
        let mut app = test_app();
        app.input.clear();
        app.voice_recording = false;
        app.voice_input_prefix = None;

        handle_app_event(
            &mut app,
            AppEvent::VoiceFrame {
                session_id: 0,
                result: Ok(BridgeVoiceFrame {
                    kind: String::from("final"),
                    delta: None,
                    transcript: Some(String::from("close Spark")),
                    error: None,
                }),
            },
        );

        assert!(app.input.is_empty());
        assert_eq!(app.status, "Ready");
    }

    #[test]
    fn compatible_provider_endpoint_requirement_is_preserved_in_snapshot() {
        let mut session = anthropic_session();
        session.provider = String::from("openai-compatible");
        session.configuration = String::from(
            "OpenAI-compatible provider is not configured. Set AGENT_OPENAI_COMPAT_BASE_URL.",
        );
        session.configuration_state = String::from("endpoint_required");
        assert_eq!(session.configuration_state, "endpoint_required");
    }

    #[test]
    fn model_palette_window_uses_rendered_line_heights() {
        let mut palette = BridgeCompletions {
            title: String::from("Models"),
            selected_index: Some(0),
            entries: Vec::new(),
        };
        for index in 0..10 {
            palette.entries.push(BridgeCompletionEntry {
                value: format!("model-{index}"),
                label: format!("model-{index}"),
                description: String::from("Provider · Ready"),
                complete_to: format!("/model provider model-{index}"),
                execute: true,
            });
        }

        assert_eq!(palette_visible_window(&palette, "", 0, 8), (0, 4, 10));
        assert_eq!(palette_visible_window(&palette, "", 4, 8), (1, 5, 10));
        assert_eq!(palette_visible_window(&palette, "", 9, 8), (6, 10, 10));
    }

    #[test]
    fn palette_requests_follow_input_and_non_commands_clear_results() {
        let mut app = test_app();
        let (tx, mut rx) = watch::channel(None::<Arc<str>>);

        app.input = String::from("/mo");
        request_palette_refresh(&tx, &mut app);
        let request = rx
            .borrow_and_update()
            .clone()
            .expect("slash command request");
        assert_eq!(request.as_ref(), "/");

        app.palette.entries.push(test_palette_entry("model", true));
        app.input = String::from("ordinary prompt");
        request_palette_refresh(&tx, &mut app);
        assert!(app.palette.entries.is_empty());
        assert!(!rx.has_changed().expect("palette sender remains available"));

        app.input = String::from("/");
        request_palette_refresh(&tx, &mut app);
        assert_eq!(
            rx.borrow_and_update().as_deref().expect("new root request"),
            "/"
        );
    }

    #[test]
    fn palette_reuses_a_source_and_filters_entries_without_new_requests() {
        let mut app = test_app();
        let (tx, mut rx) = watch::channel(None::<Arc<str>>);

        app.input = String::from("/model ");
        request_palette_refresh(&tx, &mut app);
        assert_eq!(
            rx.borrow_and_update().as_deref().expect("model request"),
            "/model "
        );

        app.palette.entries = vec![
            BridgeCompletionEntry {
                value: String::from("openai/gpt-small"),
                label: String::from("gpt-small"),
                description: String::new(),
                complete_to: String::from("/model openai gpt-small"),
                execute: true,
            },
            BridgeCompletionEntry {
                value: String::from("anthropic/claude-small"),
                label: String::from("claude-small"),
                description: String::new(),
                complete_to: String::from("/model anthropic claude-small"),
                execute: true,
            },
        ];
        app.input = String::from("/model gpt");
        request_palette_refresh(&tx, &mut app);

        assert!(!rx.has_changed().expect("palette sender remains available"));
        let context = palette_context(&app.input).expect("palette context");
        assert_eq!(
            visible_palette_indices(&app.palette, context.query).collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn palette_context_does_not_request_completion_for_complete_commands() {
        assert!(palette_context("/model openai gpt").is_none());
        assert!(palette_context("plain text").is_none());
        assert!(palette_context("/name ").is_none());
        assert!(palette_context("/name Nymi").is_none());
        assert!(matches!(
            palette_context("/model ").map(|context| context.source),
            Some(PaletteSource::Command("/model"))
        ));
    }

    #[test]
    fn stale_palette_results_cannot_replace_current_input_results() {
        let mut app = test_app();
        app.input = String::from("/current");
        app.palette
            .entries
            .push(test_palette_entry("current", true));

        apply_palette_result(&mut app, "/stale", Err(anyhow::anyhow!("stale")));

        assert_eq!(app.palette.entries[0].label, "current");
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

        let title = palette_title(&palette, "", 12, 8);

        assert!(title.contains("Models · 6-13/30"));
        assert!(title.contains("PgUp/PgDn"));
        assert!(title.contains("wheel"));
    }

    #[test]
    fn model_palette_keeps_full_long_model_names_visible() {
        let model = "Qwen/Qwen2.5-Coder-32B-Instruct";
        let palette = BridgeCompletions {
            title: String::from("Models"),
            selected_index: Some(0),
            entries: vec![BridgeCompletionEntry {
                value: format!("vllm/{model}"),
                label: String::from(model),
                description: String::from("vLLM · Ready · 32.5B params · ~65 GB · 128K ctx"),
                complete_to: format!("/model vllm {model}"),
                execute: true,
            }],
        };

        let rendered = rendered_text(&palette_text(&palette, "", 0, 1, 24));

        assert!(rendered.contains(model));
        assert!(!rendered.contains("..."));
    }

    #[test]
    fn model_palette_keeps_keyboard_cursor_and_active_model_visible() {
        let mut palette = BridgeCompletions {
            title: String::from("Models"),
            selected_index: Some(0),
            entries: Vec::new(),
        };
        for index in 0..8 {
            palette.entries.push(BridgeCompletionEntry {
                value: format!("model-{index}"),
                label: format!("model-{index}"),
                description: String::from("Provider · Ready"),
                complete_to: format!("/model provider model-{index}"),
                execute: true,
            });
        }

        let near_top = palette_text(&palette, "", 2, 8, 80);
        let active_line = near_top
            .lines
            .iter()
            .find(|line| line.spans.iter().any(|span| span.content == "model-0"))
            .expect("active model row");
        assert_eq!(active_line.spans[1].content, "✓ ");
        let selected_line = near_top
            .lines
            .iter()
            .find(|line| line.spans.iter().any(|span| span.content == "model-2"))
            .expect("selected model row");
        assert_eq!(selected_line.spans[0].content, " > ");

        let below_fold = palette_text(&palette, "", 6, 6, 80);
        let selected_line = below_fold
            .lines
            .iter()
            .find(|line| line.spans.iter().any(|span| span.content == "model-6"))
            .expect("selected model remains inside viewport");
        assert_eq!(selected_line.spans[0].content, " > ");
        assert!(below_fold.lines.len() <= 6);
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

        assert_eq!(closest_selectable_palette_index(&palette, "", 0), Some(1));
        assert_eq!(previous_palette_index(&palette, "", 4), 2);
        assert_eq!(next_palette_index(&palette, "", 2), 4);
        assert_eq!(move_palette_index(&palette, "", 0, 3), 4);
        assert_eq!(last_selectable_palette_index(&palette, ""), Some(4));
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
                test_palette_entry("/skills", true),
            ],
        };

        assert_eq!(closest_selectable_palette_index(&palette, "", 0), Some(0));
        assert_eq!(next_palette_index(&palette, "", 0), 1);
        assert_eq!(next_palette_index(&palette, "", 1), 2);
        assert_eq!(next_palette_index(&palette, "", 2), 3);
        assert_eq!(previous_palette_index(&palette, "", 3), 2);
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
    fn model_palette_mouse_targets_cover_labels_and_descriptions() {
        let palette = BridgeCompletions {
            title: String::from("Models"),
            selected_index: None,
            entries: vec![
                BridgeCompletionEntry {
                    value: String::from("section:ollama"),
                    label: String::from("Ollama"),
                    description: String::new(),
                    complete_to: String::from("/model "),
                    execute: false,
                },
                BridgeCompletionEntry {
                    value: String::from("ollama/qwen2.5-coder:3b"),
                    label: String::from("qwen2.5-coder:3b"),
                    description: String::from("Ollama · Ready · local"),
                    complete_to: String::from("/model ollama qwen2.5-coder:3b"),
                    execute: true,
                },
                BridgeCompletionEntry {
                    value: String::from("ollama/qwen3"),
                    label: String::from("qwen3"),
                    description: String::from("Ollama · Not installed"),
                    complete_to: String::from("/model ollama qwen3"),
                    execute: true,
                },
            ],
        };
        let targets = palette_hit_areas(&palette, "", 1, 6, Rect::new(10, 5, 80, 8));

        assert_eq!(targets.len(), 2);
        assert_eq!(targets[0].index, 1);
        assert_eq!(targets[0].area, Rect::new(11, 7, 78, 2));
        assert_eq!(targets[1].index, 2);
        assert_eq!(targets[1].area, Rect::new(11, 9, 78, 2));

        let mut app = test_app();
        app.palette = palette;
        app.palette_hit_areas = targets;
        assert_eq!(palette_index_for_mouse(&app, 20, 7), Some(1));
        assert_eq!(palette_index_for_mouse(&app, 20, 8), Some(1));
        assert_eq!(palette_index_for_mouse(&app, 20, 9), Some(2));
        assert_eq!(palette_index_for_mouse(&app, 20, 6), None);
    }

    #[test]
    fn approval_controls_take_priority_while_agent_is_waiting() {
        assert_eq!(
            footer_help_text(false, true, true, true),
            "Enter/Y approve  N/Esc deny  Ctrl+N/P select"
        );
    }

    #[test]
    fn attachment_control_is_icon_click_target_with_keyboard_shortcut_in_footer() {
        let text = rendered_text(&attachment_button_text(true));

        assert!(text.contains("+"));
        assert!(!text.contains("Add file"));
        assert!(!text.contains("F4"));
        assert!(footer_help_text(false, false, false, true).contains("+/Ctrl+O/F4 attach"));
        assert!(footer_help_text(false, false, false, true).contains("drag select"));
        assert!(footer_help_text(false, false, false, true).contains("Ctrl+C copy"));
        assert!(footer_help_text(false, false, false, true).contains("/help"));
        assert!(!footer_help_text(false, false, false, false).contains("drag select"));
    }

    #[test]
    fn pending_attachment_renders_as_clickable_file_chip() {
        let attachment = BridgeAttachment {
            id: Some(String::from("attachment-1")),
            filename: String::from("quarterly-report.pdf"),
            mime: String::from("application/pdf"),
            size_bytes: 123,
            storage_path: Some(String::from("/private/attachments/report")),
        };

        let (line, targets) = attachment_chip_line(&[attachment], 10, 20, 40);
        let rendered = rendered_text(&Text::from(line));

        assert!(rendered.contains("quarterly-report.pdf"));
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].area.x, 10);
        assert_eq!(targets[0].area.y, 20);
        assert!(rendered.contains('×'));
        assert_eq!(targets[0].attachment_id.as_deref(), Some("attachment-1"));
        assert!(targets[0].remove_area.is_some());
        assert_eq!(
            targets[0].storage_path.as_deref(),
            Some("/private/attachments/report")
        );
    }

    #[test]
    fn sent_attachment_remains_clickable_in_visible_transcript() {
        let mut app = test_app();
        app.snapshot.messages.push(BridgeMessage {
            role: String::from("user"),
            content: String::from("Review this"),
            created_at: String::from("now"),
            attachments: vec![BridgeAttachment {
                id: Some(String::from("attachment-1")),
                filename: String::from("report.pdf"),
                mime: String::from("application/pdf"),
                size_bytes: 123,
                storage_path: Some(String::from("/private/attachments/report")),
            }],
        });
        let text = transcript_text(&app.snapshot, &app, true);

        let targets =
            transcript_attachment_hit_areas(&text, &app.snapshot, Rect::new(0, 0, 80, 12), 0);

        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].filename, "report.pdf");
        assert_eq!(targets[0].area.y, 3);
    }

    #[test]
    fn attachment_click_opens_inline_text_preview_without_launching_viewer() {
        let path = std::env::temp_dir().join(format!(
            "nym-attachment-preview-{}-{}.txt",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, "first line\nsecond line").expect("write preview fixture");
        let mut app = test_app();
        app.attachment_hit_areas.push(AttachmentHitArea {
            area: Rect::new(5, 6, 20, 1),
            remove_area: None,
            attachment_id: None,
            filename: String::from("notes.txt"),
            mime: String::from("text/plain"),
            size_bytes: 22,
            storage_path: Some(path.to_string_lossy().into_owned()),
        });

        open_clicked_attachment(&mut app, 8, 6);

        let preview = app.attachment_preview.as_ref().expect("inline preview");
        assert_eq!(preview.filename, "notes.txt");
        assert!(preview
            .text
            .as_deref()
            .is_some_and(|text| text.contains("second line")));
        assert_eq!(app.status, "Previewing notes.txt");
        std::fs::remove_file(path).expect("remove preview fixture");
    }

    #[test]
    fn attachment_preview_file_sizes_are_human_readable() {
        assert_eq!(human_file_size(12), "12 B");
        assert_eq!(human_file_size(1536), "1.5 KiB");
        assert_eq!(human_file_size(2 * 1024 * 1024), "2.0 MiB");
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
            transient: false,
            pending_action: None,
        };

        assert!(result.setup_required);
        assert_eq!(
            command_result_status(&result),
            "Error — local model installation failed"
        );
    }

    #[test]
    fn incompatible_local_model_has_explicit_status() {
        let result = BridgeCommandResult {
            code: BridgeCommandCode::Incompatible,
            setup_required: true,
            error: true,
            secret_provider: None,
            next_command: None,
            transient: false,
            pending_action: None,
        };

        assert_eq!(
            command_result_status(&result),
            "Model is incompatible with native agent tool use"
        );
    }

    #[test]
    fn unselected_model_uses_neutral_startup_surface() {
        let mut session = anthropic_session();
        session.model_selected = false;
        session.configuration_state = String::from("model_required");

        let surface = session_setup_surface(&session).expect("model setup surface");

        assert_eq!(surface.title, "Choose a model");
        assert!(surface.text.contains("/model"));
        assert!(!surface.text.contains("API key"));
    }

    #[test]
    fn install_preview_uses_typed_pending_action() {
        let result = BridgeCommandResult {
            code: BridgeCommandCode::InstallConfirmationRequired,
            setup_required: false,
            error: false,
            secret_provider: None,
            next_command: None,
            transient: true,
            pending_action: Some(BridgePendingAction {
                kind: BridgePendingActionKind::InstallModel,
                label: String::from("Install"),
                command: String::from("/install ollama qwen3 --yes"),
            }),
        };

        assert_eq!(
            result
                .pending_action
                .as_ref()
                .map(|action| action.command.as_str()),
            Some("/install ollama qwen3 --yes")
        );
    }

    #[test]
    fn unconfirmed_install_is_presented_as_preview_not_download() {
        let mut app = test_app();
        app.submitting = true;

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(Box::new(BridgeStreamFrame {
                kind: String::from("submitted"),
                prompt: Some(String::from("/install ollama qwen3")),
                answer: None,
                error: None,
                event: None,
                snapshot: None,
                command_result: None,
            }))),
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
        let answer = "qwen3 via Ollama\nDownload ~5.2 GB · Memory 8 GB+ RAM\n\
                      Nothing has been downloaded.";

        handle_app_event(
            &mut app,
            AppEvent::StreamFrame(Ok(Box::new(BridgeStreamFrame {
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
                    next_command: None,
                    transient: true,
                    pending_action: Some(BridgePendingAction {
                        kind: BridgePendingActionKind::InstallModel,
                        label: String::from("Install"),
                        command: String::from("/install ollama qwen3 --yes"),
                    }),
                }),
            }))),
        );

        assert!(!app.submitting);
        assert!(app.input.is_empty());
        assert_eq!(app.snapshot.messages.len(), 0);
        assert_eq!(
            app.setup_surface
                .as_ref()
                .and_then(|surface| surface.pending_action.as_ref())
                .map(|action| action.command.as_str()),
            Some("/install ollama qwen3 --yes")
        );
        assert_eq!(ui_state_badge(&app).0, "confirm");
        assert!(app.status.contains("press Enter to install"));
    }

    #[test]
    fn footer_cost_text_formats_session_cost_for_corner_status() {
        assert_eq!(footer_cost_text(0.0), "cost $0");
        assert_eq!(footer_cost_text(0.0042), "cost $0.0042");
        assert_eq!(footer_cost_text(1.234), "cost $1.23");
    }

    #[test]
    fn cost_detail_rows_show_token_counts_and_component_costs() {
        let rendered = rendered_text(&Text::from(cost_detail_line(
            "Input",
            Some(12_345),
            0.0042,
            true,
        )));

        assert!(rendered.contains("Input"));
        assert!(rendered.contains("12,345 tokens"));
        assert!(rendered.contains("$0.0042"));
        assert_eq!(format_token_count(1_000_000), "1,000,000");
    }

    #[test]
    fn bridge_costs_keep_input_and_output_totals_separate() {
        let costs = BridgeCosts {
            input: 0.30,
            cached_input: 0.05,
            cache_write: 0.10,
            output: 0.55,
        };

        assert!((costs.input_total() - 0.45).abs() < f64::EPSILON);
        assert!((costs.total() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn sidebar_is_reserved_only_for_pending_approvals() {
        let mut app = test_app();

        assert!(!should_show_sidebar(120, &app));

        app.snapshot.approvals.push(BridgeApproval {
            id: String::from("req-1"),
            tool: String::from("delete_path"),
            reason: String::from("needs approval"),
            requested_path: Some(String::from("/tmp/example.txt")),
            display_path: None,
            translated_path: None,
            resolved_path: None,
        });

        assert!(should_show_sidebar(120, &app));
        assert!(!should_show_sidebar(80, &app));
    }
}
