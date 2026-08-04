use agent_rust::compact_whitespace;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sqlx::sqlite::{
    SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteRow, SqliteSynchronous,
};
use sqlx::{Executor, QueryBuilder, Row, Sqlite, SqliteConnection, SqlitePool};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::future::Future;
use std::path::{Component, Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{RwLock, Semaphore};

pub const DEFAULT_WRITE_LOCK_TIMEOUT_MS: u64 = 60_000;
const SCHEMA_VERSION: u32 = 1;
const DEFAULT_MAX_POOL_CONNECTIONS: u32 = 4;
const DEFAULT_SESSION_PRUNE_AFTER_DAYS: u32 = 30;
const DEFAULT_SESSION_MAX_ENTRIES: u32 = 500;
const STRICT_ENTRY_MAINTENANCE_MAX_ENTRIES: u32 = 49;
const MIN_BATCHED_ENTRY_MAINTENANCE_SLACK: u32 = 25;

const SCHEMA: &str = r#"
create table if not exists projects (
    id text primary key,
    root text not null unique,
    title text not null,
    created_at text not null,
    updated_at text not null
);

create table if not exists workspaces (
    id text primary key,
    project_id text not null,
    root text not null,
    cwd text not null,
    created_at text not null,
    updated_at text not null,
    foreign key (project_id) references projects(id) on delete cascade,
    unique (project_id, root, cwd)
);

create table if not exists sessions (
    id text primary key,
    project_id text,
    workspace_id text,
    title text not null,
    workspace_root text not null,
    cwd text,
    created_at text not null,
    updated_at text not null,
    provider text,
    model text,
    agent text,
    permission_json text,
    cost_usd real not null default 0,
    tokens_input integer not null default 0,
    tokens_output integer not null default 0,
    tokens_reasoning integer not null default 0,
    tokens_cache_read integer not null default 0,
    tokens_cache_write integer not null default 0,
    summary text,
    active_root text,
    focus_path text,
    last_prompt text,
    state_json text,
    foreign key (project_id) references projects(id),
    foreign key (workspace_id) references workspaces(id)
);

create table if not exists messages (
    id integer primary key autoincrement,
    session_id text not null,
    seq integer not null,
    role text not null,
    content text not null,
    created_at text not null,
    foreign key (session_id) references sessions(id) on delete cascade
);

create table if not exists attachments (
    id text primary key,
    filename text not null,
    mime text not null,
    size_bytes integer not null,
    sha256 text not null,
    storage_path text not null,
    source text not null,
    created_at text not null
);

create table if not exists message_attachments (
    message_id integer not null,
    attachment_id text not null,
    position integer not null,
    primary key (message_id, attachment_id),
    foreign key (message_id) references messages(id) on delete cascade,
    foreign key (attachment_id) references attachments(id) on delete cascade
);

create table if not exists events (
    id integer primary key autoincrement,
    session_id text not null,
    seq integer not null,
    event_type text not null,
    tool text,
    path text,
    summary text not null,
    args_json text,
    data_json text,
    created_at text not null,
    foreign key (session_id) references sessions(id) on delete cascade
);

create table if not exists session_routes (
    route_key text primary key,
    session_id text not null,
    agent_id text not null,
    scope text not null,
    channel text not null,
    account_id text not null,
    peer_kind text,
    peer_id text,
    sender_id text,
    guild_id text,
    team_id text,
    created_at text not null,
    updated_at text not null,
    foreign key (session_id) references sessions(id) on delete cascade
);

create index if not exists idx_sessions_updated_at
    on sessions(updated_at desc);
create index if not exists idx_sessions_project_updated_at
    on sessions(project_id, updated_at desc);
create index if not exists idx_messages_session_seq
    on messages(session_id, seq);
create index if not exists idx_message_attachments_message_id
    on message_attachments(message_id, position);
create index if not exists idx_events_session_seq
    on events(session_id, seq);
create index if not exists idx_events_session_created_at
    on events(session_id, created_at desc);
create index if not exists idx_session_routes_session_id
    on session_routes(session_id);
"#;

const DEFAULT_PERMISSION_JSON: &str =
    r#"{"read_files":"allow","write_files":"ask","shell":"ask","network":"ask"}"#;

type StoreResult<T> = Result<T, StoreError>;
type StoreFuture<'a, T> = Pin<Box<dyn Future<Output = StoreResult<T>> + Send + 'a>>;

#[derive(Clone, Debug, Deserialize)]
pub struct SessionStoreCall {
    pub db_path: PathBuf,
    #[serde(default = "default_write_lock_timeout_ms")]
    pub busy_timeout_ms: u64,
    #[serde(flatten)]
    pub request: SessionRequest,
}

fn default_write_lock_timeout_ms() -> u64 {
    DEFAULT_WRITE_LOCK_TIMEOUT_MS
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "operation", content = "params", rename_all = "snake_case")]
pub enum SessionRequest {
    Initialize,
    CreateSession {
        workspace_root: String,
        provider: Option<String>,
        model: Option<String>,
        title: Option<String>,
        #[serde(default = "default_agent_id")]
        agent_id: String,
    },
    GetRoute {
        route_key: String,
    },
    ListRoutesForSession {
        session_id: String,
    },
    ListRoutes {
        #[serde(default = "default_route_limit")]
        limit: u32,
    },
    GetOrCreateRoutedSession {
        route_key: String,
        workspace_root: String,
        agent_id: String,
        scope: String,
        channel: String,
        account_id: String,
        peer_kind: Option<String>,
        peer_id: Option<String>,
        sender_id: Option<String>,
        guild_id: Option<String>,
        team_id: Option<String>,
        provider: Option<String>,
        model: Option<String>,
        title: Option<String>,
    },
    ApplyMaintenance {
        #[serde(default = "default_session_max_entries")]
        max_entries: u32,
        #[serde(default = "default_session_prune_after_days")]
        prune_after_days: u32,
        active_session_id: Option<String>,
        #[serde(default)]
        force: bool,
        #[serde(default = "default_maintenance_mode")]
        mode: String,
    },
    GetSession {
        session_id: String,
    },
    ListSessions {
        #[serde(default = "default_session_list_limit")]
        limit: Option<u32>,
        agent_id: Option<String>,
        updated_after: Option<String>,
    },
    CountSessions {
        agent_id: Option<String>,
        updated_after: Option<String>,
    },
    UpdateLlmConfig {
        session_id: String,
        provider: Option<String>,
        model: Option<String>,
    },
    PatchSessionMetadata {
        session_id: String,
        title: Option<String>,
        provider: Option<String>,
        model: Option<String>,
        state_patch: Option<Map<String, Value>>,
    },
    ResetRoutedSession {
        route_key: String,
        #[serde(default = "default_reset_reason")]
        reason: String,
    },
    DeleteRoutedSession {
        route_key: String,
    },
    CompactRoutedSession {
        route_key: String,
        max_messages: u32,
    },
    ResolveSessionId {
        prefix: String,
    },
    AddMessage {
        session_id: String,
        role: String,
        content: String,
        expected_route_key: Option<String>,
    },
    AddMessages {
        session_id: String,
        messages: Vec<MessageInput>,
        last_prompt: Option<String>,
        expected_route_key: Option<String>,
    },
    ListMessages {
        session_id: String,
        #[serde(default = "default_message_limit")]
        limit: Option<u32>,
    },
    UpdateLastPrompt {
        session_id: String,
        prompt: String,
    },
    SaveAgentState {
        session_id: String,
        state: Map<String, Value>,
    },
    AddUsage {
        session_id: String,
        tokens: TokenUsage,
        #[serde(default)]
        cost_usd: f64,
    },
    AddEvent {
        session_id: String,
        event_type: String,
        summary: String,
        tool: Option<String>,
        args: Option<Map<String, Value>>,
        path: Option<String>,
        data: Option<Map<String, Value>>,
    },
    ListEvents {
        session_id: String,
        #[serde(default = "default_event_limit")]
        limit: u32,
    },
    CountSessionCompactionCheckpoints {
        session_id: String,
    },
    ListSessionCompactionCheckpoints {
        session_id: String,
        session_key: String,
        #[serde(default = "default_route_limit")]
        limit: u32,
    },
    GetSessionCompactionCheckpoint {
        session_id: String,
        session_key: String,
        checkpoint_id: String,
    },
    BranchRoutedSessionFromCompactionCheckpoint {
        route_key: String,
        checkpoint_id: String,
    },
    RestoreRoutedSessionFromCompactionCheckpoint {
        route_key: String,
        checkpoint_id: String,
    },
}

fn default_agent_id() -> String {
    "agent".to_owned()
}

fn default_route_limit() -> u32 {
    100
}

fn default_session_list_limit() -> Option<u32> {
    Some(50)
}

fn default_message_limit() -> Option<u32> {
    Some(20)
}

fn default_event_limit() -> u32 {
    20
}

fn default_session_max_entries() -> u32 {
    DEFAULT_SESSION_MAX_ENTRIES
}

fn default_session_prune_after_days() -> u32 {
    DEFAULT_SESSION_PRUNE_AFTER_DAYS
}

fn default_maintenance_mode() -> String {
    "enforce".to_owned()
}

fn default_reset_reason() -> String {
    "reset".to_owned()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub attachments: Vec<AttachmentInput>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AttachmentInput {
    pub id: String,
    pub filename: String,
    pub mime: String,
    pub size_bytes: i64,
    pub sha256: String,
    pub storage_path: String,
    pub source: String,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct TokenUsage {
    #[serde(default)]
    pub input: i64,
    #[serde(default)]
    pub output: i64,
    #[serde(default)]
    pub reasoning: i64,
    #[serde(default)]
    pub cache_read: i64,
    #[serde(default)]
    pub cache_write: i64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProjectInfo {
    pub id: String,
    pub root: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct WorkspaceInfo {
    pub id: String,
    pub project_id: String,
    pub root: String,
    pub cwd: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionInfo {
    pub id: String,
    pub project_id: Option<String>,
    pub workspace_id: Option<String>,
    pub title: String,
    pub workspace_root: String,
    pub cwd: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub agent: Option<String>,
    pub permission: Option<Map<String, Value>>,
    pub cost_usd: f64,
    pub tokens: TokenUsage,
    pub summary: Option<String>,
    pub active_root: Option<String>,
    pub focus_path: Option<String>,
    pub last_prompt: Option<String>,
    pub state: Option<Map<String, Value>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MessageInfo {
    pub id: i64,
    pub session_id: String,
    pub seq: i64,
    pub role: String,
    pub content: String,
    pub created_at: String,
    pub attachments: Vec<AttachmentInfo>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AttachmentInfo {
    pub id: String,
    pub filename: String,
    pub mime: String,
    pub size_bytes: i64,
    pub sha256: String,
    pub storage_path: String,
    pub source: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct EventInfo {
    pub id: i64,
    pub session_id: String,
    pub seq: i64,
    pub event_type: String,
    pub tool: Option<String>,
    pub args: Option<Map<String, Value>>,
    pub path: Option<String>,
    pub summary: String,
    pub data: Option<Map<String, Value>>,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionRouteInfo {
    pub route_key: String,
    pub session_id: String,
    pub agent_id: String,
    pub scope: String,
    pub channel: String,
    pub account_id: String,
    pub peer_kind: Option<String>,
    pub peer_id: Option<String>,
    pub sender_id: Option<String>,
    pub guild_id: Option<String>,
    pub team_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionMaintenanceReport {
    pub mode: String,
    pub before_count: i64,
    pub after_count: i64,
    pub pruned: usize,
    pub capped: usize,
    pub applied: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreErrorCode {
    InvalidArgument,
    NotFound,
    Ambiguous,
    RouteRebound,
    DatabaseBusy,
    Constraint,
    MigrationFailed,
    Internal,
}

#[derive(Clone, Debug, Serialize)]
pub struct StoreError {
    pub code: StoreErrorCode,
    pub message: String,
    pub retryable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

impl StoreError {
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StoreErrorCode::InvalidArgument, message, false)
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self::new(StoreErrorCode::NotFound, message, false)
    }

    fn ambiguous(message: impl Into<String>) -> Self {
        Self::new(StoreErrorCode::Ambiguous, message, false)
    }

    fn route_rebound(message: impl Into<String>, details: Value) -> Self {
        let mut error = Self::new(StoreErrorCode::RouteRebound, message, false);
        error.details = Some(details);
        error
    }

    fn migration(error: sqlx::Error) -> Self {
        Self::new(
            StoreErrorCode::MigrationFailed,
            format!(
                "Session database migration failed: {}",
                safe_sqlx_message(&error)
            ),
            false,
        )
    }

    fn internal(message: impl Into<String>) -> Self {
        Self::new(StoreErrorCode::Internal, message, false)
    }

    fn new(code: StoreErrorCode, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code,
            message: message.into(),
            retryable,
            details: None,
        }
    }
}

impl fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for StoreError {}

impl From<sqlx::Error> for StoreError {
    fn from(error: sqlx::Error) -> Self {
        if matches!(error, sqlx::Error::PoolTimedOut) {
            return Self::new(
                StoreErrorCode::DatabaseBusy,
                "Timed out waiting for a session database connection.",
                true,
            );
        }
        if let Some(database) = error.as_database_error() {
            let code = database.code();
            let code = code.as_deref().unwrap_or_default();
            if matches!(code, "5" | "6" | "261" | "262" | "517" | "518") {
                return Self::new(
                    StoreErrorCode::DatabaseBusy,
                    "The session database is busy.",
                    true,
                );
            }
            if code.starts_with("19") || matches!(code, "787" | "1555" | "2067") {
                return Self::new(
                    StoreErrorCode::Constraint,
                    "The session database rejected a conflicting change.",
                    false,
                );
            }
        }
        Self::new(
            StoreErrorCode::Internal,
            format!(
                "Session database operation failed: {}",
                safe_sqlx_message(&error)
            ),
            false,
        )
    }
}

fn safe_sqlx_message(error: &sqlx::Error) -> &'static str {
    match error {
        sqlx::Error::Configuration(_) => "invalid database configuration",
        sqlx::Error::Database(_) => "database error",
        sqlx::Error::Io(_) => "database I/O error",
        sqlx::Error::Tls(_) => "database transport error",
        sqlx::Error::Protocol(_) => "database protocol error",
        sqlx::Error::RowNotFound => "database row not found",
        sqlx::Error::TypeNotFound { .. } => "database type error",
        sqlx::Error::ColumnIndexOutOfBounds { .. }
        | sqlx::Error::ColumnNotFound(_)
        | sqlx::Error::ColumnDecode { .. }
        | sqlx::Error::Decode(_) => "database row decode error",
        sqlx::Error::AnyDriverError(_) => "database driver error",
        sqlx::Error::PoolTimedOut => "database pool timeout",
        sqlx::Error::PoolClosed => "database pool closed",
        sqlx::Error::WorkerCrashed => "database worker crashed",
        _ => "database error",
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct StoreKey {
    path: PathBuf,
    busy_timeout_ms: u64,
}

#[derive(Default)]
pub struct SessionStoreRegistry {
    stores: RwLock<HashMap<StoreKey, Arc<SessionStore>>>,
}

impl SessionStoreRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn handle(&self, call: SessionStoreCall) -> StoreResult<Value> {
        if call.busy_timeout_ms == 0 {
            return Err(StoreError::invalid(
                "busy_timeout_ms must be a positive integer.",
            ));
        }
        let path = normalize_db_path(&call.db_path)?;
        let key = StoreKey {
            path: path.clone(),
            busy_timeout_ms: call.busy_timeout_ms,
        };
        let store = self.open(key).await?;
        let result = if matches!(call.request, SessionRequest::Initialize) {
            serde_value(json!({
                "path": path,
                "busy_timeout_ms": call.busy_timeout_ms,
                "schema_version": SCHEMA_VERSION,
            }))
        } else {
            store.handle(call.request).await
        };
        set_private_database_permissions(&store.path)?;
        result
    }

    async fn open(&self, key: StoreKey) -> StoreResult<Arc<SessionStore>> {
        if let Some(store) = self.stores.read().await.get(&key).cloned() {
            return Ok(store);
        }
        let candidate = Arc::new(SessionStore::open(key.path.clone(), key.busy_timeout_ms).await?);
        let mut stores = self.stores.write().await;
        Ok(stores.entry(key).or_insert_with(|| candidate).clone())
    }

    pub async fn close(&self) {
        let stores = {
            let mut guard = self.stores.write().await;
            std::mem::take(&mut *guard)
        };
        for store in stores.into_values() {
            store.pool.close().await;
        }
    }
}

pub struct SessionStore {
    path: PathBuf,
    busy_timeout_ms: u64,
    pool: SqlitePool,
    write_gate: Semaphore,
}

impl SessionStore {
    async fn open(path: PathBuf, busy_timeout_ms: u64) -> StoreResult<Self> {
        let options = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true)
            .foreign_keys(true)
            .journal_mode(SqliteJournalMode::Wal)
            .synchronous(SqliteSynchronous::Full)
            .busy_timeout(Duration::from_millis(busy_timeout_ms));
        let acquire_timeout = Duration::from_millis(busy_timeout_ms.saturating_add(5_000));
        let pool = SqlitePoolOptions::new()
            .min_connections(1)
            .max_connections(DEFAULT_MAX_POOL_CONNECTIONS)
            .acquire_timeout(acquire_timeout)
            .connect_with(options)
            .await
            .map_err(StoreError::migration)?;
        set_private_database_permissions(&path)?;
        let store = Self {
            path,
            busy_timeout_ms,
            pool,
            write_gate: Semaphore::new(1),
        };
        store.initialize_schema().await?;
        set_private_database_permissions(&store.path)?;
        Ok(store)
    }

    pub async fn handle(&self, request: SessionRequest) -> StoreResult<Value> {
        match request {
            SessionRequest::Initialize => serde_value(json!({
                "path": self.path,
                "busy_timeout_ms": self.busy_timeout_ms,
                "schema_version": SCHEMA_VERSION,
            })),
            SessionRequest::CreateSession {
                workspace_root,
                provider,
                model,
                title,
                agent_id,
            } => serde_value(
                self.create_session(workspace_root, provider, model, title, agent_id)
                    .await?,
            ),
            SessionRequest::GetRoute { route_key } => {
                serde_value(self.get_route(&route_key).await?)
            }
            SessionRequest::ListRoutesForSession { session_id } => {
                serde_value(self.list_routes_for_session(&session_id).await?)
            }
            SessionRequest::ListRoutes { limit } => serde_value(self.list_routes(limit).await?),
            SessionRequest::GetOrCreateRoutedSession {
                route_key,
                workspace_root,
                agent_id,
                scope,
                channel,
                account_id,
                peer_kind,
                peer_id,
                sender_id,
                guild_id,
                team_id,
                provider,
                model,
                title,
            } => {
                let (session, created) = self
                    .get_or_create_routed_session(
                        route_key,
                        workspace_root,
                        agent_id,
                        scope,
                        channel,
                        account_id,
                        peer_kind,
                        peer_id,
                        sender_id,
                        guild_id,
                        team_id,
                        provider,
                        model,
                        title,
                    )
                    .await?;
                serde_value(json!({"session": session, "created": created}))
            }
            SessionRequest::ApplyMaintenance {
                max_entries,
                prune_after_days,
                active_session_id,
                force,
                mode,
            } => serde_value(
                self.apply_maintenance(
                    max_entries,
                    prune_after_days,
                    active_session_id,
                    force,
                    mode,
                )
                .await?,
            ),
            SessionRequest::GetSession { session_id } => {
                serde_value(self.get_session(&session_id).await?)
            }
            SessionRequest::ListSessions {
                limit,
                agent_id,
                updated_after,
            } => serde_value(
                self.list_sessions(limit, agent_id.as_deref(), updated_after.as_deref())
                    .await?,
            ),
            SessionRequest::CountSessions {
                agent_id,
                updated_after,
            } => serde_value(
                self.count_sessions(agent_id.as_deref(), updated_after.as_deref())
                    .await?,
            ),
            SessionRequest::UpdateLlmConfig {
                session_id,
                provider,
                model,
            } => {
                self.update_llm_config(&session_id, provider, model).await?;
                Ok(Value::Null)
            }
            SessionRequest::PatchSessionMetadata {
                session_id,
                title,
                provider,
                model,
                state_patch,
            } => serde_value(
                self.patch_session_metadata(session_id, title, provider, model, state_patch)
                    .await?,
            ),
            SessionRequest::ResetRoutedSession { route_key, reason } => {
                let (old_session, new_session) =
                    self.reset_routed_session(route_key, reason).await?;
                serde_value(json!({
                    "old_session": old_session,
                    "new_session": new_session,
                }))
            }
            SessionRequest::DeleteRoutedSession { route_key } => {
                serde_value(self.delete_routed_session(route_key).await?)
            }
            SessionRequest::CompactRoutedSession {
                route_key,
                max_messages,
            } => serde_value(self.compact_routed_session(route_key, max_messages).await?),
            SessionRequest::ResolveSessionId { prefix } => {
                serde_value(self.resolve_session_id(&prefix).await?)
            }
            SessionRequest::AddMessage {
                session_id,
                role,
                content,
                expected_route_key,
            } => {
                let mut messages = self
                    .add_messages(
                        session_id,
                        vec![MessageInput {
                            role,
                            content,
                            attachments: Vec::new(),
                        }],
                        None,
                        expected_route_key,
                    )
                    .await?;
                let message = messages
                    .pop()
                    .ok_or_else(|| StoreError::internal("Message insert returned no row."))?;
                serde_value(message)
            }
            SessionRequest::AddMessages {
                session_id,
                messages,
                last_prompt,
                expected_route_key,
            } => serde_value(
                self.add_messages(session_id, messages, last_prompt, expected_route_key)
                    .await?,
            ),
            SessionRequest::ListMessages { session_id, limit } => {
                serde_value(self.list_messages(&session_id, limit).await?)
            }
            SessionRequest::UpdateLastPrompt { session_id, prompt } => {
                self.update_last_prompt(session_id, prompt).await?;
                Ok(Value::Null)
            }
            SessionRequest::SaveAgentState { session_id, state } => {
                self.save_agent_state(session_id, state).await?;
                Ok(Value::Null)
            }
            SessionRequest::AddUsage {
                session_id,
                tokens,
                cost_usd,
            } => {
                self.add_usage(session_id, tokens, cost_usd).await?;
                Ok(Value::Null)
            }
            SessionRequest::AddEvent {
                session_id,
                event_type,
                summary,
                tool,
                args,
                path,
                data,
            } => serde_value(
                self.add_event(session_id, event_type, summary, tool, args, path, data)
                    .await?,
            ),
            SessionRequest::ListEvents { session_id, limit } => {
                serde_value(self.list_events(&session_id, limit).await?)
            }
            SessionRequest::CountSessionCompactionCheckpoints { session_id } => serde_value(
                self.count_session_compaction_checkpoints(&session_id)
                    .await?,
            ),
            SessionRequest::ListSessionCompactionCheckpoints {
                session_id,
                session_key,
                limit,
            } => serde_value(
                self.list_session_compaction_checkpoints(&session_id, &session_key, limit)
                    .await?,
            ),
            SessionRequest::GetSessionCompactionCheckpoint {
                session_id,
                session_key,
                checkpoint_id,
            } => serde_value(
                self.get_session_compaction_checkpoint(&session_id, &session_key, &checkpoint_id)
                    .await?,
            ),
            SessionRequest::BranchRoutedSessionFromCompactionCheckpoint {
                route_key,
                checkpoint_id,
            } => serde_value(
                self.branch_routed_session_from_compaction_checkpoint(route_key, checkpoint_id)
                    .await?,
            ),
            SessionRequest::RestoreRoutedSessionFromCompactionCheckpoint {
                route_key,
                checkpoint_id,
            } => serde_value(
                self.restore_routed_session_from_compaction_checkpoint(route_key, checkpoint_id)
                    .await?,
            ),
        }
    }
}

fn serde_value<T: Serialize>(value: T) -> StoreResult<Value> {
    serde_json::to_value(value)
        .map_err(|_| StoreError::internal("Could not serialize session database result."))
}

fn normalize_db_path(path: &Path) -> StoreResult<PathBuf> {
    if path.as_os_str().is_empty() {
        return Err(StoreError::invalid("db_path cannot be empty."));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|_| StoreError::internal("Could not resolve the session database path."))?
            .join(path)
    };
    let parent = absolute
        .parent()
        .ok_or_else(|| StoreError::invalid("db_path must have a parent directory."))?;
    let parent_existed = parent.exists();
    std::fs::create_dir_all(parent)
        .map_err(|_| StoreError::internal("Could not create the session database directory."))?;
    if !parent_existed {
        set_private_directory_permissions(parent)?;
    }
    let parent = parent
        .canonicalize()
        .map_err(|_| StoreError::internal("Could not resolve the session database directory."))?;
    let file_name = absolute
        .file_name()
        .ok_or_else(|| StoreError::invalid("db_path must name a database file."))?;
    Ok(lexically_normalize(&parent.join(file_name)))
}

#[cfg(unix)]
fn set_private_directory_permissions(path: &Path) -> StoreResult<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .map_err(|_| StoreError::internal("Could not secure the session database directory."))
}

#[cfg(not(unix))]
fn set_private_directory_permissions(_path: &Path) -> StoreResult<()> {
    Ok(())
}

#[cfg(unix)]
fn set_private_database_permissions(path: &Path) -> StoreResult<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|_| StoreError::internal("Could not secure the session database file."))?;
    for suffix in ["-wal", "-shm"] {
        let mut sidecar = path.as_os_str().to_os_string();
        sidecar.push(suffix);
        let sidecar = PathBuf::from(sidecar);
        if sidecar.exists() {
            std::fs::set_permissions(&sidecar, std::fs::Permissions::from_mode(0o600)).map_err(
                |_| StoreError::internal("Could not secure a session database sidecar."),
            )?;
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn set_private_database_permissions(_path: &Path) -> StoreResult<()> {
    Ok(())
}

fn lexically_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

impl SessionStore {
    async fn initialize_schema(&self) -> StoreResult<()> {
        self.immediate(|conn| {
            Box::pin(async move {
                conn.execute(SCHEMA).await.map_err(StoreError::migration)?;
                migrate_legacy_session_columns(conn).await?;
                backfill_session_context(conn).await?;
                Ok(())
            })
        })
        .await
    }

    async fn immediate<T, F>(&self, operation: F) -> StoreResult<T>
    where
        T: Send,
        F: for<'a> FnOnce(&'a mut SqliteConnection) -> StoreFuture<'a, T> + Send,
    {
        let _writer = self
            .write_gate
            .acquire()
            .await
            .map_err(|_| StoreError::internal("Session database writer is closed."))?;
        let mut connection = self.pool.acquire().await?;
        (&mut *connection).execute("BEGIN IMMEDIATE").await?;
        match operation(&mut connection).await {
            Ok(value) => {
                (&mut *connection).execute("COMMIT").await?;
                Ok(value)
            }
            Err(error) => {
                let _ = (&mut *connection).execute("ROLLBACK").await;
                Err(error)
            }
        }
    }

    async fn create_session(
        &self,
        workspace_root: String,
        provider: Option<String>,
        model: Option<String>,
        title: Option<String>,
        agent_id: String,
    ) -> StoreResult<SessionInfo> {
        let root = required_text(workspace_root, "workspace_root")?;
        let agent_id = required_text(agent_id, "agent_id")?;
        let session_id = self
            .immediate(|conn| {
                Box::pin(async move {
                    let now = db_now(conn).await?;
                    let project = ensure_project(conn, &root, &now).await?;
                    let workspace = ensure_workspace(conn, &project.id, &root, &root, &now).await?;
                    let session_id = db_id(conn).await?;
                    insert_session(
                        conn,
                        &session_id,
                        &project.id,
                        &workspace.id,
                        clean_title(title.as_deref()).unwrap_or_else(|| "New session".to_owned()),
                        &root,
                        &now,
                        provider.as_deref(),
                        model.as_deref(),
                        &agent_id,
                        None,
                    )
                    .await?;
                    Ok(session_id)
                })
            })
            .await?;
        let _ = self
            .apply_maintenance(
                DEFAULT_SESSION_MAX_ENTRIES,
                DEFAULT_SESSION_PRUNE_AFTER_DAYS,
                Some(session_id.clone()),
                false,
                "enforce".to_owned(),
            )
            .await?;
        self.get_session(&session_id).await
    }

    async fn get_route(&self, route_key: &str) -> StoreResult<SessionRouteInfo> {
        let route_key = required_ref(route_key, "route_key")?;
        let row = sqlx::query(
            "select route_key, session_id, agent_id, scope, channel, account_id,
                    peer_kind, peer_id, sender_id, guild_id, team_id, created_at, updated_at
             from session_routes where route_key = ?",
        )
        .bind(route_key)
        .fetch_optional(&self.pool)
        .await?;
        row.map(route_from_row)
            .transpose()?
            .ok_or_else(|| StoreError::not_found(format!("Session route not found: {route_key}")))
    }

    async fn list_routes_for_session(
        &self,
        session_id: &str,
    ) -> StoreResult<Vec<SessionRouteInfo>> {
        let rows = sqlx::query(
            "select route_key, session_id, agent_id, scope, channel, account_id,
                    peer_kind, peer_id, sender_id, guild_id, team_id, created_at, updated_at
             from session_routes where session_id = ? order by created_at",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(route_from_row).collect()
    }

    async fn list_routes(&self, limit: u32) -> StoreResult<Vec<SessionRouteInfo>> {
        let limit = limit.clamp(1, 500);
        let rows = sqlx::query(
            "select route_key, session_id, agent_id, scope, channel, account_id,
                    peer_kind, peer_id, sender_id, guild_id, team_id, created_at, updated_at
             from session_routes order by updated_at desc limit ?",
        )
        .bind(i64::from(limit))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(route_from_row).collect()
    }

    #[allow(clippy::too_many_arguments)]
    async fn get_or_create_routed_session(
        &self,
        route_key: String,
        workspace_root: String,
        agent_id: String,
        scope: String,
        channel: String,
        account_id: String,
        peer_kind: Option<String>,
        peer_id: Option<String>,
        sender_id: Option<String>,
        guild_id: Option<String>,
        team_id: Option<String>,
        provider: Option<String>,
        model: Option<String>,
        title: Option<String>,
    ) -> StoreResult<(SessionInfo, bool)> {
        let route_key = required_text(route_key, "route_key")?;
        let root = required_text(workspace_root, "workspace_root")?;
        let agent_id = required_text(agent_id, "agent_id")?;
        let scope = required_text(scope, "scope")?;
        let channel = required_text(channel, "channel")?;
        let account_id = required_text(account_id, "account_id")?;
        let (session_id, created) = self
            .immediate(|conn| {
                Box::pin(async move {
                    let now = db_now(conn).await?;
                    if let Some(row) =
                        sqlx::query("select session_id from session_routes where route_key = ?")
                            .bind(&route_key)
                            .fetch_optional(&mut *conn)
                            .await?
                    {
                        let session_id: String = row.try_get("session_id")?;
                        sqlx::query("update session_routes set updated_at = ? where route_key = ?")
                            .bind(&now)
                            .bind(&route_key)
                            .execute(&mut *conn)
                            .await?;
                        return Ok((session_id, false));
                    }
                    let project = ensure_project(conn, &root, &now).await?;
                    let workspace = ensure_workspace(conn, &project.id, &root, &root, &now).await?;
                    let session_id = db_id(conn).await?;
                    insert_session(
                        conn,
                        &session_id,
                        &project.id,
                        &workspace.id,
                        clean_title(title.as_deref()).unwrap_or_else(|| "New session".to_owned()),
                        &root,
                        &now,
                        provider.as_deref(),
                        model.as_deref(),
                        &agent_id,
                        None,
                    )
                    .await?;
                    sqlx::query(
                        "insert into session_routes (
                            route_key, session_id, agent_id, scope, channel, account_id,
                            peer_kind, peer_id, sender_id, guild_id, team_id, created_at, updated_at
                         ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    )
                    .bind(&route_key)
                    .bind(&session_id)
                    .bind(&agent_id)
                    .bind(&scope)
                    .bind(&channel)
                    .bind(&account_id)
                    .bind(peer_kind)
                    .bind(peer_id)
                    .bind(sender_id)
                    .bind(guild_id)
                    .bind(team_id)
                    .bind(&now)
                    .bind(&now)
                    .execute(&mut *conn)
                    .await?;
                    Ok((session_id, true))
                })
            })
            .await?;
        let _ = self
            .apply_maintenance(
                DEFAULT_SESSION_MAX_ENTRIES,
                DEFAULT_SESSION_PRUNE_AFTER_DAYS,
                Some(session_id.clone()),
                false,
                "enforce".to_owned(),
            )
            .await?;
        Ok((self.get_session(&session_id).await?, created))
    }

    async fn get_session(&self, session_id: &str) -> StoreResult<SessionInfo> {
        fetch_session(&self.pool, session_id).await
    }

    async fn list_sessions(
        &self,
        limit: Option<u32>,
        agent_id: Option<&str>,
        updated_after: Option<&str>,
    ) -> StoreResult<Vec<SessionInfo>> {
        let limit = limit.map(|value| value.clamp(1, 500));
        let mut query = QueryBuilder::<Sqlite>::new(SESSION_SELECT);
        push_session_filters(&mut query, agent_id, updated_after);
        query.push(" order by updated_at desc");
        if let Some(limit) = limit {
            query.push(" limit ").push_bind(i64::from(limit));
        }
        let rows = query.build().fetch_all(&self.pool).await?;
        rows.into_iter().map(session_from_row).collect()
    }

    async fn count_sessions(
        &self,
        agent_id: Option<&str>,
        updated_after: Option<&str>,
    ) -> StoreResult<i64> {
        let mut query = QueryBuilder::<Sqlite>::new("select count(*) as count from sessions");
        push_session_filters(&mut query, agent_id, updated_after);
        let row = query.build().fetch_one(&self.pool).await?;
        Ok(row.try_get("count")?)
    }

    async fn resolve_session_id(&self, prefix: &str) -> StoreResult<String> {
        let prefix = required_ref(prefix, "prefix")?;
        let rows = sqlx::query(
            "select id from sessions where id like ? order by updated_at desc limit 11",
        )
        .bind(format!("{prefix}%"))
        .fetch_all(&self.pool)
        .await?;
        match rows.as_slice() {
            [] => Err(StoreError::not_found(format!(
                "No session matches prefix: {prefix}"
            ))),
            [row] => Ok(row.try_get("id")?),
            rows => {
                let mut matches = String::new();
                for row in rows.iter().take(10) {
                    let id = row.try_get::<String, _>("id")?;
                    if !matches.is_empty() {
                        matches.push_str(", ");
                    }
                    matches.push_str(&id);
                }
                Err(StoreError::ambiguous(format!(
                    "Session prefix is ambiguous: {prefix}. Matches: {matches}"
                )))
            }
        }
    }

    async fn apply_maintenance(
        &self,
        max_entries: u32,
        prune_after_days: u32,
        active_session_id: Option<String>,
        force: bool,
        mode: String,
    ) -> StoreResult<SessionMaintenanceReport> {
        if max_entries == 0 {
            return Err(StoreError::invalid(
                "max_entries must be a positive integer.",
            ));
        }
        if prune_after_days == 0 {
            return Err(StoreError::invalid(
                "prune_after_days must be a positive integer.",
            ));
        }
        let mode = mode.trim().to_ascii_lowercase();
        if !matches!(mode.as_str(), "enforce" | "warn") {
            return Err(StoreError::invalid(
                "Session maintenance mode must be 'enforce' or 'warn'.",
            ));
        }
        self.immediate(|conn| {
            Box::pin(async move {
                let before_count = session_count(conn).await?;
                let high_water = maintenance_high_water(max_entries);
                if !force && before_count < i64::from(high_water) {
                    return Ok(SessionMaintenanceReport {
                        mode,
                        before_count,
                        after_count: before_count,
                        pruned: 0,
                        capped: 0,
                        applied: false,
                    });
                }
                let route_rows = sqlx::query("select distinct session_id from session_routes")
                    .fetch_all(&mut *conn)
                    .await?;
                let mut preserve = route_rows
                    .into_iter()
                    .map(|row| row.try_get::<String, _>("session_id"))
                    .collect::<Result<HashSet<_>, _>>()?;
                if let Some(active) = active_session_id {
                    preserve.insert(active);
                }
                let cutoff_modifier = format!("-{prune_after_days} days");
                let cutoff: String =
                    sqlx::query("select strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now', ?) as cutoff")
                        .bind(cutoff_modifier)
                        .fetch_one(&mut *conn)
                        .await?
                        .try_get("cutoff")?;
                let stale_rows = sqlx::query(
                    "select id from sessions where updated_at < ?
                     order by updated_at asc, created_at asc, id asc",
                )
                .bind(cutoff)
                .fetch_all(&mut *conn)
                .await?;
                let mut stale = Vec::with_capacity(stale_rows.len());
                for row in stale_rows {
                    let id = row.try_get::<String, _>("id")?;
                    if !preserve.contains(&id) {
                        stale.push(id);
                    }
                }
                let stale_set = stale.iter().map(String::as_str).collect::<HashSet<_>>();
                let all_rows = sqlx::query(
                    "select id from sessions
                     order by updated_at desc, created_at desc, id desc",
                )
                .fetch_all(&mut *conn)
                .await?;
                let mut preserved_count = 0;
                let mut removable = Vec::with_capacity(all_rows.len());
                for row in all_rows {
                    let id = row.try_get::<String, _>("id")?;
                    if preserve.contains(&id) {
                        preserved_count += 1;
                    } else if !stale_set.contains(id.as_str()) {
                        removable.push(id);
                    }
                }
                let removable_budget = usize::try_from(max_entries)
                    .unwrap_or(usize::MAX)
                    .saturating_sub(preserved_count);
                let capped = if removable_budget < removable.len() {
                    removable.split_off(removable_budget)
                } else {
                    Vec::new()
                };
                if mode == "enforce" {
                    for id in stale.iter().chain(capped.iter()) {
                        sqlx::query("delete from sessions where id = ?")
                            .bind(id)
                            .execute(&mut *conn)
                            .await?;
                    }
                }
                let after_count = if mode == "enforce" {
                    session_count(conn).await?
                } else {
                    before_count
                        .saturating_sub(i64::try_from(stale.len()).unwrap_or(i64::MAX))
                        .saturating_sub(i64::try_from(capped.len()).unwrap_or(i64::MAX))
                };
                Ok(SessionMaintenanceReport {
                    mode,
                    before_count,
                    after_count,
                    pruned: stale.len(),
                    capped: capped.len(),
                    applied: !stale.is_empty() || !capped.is_empty(),
                })
            })
        })
        .await
    }

    async fn update_llm_config(
        &self,
        session_id: &str,
        provider: Option<String>,
        model: Option<String>,
    ) -> StoreResult<()> {
        let session_id = session_id.to_owned();
        self.immediate(|conn| {
            Box::pin(async move {
                let now = db_now(conn).await?;
                let result = sqlx::query(
                    "update sessions set provider = ?, model = ?, updated_at = ? where id = ?",
                )
                .bind(provider)
                .bind(model)
                .bind(now)
                .bind(&session_id)
                .execute(conn)
                .await?;
                require_changed(result.rows_affected(), "Session", &session_id)
            })
        })
        .await
    }

    async fn patch_session_metadata(
        &self,
        session_id: String,
        title: Option<String>,
        provider: Option<String>,
        model: Option<String>,
        state_patch: Option<Map<String, Value>>,
    ) -> StoreResult<SessionInfo> {
        let requested_title = title
            .as_deref()
            .map(|value| {
                clean_title(Some(value))
                    .ok_or_else(|| StoreError::invalid("Session title cannot be empty."))
            })
            .transpose()?;
        self.immediate(|conn| {
            Box::pin(async move {
                let row = sqlx::query(
                    "select title, provider, model, state_json, active_root, focus_path
                     from sessions where id = ?",
                )
                .bind(&session_id)
                .fetch_optional(&mut *conn)
                .await?
                .ok_or_else(|| StoreError::not_found(format!("Session not found: {session_id}")))?;
                let mut state = json_object(row.try_get("state_json")?).unwrap_or_default();
                if let Some(patch) = state_patch {
                    state.extend(patch);
                }
                let active_root = state
                    .get("active_root")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .or(row.try_get("active_root")?);
                let focus_path = state
                    .get("focus_path")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned)
                    .or(row.try_get("focus_path")?);
                let state_json = if state.is_empty() {
                    None
                } else {
                    Some(serialize_json(&state)?)
                };
                let current_title: String = row.try_get("title")?;
                let current_provider: Option<String> = row.try_get("provider")?;
                let current_model: Option<String> = row.try_get("model")?;
                let now = db_now(conn).await?;
                sqlx::query(
                    "update sessions
                     set title = ?, provider = ?, model = ?, state_json = ?,
                         active_root = ?, focus_path = ?, updated_at = ?
                     where id = ?",
                )
                .bind(requested_title.unwrap_or(current_title))
                .bind(provider.or(current_provider))
                .bind(model.or(current_model))
                .bind(state_json)
                .bind(active_root)
                .bind(focus_path)
                .bind(now)
                .bind(&session_id)
                .execute(&mut *conn)
                .await?;
                fetch_session(&mut *conn, &session_id).await
            })
        })
        .await
    }

    async fn reset_routed_session(
        &self,
        route_key: String,
        reason: String,
    ) -> StoreResult<(SessionInfo, SessionInfo)> {
        let route_key = required_text(route_key, "route_key")?;
        if !matches!(reason.as_str(), "new" | "reset") {
            return Err(StoreError::invalid(
                "Session reset reason must be 'new' or 'reset'.",
            ));
        }
        let (old_id, new_id) = self
            .immediate(|conn| {
                Box::pin(async move {
                    let route =
                        sqlx::query("select session_id from session_routes where route_key = ?")
                            .bind(&route_key)
                            .fetch_optional(&mut *conn)
                            .await?
                            .ok_or_else(|| {
                                StoreError::not_found(format!(
                                    "Session route not found: {route_key}"
                                ))
                            })?;
                    let old_id: String = route.try_get("session_id")?;
                    let old = sqlx::query(
                        "select project_id, workspace_id, title, workspace_root, cwd,
                                provider, model, agent, permission_json, state_json
                         from sessions where id = ?",
                    )
                    .bind(&old_id)
                    .fetch_optional(&mut *conn)
                    .await?
                    .ok_or_else(|| StoreError::not_found(format!("Session not found: {old_id}")))?;
                    let state = json_object(old.try_get("state_json")?).unwrap_or_default();
                    let preserved_state = state
                        .get("reasoning_effort")
                        .cloned()
                        .map(|value| Map::from_iter([("reasoning_effort".to_owned(), value)]));
                    let state_json = preserved_state.as_ref().map(serialize_json).transpose()?;
                    let new_id = db_id(conn).await?;
                    let now = db_now(conn).await?;
                    sqlx::query(
                        "insert into sessions (
                            id, project_id, workspace_id, title, workspace_root, cwd,
                            created_at, updated_at, provider, model, agent, permission_json,
                            cost_usd, tokens_input, tokens_output, tokens_reasoning,
                            tokens_cache_read, tokens_cache_write, summary, active_root,
                            focus_path, last_prompt, state_json
                         ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0,
                                   null, null, null, null, ?)",
                    )
                    .bind(&new_id)
                    .bind(old.try_get::<Option<String>, _>("project_id")?)
                    .bind(old.try_get::<Option<String>, _>("workspace_id")?)
                    .bind(old.try_get::<String, _>("title")?)
                    .bind(old.try_get::<String, _>("workspace_root")?)
                    .bind(old.try_get::<Option<String>, _>("cwd")?)
                    .bind(&now)
                    .bind(&now)
                    .bind(old.try_get::<Option<String>, _>("provider")?)
                    .bind(old.try_get::<Option<String>, _>("model")?)
                    .bind(old.try_get::<Option<String>, _>("agent")?)
                    .bind(old.try_get::<Option<String>, _>("permission_json")?)
                    .bind(state_json)
                    .execute(&mut *conn)
                    .await?;
                    sqlx::query(
                        "update session_routes set session_id = ?, updated_at = ?
                         where route_key = ?",
                    )
                    .bind(&new_id)
                    .bind(now)
                    .bind(&route_key)
                    .execute(&mut *conn)
                    .await?;
                    Ok((old_id, new_id))
                })
            })
            .await?;
        let _ = self
            .apply_maintenance(
                DEFAULT_SESSION_MAX_ENTRIES,
                DEFAULT_SESSION_PRUNE_AFTER_DAYS,
                Some(new_id.clone()),
                false,
                "enforce".to_owned(),
            )
            .await?;
        Ok((
            self.get_session(&old_id).await?,
            self.get_session(&new_id).await?,
        ))
    }

    async fn delete_routed_session(&self, route_key: String) -> StoreResult<Value> {
        let route_key = required_text(route_key, "route_key")?;
        self.immediate(|conn| {
            Box::pin(async move {
                let route =
                    sqlx::query("select session_id from session_routes where route_key = ?")
                        .bind(&route_key)
                        .fetch_optional(&mut *conn)
                        .await?
                        .ok_or_else(|| {
                            StoreError::not_found(format!("Session route not found: {route_key}"))
                        })?;
                let session_id: String = route.try_get("session_id")?;
                let messages = count_rows(conn, "messages", &session_id).await?;
                let events = count_rows(conn, "events", &session_id).await?;
                let routes = count_rows(conn, "session_routes", &session_id).await?;
                let result = sqlx::query("delete from sessions where id = ?")
                    .bind(&session_id)
                    .execute(&mut *conn)
                    .await?;
                require_changed(result.rows_affected(), "Session", &session_id)?;
                Ok(json!({
                    "session_id": session_id,
                    "messages_deleted": messages,
                    "events_deleted": events,
                    "routes_deleted": routes,
                }))
            })
        })
        .await
    }

    async fn add_messages(
        &self,
        session_id: String,
        messages: Vec<MessageInput>,
        last_prompt: Option<String>,
        expected_route_key: Option<String>,
    ) -> StoreResult<Vec<MessageInfo>> {
        if messages.is_empty() {
            return Ok(Vec::new());
        }
        for message in &messages {
            validate_role(&message.role)?;
            for attachment in &message.attachments {
                validate_attachment(attachment)?;
            }
        }
        self.immediate(|conn| {
            Box::pin(async move {
                if let Some(route_key) = expected_route_key.as_deref() {
                    let route = sqlx::query(
                        "select session_id from session_routes where route_key = ?",
                    )
                    .bind(route_key)
                    .fetch_optional(&mut *conn)
                    .await?;
                    let current = route
                        .as_ref()
                        .map(|row| row.try_get::<String, _>("session_id"))
                        .transpose()?;
                    if current.as_deref() != Some(session_id.as_str()) {
                        return Err(StoreError::route_rebound(
                            format!(
                                "Session route rebound: {route_key} no longer points to {session_id}"
                            ),
                            json!({
                                "route_key": route_key,
                                "expected_session_id": session_id,
                                "current_session_id": current,
                            }),
                        ));
                    }
                }
                let session = sqlx::query("select title from sessions where id = ?")
                    .bind(&session_id)
                    .fetch_optional(&mut *conn)
                    .await?
                    .ok_or_else(|| {
                        StoreError::not_found(format!("Session not found: {session_id}"))
                    })?;
                let now = db_now(conn).await?;
                let start = next_seq(conn, "messages", &session_id).await?;
                let mut inserted = Vec::with_capacity(messages.len());
                for (offset, message) in messages.into_iter().enumerate() {
                    let seq = start + i64::try_from(offset).unwrap_or(i64::MAX);
                    let result = sqlx::query(
                        "insert into messages (session_id, seq, role, content, created_at)
                         values (?, ?, ?, ?, ?)",
                    )
                    .bind(&session_id)
                    .bind(seq)
                    .bind(&message.role)
                    .bind(&message.content)
                    .bind(&now)
                    .execute(&mut *conn)
                    .await?;
                    let message_id = result.last_insert_rowid();
                    let mut attached = Vec::with_capacity(message.attachments.len());
                    for (position, attachment) in message.attachments.into_iter().enumerate() {
                        sqlx::query(
                            "insert into attachments (id, filename, mime, size_bytes, sha256, storage_path, source, created_at)
                             values (?, ?, ?, ?, ?, ?, ?, ?)",
                        )
                        .bind(&attachment.id)
                        .bind(&attachment.filename)
                        .bind(&attachment.mime)
                        .bind(attachment.size_bytes)
                        .bind(&attachment.sha256)
                        .bind(&attachment.storage_path)
                        .bind(&attachment.source)
                        .bind(&now)
                        .execute(&mut *conn)
                        .await?;
                        sqlx::query(
                            "insert into message_attachments (message_id, attachment_id, position) values (?, ?, ?)",
                        )
                        .bind(message_id)
                        .bind(&attachment.id)
                        .bind(i64::try_from(position).unwrap_or(i64::MAX))
                        .execute(&mut *conn)
                        .await?;
                        attached.push(AttachmentInfo {
                            id: attachment.id,
                            filename: attachment.filename,
                            mime: attachment.mime,
                            size_bytes: attachment.size_bytes,
                            sha256: attachment.sha256,
                            storage_path: attachment.storage_path,
                            source: attachment.source,
                            created_at: now.clone(),
                        });
                    }
                    inserted.push(MessageInfo {
                        id: message_id,
                        session_id: session_id.clone(),
                        seq,
                        role: message.role,
                        content: message.content,
                        created_at: now.clone(),
                        attachments: attached,
                    });
                }
                let current_title: String = session.try_get("title")?;
                if let Some(prompt) = last_prompt {
                    let next_title = if current_title == "New session" {
                        clean_title(Some(&prompt)).unwrap_or(current_title)
                    } else {
                        current_title
                    };
                    sqlx::query(
                        "update sessions set title = ?, last_prompt = ?, updated_at = ?
                         where id = ?",
                    )
                    .bind(next_title)
                    .bind(prompt)
                    .bind(now)
                    .bind(&session_id)
                    .execute(&mut *conn)
                    .await?;
                } else {
                    sqlx::query("update sessions set updated_at = ? where id = ?")
                        .bind(now)
                        .bind(&session_id)
                        .execute(&mut *conn)
                        .await?;
                }
                Ok(inserted)
            })
        })
        .await
    }

    async fn list_messages(
        &self,
        session_id: &str,
        limit: Option<u32>,
    ) -> StoreResult<Vec<MessageInfo>> {
        let rows = if let Some(limit) = limit {
            sqlx::query(
                "select id, session_id, seq, role, content, created_at from (
                    select id, session_id, seq, role, content, created_at
                    from messages where session_id = ? order by seq desc limit ?
                 ) order by seq asc",
            )
            .bind(session_id)
            .bind(i64::from(limit.clamp(1, 500)))
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query(
                "select id, session_id, seq, role, content, created_at
                 from messages where session_id = ? order by seq asc",
            )
            .bind(session_id)
            .fetch_all(&self.pool)
            .await?
        };
        let mut messages = Vec::with_capacity(rows.len());
        for row in rows {
            let mut message = message_from_row(row)?;
            message.attachments = list_message_attachments(&self.pool, message.id).await?;
            messages.push(message);
        }
        Ok(messages)
    }

    async fn update_last_prompt(&self, session_id: String, prompt: String) -> StoreResult<()> {
        self.immediate(|conn| {
            Box::pin(async move {
                let row = sqlx::query("select title from sessions where id = ?")
                    .bind(&session_id)
                    .fetch_optional(&mut *conn)
                    .await?
                    .ok_or_else(|| {
                        StoreError::not_found(format!("Session not found: {session_id}"))
                    })?;
                let title: String = row.try_get("title")?;
                let title = if title == "New session" {
                    clean_title(Some(&prompt)).unwrap_or(title)
                } else {
                    title
                };
                let now = db_now(conn).await?;
                sqlx::query(
                    "update sessions set title = ?, last_prompt = ?, updated_at = ?
                     where id = ?",
                )
                .bind(title)
                .bind(prompt)
                .bind(now)
                .bind(session_id)
                .execute(conn)
                .await?;
                Ok(())
            })
        })
        .await
    }

    async fn save_agent_state(
        &self,
        session_id: String,
        state: Map<String, Value>,
    ) -> StoreResult<()> {
        let state_json = serialize_json(&state)?;
        let active_root = state
            .get("active_root")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let focus_path = state
            .get("focus_path")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        self.immediate(|conn| {
            Box::pin(async move {
                let now = db_now(conn).await?;
                let result = sqlx::query(
                    "update sessions set state_json = ?, active_root = ?, focus_path = ?,
                         updated_at = ? where id = ?",
                )
                .bind(state_json)
                .bind(active_root)
                .bind(focus_path)
                .bind(now)
                .bind(&session_id)
                .execute(conn)
                .await?;
                require_changed(result.rows_affected(), "Session", &session_id)
            })
        })
        .await
    }

    async fn add_usage(
        &self,
        session_id: String,
        tokens: TokenUsage,
        cost_usd: f64,
    ) -> StoreResult<()> {
        if !cost_usd.is_finite() {
            return Err(StoreError::invalid("cost_usd must be finite."));
        }
        if tokens.input == 0
            && tokens.output == 0
            && tokens.reasoning == 0
            && tokens.cache_read == 0
            && tokens.cache_write == 0
            && cost_usd == 0.0
        {
            return Ok(());
        }
        self.immediate(|conn| {
            Box::pin(async move {
                let now = db_now(conn).await?;
                let result = sqlx::query(
                    "update sessions
                     set tokens_input = tokens_input + ?,
                         tokens_output = tokens_output + ?,
                         tokens_reasoning = tokens_reasoning + ?,
                         tokens_cache_read = tokens_cache_read + ?,
                         tokens_cache_write = tokens_cache_write + ?,
                         cost_usd = cost_usd + ?,
                         updated_at = ?
                     where id = ?",
                )
                .bind(tokens.input)
                .bind(tokens.output)
                .bind(tokens.reasoning)
                .bind(tokens.cache_read)
                .bind(tokens.cache_write)
                .bind(cost_usd)
                .bind(now)
                .bind(&session_id)
                .execute(conn)
                .await?;
                require_changed(result.rows_affected(), "Session", &session_id)
            })
        })
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn add_event(
        &self,
        session_id: String,
        event_type: String,
        summary: String,
        tool: Option<String>,
        args: Option<Map<String, Value>>,
        path: Option<String>,
        data: Option<Map<String, Value>>,
    ) -> StoreResult<EventInfo> {
        let event_type = required_text(event_type, "event_type")?;
        let summary = compact_whitespace(&summary);
        let summary = required_text(summary.into_owned(), "summary")?;
        let args_json = args.as_ref().map(serialize_json).transpose()?;
        let data_json = data.as_ref().map(serialize_json).transpose()?;
        self.immediate(|conn| {
            Box::pin(async move {
                require_session(conn, &session_id).await?;
                let now = db_now(conn).await?;
                let seq = next_seq(conn, "events", &session_id).await?;
                let result = sqlx::query(
                    "insert into events (
                        session_id, seq, event_type, tool, path, summary,
                        args_json, data_json, created_at
                     ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                )
                .bind(&session_id)
                .bind(seq)
                .bind(&event_type)
                .bind(&tool)
                .bind(&path)
                .bind(&summary)
                .bind(args_json)
                .bind(data_json)
                .bind(&now)
                .execute(&mut *conn)
                .await?;
                sqlx::query("update sessions set updated_at = ? where id = ?")
                    .bind(&now)
                    .bind(&session_id)
                    .execute(&mut *conn)
                    .await?;
                Ok(EventInfo {
                    id: result.last_insert_rowid(),
                    session_id,
                    seq,
                    event_type,
                    tool,
                    args,
                    path,
                    summary,
                    data,
                    created_at: now,
                })
            })
        })
        .await
    }

    async fn list_events(&self, session_id: &str, limit: u32) -> StoreResult<Vec<EventInfo>> {
        let rows = sqlx::query(
            "select id, session_id, seq, event_type, tool, args_json,
                    path, summary, data_json, created_at
             from events where session_id = ? order by seq desc limit ?",
        )
        .bind(session_id)
        .bind(i64::from(limit.clamp(1, 500)))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter().map(event_from_row).collect()
    }

    async fn compact_routed_session(
        &self,
        route_key: String,
        max_messages: u32,
    ) -> StoreResult<Value> {
        let route_key = required_text(route_key, "route_key")?;
        if max_messages == 0 {
            return Err(StoreError::invalid(
                "max_messages must be a positive integer.",
            ));
        }
        self.immediate(|conn| {
            Box::pin(async move {
                let route = sqlx::query(
                    "select session_id from session_routes where route_key = ?",
                )
                .bind(&route_key)
                .fetch_optional(&mut *conn)
                .await?
                .ok_or_else(|| {
                    StoreError::not_found(format!("Session route not found: {route_key}"))
                })?;
                let session_id: String = route.try_get("session_id")?;
                require_session(conn, &session_id).await?;
                let rows = sqlx::query(
                    "select id, seq, role, content, created_at
                     from messages where session_id = ? order by seq asc",
                )
                .bind(&session_id)
                .fetch_all(&mut *conn)
                .await?;
                let before = rows.len();
                let max = usize::try_from(max_messages).unwrap_or(usize::MAX);
                if before <= max {
                    return Ok(json!({
                        "session_id": session_id,
                        "compacted": false,
                        "lines_before": before,
                        "lines_after": before,
                        "kept": before,
                        "pruned": 0,
                        "archived_event_id": Value::Null,
                    }));
                }
                let split = before - max;
                let (pruned, kept) = rows.split_at(split);
                let archived = pruned
                    .iter()
                    .map(|row| {
                        Ok(json!({
                            "id": row.try_get::<i64, _>("id")?,
                            "seq": row.try_get::<i64, _>("seq")?,
                            "role": row.try_get::<String, _>("role")?,
                            "content": row.try_get::<String, _>("content")?,
                            "created_at": row.try_get::<String, _>("created_at")?,
                        }))
                    })
                    .collect::<StoreResult<Vec<_>>>()?;
                let first_kept_seq = kept
                    .first()
                    .map(|row| row.try_get::<i64, _>("seq"))
                    .transpose()?;
                let archive = json!({
                    "kind": "sessions.compact.maxLines.archive",
                    "route_key": route_key,
                    "session_id": session_id,
                    "max_messages": max_messages,
                    "lines_before": before,
                    "lines_after": kept.len(),
                    "first_kept_seq": first_kept_seq,
                    "messages": archived,
                });
                let now = db_now(conn).await?;
                let event_seq = next_seq(conn, "events", &session_id).await?;
                let summary = format!(
                    "Compacted transcript to last {max_messages} message(s); archived {} pruned message(s)",
                    pruned.len()
                );
                let event = sqlx::query(
                    "insert into events (
                        session_id, seq, event_type, tool, path, summary,
                        args_json, data_json, created_at
                     ) values (?, ?, 'session_compacted', null, null, ?, ?, ?, ?)",
                )
                .bind(&session_id)
                .bind(event_seq)
                .bind(summary)
                .bind(serialize_json(&json!({"max_messages": max_messages}))?)
                .bind(serialize_json(&archive)?)
                .bind(&now)
                .execute(&mut *conn)
                .await?;
                for row in pruned {
                    sqlx::query("delete from messages where id = ?")
                        .bind(row.try_get::<i64, _>("id")?)
                        .execute(&mut *conn)
                        .await?;
                }
                sqlx::query("update sessions set updated_at = ? where id = ?")
                    .bind(now)
                    .bind(&session_id)
                    .execute(&mut *conn)
                    .await?;
                Ok(json!({
                    "session_id": session_id,
                    "compacted": true,
                    "lines_before": before,
                    "lines_after": kept.len(),
                    "kept": kept.len(),
                    "pruned": pruned.len(),
                    "archived_event_id": event.last_insert_rowid(),
                }))
            })
        })
        .await
    }

    async fn count_session_compaction_checkpoints(&self, session_id: &str) -> StoreResult<i64> {
        let row = sqlx::query(
            "select count(*) as count from events
             where session_id = ? and event_type = 'session_compacted'",
        )
        .bind(session_id)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.try_get("count")?)
    }

    async fn list_session_compaction_checkpoints(
        &self,
        session_id: &str,
        session_key: &str,
        limit: u32,
    ) -> StoreResult<Vec<Value>> {
        require_nonempty_session_key(session_key)?;
        ensure_session_in_pool(&self.pool, session_id).await?;
        let rows = sqlx::query(
            "select id, session_id, seq, event_type, tool, args_json,
                    path, summary, data_json, created_at
             from events
             where session_id = ? and event_type = 'session_compacted'
             order by created_at desc, id desc limit ?",
        )
        .bind(session_id)
        .bind(i64::from(limit.clamp(1, 500)))
        .fetch_all(&self.pool)
        .await?;
        rows.into_iter()
            .map(|row| checkpoint_from_row(row, session_key))
            .collect()
    }

    async fn get_session_compaction_checkpoint(
        &self,
        session_id: &str,
        session_key: &str,
        checkpoint_id: &str,
    ) -> StoreResult<Value> {
        require_nonempty_session_key(session_key)?;
        ensure_session_in_pool(&self.pool, session_id).await?;
        let event_id = checkpoint_event_id(checkpoint_id)?;
        let row = sqlx::query(
            "select id, session_id, seq, event_type, tool, args_json,
                    path, summary, data_json, created_at
             from events
             where id = ? and session_id = ? and event_type = 'session_compacted'",
        )
        .bind(event_id)
        .bind(session_id)
        .fetch_optional(&self.pool)
        .await?
        .ok_or_else(|| {
            StoreError::not_found(format!("Compaction checkpoint not found: {checkpoint_id}"))
        })?;
        checkpoint_from_row(row, session_key)
    }

    async fn branch_routed_session_from_compaction_checkpoint(
        &self,
        route_key: String,
        checkpoint_id: String,
    ) -> StoreResult<Value> {
        let route_key = required_text(route_key, "route_key")?;
        let event_id = checkpoint_event_id(&checkpoint_id)?;
        let route_key_for_tx = route_key.clone();
        let checkpoint_for_tx = checkpoint_id.clone();
        let (source_id, branch_id, branch_key, copied) = self
            .immediate(|conn| {
                Box::pin(async move {
                    let route = sqlx::query(
                        "select route_key, session_id, agent_id, scope, channel, account_id,
                                peer_kind, peer_id, sender_id, guild_id, team_id
                         from session_routes where route_key = ?",
                    )
                    .bind(&route_key_for_tx)
                    .fetch_optional(&mut *conn)
                    .await?
                    .ok_or_else(|| {
                        StoreError::not_found(format!(
                            "Session route not found: {route_key_for_tx}"
                        ))
                    })?;
                    let source_id: String = route.try_get("session_id")?;
                    let source =
                        checkpoint_source(conn, &source_id, event_id, &checkpoint_for_tx).await?;
                    let messages =
                        reconstruct_checkpoint_messages(conn, &source_id, &source.checkpoint)
                            .await?;
                    let branch_id = db_id(conn).await?;
                    let branch_key = format!("{route_key_for_tx}:checkpoint:{branch_id}");
                    let now = db_now(conn).await?;
                    let state = branch_state(
                        source.state_json.as_deref(),
                        &route_key_for_tx,
                        &source_id,
                        &checkpoint_for_tx,
                    );
                    insert_session_copy(
                        conn,
                        &branch_id,
                        &source,
                        clean_title(Some(&format!("{} (checkpoint)", source.title)))
                            .unwrap_or_else(|| "Checkpoint branch".to_owned()),
                        &now,
                        Some(serialize_json(&state)?),
                    )
                    .await?;
                    sqlx::query(
                        "insert into session_routes (
                            route_key, session_id, agent_id, scope, channel, account_id,
                            peer_kind, peer_id, sender_id, guild_id, team_id,
                            created_at, updated_at
                         ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    )
                    .bind(&branch_key)
                    .bind(&branch_id)
                    .bind(route.try_get::<String, _>("agent_id")?)
                    .bind(route.try_get::<String, _>("scope")?)
                    .bind(route.try_get::<String, _>("channel")?)
                    .bind(route.try_get::<String, _>("account_id")?)
                    .bind(route.try_get::<Option<String>, _>("peer_kind")?)
                    .bind(route.try_get::<Option<String>, _>("peer_id")?)
                    .bind(route.try_get::<Option<String>, _>("sender_id")?)
                    .bind(route.try_get::<Option<String>, _>("guild_id")?)
                    .bind(route.try_get::<Option<String>, _>("team_id")?)
                    .bind(&now)
                    .bind(&now)
                    .execute(&mut *conn)
                    .await?;
                    insert_snapshot_messages(conn, &branch_id, &messages).await?;
                    Ok((source_id, branch_id, branch_key, messages.len()))
                })
            })
            .await?;
        let _ = self
            .apply_maintenance(
                DEFAULT_SESSION_MAX_ENTRIES,
                DEFAULT_SESSION_PRUNE_AFTER_DAYS,
                Some(branch_id.clone()),
                false,
                "enforce".to_owned(),
            )
            .await?;
        let checkpoint = self
            .get_session_compaction_checkpoint(&source_id, &route_key, &checkpoint_id)
            .await?;
        Ok(json!({
            "source_session_id": source_id,
            "session_id": branch_id,
            "key": branch_key,
            "messages_copied": copied,
            "checkpoint": checkpoint,
        }))
    }

    async fn restore_routed_session_from_compaction_checkpoint(
        &self,
        route_key: String,
        checkpoint_id: String,
    ) -> StoreResult<Value> {
        let route_key = required_text(route_key, "route_key")?;
        let event_id = checkpoint_event_id(&checkpoint_id)?;
        let route_key_for_tx = route_key.clone();
        let checkpoint_for_tx = checkpoint_id.clone();
        let (previous_id, restored_id, copied) = self
            .immediate(|conn| {
                Box::pin(async move {
                    let route =
                        sqlx::query("select session_id from session_routes where route_key = ?")
                            .bind(&route_key_for_tx)
                            .fetch_optional(&mut *conn)
                            .await?
                            .ok_or_else(|| {
                                StoreError::not_found(format!(
                                    "Session route not found: {route_key_for_tx}"
                                ))
                            })?;
                    let previous_id: String = route.try_get("session_id")?;
                    let source =
                        checkpoint_source(conn, &previous_id, event_id, &checkpoint_for_tx).await?;
                    let messages =
                        reconstruct_checkpoint_messages(conn, &previous_id, &source.checkpoint)
                            .await?;
                    let restored_id = db_id(conn).await?;
                    let now = db_now(conn).await?;
                    let state = restore_state(
                        source.state_json.as_deref(),
                        &previous_id,
                        &checkpoint_for_tx,
                    );
                    insert_session_copy(
                        conn,
                        &restored_id,
                        &source,
                        source.title.clone(),
                        &now,
                        Some(serialize_json(&state)?),
                    )
                    .await?;
                    insert_snapshot_messages(conn, &restored_id, &messages).await?;
                    sqlx::query(
                        "update events set session_id = ?
                         where session_id = ? and event_type = 'session_compacted'",
                    )
                    .bind(&restored_id)
                    .bind(&previous_id)
                    .execute(&mut *conn)
                    .await?;
                    sqlx::query(
                        "update session_routes set session_id = ?, updated_at = ?
                         where route_key = ?",
                    )
                    .bind(&restored_id)
                    .bind(now)
                    .bind(&route_key_for_tx)
                    .execute(&mut *conn)
                    .await?;
                    Ok((previous_id, restored_id, messages.len()))
                })
            })
            .await?;
        let _ = self
            .apply_maintenance(
                DEFAULT_SESSION_MAX_ENTRIES,
                DEFAULT_SESSION_PRUNE_AFTER_DAYS,
                Some(restored_id.clone()),
                false,
                "enforce".to_owned(),
            )
            .await?;
        let checkpoint = self
            .get_session_compaction_checkpoint(&restored_id, &route_key, &checkpoint_id)
            .await?;
        Ok(json!({
            "previous_session_id": previous_id,
            "session_id": restored_id,
            "key": route_key,
            "messages_restored": copied,
            "checkpoint": checkpoint,
        }))
    }
}

fn require_changed(rows: u64, entity: &str, id: &str) -> StoreResult<()> {
    if rows == 0 {
        return Err(StoreError::not_found(format!("{entity} not found: {id}")));
    }
    Ok(())
}

async fn require_session(conn: &mut SqliteConnection, session_id: &str) -> StoreResult<()> {
    let exists: i64 = sqlx::query("select exists(select 1 from sessions where id = ?) as present")
        .bind(session_id)
        .fetch_one(conn)
        .await?
        .try_get("present")?;
    if exists == 0 {
        return Err(StoreError::not_found(format!(
            "Session not found: {session_id}"
        )));
    }
    Ok(())
}

async fn next_seq(
    conn: &mut SqliteConnection,
    table: &'static str,
    session_id: &str,
) -> StoreResult<i64> {
    let statement = match table {
        "messages" => {
            "select coalesce(max(seq), 0) + 1 as next_seq
             from messages where session_id = ?"
        }
        "events" => {
            "select coalesce(max(seq), 0) + 1 as next_seq
             from events where session_id = ?"
        }
        _ => return Err(StoreError::internal("Unsupported sequence table.")),
    };
    let row = sqlx::query(statement)
        .bind(session_id)
        .fetch_one(conn)
        .await?;
    Ok(row.try_get("next_seq")?)
}

async fn count_rows(
    conn: &mut SqliteConnection,
    table: &'static str,
    session_id: &str,
) -> StoreResult<i64> {
    let statement = match table {
        "messages" => "select count(*) as count from messages where session_id = ?",
        "events" => "select count(*) as count from events where session_id = ?",
        "session_routes" => "select count(*) as count from session_routes where session_id = ?",
        _ => return Err(StoreError::internal("Unsupported count table.")),
    };
    let row = sqlx::query(statement)
        .bind(session_id)
        .fetch_one(conn)
        .await?;
    Ok(row.try_get("count")?)
}

fn validate_role(role: &str) -> StoreResult<()> {
    if matches!(role, "user" | "assistant" | "tool" | "system") {
        return Ok(());
    }
    Err(StoreError::invalid(format!(
        "Unsupported message role: {role}"
    )))
}

fn validate_attachment(attachment: &AttachmentInput) -> StoreResult<()> {
    if attachment.id.trim().is_empty()
        || attachment.filename.trim().is_empty()
        || attachment.mime.trim().is_empty()
        || attachment.sha256.len() != 64
        || attachment.storage_path.trim().is_empty()
        || attachment.source.trim().is_empty()
        || attachment.size_bytes < 0
    {
        return Err(StoreError::invalid(
            "Attachment metadata is incomplete or invalid.",
        ));
    }
    if !attachment
        .sha256
        .bytes()
        .all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(StoreError::invalid(
            "Attachment digest must be a SHA-256 hex string.",
        ));
    }
    Ok(())
}

fn serialize_json<T: Serialize>(value: &T) -> StoreResult<String> {
    serde_json::to_string(value)
        .map_err(|_| StoreError::invalid("Value could not be encoded as JSON."))
}

fn message_from_row(row: SqliteRow) -> StoreResult<MessageInfo> {
    Ok(MessageInfo {
        id: row.try_get("id")?,
        session_id: row.try_get("session_id")?,
        seq: row.try_get("seq")?,
        role: row.try_get("role")?,
        content: row.try_get("content")?,
        created_at: row.try_get("created_at")?,
        attachments: Vec::new(),
    })
}

async fn list_message_attachments(
    pool: &SqlitePool,
    message_id: i64,
) -> StoreResult<Vec<AttachmentInfo>> {
    let rows = sqlx::query(
        "select a.id, a.filename, a.mime, a.size_bytes, a.sha256, a.storage_path, a.source, a.created_at
         from message_attachments ma join attachments a on a.id = ma.attachment_id
         where ma.message_id = ? order by ma.position asc",
    )
    .bind(message_id)
    .fetch_all(pool)
    .await?;
    rows.into_iter()
        .map(|row| {
            Ok(AttachmentInfo {
                id: row.try_get("id")?,
                filename: row.try_get("filename")?,
                mime: row.try_get("mime")?,
                size_bytes: row.try_get("size_bytes")?,
                sha256: row.try_get("sha256")?,
                storage_path: row.try_get("storage_path")?,
                source: row.try_get("source")?,
                created_at: row.try_get("created_at")?,
            })
        })
        .collect()
}

fn event_from_row(row: SqliteRow) -> StoreResult<EventInfo> {
    Ok(EventInfo {
        id: row.try_get("id")?,
        session_id: row.try_get("session_id")?,
        seq: row.try_get("seq")?,
        event_type: row.try_get("event_type")?,
        tool: row.try_get("tool")?,
        args: json_object(row.try_get("args_json")?),
        path: row.try_get("path")?,
        summary: row.try_get("summary")?,
        data: json_object(row.try_get("data_json")?),
        created_at: row.try_get("created_at")?,
    })
}

async fn session_count(conn: &mut SqliteConnection) -> StoreResult<i64> {
    let row = sqlx::query("select count(*) as count from sessions")
        .fetch_one(conn)
        .await?;
    Ok(row.try_get("count")?)
}

fn maintenance_high_water(max_entries: u32) -> u32 {
    if max_entries <= STRICT_ENTRY_MAINTENANCE_MAX_ENTRIES {
        return max_entries.saturating_add(1);
    }
    let ratio_slack = max_entries.saturating_add(9) / 10;
    max_entries.saturating_add(ratio_slack.max(MIN_BATCHED_ENTRY_MAINTENANCE_SLACK))
}

async fn ensure_session_in_pool(pool: &SqlitePool, session_id: &str) -> StoreResult<()> {
    let row = sqlx::query("select exists(select 1 from sessions where id = ?) as present")
        .bind(session_id)
        .fetch_one(pool)
        .await?;
    if row.try_get::<i64, _>("present")? == 0 {
        return Err(StoreError::not_found(format!(
            "Session not found: {session_id}"
        )));
    }
    Ok(())
}

fn require_nonempty_session_key(session_key: &str) -> StoreResult<()> {
    if session_key.trim().is_empty() {
        return Err(StoreError::invalid("Session key cannot be empty."));
    }
    Ok(())
}

fn checkpoint_event_id(checkpoint_id: &str) -> StoreResult<i64> {
    let raw = checkpoint_id.strip_prefix("sqlite:event:").ok_or_else(|| {
        StoreError::not_found(format!("Compaction checkpoint not found: {checkpoint_id}"))
    })?;
    raw.parse::<i64>().map_err(|_| {
        StoreError::not_found(format!("Compaction checkpoint not found: {checkpoint_id}"))
    })
}

fn checkpoint_from_row(row: SqliteRow, session_key: &str) -> StoreResult<Value> {
    let event_id: i64 = row.try_get("id")?;
    let session_id: String = row.try_get("session_id")?;
    let created_at: String = row.try_get("created_at")?;
    let summary: String = row.try_get("summary")?;
    let data = json_value(row.try_get("data_json")?).unwrap_or_else(|| json!({}));
    let object = data.as_object();
    let messages = object
        .and_then(|value| value.get("messages"))
        .and_then(Value::as_array);
    let last_pruned_seq = messages
        .and_then(|messages| messages.last())
        .and_then(Value::as_object)
        .and_then(|message| message.get("seq"))
        .and_then(Value::as_i64);
    let first_kept_seq = object
        .and_then(|value| value.get("first_kept_seq"))
        .and_then(Value::as_i64);
    let mut checkpoint = Map::from_iter([
        (
            "checkpointId".to_owned(),
            Value::String(format!("sqlite:event:{event_id}")),
        ),
        (
            "sessionKey".to_owned(),
            Value::String(session_key.to_owned()),
        ),
        ("sessionId".to_owned(), Value::String(session_id.clone())),
        (
            "createdAt".to_owned(),
            Value::from(iso_timestamp_ms(&created_at)),
        ),
        ("reason".to_owned(), Value::String("manual".to_owned())),
        ("summary".to_owned(), Value::String(summary)),
        (
            "preCompaction".to_owned(),
            compact_object([
                ("sessionId", Some(Value::String(session_id.clone()))),
                (
                    "entryId",
                    last_pruned_seq.map(|value| Value::String(value.to_string())),
                ),
            ]),
        ),
        (
            "postCompaction".to_owned(),
            compact_object([
                ("sessionId", Some(Value::String(session_id))),
                (
                    "entryId",
                    first_kept_seq.map(|value| Value::String(value.to_string())),
                ),
            ]),
        ),
        (
            "pruned".to_owned(),
            Value::from(
                messages
                    .map(|messages| u64::try_from(messages.len()).unwrap_or(u64::MAX))
                    .unwrap_or(0),
            ),
        ),
    ]);
    for (source, target) in [
        ("lines_before", "linesBefore"),
        ("lines_after", "linesAfter"),
        ("max_messages", "maxMessages"),
    ] {
        if let Some(value) = object
            .and_then(|data| data.get(source))
            .and_then(Value::as_i64)
        {
            checkpoint.insert(target.to_owned(), Value::from(value));
        }
    }
    if let Some(value) = first_kept_seq {
        checkpoint.insert(
            "firstKeptEntryId".to_owned(),
            Value::String(value.to_string()),
        );
    }
    Ok(Value::Object(checkpoint))
}

fn compact_object<const N: usize>(values: [(&str, Option<Value>); N]) -> Value {
    Value::Object(
        values
            .into_iter()
            .filter_map(|(key, value)| value.map(|value| (key.to_owned(), value)))
            .collect(),
    )
}

fn json_value(raw: Option<String>) -> Option<Value> {
    raw.and_then(|text| serde_json::from_str(&text).ok())
}

#[derive(Clone, Debug)]
struct SnapshotMessage {
    seq: i64,
    role: String,
    content: String,
    created_at: String,
}

struct CheckpointSource {
    project_id: Option<String>,
    workspace_id: Option<String>,
    title: String,
    workspace_root: String,
    cwd: Option<String>,
    provider: Option<String>,
    model: Option<String>,
    agent: Option<String>,
    permission_json: Option<String>,
    state_json: Option<String>,
    checkpoint: SqliteRow,
}

async fn checkpoint_source(
    conn: &mut SqliteConnection,
    session_id: &str,
    event_id: i64,
    checkpoint_id: &str,
) -> StoreResult<CheckpointSource> {
    let source = sqlx::query(
        "select project_id, workspace_id, title, workspace_root, cwd,
                provider, model, agent, permission_json, state_json
         from sessions where id = ?",
    )
    .bind(session_id)
    .fetch_optional(&mut *conn)
    .await?
    .ok_or_else(|| StoreError::not_found(format!("Session not found: {session_id}")))?;
    let checkpoint = sqlx::query(
        "select id, session_id, seq, event_type, tool, args_json,
                path, summary, data_json, created_at
         from events
         where id = ? and session_id = ? and event_type = 'session_compacted'",
    )
    .bind(event_id)
    .bind(session_id)
    .fetch_optional(&mut *conn)
    .await?
    .ok_or_else(|| {
        StoreError::not_found(format!("Compaction checkpoint not found: {checkpoint_id}"))
    })?;
    Ok(CheckpointSource {
        project_id: source.try_get("project_id")?,
        workspace_id: source.try_get("workspace_id")?,
        title: source.try_get("title")?,
        workspace_root: source.try_get("workspace_root")?,
        cwd: source.try_get("cwd")?,
        provider: source.try_get("provider")?,
        model: source.try_get("model")?,
        agent: source.try_get("agent")?,
        permission_json: source.try_get("permission_json")?,
        state_json: source.try_get("state_json")?,
        checkpoint,
    })
}

async fn reconstruct_checkpoint_messages(
    conn: &mut SqliteConnection,
    session_id: &str,
    checkpoint: &SqliteRow,
) -> StoreResult<Vec<SnapshotMessage>> {
    let data = json_value(checkpoint.try_get("data_json")?)
        .and_then(|value| value.as_object().cloned())
        .ok_or_else(|| StoreError::internal("Checkpoint archive is invalid."))?;
    let first_kept_seq = data
        .get("first_kept_seq")
        .and_then(Value::as_i64)
        .ok_or_else(|| StoreError::internal("Checkpoint snapshot is not reconstructable."))?;
    let lines_after = data
        .get("lines_after")
        .and_then(Value::as_i64)
        .filter(|value| *value >= 0)
        .ok_or_else(|| StoreError::internal("Checkpoint snapshot is not reconstructable."))?;
    let archived = data
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| StoreError::internal("Checkpoint archive does not contain message rows."))?;
    let mut messages = Vec::with_capacity(
        archived
            .len()
            .saturating_add(usize::try_from(lines_after).unwrap_or(0)),
    );
    for item in archived {
        let item = item.as_object().ok_or_else(|| {
            StoreError::internal("Checkpoint archive contains an invalid message row.")
        })?;
        let role = item.get("role").and_then(Value::as_str).unwrap_or_default();
        validate_role(role)?;
        messages.push(SnapshotMessage {
            seq: item
                .get("seq")
                .and_then(Value::as_i64)
                .ok_or_else(|| StoreError::internal("Checkpoint message has no sequence."))?,
            role: role.to_owned(),
            content: item
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| StoreError::internal("Checkpoint message has no content."))?
                .to_owned(),
            created_at: item
                .get("created_at")
                .and_then(Value::as_str)
                .ok_or_else(|| StoreError::internal("Checkpoint message has no timestamp."))?
                .to_owned(),
        });
    }
    let kept = sqlx::query(
        "select seq, role, content, created_at from messages
         where session_id = ? and seq >= ? and seq < ? order by seq asc",
    )
    .bind(session_id)
    .bind(first_kept_seq)
    .bind(first_kept_seq.saturating_add(lines_after))
    .fetch_all(&mut *conn)
    .await?;
    if i64::try_from(kept.len()).unwrap_or(i64::MAX) != lines_after {
        return Err(StoreError::internal(
            "Checkpoint snapshot tail is missing from the active transcript.",
        ));
    }
    for row in kept {
        messages.push(SnapshotMessage {
            seq: row.try_get("seq")?,
            role: row.try_get("role")?,
            content: row.try_get("content")?,
            created_at: row.try_get("created_at")?,
        });
    }
    Ok(messages)
}

async fn insert_snapshot_messages(
    conn: &mut SqliteConnection,
    session_id: &str,
    messages: &[SnapshotMessage],
) -> StoreResult<()> {
    for message in messages {
        sqlx::query(
            "insert into messages (session_id, seq, role, content, created_at)
             values (?, ?, ?, ?, ?)",
        )
        .bind(session_id)
        .bind(message.seq)
        .bind(&message.role)
        .bind(&message.content)
        .bind(&message.created_at)
        .execute(&mut *conn)
        .await?;
    }
    Ok(())
}

async fn insert_session_copy(
    conn: &mut SqliteConnection,
    session_id: &str,
    source: &CheckpointSource,
    title: String,
    now: &str,
    state_json: Option<String>,
) -> StoreResult<()> {
    sqlx::query(
        "insert into sessions (
            id, project_id, workspace_id, title, workspace_root, cwd,
            created_at, updated_at, provider, model, agent, permission_json,
            cost_usd, tokens_input, tokens_output, tokens_reasoning,
            tokens_cache_read, tokens_cache_write, summary, active_root,
            focus_path, last_prompt, state_json
         ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0,
                   null, null, null, null, ?)",
    )
    .bind(session_id)
    .bind(&source.project_id)
    .bind(&source.workspace_id)
    .bind(title)
    .bind(&source.workspace_root)
    .bind(&source.cwd)
    .bind(now)
    .bind(now)
    .bind(&source.provider)
    .bind(&source.model)
    .bind(&source.agent)
    .bind(&source.permission_json)
    .bind(state_json)
    .execute(conn)
    .await?;
    Ok(())
}

fn branch_state(
    raw: Option<&str>,
    route_key: &str,
    source_session_id: &str,
    checkpoint_id: &str,
) -> Map<String, Value> {
    let source = raw
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let mut state = Map::from_iter([
        (
            "parent_session_key".to_owned(),
            Value::String(route_key.to_owned()),
        ),
        (
            "source_session_id".to_owned(),
            Value::String(source_session_id.to_owned()),
        ),
        (
            "checkpoint_id".to_owned(),
            Value::String(checkpoint_id.to_owned()),
        ),
    ]);
    if let Some(value) = source.get("reasoning_effort") {
        state.insert("reasoning_effort".to_owned(), value.clone());
    }
    state
}

fn restore_state(
    raw: Option<&str>,
    source_session_id: &str,
    checkpoint_id: &str,
) -> Map<String, Value> {
    let source = raw
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default();
    let mut state = Map::from_iter([
        (
            "restored_from_session_id".to_owned(),
            Value::String(source_session_id.to_owned()),
        ),
        (
            "restored_checkpoint_id".to_owned(),
            Value::String(checkpoint_id.to_owned()),
        ),
    ]);
    for key in ["reasoning_effort", "parent_session_key"] {
        if let Some(value) = source.get(key) {
            state.insert(key.to_owned(), value.clone());
        }
    }
    state
}

fn iso_timestamp_ms(value: &str) -> i64 {
    let bytes = value.as_bytes();
    if bytes.len() < 19 {
        return 0;
    }
    let number = |start: usize, end: usize| {
        value
            .get(start..end)
            .and_then(|part| part.parse::<i64>().ok())
    };
    let Some(year) = number(0, 4) else { return 0 };
    let Some(month) = number(5, 7) else { return 0 };
    let Some(day) = number(8, 10) else { return 0 };
    let Some(hour) = number(11, 13) else { return 0 };
    let Some(minute) = number(14, 16) else {
        return 0;
    };
    let Some(second) = number(17, 19) else {
        return 0;
    };
    let mut millis = 0;
    let mut millisecond_digits = 0;
    if let Some(fraction) = value.get(19..).and_then(|tail| tail.strip_prefix('.')) {
        for character in fraction
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .take(3)
        {
            millis = millis * 10 + i64::from(character as u8 - b'0');
            millisecond_digits += 1;
        }
    }
    for _ in millisecond_digits..3 {
        millis *= 10;
    }
    let days = days_from_civil(year, month, day);
    days.saturating_mul(86_400_000)
        .saturating_add(hour.saturating_mul(3_600_000))
        .saturating_add(minute.saturating_mul(60_000))
        .saturating_add(second.saturating_mul(1_000))
        .saturating_add(millis)
}

fn days_from_civil(mut year: i64, month: i64, day: i64) -> i64 {
    year -= i64::from(month <= 2);
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

const SESSION_SELECT: &str = "select id, project_id, workspace_id, title, workspace_root, cwd,
    created_at, updated_at, provider, model, agent, permission_json,
    cost_usd, tokens_input, tokens_output, tokens_reasoning,
    tokens_cache_read, tokens_cache_write, summary, active_root, focus_path,
    last_prompt, state_json from sessions";

fn push_session_filters(
    query: &mut QueryBuilder<Sqlite>,
    agent_id: Option<&str>,
    updated_after: Option<&str>,
) {
    let mut has_filter = false;
    if let Some(agent_id) = agent_id {
        query
            .push(" where coalesce(agent, ")
            .push_bind("agent")
            .push(") = ")
            .push_bind(agent_id);
        has_filter = true;
    }
    if let Some(updated_after) = updated_after {
        query
            .push(if has_filter { " and " } else { " where " })
            .push("updated_at >= ")
            .push_bind(updated_after);
    }
}

async fn fetch_session<'e, E>(executor: E, session_id: &str) -> StoreResult<SessionInfo>
where
    E: Executor<'e, Database = Sqlite>,
{
    let row = sqlx::query(sqlx::AssertSqlSafe(format!(
        "{SESSION_SELECT} where id = ?"
    )))
    .bind(session_id)
    .fetch_optional(executor)
    .await?;
    row.map(session_from_row)
        .transpose()?
        .ok_or_else(|| StoreError::not_found(format!("Session not found: {session_id}")))
}

fn session_from_row(row: SqliteRow) -> StoreResult<SessionInfo> {
    Ok(SessionInfo {
        id: row.try_get("id")?,
        project_id: row.try_get("project_id")?,
        workspace_id: row.try_get("workspace_id")?,
        title: row.try_get("title")?,
        workspace_root: row.try_get("workspace_root")?,
        cwd: row.try_get("cwd")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        provider: row.try_get("provider")?,
        model: row.try_get("model")?,
        agent: row.try_get("agent")?,
        permission: json_object(row.try_get("permission_json")?),
        cost_usd: row.try_get::<Option<f64>, _>("cost_usd")?.unwrap_or(0.0),
        tokens: TokenUsage {
            input: row.try_get::<Option<i64>, _>("tokens_input")?.unwrap_or(0),
            output: row.try_get::<Option<i64>, _>("tokens_output")?.unwrap_or(0),
            reasoning: row
                .try_get::<Option<i64>, _>("tokens_reasoning")?
                .unwrap_or(0),
            cache_read: row
                .try_get::<Option<i64>, _>("tokens_cache_read")?
                .unwrap_or(0),
            cache_write: row
                .try_get::<Option<i64>, _>("tokens_cache_write")?
                .unwrap_or(0),
        },
        summary: row.try_get("summary")?,
        active_root: row.try_get("active_root")?,
        focus_path: row.try_get("focus_path")?,
        last_prompt: row.try_get("last_prompt")?,
        state: json_object(row.try_get("state_json")?),
    })
}

fn route_from_row(row: SqliteRow) -> StoreResult<SessionRouteInfo> {
    Ok(SessionRouteInfo {
        route_key: row.try_get("route_key")?,
        session_id: row.try_get("session_id")?,
        agent_id: row.try_get("agent_id")?,
        scope: row.try_get("scope")?,
        channel: row.try_get("channel")?,
        account_id: row.try_get("account_id")?,
        peer_kind: row.try_get("peer_kind")?,
        peer_id: row.try_get("peer_id")?,
        sender_id: row.try_get("sender_id")?,
        guild_id: row.try_get("guild_id")?,
        team_id: row.try_get("team_id")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

fn json_object(raw: Option<String>) -> Option<Map<String, Value>> {
    raw.and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|value| value.as_object().cloned())
}

async fn migrate_legacy_session_columns(conn: &mut SqliteConnection) -> StoreResult<()> {
    let rows = sqlx::query("pragma table_info(sessions)")
        .fetch_all(&mut *conn)
        .await
        .map_err(StoreError::migration)?;
    let columns = rows
        .into_iter()
        .map(|row| row.try_get::<String, _>("name"))
        .collect::<Result<HashSet<_>, _>>()
        .map_err(StoreError::migration)?;
    const REQUIRED: &[(&str, &str)] = &[
        ("project_id", "text"),
        ("workspace_id", "text"),
        ("cwd", "text"),
        ("provider", "text"),
        ("agent", "text"),
        ("permission_json", "text"),
        ("cost_usd", "real not null default 0"),
        ("tokens_input", "integer not null default 0"),
        ("tokens_output", "integer not null default 0"),
        ("tokens_reasoning", "integer not null default 0"),
        ("tokens_cache_read", "integer not null default 0"),
        ("tokens_cache_write", "integer not null default 0"),
    ];
    for (column, definition) in REQUIRED {
        if !columns.contains(*column) {
            let statement = format!("alter table sessions add column {column} {definition}");
            sqlx::query(sqlx::AssertSqlSafe(statement))
                .execute(&mut *conn)
                .await
                .map_err(StoreError::migration)?;
        }
    }
    Ok(())
}

async fn backfill_session_context(conn: &mut SqliteConnection) -> StoreResult<()> {
    let rows = sqlx::query(
        "select id, workspace_root, cwd, project_id, workspace_id, permission_json
         from sessions
         where project_id is null or workspace_id is null or cwd is null
            or agent is null or permission_json is null",
    )
    .fetch_all(&mut *conn)
    .await
    .map_err(StoreError::migration)?;
    for row in rows {
        let session_id: String = row.try_get("id").map_err(StoreError::migration)?;
        let root: String = row
            .try_get("workspace_root")
            .map_err(StoreError::migration)?;
        let cwd: Option<String> = row.try_get("cwd").map_err(StoreError::migration)?;
        let cwd = cwd.unwrap_or_else(|| root.clone());
        let now = db_now(conn).await?;
        let project = ensure_project(conn, &root, &now).await?;
        let workspace = ensure_workspace(conn, &project.id, &root, &cwd, &now).await?;
        sqlx::query(
            "update sessions
             set project_id = coalesce(project_id, ?),
                 workspace_id = coalesce(workspace_id, ?),
                 cwd = coalesce(cwd, ?),
                 agent = coalesce(agent, 'agent'),
                 permission_json = coalesce(permission_json, ?)
             where id = ?",
        )
        .bind(project.id)
        .bind(workspace.id)
        .bind(cwd)
        .bind(DEFAULT_PERMISSION_JSON)
        .bind(session_id)
        .execute(&mut *conn)
        .await
        .map_err(StoreError::migration)?;
    }
    Ok(())
}

async fn db_now(conn: &mut SqliteConnection) -> StoreResult<String> {
    let row = sqlx::query("select strftime('%Y-%m-%dT%H:%M:%f+00:00', 'now') as timestamp")
        .fetch_one(conn)
        .await?;
    Ok(row.try_get("timestamp")?)
}

async fn db_id(conn: &mut SqliteConnection) -> StoreResult<String> {
    let row = sqlx::query("select lower(hex(randomblob(6))) as id")
        .fetch_one(conn)
        .await?;
    Ok(row.try_get("id")?)
}

async fn ensure_project(
    conn: &mut SqliteConnection,
    root: &str,
    now: &str,
) -> StoreResult<ProjectInfo> {
    if let Some(row) =
        sqlx::query("select id, root, title, created_at, updated_at from projects where root = ?")
            .bind(root)
            .fetch_optional(&mut *conn)
            .await?
    {
        return project_from_row(row);
    }
    let id = db_id(conn).await?;
    let title = Path::new(root)
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(root);
    sqlx::query(
        "insert into projects (id, root, title, created_at, updated_at)
         values (?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(root)
    .bind(title)
    .bind(now)
    .bind(now)
    .execute(&mut *conn)
    .await?;
    Ok(ProjectInfo {
        id,
        root: root.to_owned(),
        title: title.to_owned(),
        created_at: now.to_owned(),
        updated_at: now.to_owned(),
    })
}

fn project_from_row(row: SqliteRow) -> StoreResult<ProjectInfo> {
    Ok(ProjectInfo {
        id: row.try_get("id")?,
        root: row.try_get("root")?,
        title: row.try_get("title")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

async fn ensure_workspace(
    conn: &mut SqliteConnection,
    project_id: &str,
    root: &str,
    cwd: &str,
    now: &str,
) -> StoreResult<WorkspaceInfo> {
    if let Some(row) = sqlx::query(
        "select id, project_id, root, cwd, created_at, updated_at
         from workspaces where project_id = ? and root = ? and cwd = ?",
    )
    .bind(project_id)
    .bind(root)
    .bind(cwd)
    .fetch_optional(&mut *conn)
    .await?
    {
        return workspace_from_row(row);
    }
    let id = db_id(conn).await?;
    sqlx::query(
        "insert into workspaces (id, project_id, root, cwd, created_at, updated_at)
         values (?, ?, ?, ?, ?, ?)",
    )
    .bind(&id)
    .bind(project_id)
    .bind(root)
    .bind(cwd)
    .bind(now)
    .bind(now)
    .execute(&mut *conn)
    .await?;
    Ok(WorkspaceInfo {
        id,
        project_id: project_id.to_owned(),
        root: root.to_owned(),
        cwd: cwd.to_owned(),
        created_at: now.to_owned(),
        updated_at: now.to_owned(),
    })
}

fn workspace_from_row(row: SqliteRow) -> StoreResult<WorkspaceInfo> {
    Ok(WorkspaceInfo {
        id: row.try_get("id")?,
        project_id: row.try_get("project_id")?,
        root: row.try_get("root")?,
        cwd: row.try_get("cwd")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

#[allow(clippy::too_many_arguments)]
async fn insert_session(
    conn: &mut SqliteConnection,
    session_id: &str,
    project_id: &str,
    workspace_id: &str,
    title: String,
    workspace_root: &str,
    now: &str,
    provider: Option<&str>,
    model: Option<&str>,
    agent_id: &str,
    state_json: Option<&str>,
) -> StoreResult<()> {
    sqlx::query(
        "insert into sessions (
            id, project_id, workspace_id, title, workspace_root, cwd,
            created_at, updated_at, provider, model, agent, permission_json,
            cost_usd, tokens_input, tokens_output, tokens_reasoning,
            tokens_cache_read, tokens_cache_write, summary, active_root,
            focus_path, last_prompt, state_json
         ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 0, 0, 0,
                   null, null, null, null, ?)",
    )
    .bind(session_id)
    .bind(project_id)
    .bind(workspace_id)
    .bind(title)
    .bind(workspace_root)
    .bind(workspace_root)
    .bind(now)
    .bind(now)
    .bind(provider)
    .bind(model)
    .bind(agent_id)
    .bind(DEFAULT_PERMISSION_JSON)
    .bind(state_json)
    .execute(conn)
    .await?;
    Ok(())
}

fn required_ref<'a>(value: &'a str, name: &str) -> StoreResult<&'a str> {
    let value = value.trim();
    if value.is_empty() {
        return Err(StoreError::invalid(format!("{name} cannot be empty.")));
    }
    Ok(value)
}

fn required_text(value: String, name: &str) -> StoreResult<String> {
    Ok(required_ref(&value, name)?.to_owned())
}

fn clean_title(value: Option<&str>) -> Option<String> {
    let text = compact_whitespace(value?);
    if text.is_empty() {
        return None;
    }
    if text.chars().count() <= 80 {
        return Some(text.into_owned());
    }
    let mut shortened = text.chars().take(79).collect::<String>();
    shortened.push_str("...");
    Some(shortened)
}
