use anyhow::Result;
use clap::{Parser, Subcommand};
use nym_rust::{
    delete_path, edit_file, glob_files, grep_files, read_path, resolve_search_roots,
    resolve_target, search_files, system_search_roots, write_file, DeletePathOptions,
    EditFileOptions, FileSearchOptions, GlobKind, GlobOptions, GrepOptions, ReadLimits,
    ReadPathOptions, ResolveTargetOptions, SearchKind, SearchMode, SearchStrategy, TargetKind,
    WriteFileOptions,
};
use std::num::NonZeroUsize;
use std::path::PathBuf;

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
