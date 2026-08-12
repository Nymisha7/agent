// lib.rs
mod delete_path;
mod edit_file;
mod glob;
mod grep;
mod inspect_tree;
mod read_path;
mod ripgrep;
mod target_resolver;
mod text;
mod write_file;

pub use glob::{glob_files, GlobKind, GlobMatch, GlobOptions, GlobResult};
pub use grep::{grep_files, GrepMatch, GrepOptions, GrepResult};
pub use inspect_tree::{inspect_tree, InspectTreeOptions, InspectTreeResult};
pub use read_path::{
    read_path, ContentDetection, ReadContentKind, ReadLimits, ReadPathOptions, ReadPathResult,
};

pub mod search_file;

pub use search_file::{
    search_files, search_files_staged, FileMatch, FileSearchOptions, MatchType, SearchKind,
    SearchMode, SearchStrategy,
};

pub mod roots;

pub use delete_path::{delete_path, DeletePathOptions, DeletePathResult};
pub use edit_file::{edit_file, EditFileOptions, EditFileResult};
pub use roots::{resolve_search_roots, system_search_roots};
pub use target_resolver::{
    resolve_target, ResolveConfidence, ResolveSource, ResolveTargetOptions, ResolveTargetResult,
    ResolvedKind, TargetKind,
};
pub use text::compact_whitespace;
pub use write_file::{write_file, WriteFileOptions, WriteFileResult};
