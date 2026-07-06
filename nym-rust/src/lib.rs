// lib.rs
mod read_path;
mod glob;
mod grep;
mod target_resolver;
mod write_file;
mod edit_file;
mod delete_path;

pub use glob::{glob_files, GlobKind, GlobMatch, GlobOptions, GlobResult};
pub use grep::{grep_files, GrepMatch, GrepOptions, GrepResult};
pub use read_path::{
    read_path, ContentDetection, ReadContentKind, ReadLimits, ReadPathOptions, ReadPathResult,
};


pub mod search_file;

pub use search_file::{
    search_files, search_files_staged, FileMatch, FileSearchOptions, MatchType, SearchKind,
    SearchMode, SearchStrategy,
};

pub mod roots;

pub use roots::{resolve_search_roots, system_search_roots};
pub use target_resolver::{
    resolve_target, ResolveConfidence, ResolveSource, ResolveTargetOptions, ResolveTargetResult,
    ResolvedKind, TargetKind,
};
pub use write_file::{write_file, WriteFileOptions, WriteFileResult};
pub use edit_file::{edit_file, EditFileOptions, EditFileResult};
pub use delete_path::{delete_path, DeletePathOptions, DeletePathResult};
