use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{
    search_files_staged, FileMatch, FileSearchOptions, MatchType, SearchKind, SearchMode,
    SearchStrategy,
};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    Any,
    File,
    Directory,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolvedKind {
    File,
    Directory,
    Other,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolveSource {
    PlatformTranslated,
    RawAbsolute,
    FocusRelative,
    WorkspaceRelative,
    WorkspaceExactSearch,
    WorkspaceContainsSearch,
    SystemExactSearch,
    SystemContainsSearch,
    SystemFuzzySearch,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolveConfidence {
    ExactPath,
    ExactSearch,
    ContainsSearch,
    FuzzySearch,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolvedTarget {
    pub path: PathBuf,
    pub kind: ResolvedKind,
    pub source: ResolveSource,
    pub confidence: ResolveConfidence,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResolveAttempt {
    pub strategy: String,
    pub input: String,
    pub ok: bool,
    pub path: Option<PathBuf>,
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ResolveTargetResult {
    Resolved {
        target: ResolvedTarget,
        attempts: Vec<ResolveAttempt>,
    },
    Candidates {
        query: String,
        candidates: Vec<FileMatch>,
        source: ResolveSource,
        confidence: ResolveConfidence,
        attempts: Vec<ResolveAttempt>,
        reason: String,
    },
    NotFound {
        query: String,
        attempts: Vec<ResolveAttempt>,
        message: String,
    },
}

#[derive(Debug, Clone)]
pub struct ResolveTargetOptions {
    pub raw_target: String,
    pub workspace_root: PathBuf,
    pub focus_path: Option<PathBuf>,
    pub kind: TargetKind,
    pub limit: usize,
    pub allow_system_fallback: bool,
    pub allow_contains_fallback: bool,
    pub allow_fuzzy_fallback: bool,
}

#[derive(Debug, Clone)]
struct DirectCandidate {
    path: PathBuf,
    source: ResolveSource,
    strategy: &'static str,
}

#[derive(Debug, Clone)]
struct ResolveSearchStage {
    strategy: SearchStrategy,
    source: ResolveSource,
    confidence: ResolveConfidence,
}

pub fn resolve_target(options: ResolveTargetOptions) -> Result<ResolveTargetResult> {
    let raw = clean_target(&options.raw_target);

    if raw.is_empty() {
        anyhow::bail!("target is empty");
    }

    let mut attempts = Vec::new();

    for candidate in direct_path_candidates(
        &raw,
        &options.workspace_root,
        options.focus_path.as_deref(),
        &mut attempts,
    ) {
        match candidate.path.canonicalize() {
            Ok(canonical) => {
                let kind_ok = kind_matches(&canonical, &options.kind);
                attempts.push(ResolveAttempt {
                    strategy: candidate.strategy.to_string(),
                    input: raw.clone(),
                    ok: kind_ok,
                    path: Some(canonical.clone()),
                    error: if kind_ok {
                        None
                    } else {
                        Some(format!(
                            "kind mismatch: resolved path is {}, expected {:?}",
                            path_kind_string(&canonical),
                            options.kind
                        ))
                    },
                });

                if kind_ok {
                    return Ok(ResolveTargetResult::Resolved {
                        target: ResolvedTarget {
                            path: canonical.clone(),
                            kind: resolved_kind(&canonical),
                            source: candidate.source,
                            confidence: ResolveConfidence::ExactPath,
                        },
                        attempts,
                    });
                }
            }
            Err(error) => {
                attempts.push(ResolveAttempt {
                    strategy: candidate.strategy.to_string(),
                    input: raw.clone(),
                    ok: false,
                    path: Some(candidate.path.clone()),
                    error: Some(error.to_string()),
                });
            }
        }
    }

    let hint = extract_search_hint(&raw);
    let query = hint.basename.clone();

    if query.trim().is_empty() {
        return Ok(ResolveTargetResult::NotFound {
            query,
            attempts,
            message: "Could not extract a usable search query from target.".to_string(),
        });
    }

    let mut workspace_stages = vec![ResolveSearchStage {
        strategy: SearchStrategy::ExactName,
        source: ResolveSource::WorkspaceExactSearch,
        confidence: ResolveConfidence::ExactSearch,
    }];
    if options.allow_contains_fallback {
        workspace_stages.push(ResolveSearchStage {
            strategy: SearchStrategy::ContainsName,
            source: ResolveSource::WorkspaceContainsSearch,
            confidence: ResolveConfidence::ContainsSearch,
        });
    }
    if options.allow_fuzzy_fallback {
        workspace_stages.push(ResolveSearchStage {
            strategy: SearchStrategy::FuzzyPath,
            source: ResolveSource::WorkspaceContainsSearch,
            confidence: ResolveConfidence::FuzzySearch,
        });
    }

    if let Some(result) = search_scope_staged(
        &query,
        vec![options.workspace_root.clone()],
        &options.kind,
        &workspace_stages,
        options.limit,
        &hint,
        &mut attempts,
    )? {
        return Ok(result);
    }

    if !options.allow_system_fallback {
        return Ok(ResolveTargetResult::NotFound {
            query,
            attempts,
            message: "No matching target found in direct path or workspace search.".to_string(),
        });
    }

    let system_roots = crate::system_search_roots()?;

    let mut system_stages = vec![ResolveSearchStage {
        strategy: SearchStrategy::ExactName,
        source: ResolveSource::SystemExactSearch,
        confidence: ResolveConfidence::ExactSearch,
    }];
    if options.allow_contains_fallback {
        system_stages.push(ResolveSearchStage {
            strategy: SearchStrategy::ContainsName,
            source: ResolveSource::SystemContainsSearch,
            confidence: ResolveConfidence::ContainsSearch,
        });
    }

    if options.allow_fuzzy_fallback {
        system_stages.push(ResolveSearchStage {
            strategy: SearchStrategy::FuzzyPath,
            source: ResolveSource::SystemFuzzySearch,
            confidence: ResolveConfidence::FuzzySearch,
        });
    }

    if let Some(result) = search_scope_staged(
        &query,
        system_roots,
        &options.kind,
        &system_stages,
        options.limit,
        &hint,
        &mut attempts,
    )? {
        return Ok(result);
    }

    Ok(ResolveTargetResult::NotFound {
        query,
        attempts,
        message: "No matching target found.".to_string(),
    })
}

fn search_scope_staged(
    query: &str,
    roots: Vec<PathBuf>,
    kind: &TargetKind,
    stages: &[ResolveSearchStage],
    limit: usize,
    hint: &SearchHint,
    attempts: &mut Vec<ResolveAttempt>,
) -> Result<Option<ResolveTargetResult>> {
    let root_list = roots
        .iter()
        .map(|root| root.display().to_string())
        .collect::<Vec<_>>()
        .join(",");

    let mut options = FileSearchOptions::new(query.to_string());
    options.roots = roots;
    options.limit = NonZeroUsize::new(limit.max(1)).expect("limit is non-zero");
    options.search_mode = SearchMode::Interactive;
    options.kind = to_search_kind(kind);
    options.include_hidden = false;
    options.include_generated = false;

    let strategies = stages
        .iter()
        .map(|stage| stage.strategy)
        .collect::<Vec<_>>();
    let staged_results = search_files_staged(options, &strategies)?;
    for (stage, mut staged) in stages.iter().zip(staged_results) {
        rank_matches_by_hint(&mut staged.matches, hint);

        attempts.push(ResolveAttempt {
            strategy: format!("{:?}", stage.source),
            input: query.to_string(),
            ok: !staged.matches.is_empty(),
            path: None,
            error: if staged.matches.is_empty() {
                Some(format!("no matches in roots: {root_list}"))
            } else {
                None
            },
        });

        if staged.matches.is_empty() {
            continue;
        }

        if staged.matches.len() == 1 && !matches!(stage.confidence, ResolveConfidence::FuzzySearch)
        {
            let item = &staged.matches[0];

            return Ok(Some(ResolveTargetResult::Resolved {
                target: ResolvedTarget {
                    path: item.root.join(&item.path),
                    kind: match_type_to_resolved_kind(&item.match_type),
                    source: stage.source.clone(),
                    confidence: stage.confidence.clone(),
                },
                attempts: attempts.clone(),
            }));
        }

        let reason = if staged.matches.len() == 1 {
            "single low-confidence fuzzy target found"
        } else {
            "multiple matching targets found"
        };
        return Ok(Some(ResolveTargetResult::Candidates {
            query: query.to_string(),
            candidates: staged.matches,
            source: stage.source.clone(),
            confidence: stage.confidence.clone(),
            attempts: attempts.clone(),
            reason: reason.to_string(),
        }));
    }

    Ok(None)
}

fn direct_path_candidates(
    raw: &str,
    workspace_root: &Path,
    focus_path: Option<&Path>,
    attempts: &mut Vec<ResolveAttempt>,
) -> Vec<DirectCandidate> {
    let mut candidates = Vec::new();

    if looks_like_windows_drive_path(raw) {
        match platform_translate_windows_path(raw) {
            Ok(path) => {
                attempts.push(ResolveAttempt {
                    strategy: "platform_translate_windows_path".to_string(),
                    input: raw.to_string(),
                    ok: true,
                    path: Some(path.clone()),
                    error: None,
                });

                candidates.push(DirectCandidate {
                    path,
                    source: ResolveSource::PlatformTranslated,
                    strategy: "platform_translated",
                });
            }
            Err(error) => {
                attempts.push(ResolveAttempt {
                    strategy: "platform_translate_windows_path".to_string(),
                    input: raw.to_string(),
                    ok: false,
                    path: None,
                    error: Some(error.to_string()),
                });
            }
        }
    }

    let raw_path = expand_tilde(Path::new(raw));

    if raw_path.is_absolute() {
        candidates.push(DirectCandidate {
            path: raw_path.clone(),
            source: ResolveSource::RawAbsolute,
            strategy: "raw_absolute",
        });
    }

    if let Some(focus) = focus_path {
        if !raw_path.is_absolute() {
            candidates.push(DirectCandidate {
                path: focus.join(&raw_path),
                source: ResolveSource::FocusRelative,
                strategy: "focus_relative",
            });
        }
    }

    if !raw_path.is_absolute() {
        candidates.push(DirectCandidate {
            path: workspace_root.join(&raw_path),
            source: ResolveSource::WorkspaceRelative,
            strategy: "workspace_relative",
        });
    }

    dedupe_candidates(candidates)
}

fn clean_target(raw: &str) -> String {
    raw.trim()
        .trim_matches('"')
        .trim_matches('\'')
        .trim()
        .to_string()
}

fn looks_like_windows_drive_path(value: &str) -> bool {
    let bytes = value.as_bytes();

    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

fn platform_translate_windows_path(value: &str) -> Result<PathBuf> {
    if is_wsl() {
        let output = Command::new("wslpath")
            .arg("-u")
            .arg(value)
            .output()
            .context("failed to run wslpath")?;

        if output.status.success() {
            let converted = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !converted.is_empty() {
                return Ok(PathBuf::from(converted));
            }
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("wslpath failed: {stderr}");
    }

    anyhow::bail!("Windows path provided, but this runtime is not WSL");
}

fn is_wsl() -> bool {
    std::env::var_os("WSL_INTEROP").is_some()
        || std::fs::read_to_string("/proc/version")
            .map(|text| text.to_lowercase().contains("microsoft"))
            .unwrap_or(false)
}

fn expand_tilde(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();

    if raw == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }

    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }

    path.to_path_buf()
}

fn dedupe_candidates(candidates: Vec<DirectCandidate>) -> Vec<DirectCandidate> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for candidate in candidates {
        if seen.insert(candidate.path.clone()) {
            result.push(candidate);
        }
    }

    result
}

fn kind_matches(path: &Path, kind: &TargetKind) -> bool {
    match kind {
        TargetKind::Any => path.exists(),
        TargetKind::File => path.is_file(),
        TargetKind::Directory => path.is_dir(),
    }
}

fn resolved_kind(path: &Path) -> ResolvedKind {
    if path.is_dir() {
        ResolvedKind::Directory
    } else if path.is_file() {
        ResolvedKind::File
    } else {
        ResolvedKind::Other
    }
}

fn path_kind_string(path: &Path) -> &'static str {
    if path.is_dir() {
        "directory"
    } else if path.is_file() {
        "file"
    } else {
        "other"
    }
}

fn match_type_to_resolved_kind(match_type: &MatchType) -> ResolvedKind {
    match match_type {
        MatchType::File => ResolvedKind::File,
        MatchType::Directory => ResolvedKind::Directory,
    }
}

fn to_search_kind(kind: &TargetKind) -> SearchKind {
    match kind {
        TargetKind::Any => SearchKind::Any,
        TargetKind::File => SearchKind::File,
        TargetKind::Directory => SearchKind::Directory,
    }
}

#[derive(Debug, Clone)]
struct SearchHint {
    basename: String,
    path_tokens: Vec<String>,
}

fn extract_search_hint(raw: &str) -> SearchHint {
    let normalized = raw.replace('\\', "/");
    let path = Path::new(&normalized);

    let basename = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .filter(|name| !name.trim().is_empty())
        .unwrap_or_else(|| raw.to_string());

    let path_tokens = normalized
        .split('/')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .filter(|part| *part != basename)
        .map(|part| part.to_lowercase())
        .collect();

    SearchHint {
        basename,
        path_tokens,
    }
}

fn rank_matches_by_hint(matches: &mut [FileMatch], hint: &SearchHint) {
    if hint.path_tokens.is_empty() {
        return;
    }

    matches.sort_by(|a, b| {
        let a_score = path_hint_score(&a.path, hint);
        let b_score = path_hint_score(&b.path, hint);

        b_score
            .cmp(&a_score)
            .then_with(|| b.score.cmp(&a.score))
            .then_with(|| a.path.cmp(&b.path))
    });
}

fn path_hint_score(path: &Path, hint: &SearchHint) -> usize {
    let path_text = path.to_string_lossy().to_lowercase();

    hint.path_tokens
        .iter()
        .filter(|token| path_text.contains(token.as_str()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn single_fuzzy_match_stays_a_candidate() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "agent-target-resolver-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("calculator")).expect("create test root");
        fs::write(root.join("calculator/make_projected.h"), "").expect("create fuzzy candidate");

        let result = resolve_target(ResolveTargetOptions {
            raw_target: "calculator_project".to_string(),
            workspace_root: root.clone(),
            focus_path: None,
            kind: TargetKind::Any,
            limit: 10,
            allow_system_fallback: false,
            allow_contains_fallback: true,
            allow_fuzzy_fallback: true,
        })
        .expect("resolve target");

        fs::remove_dir_all(root).expect("remove test root");
        match result {
            ResolveTargetResult::Candidates {
                confidence,
                candidates,
                ..
            } => {
                assert!(matches!(confidence, ResolveConfidence::FuzzySearch));
                assert_eq!(candidates.len(), 1);
            }
            other => panic!("expected a low-confidence candidate, got {other:?}"),
        }
    }
}
