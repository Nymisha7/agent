use anyhow::{Context, Result};
use nucleo_matcher::pattern::{CaseMatching, Normalization, Pattern};
use nucleo_matcher::{Config, Matcher, Utf32Str};
use serde::Serialize;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use crate::ripgrep::{ripgrep_paths, RipgrepFilesOptions, RipgrepPath, RipgrepPathKind};

#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    Interactive,
    Balanced,
    Aggressive,
    Background,
    Manual(usize),
}

#[derive(Debug, Clone, Copy)]
pub enum SearchStrategy {
    ExactName,
    ContainsName,
    FuzzyPath,
}

#[derive(Debug, Clone, Copy)]
pub enum SearchKind {
    Any,
    File,
    Directory,
}

#[derive(Debug, Clone)]
pub struct FileSearchOptions {
    pub roots: Vec<PathBuf>, // is owned and mutable || to search multiple roots
    pub query: String,
    pub limit: NonZeroUsize,
    pub include_hidden: bool, //whether to include hidden files
    pub follow_links: bool,
    pub search_mode: SearchMode, //symbolic link - pointer to another file or directory
    pub strategy: SearchStrategy,
    pub kind: SearchKind,
    pub include_generated: bool,
    //pub max_workers: NonZeroUsize, //better to set max_workers to the number of physical cores (num_cpus::get_physical()), not logical threads, if you want to avoid oversubscription.
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MatchType {
    File,
    Directory,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileMatch {
    pub score: u32,    //relevance score from the fuzzy matcher, higher is better
    pub path: PathBuf, //full filesystem path to the matched file
    //pub match type: MatchType, // enum describing how the match was found
    pub match_type: MatchType,
    pub root: PathBuf, // the root dir from which this match originated
}

#[derive(Debug, Clone)]
pub struct StagedSearchResult {
    pub matches: Vec<FileMatch>,
}

#[derive(Debug, Clone)]
struct Candidate {
    root: PathBuf,
    relative_path: PathBuf,
    display_path: String,
    match_type: MatchType,
}

impl FileSearchOptions {
    // implementation block- its where we attach methds(functions) to a struct
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            roots: default_search_roots(),
            query: query.into(),
            limit: NonZeroUsize::new(50).expect("50 is non zero"),
            include_hidden: false,
            follow_links: false,
            search_mode: SearchMode::Interactive,
            strategy: SearchStrategy::FuzzyPath,
            kind: SearchKind::Any,
            include_generated: false,
        }
    }
}

fn default_search_roots() -> Vec<PathBuf> {
    vec![std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))]
}

pub fn search_files(options: FileSearchOptions) -> Result<Vec<FileMatch>> {
    let query = options.query.trim(); // Removes leading/trailing whitespace from the search string

    if query.is_empty() {
        return Ok(Vec::new());
    } //If the query is empty after trimming, return an empty result immediately.

    let mut collector = SearchCollector::new(query, options.strategy, options.limit);
    collect_ranked_matches(&options, std::slice::from_mut(&mut collector))?;
    let mut matches = collector.finish();

    matches.truncate(options.limit.get());

    Ok(matches)
}

pub fn search_files_staged(
    options: FileSearchOptions,
    strategies: &[SearchStrategy],
) -> Result<Vec<StagedSearchResult>> {
    let query = options.query.trim();
    if query.is_empty() {
        return Ok(strategies
            .iter()
            .map(|_| StagedSearchResult {
                matches: Vec::new(),
            })
            .collect());
    }

    let mut collectors = strategies
        .iter()
        .copied()
        .map(|strategy| SearchCollector::new(query, strategy, options.limit))
        .collect::<Vec<_>>();
    collect_ranked_matches(&options, &mut collectors)?;

    let staged = collectors
        .into_iter()
        .map(|collector| StagedSearchResult {
            matches: collector.finish(),
        })
        .collect();

    Ok(staged)
}

fn collect_ranked_matches(
    options: &FileSearchOptions,
    collectors: &mut [SearchCollector],
) -> Result<()> {
    for root in &options.roots {
        let root = expand_tilde(root);

        let root = root
            .canonicalize()
            .with_context(|| format!("Invalid search root: {}", root.display()))?;

        collect_candidates(&root, options, |candidate| {
            for collector in collectors.iter_mut() {
                collector.observe(&candidate);
            }
        })?;
    }

    Ok(())
}

struct SearchCollector {
    inner: SearchCollectorInner,
}

enum SearchCollectorInner {
    Exact(ExactCollector),
    Contains(ScoredCollector),
    Fuzzy(FuzzyCollector),
}

impl SearchCollector {
    fn new(query: &str, strategy: SearchStrategy, limit: NonZeroUsize) -> Self {
        let inner = match strategy {
            SearchStrategy::ExactName => {
                SearchCollectorInner::Exact(ExactCollector::new(query.to_lowercase(), limit))
            }
            SearchStrategy::ContainsName => {
                SearchCollectorInner::Contains(ScoredCollector::new(query.to_lowercase(), limit))
            }
            SearchStrategy::FuzzyPath => {
                SearchCollectorInner::Fuzzy(FuzzyCollector::new(query, limit))
            }
        };
        Self { inner }
    }

    fn observe(&mut self, candidate: &Candidate) {
        match &mut self.inner {
            SearchCollectorInner::Exact(collector) => collector.observe(candidate),
            SearchCollectorInner::Contains(collector) => collector.observe(candidate),
            SearchCollectorInner::Fuzzy(collector) => collector.observe(candidate),
        }
    }

    fn finish(self) -> Vec<FileMatch> {
        match self.inner {
            SearchCollectorInner::Exact(collector) => collector.finish(),
            SearchCollectorInner::Contains(collector) => collector.finish(),
            SearchCollectorInner::Fuzzy(collector) => collector.finish(),
        }
    }
}

struct ExactCollector {
    query: String,
    limit: NonZeroUsize,
    matches: Vec<FileMatch>,
}

impl ExactCollector {
    fn new(query: String, limit: NonZeroUsize) -> Self {
        Self {
            query,
            limit,
            matches: Vec::new(),
        }
    }

    fn observe(&mut self, candidate: &Candidate) {
        if candidate_name(candidate).to_lowercase() != self.query {
            return;
        }

        self.matches.push(FileMatch {
            score: 1_000,
            path: candidate.relative_path.clone(),
            match_type: candidate.match_type,
            root: candidate.root.clone(),
        });
    }

    fn finish(mut self) -> Vec<FileMatch> {
        self.matches.sort_by(|a, b| a.path.cmp(&b.path));
        self.matches.truncate(self.limit.get());
        self.matches
    }
}

struct ScoredCollector {
    query: String,
    limit: NonZeroUsize,
    heap: BinaryHeap<Reverse<ScoredCandidate>>,
}

impl ScoredCollector {
    fn new(query: String, limit: NonZeroUsize) -> Self {
        Self {
            query,
            limit,
            heap: BinaryHeap::new(),
        }
    }

    fn observe(&mut self, candidate: &Candidate) {
        let name = candidate_name(candidate);
        let name_lower = name.to_lowercase();
        let Some(position) = name_lower.find(&self.query) else {
            return;
        };

        let position_score = 10_000u32.saturating_sub((position as u32) * 100);
        let length_bonus =
            1_000u32.saturating_sub(name.len().saturating_sub(self.query.len()) as u32);
        let score = 500 + position_score + length_bonus;

        self.push(ScoredCandidate::new(score, candidate));
    }

    fn push(&mut self, item: ScoredCandidate) {
        self.heap.push(Reverse(item));
        if self.heap.len() > self.limit.get() {
            self.heap.pop();
        }
    }

    fn finish(self) -> Vec<FileMatch> {
        heap_into_sorted_matches(self.heap)
    }
}

struct FuzzyCollector {
    limit: NonZeroUsize,
    pattern: Pattern,
    matcher: Matcher,
    buf: Vec<char>,
    heap: BinaryHeap<Reverse<ScoredCandidate>>,
}

impl FuzzyCollector {
    fn new(query: &str, limit: NonZeroUsize) -> Self {
        Self {
            limit,
            pattern: Pattern::parse(query, CaseMatching::Ignore, Normalization::Smart),
            matcher: Matcher::new(Config::DEFAULT.match_paths()),
            buf: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    fn observe(&mut self, candidate: &Candidate) {
        let haystack = Utf32Str::new(candidate.display_path.as_str(), &mut self.buf);
        let Some(score) = self.pattern.score(haystack, &mut self.matcher) else {
            return;
        };

        self.heap
            .push(Reverse(ScoredCandidate::new(score, candidate)));
        if self.heap.len() > self.limit.get() {
            self.heap.pop();
        }
    }

    fn finish(self) -> Vec<FileMatch> {
        heap_into_sorted_matches(self.heap)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ScoredCandidate {
    score: u32,
    path: PathBuf,
    match_type: MatchType,
    root: PathBuf,
}

impl ScoredCandidate {
    fn new(score: u32, candidate: &Candidate) -> Self {
        Self {
            score,
            path: candidate.relative_path.clone(),
            match_type: candidate.match_type,
            root: candidate.root.clone(),
        }
    }

    fn into_file_match(self) -> FileMatch {
        FileMatch {
            score: self.score,
            path: self.path,
            match_type: self.match_type,
            root: self.root,
        }
    }
}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| other.path.cmp(&self.path))
            .then_with(|| other.root.cmp(&self.root))
    }
}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn heap_into_sorted_matches(heap: BinaryHeap<Reverse<ScoredCandidate>>) -> Vec<FileMatch> {
    let mut matches = heap
        .into_iter()
        .map(|Reverse(item)| item.into_file_match())
        .collect::<Vec<_>>();
    matches.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
    matches
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

fn collect_candidates<F>(
    root: &Path,
    options: &FileSearchOptions,
    mut on_candidate: F,
) -> Result<()>
where
    F: FnMut(Candidate),
{
    for path in ripgrep_paths(
        root,
        RipgrepFilesOptions {
            include_hidden: options.include_hidden,
            follow_links: options.follow_links,
            include_ignored: options.include_generated,
            threads: match options.search_mode {
                SearchMode::Background => Some(1),
                SearchMode::Manual(threads) => Some(threads),
                SearchMode::Interactive | SearchMode::Balanced | SearchMode::Aggressive => None,
            },
        },
    )? {
        if let Some(candidate) = candidate_from_ripgrep_path(root, options, path) {
            on_candidate(candidate);
        }
    }

    Ok(())
}

fn candidate_from_ripgrep_path(
    root: &Path,
    options: &FileSearchOptions,
    path: RipgrepPath,
) -> Option<Candidate> {
    let match_type = match path.kind {
        RipgrepPathKind::File => MatchType::File,
        RipgrepPathKind::Directory => MatchType::Directory,
    };

    if !matches_kind(options.kind, match_type) {
        return None;
    }

    let display_path = path.relative.to_string_lossy().replace('\\', "/");
    if display_path.is_empty() {
        return None;
    }

    Some(Candidate {
        root: root.to_path_buf(),
        relative_path: path.relative,
        display_path,
        match_type,
    })
}

fn matches_kind(kind: SearchKind, match_type: MatchType) -> bool {
    match kind {
        SearchKind::Any => true,
        SearchKind::File => matches!(match_type, MatchType::File),
        SearchKind::Directory => matches!(match_type, MatchType::Directory),
    }
}

fn candidate_name(candidate: &Candidate) -> String {
    candidate
        .relative_path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| candidate.display_path.clone())
}
