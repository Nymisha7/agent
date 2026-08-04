use anyhow::{Context, Result};
use chardetng::{EncodingDetector, Iso2022JpDetection, Utf8Detection};
use content_inspector::ContentType;
use encoding_rs::Encoding;
use serde::Serialize;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ReadLimits {
    pub default_limit: usize,
    pub max_limit: usize,
    pub max_line_length: usize,
    pub max_bytes: usize,
    pub sample_bytes: usize,
}

impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            default_limit: 200,
            max_limit: 2_000,
            max_line_length: 2_000,
            max_bytes: 50 * 1024,
            sample_bytes: 4_096,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReadPathOptions {
    pub path: PathBuf,
    pub offset: usize,
    pub limit: Option<usize>,
    pub limits: ReadLimits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadContentKind {
    Empty,
    Utf8Text,
    Utf16LeText,
    Utf16BeText,
    Directory,
    Image,
    Pdf,
    Binary,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReadPathResult {
    pub path: PathBuf,
    pub detection: ContentDetection,
    pub offset: usize,
    pub end_line: usize,
    pub lines_seen: usize,
    pub total_lines: Option<usize>,
    pub truncated: bool,
    pub bytes_read: usize,
    pub content: String,
}

fn effective_offset(offset: usize) -> usize {
    offset.max(1)
}

fn effective_limit(options: &ReadPathOptions) -> usize {
    let requested = options.limit.unwrap_or(options.limits.default_limit);
    requested.clamp(1, options.limits.max_limit)
}

//path resolution and validation

fn resolve_existing_path(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("Invalid path: {}", path.display()))
}

fn read_path_metadata_kind(path: &Path) -> Result<(std::fs::Metadata, ReadContentKind)> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Failed to read metadata: {}", path.display()))?;

    if metadata.is_dir() {
        return Ok((metadata, ReadContentKind::Directory));
    }

    if metadata.is_file() {
        return Ok((metadata, ReadContentKind::Unknown));
    }

    anyhow::bail!(
        "Path is neither a regular file nor directory: {}",
        path.display()
    );
}

// before treating file as text we inspect a small byte sample , so this will help is in avoiding reading b files as text

fn read_sample(path: &Path, sample_bytes: usize) -> Result<Vec<u8>> {
    let mut file = File::open(path)
        .with_context(|| format!("Failed to open file for sampling: {}", path.display()))?;

    let mut buffer = vec![0_u8; sample_bytes];
    let bytes_read = file
        .read(&mut buffer)
        .with_context(|| format!("Failed to sample file: {}", path.display()))?;

    buffer.truncate(bytes_read);
    Ok(buffer)
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentDetection {
    pub kind: ReadContentKind,
    pub mime: Option<String>,
    pub encoding: Option<String>,
    pub reason: String,
    pub read_strategy: String,
}

pub fn detect_content_from_sample(_path: &Path, sample: &[u8]) -> ContentDetection {
    if sample.is_empty() {
        return ContentDetection {
            kind: ReadContentKind::Empty,
            mime: Some("text/plain".to_string()),
            encoding: Some("utf-8".to_string()),
            reason: "file sample is empty".to_string(),
            read_strategy: "empty".to_string(),
        };
    }

    let inferred_mime = infer::get(sample).map(|file_type| file_type.mime_type().to_string());

    if let Some(mime) = inferred_mime.as_deref() {
        if mime == "application/pdf" {
            return ContentDetection {
                kind: ReadContentKind::Pdf,
                mime: inferred_mime,
                encoding: None,
                reason: "infer matched PDF magic bytes".to_string(),
                read_strategy: "attachment_or_metadata".to_string(),
            };
        }

        if mime == "image/svg+xml" {
            return text_detection(
                ReadContentKind::Utf8Text,
                inferred_mime,
                Some("utf-8"),
                "infer matched SVG; treat as text",
            );
        }

        if mime.starts_with("image/") {
            return ContentDetection {
                kind: ReadContentKind::Image,
                mime: inferred_mime,
                encoding: None,
                reason: "infer matched image magic bytes".to_string(),
                read_strategy: "attachment_or_metadata".to_string(),
            };
        }

        if !is_text_like_mime(mime) {
            return ContentDetection {
                kind: ReadContentKind::Binary,
                mime: inferred_mime,
                encoding: None,
                reason: "infer matched a non-text file type".to_string(),
                read_strategy: "binary_summary".to_string(),
            };
        }
    }

    match content_inspector::inspect(sample) {
        ContentType::UTF_8 => text_detection(
            ReadContentKind::Utf8Text,
            inferred_mime.or_else(|| Some("text/plain".to_string())),
            Some("utf-8"),
            "content_inspector detected UTF-8 text",
        ),
        ContentType::UTF_8_BOM => text_detection(
            ReadContentKind::Utf8Text,
            inferred_mime.or_else(|| Some("text/plain".to_string())),
            Some("utf-8"),
            "content_inspector detected UTF-8 text with BOM",
        ),
        ContentType::UTF_16LE => text_detection(
            ReadContentKind::Utf16LeText,
            inferred_mime.or_else(|| Some("text/plain".to_string())),
            Some("utf-16le"),
            "content_inspector detected UTF-16LE text",
        ),
        ContentType::UTF_16BE => text_detection(
            ReadContentKind::Utf16BeText,
            inferred_mime.or_else(|| Some("text/plain".to_string())),
            Some("utf-16be"),
            "content_inspector detected UTF-16BE text",
        ),
        ContentType::UTF_32LE => text_detection(
            ReadContentKind::Unknown,
            inferred_mime.or_else(|| Some("text/plain".to_string())),
            Some("utf-32le"),
            "content_inspector detected UTF-32LE text; decoding is not wired yet",
        ),
        ContentType::UTF_32BE => text_detection(
            ReadContentKind::Unknown,
            inferred_mime.or_else(|| Some("text/plain".to_string())),
            Some("utf-32be"),
            "content_inspector detected UTF-32BE text; decoding is not wired yet",
        ),
        ContentType::BINARY => {
            if let Some(kind) = detect_utf16_by_null_pattern(sample) {
                let encoding = match kind {
                    ReadContentKind::Utf16LeText => "utf-16le",
                    ReadContentKind::Utf16BeText => "utf-16be",
                    _ => "utf-8",
                };

                return text_detection(
                    kind,
                    inferred_mime.or_else(|| Some("text/plain".to_string())),
                    Some(encoding),
                    "sample has UTF-16 null-byte pattern without BOM",
                );
            }

            ContentDetection {
                kind: ReadContentKind::Binary,
                mime: inferred_mime.or_else(|| Some("application/octet-stream".to_string())),
                encoding: None,
                reason: "content_inspector detected binary content".to_string(),
                read_strategy: "binary_summary".to_string(),
            }
        }
    }
}

pub fn decode_text_bytes(bytes: &[u8], detection: &ContentDetection) -> (String, String, bool) {
    let encoding = detection
        .encoding
        .as_deref()
        .and_then(|label| Encoding::for_label(label.as_bytes()))
        .unwrap_or_else(|| guess_legacy_encoding(bytes, true));

    let (text, used_encoding, had_errors) = encoding.decode(bytes);
    (
        text.into_owned(),
        used_encoding.name().to_string(),
        had_errors,
    )
}

fn text_detection(
    kind: ReadContentKind,
    mime: Option<String>,
    encoding: Option<&str>,
    reason: &str,
) -> ContentDetection {
    ContentDetection {
        kind,
        mime,
        encoding: encoding.map(str::to_string),
        reason: reason.to_string(),
        read_strategy: "text_lines".to_string(),
    }
}

fn is_text_like_mime(mime: &str) -> bool {
    mime.starts_with("text/")
        || matches!(
            mime,
            "application/json"
                | "application/javascript"
                | "application/xml"
                | "application/xhtml+xml"
                | "image/svg+xml"
        )
        || mime.ends_with("+json")
        || mime.ends_with("+xml")
}

fn detect_utf16_by_null_pattern(sample: &[u8]) -> Option<ReadContentKind> {
    let pairs = sample.len() / 2;

    if pairs < 8 {
        return None;
    }

    let even_nulls = sample.iter().step_by(2).filter(|byte| **byte == 0).count();
    let odd_nulls = sample
        .iter()
        .skip(1)
        .step_by(2)
        .filter(|byte| **byte == 0)
        .count();
    let high_threshold = (pairs * 6) / 10;
    let low_threshold = pairs / 5;

    if odd_nulls >= high_threshold && even_nulls <= low_threshold {
        return Some(ReadContentKind::Utf16LeText);
    }

    if even_nulls >= high_threshold && odd_nulls <= low_threshold {
        return Some(ReadContentKind::Utf16BeText);
    }

    None
}

fn guess_legacy_encoding(bytes: &[u8], last: bool) -> &'static Encoding {
    let mut detector = EncodingDetector::new(Iso2022JpDetection::Deny);
    detector.feed(bytes, last);
    detector.guess(None, Utf8Detection::Allow)
}

// entry point

pub fn read_path(options: ReadPathOptions) -> Result<ReadPathResult> {
    let path = resolve_existing_path(&options.path)?;
    let (_metadata, metadata_kind) = read_path_metadata_kind(&path)?;

    if metadata_kind == ReadContentKind::Directory {
        return read_directory_listing(&path, &options);
    }

    let sample = read_sample(&path, options.limits.sample_bytes)?;
    let detection = detect_content_from_sample(&path, &sample);

    match detection.kind {
        ReadContentKind::Empty => read_empty_result(&path, &options, detection),

        ReadContentKind::Utf8Text => read_utf8_text_result(&path, &options, detection),
        ReadContentKind::Utf16LeText | ReadContentKind::Utf16BeText => {
            read_text_result_buffered(&path, &options, detection)
        }

        ReadContentKind::Image
        | ReadContentKind::Pdf
        | ReadContentKind::Binary
        | ReadContentKind::Unknown => read_metadata_result(&path, &options, detection),

        ReadContentKind::Directory => read_directory_listing(&path, &options),
    }
}

// resolve path
// -> read metadata + high-level kind
// -> destructure tuple
// -> compare only ReadContentKind
// -> if file, sample bytes
// -> classify file content
// -> route to handler

fn read_directory_listing(path: &Path, options: &ReadPathOptions) -> Result<ReadPathResult> {
    let mut entries = Vec::new();

    for entry in path
        .read_dir()
        .with_context(|| format!("Failed to read directory: {}", path.display()))?
    {
        let entry = entry
            .with_context(|| format!("Failed to read directory entry in: {}", path.display()))?;

        let mut name = entry.file_name().to_string_lossy().into_owned();

        if entry
            .file_type()
            .with_context(|| format!("Failed to inspect entry: {}", entry.path().display()))?
            .is_dir()
        {
            name.push('/');
        }

        entries.push(name);
    }

    entries.sort();

    let offset = effective_offset(options.offset);
    let limit = effective_limit(options);
    let start = offset.saturating_sub(1);
    let total_entries = entries.len();

    let mut content = String::new();
    let mut selected_len = 0;
    for entry in entries.into_iter().skip(start).take(limit) {
        if selected_len > 0 {
            content.push('\n');
        }
        content.push_str(&entry);
        selected_len += 1;
    }
    let end_line = end_line_for(offset, selected_len);

    Ok(ReadPathResult {
        path: path.to_path_buf(),
        detection: ContentDetection {
            kind: ReadContentKind::Directory,
            mime: None,
            encoding: None,
            reason: "path is a directory".to_string(),
            read_strategy: "directory_listing".to_string(),
        },
        offset,
        end_line,
        lines_seen: selected_len,
        total_lines: Some(total_entries),
        truncated: start + selected_len < total_entries,
        bytes_read: content.len(),
        content,
    })
}

fn read_empty_result(
    path: &Path,
    options: &ReadPathOptions,
    detection: ContentDetection,
) -> Result<ReadPathResult> {
    Ok(ReadPathResult {
        path: path.to_path_buf(),
        detection,
        offset: effective_offset(options.offset),
        end_line: 0,
        lines_seen: 0,
        total_lines: Some(0),
        truncated: false,
        bytes_read: 0,
        content: String::new(),
    })
}

fn read_utf8_text_result(
    path: &Path,
    options: &ReadPathOptions,
    mut detection: ContentDetection,
) -> Result<ReadPathResult> {
    let offset = effective_offset(options.offset);
    let limit = effective_limit(options);
    let max_bytes = options.limits.max_bytes.max(1);
    let max_line_length = options.limits.max_line_length.max(1);
    let file = File::open(path)
        .with_context(|| format!("Failed to open text file: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    let mut current_line = 0usize;
    let mut lines_seen = 0usize;
    let mut bytes_read = 0usize;
    let mut output_bytes = 0usize;
    let mut output_truncated = false;
    let mut line_truncated = false;
    let mut has_more_lines = false;
    let mut reached_eof = true;
    let mut content = String::new();

    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .with_context(|| format!("Failed to read text file: {}", path.display()))?;
        if read == 0 {
            break;
        }

        bytes_read += read;
        current_line += 1;

        if current_line < offset {
            continue;
        }

        if lines_seen >= limit {
            output_truncated = true;
            has_more_lines = true;
            reached_eof = false;
            break;
        }

        let normalized = line.strip_suffix('\n').unwrap_or(&line);
        let normalized = normalized.strip_suffix('\r').unwrap_or(normalized);
        let (line_text, was_line_truncated) = truncate_line(normalized, max_line_length);
        line_truncated |= was_line_truncated;

        let next_size = line_text.len() + usize::from(!content.is_empty());
        if output_bytes + next_size > max_bytes {
            output_truncated = true;
            has_more_lines = true;
            reached_eof = false;
            break;
        }

        output_bytes += next_size;
        if !content.is_empty() {
            content.push('\n');
        }
        content.push_str(&line_text);
        lines_seen += 1;
    }

    detection.encoding = Some("UTF-8".to_string());

    Ok(ReadPathResult {
        path: path.to_path_buf(),
        detection,
        offset,
        end_line: end_line_for(offset, lines_seen),
        lines_seen,
        total_lines: if reached_eof {
            Some(current_line)
        } else {
            None
        },
        truncated: output_truncated || line_truncated || has_more_lines,
        bytes_read,
        content,
    })
}

fn read_text_result_buffered(
    path: &Path,
    options: &ReadPathOptions,
    mut detection: ContentDetection,
) -> Result<ReadPathResult> {
    let max_bytes = options.limits.max_bytes.max(1);
    let file = File::open(path)
        .with_context(|| format!("Failed to open text file: {}", path.display()))?;

    let mut bytes = Vec::new();
    let mut limited = file.take((max_bytes + 1) as u64);
    limited
        .read_to_end(&mut bytes)
        .with_context(|| format!("Failed to read text file: {}", path.display()))?;

    let byte_truncated = bytes.len() > max_bytes;
    if byte_truncated {
        bytes.truncate(max_bytes);
    }

    let bytes_read = bytes.len();
    let (text, used_encoding, had_errors) = decode_text_bytes(&bytes, &detection);

    detection.encoding = Some(used_encoding);
    if had_errors {
        detection.reason = format!(
            "{}; decoder replaced invalid byte sequences",
            detection.reason
        );
    }

    let offset = effective_offset(options.offset);
    let limit = effective_limit(options);
    let start = offset.saturating_sub(1);
    let total_lines = usize::from(!text.is_empty()) * text.split('\n').count();

    let mut output_bytes = 0usize;
    let mut output_truncated = false;
    let mut line_truncated = false;
    let mut content = String::new();
    let mut lines_seen = 0usize;

    if !text.is_empty() {
        for line in text
            .split('\n')
            .map(|line| line.strip_suffix('\r').unwrap_or(line))
            .skip(start)
            .take(limit)
        {
            let (line, was_line_truncated) = truncate_line(line, options.limits.max_line_length);
            line_truncated |= was_line_truncated;

            let next_size = line.len() + usize::from(!content.is_empty());
            if output_bytes + next_size > max_bytes {
                output_truncated = true;
                break;
            }

            output_bytes += next_size;
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str(&line);
            lines_seen += 1;
        }
    }

    let end_line = end_line_for(offset, lines_seen);
    let line_limit_truncated = start + lines_seen < total_lines;

    Ok(ReadPathResult {
        path: path.to_path_buf(),
        detection,
        offset,
        end_line,
        lines_seen,
        total_lines: if byte_truncated {
            None
        } else {
            Some(total_lines)
        },
        truncated: byte_truncated || output_truncated || line_truncated || line_limit_truncated,
        bytes_read,
        content,
    })
}

fn read_metadata_result(
    path: &Path,
    options: &ReadPathOptions,
    detection: ContentDetection,
) -> Result<ReadPathResult> {
    let metadata = path
        .metadata()
        .with_context(|| format!("Failed to inspect file: {}", path.display()))?;

    let content = format!(
        "kind: {:?}\nmime: {}\nencoding: {}\nreason: {}\nstrategy: {}\nsize_bytes: {}",
        detection.kind,
        detection.mime.as_deref().unwrap_or("unknown"),
        detection.encoding.as_deref().unwrap_or("none"),
        detection.reason,
        detection.read_strategy,
        metadata.len()
    );

    Ok(ReadPathResult {
        path: path.to_path_buf(),
        detection,
        offset: effective_offset(options.offset),
        end_line: 0,
        lines_seen: 0,
        total_lines: None,
        truncated: false,
        bytes_read: 0,
        content,
    })
}

fn truncate_line(line: &str, max_line_length: usize) -> (String, bool) {
    let max_line_length = max_line_length.max(1);
    if let Some((boundary, _)) = line.char_indices().nth(max_line_length) {
        let suffix = "... [line truncated]";
        let mut output = String::with_capacity(boundary + suffix.len());
        output.push_str(&line[..boundary]);
        output.push_str(suffix);
        return (output, true);
    }

    (line.to_string(), false)
}

fn end_line_for(offset: usize, lines_seen: usize) -> usize {
    if lines_seen == 0 {
        offset.saturating_sub(1)
    } else {
        offset + lines_seen - 1
    }
}

#[cfg(test)]
mod tests {
    use super::truncate_line;

    #[test]
    fn truncation_preserves_utf8_boundaries_without_intermediate_string() {
        let (value, truncated) = truncate_line("åβγ", 2);
        assert_eq!(value, "åβ... [line truncated]");
        assert!(truncated);

        let (value, truncated) = truncate_line("åβ", 2);
        assert_eq!(value, "åβ");
        assert!(!truncated);
    }
}
