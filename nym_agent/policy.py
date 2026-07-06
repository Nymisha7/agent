from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    reason: str
    severity: str = "medium"
    message: str = ""
    redactions: tuple[str, ...] = ()


@dataclass(frozen=True)
class SecretFinding:
    path: str
    line: int
    kind: str
    name: str
    value_redacted: str
    confidence: str = "medium"


_SENSITIVE_KEY_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|SECRET_KEY|CLIENT_SECRET|TOKEN|PASSWORD|PRIVATE_KEY|ACCESS_KEY)[A-Z0-9_]*)\b"
    r"\s*[:=]\s*([\"']?)([^\"'\s;]+)\2"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._\-]{12,})\b")
_OPENAI_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{5,}\b")
_GITHUB_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")
_AWS_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_SLACK_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
_PEM_BLOCK_RE = re.compile(
    r"-----BEGIN [^-]+PRIVATE KEY-----.*?-----END [^-]+PRIVATE KEY-----",
    re.DOTALL,
)


class PolicyEngine:
    def redact_text(self, text: str) -> str:
        redacted = text
        redacted = _PEM_BLOCK_RE.sub("<redacted private key>", redacted)
        redacted = _SENSITIVE_KEY_RE.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
        redacted = _BEARER_RE.sub("Bearer <redacted>", redacted)
        redacted = _OPENAI_RE.sub("<redacted api key>", redacted)
        redacted = _GITHUB_RE.sub("<redacted token>", redacted)
        redacted = _AWS_RE.sub("<redacted access key>", redacted)
        redacted = _SLACK_RE.sub("<redacted slack token>", redacted)
        return redacted

    def sanitize_observation(self, observation: Any) -> Any:
        return self._sanitize_value(observation)

    def scan_text(self, text: str, *, path: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            findings.extend(self._scan_line(line, line_no=line_no, path=path))
        if _PEM_BLOCK_RE.search(text):
            findings.append(
                {
                    "path": path,
                    "line": 1,
                    "kind": "private_key",
                    "name": "PRIVATE KEY BLOCK",
                    "value_redacted": "<redacted private key>",
                    "confidence": "high",
                }
            )
        return findings

    def _sanitize_value(self, value: Any, *, key: str | None = None) -> Any:
        if isinstance(value, str):
            if key in {"path", "root", "file", "target", "requested_path", "resolved_path", "translated_path", "failed_path", "workspace_root"}:
                return value
            return self.redact_text(value)
        if isinstance(value, list):
            return [self._sanitize_value(item) for item in value]
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for item_key, item_value in value.items():
                sanitized[item_key] = self._sanitize_value(item_value, key=item_key)
            return sanitized
        return value

    def _scan_line(self, line: str, *, line_no: int, path: str) -> list[dict[str, Any]]:
        findings: list[dict[str, Any]] = []
        for match in _SENSITIVE_KEY_RE.finditer(line):
            key = match.group(1)
            value = match.group(3)
            findings.append(
                {
                    "path": path,
                    "line": line_no,
                    "kind": _classify_secret_key(key),
                    "name": key,
                    "value_redacted": self.redact_text(value),
                    "confidence": "high",
                }
            )
        for regex, kind, name in (
            (_BEARER_RE, "bearer_token", "Bearer token"),
            (_OPENAI_RE, "api_key", "OpenAI API key"),
            (_GITHUB_RE, "token", "GitHub token"),
            (_AWS_RE, "access_key", "AWS access key"),
            (_SLACK_RE, "token", "Slack token"),
        ):
            for match in regex.finditer(line):
                findings.append(
                    {
                        "path": path,
                        "line": line_no,
                        "kind": kind,
                        "name": name,
                        "value_redacted": self.redact_text(match.group(0)),
                        "confidence": "high",
                    }
                )
        return findings


def _classify_secret_key(key: str) -> str:
    upper = key.upper()
    if "PRIVATE_KEY" in upper:
        return "private_key"
    if "PASSWORD" in upper:
        return "password"
    if "SECRET" in upper:
        return "secret"
    if "TOKEN" in upper:
        return "token"
    if "API_KEY" in upper:
        return "api_key"
    if "ACCESS_KEY" in upper:
        return "access_key"
    return "secret"
