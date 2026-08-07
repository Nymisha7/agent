from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
DEFAULT_ATTACHMENT_RETENTION_DAYS = 30
DEFAULT_ATTACHMENT_STORE_MAX_BYTES = 1024 * 1024 * 1024
DEFAULT_ATTACHMENT_MAINTENANCE_INTERVAL_SECONDS = 24 * 60 * 60
COPY_BUFFER_BYTES = 1024 * 1024
_MAINTENANCE_MARKER = ".last-maintenance"


@dataclass(frozen=True)
class Attachment:
    id: str
    filename: str
    mime: str
    size_bytes: int
    sha256: str
    storage_path: str
    source: str

    def to_store_input(self) -> dict[str, object]:
        return {
            "id": self.id,
            "filename": self.filename,
            "mime": self.mime,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "storage_path": self.storage_path,
            "source": self.source,
        }


def attachment_from_store(value: object) -> Attachment | None:
    """Rehydrate previously queued metadata without reading attachment bytes."""
    if not isinstance(value, dict):
        return None
    fields = {
        key: value.get(key)
        for key in ("id", "filename", "mime", "sha256", "storage_path", "source")
    }
    size_bytes = value.get("size_bytes")
    if not all(isinstance(item, str) and item for item in fields.values()):
        return None
    if not isinstance(size_bytes, int) or size_bytes < 0:
        return None
    storage_path = Path(str(fields["storage_path"]))
    if not storage_path.is_file():
        return None
    return Attachment(
        id=str(fields["id"]),
        filename=str(fields["filename"]),
        mime=str(fields["mime"]),
        size_bytes=size_bytes,
        sha256=str(fields["sha256"]),
        storage_path=str(storage_path),
        source=str(fields["source"]),
    )


def attachment_data_home() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base.expanduser() / "agent" / "attachments"


def max_attachment_bytes() -> int:
    raw = os.environ.get("AGENT_MAX_ATTACHMENT_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_ATTACHMENT_BYTES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("AGENT_MAX_ATTACHMENT_BYTES must be a positive integer.") from exc
    if value <= 0:
        raise ValueError("AGENT_MAX_ATTACHMENT_BYTES must be a positive integer.")
    return value


def attachment_retention_days() -> int:
    return _positive_env_int(
        "AGENT_ATTACHMENT_RETENTION_DAYS",
        DEFAULT_ATTACHMENT_RETENTION_DAYS,
    )


def attachment_store_max_bytes() -> int:
    return _positive_env_int(
        "AGENT_ATTACHMENT_STORE_MAX_BYTES",
        DEFAULT_ATTACHMENT_STORE_MAX_BYTES,
    )


def maintain_attachment_store(*, force: bool = False, now: float | None = None) -> dict[str, int]:
    """Apply age and quota retention to the private attachment cache."""
    root = attachment_data_home()
    current_time = time.time() if now is None else now
    marker = root / _MAINTENANCE_MARKER
    if not force:
        try:
            if current_time - marker.stat().st_mtime < DEFAULT_ATTACHMENT_MAINTENANCE_INTERVAL_SECONDS:
                return {"removed": 0, "bytes_removed": 0, "bytes_kept": 0}
        except OSError:
            pass

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _set_private_permissions(root, 0o700)
    cutoff = current_time - attachment_retention_days() * 24 * 60 * 60
    entries: list[tuple[float, int, Path]] = []
    removed = 0
    bytes_removed = 0
    for directory in root.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        for candidate in directory.iterdir():
            try:
                metadata = candidate.stat(follow_symlinks=False)
            except OSError:
                continue
            if not candidate.is_file() or candidate.is_symlink():
                continue
            if metadata.st_mtime < cutoff:
                try:
                    candidate.unlink()
                except OSError:
                    continue
                removed += 1
                bytes_removed += metadata.st_size
            else:
                entries.append((metadata.st_mtime, metadata.st_size, candidate))

    kept_bytes = sum(size for _mtime, size, _path in entries)
    maximum = attachment_store_max_bytes()
    if kept_bytes > maximum:
        entries.sort(key=lambda entry: entry[0])
        for _mtime, size, candidate in entries:
            if kept_bytes <= maximum:
                break
            try:
                candidate.unlink()
            except OSError:
                continue
            removed += 1
            bytes_removed += size
            kept_bytes -= size

    for directory in root.iterdir():
        if directory.is_dir() and not directory.is_symlink():
            try:
                directory.rmdir()
            except OSError:
                pass
    try:
        marker.touch(exist_ok=True)
        _set_private_permissions(marker, 0o600)
        os.utime(marker, (current_time, current_time))
    except OSError:
        pass
    return {"removed": removed, "bytes_removed": bytes_removed, "bytes_kept": kept_bytes}


def import_attachment(path: Path | str, *, source: str) -> Attachment:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Attachment cannot be resolved: {candidate}") from exc
    if not resolved.is_file():
        raise ValueError(f"Attachment must be a regular file: {candidate}")

    limit = max_attachment_bytes()
    size = resolved.stat().st_size
    if size > limit:
        raise ValueError(f"Attachment exceeds configured limit of {limit} bytes: {resolved.name}")
    store_limit = attachment_store_max_bytes()
    if size > store_limit:
        raise ValueError(
            f"Attachment exceeds configured store quota of {store_limit} bytes: {resolved.name}"
        )
    digest = _sha256(resolved)
    mime = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
    maintain_attachment_store()
    destination = attachment_data_home() / digest[:2] / digest
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _set_private_permissions(attachment_data_home(), 0o700)
    _set_private_permissions(destination.parent, 0o700)
    if not destination.exists():
        temporary = destination.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
        try:
            with resolved.open("rb") as source_file, temporary.open("xb") as destination_file:
                shutil.copyfileobj(source_file, destination_file, COPY_BUFFER_BYTES)
            os.chmod(temporary, 0o600)
            os.replace(temporary, destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    else:
        _set_private_permissions(destination, 0o600)
        try:
            os.utime(destination)
        except OSError:
            pass
    maintain_attachment_store(force=True)
    if not destination.is_file():
        raise RuntimeError("Attachment could not be retained within the configured store quota.")
    return Attachment(
        id=uuid.uuid4().hex,
        filename=resolved.name,
        mime=mime,
        size_bytes=size,
        sha256=digest,
        storage_path=str(destination),
        source=source,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _set_private_permissions(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass
