from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .session_store import data_home


@dataclass(frozen=True)
class DeviceIdentity:
    device_id: str
    public_key_pem: str
    private_key_pem: str


def default_identity_path() -> Path:
    return data_home() / "agent" / "identity" / "device.json"


def load_or_create_device_identity(file_path: Path | None = None) -> DeviceIdentity:
    path = file_path or default_identity_path()
    try:
        parsed = _read_json_if_exists(path)
        normalized = _normalize_stored_identity(parsed)
        if normalized is not None:
            identity, stored, valid_for_read_only = normalized
            if stored is not None and not valid_for_read_only:
                try:
                    _write_private_json(path, stored)
                except OSError:
                    pass
            return identity
        if path.is_file():
            return _generate_identity()
    except Exception:
        if path.is_file():
            return _generate_identity()

    identity = _generate_identity()
    _write_private_json(path, _stored_identity(identity, created_at_ms=_now_ms()))
    return identity


def public_key_raw_base64url_from_pem(public_key_pem: str) -> str:
    return _base64url(_public_key_raw(public_key_pem))


def _normalize_stored_identity(parsed: Any) -> tuple[DeviceIdentity, dict[str, Any] | None, bool] | None:
    if not isinstance(parsed, dict):
        return None
    if (
        parsed.get("version") == 1
        and isinstance(parsed.get("deviceId"), str)
        and isinstance(parsed.get("publicKeyPem"), str)
        and isinstance(parsed.get("privateKeyPem"), str)
    ):
        public_key_pem = parsed["publicKeyPem"]
        private_key_pem = parsed["privateKeyPem"]
        if not _key_pair_matches(public_key_pem, private_key_pem):
            return None
        device_id = _fingerprint_public_key(public_key_pem)
        identity = DeviceIdentity(
            device_id=device_id,
            public_key_pem=public_key_pem,
            private_key_pem=private_key_pem,
        )
        if device_id == parsed["deviceId"]:
            return identity, None, True
        stored = dict(parsed)
        stored["deviceId"] = device_id
        return identity, stored, False
    if (
        "version" not in parsed
        and isinstance(parsed.get("deviceId"), str)
        and isinstance(parsed.get("publicKey"), str)
        and isinstance(parsed.get("privateKey"), str)
    ):
        try:
            public_raw = _base64url_decode(parsed["publicKey"])
            private_raw = _base64url_decode(parsed["privateKey"])
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_raw)
            private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_raw)
        except Exception:
            return None
        public_key_pem = _public_key_pem(public_key)
        private_key_pem = _private_key_pem(private_key)
        if not _key_pair_matches(public_key_pem, private_key_pem):
            return None
        device_id = _fingerprint_public_key(public_key_pem)
        identity = DeviceIdentity(
            device_id=device_id,
            public_key_pem=public_key_pem,
            private_key_pem=private_key_pem,
        )
        stored = _stored_identity(
            identity,
            created_at_ms=parsed.get("createdAtMs") if isinstance(parsed.get("createdAtMs"), int) else _now_ms(),
        )
        return identity, stored, device_id == parsed["deviceId"]
    return None


def _generate_identity() -> DeviceIdentity:
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_key_pem = _public_key_pem(public_key)
    private_key_pem = _private_key_pem(private_key)
    return DeviceIdentity(
        device_id=_fingerprint_public_key(public_key_pem),
        public_key_pem=public_key_pem,
        private_key_pem=private_key_pem,
    )


def _stored_identity(identity: DeviceIdentity, *, created_at_ms: int) -> dict[str, Any]:
    return {
        "version": 1,
        "deviceId": identity.device_id,
        "publicKeyPem": identity.public_key_pem,
        "privateKeyPem": identity.private_key_pem,
        "createdAtMs": created_at_ms,
    }


def _fingerprint_public_key(public_key_pem: str) -> str:
    return hashlib.sha256(_public_key_raw(public_key_pem)).hexdigest()


def _key_pair_matches(public_key_pem: str, private_key_pem: str) -> bool:
    try:
        public_raw = _public_key_raw(public_key_pem)
        private_key = _load_private_key(private_key_pem)
        derived_raw = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return public_raw == derived_raw
    except Exception:
        return False


def _public_key_raw(public_key_pem: str) -> bytes:
    public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    if not isinstance(public_key, ed25519.Ed25519PublicKey):
        raise ValueError("device identity public key must be Ed25519")
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _load_private_key(private_key_pem: str) -> ed25519.Ed25519PrivateKey:
    private_key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    if not isinstance(private_key, ed25519.Ed25519PrivateKey):
        raise ValueError("device identity private key must be Ed25519")
    return private_key


def _public_key_pem(public_key: ed25519.Ed25519PublicKey) -> str:
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def _private_key_pem(private_key: ed25519.Ed25519PrivateKey) -> str:
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _read_json_if_exists(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
    finally:
        if not fd == -1:
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass


def _now_ms() -> int:
    return int(time.time() * 1000)
