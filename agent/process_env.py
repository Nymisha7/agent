"""Environment handling for child processes that do not need Agent credentials."""
from __future__ import annotations

import os


_EXACT_SECRET_NAMES = frozenset({
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
})
_SECRET_SUFFIXES = (
    "_API_KEY",
    "_TOKEN",
    "_SECRET",
    "_PASSWORD",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PRIVATE_KEY",
    "_ENCRYPTION_KEY",
)


def credential_free_environment() -> dict[str, str]:
    """Return the current environment without credentials for helper programs."""
    environment = os.environ.copy()
    for name in tuple(environment):
        normalized = name.upper()
        if normalized in _EXACT_SECRET_NAMES or normalized.endswith(_SECRET_SUFFIXES):
            environment.pop(name, None)
    return environment
