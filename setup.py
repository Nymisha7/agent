from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:  # pragma: no cover - build_py still works without wheel installed.
    _bdist_wheel = None


ROOT = Path(__file__).resolve().parent
RUST_MANIFEST = ROOT / "agent-rust" / "Cargo.toml"


class build_py(_build_py):
    """Build and bundle the Rust backend into Python wheels."""

    def run(self) -> None:
        self._build_rust_backend()
        super().run()

    def _build_rust_backend(self) -> None:
        if os.environ.get("AGENT_SKIP_RUST_BUILD") == "1":
            return
        if not RUST_MANIFEST.is_file():
            raise RuntimeError("Cannot build bundled Agent package: agent-rust/Cargo.toml is missing.")
        cargo = shutil.which("cargo")
        if cargo is None:
            raise RuntimeError(
                "Cannot build bundled Agent package: cargo is required to compile agent-rust. "
                "Set AGENT_SKIP_RUST_BUILD=1 only for source-only development builds."
            )
        subprocess.run(
            [cargo, "build", "--release", "--manifest-path", str(RUST_MANIFEST)],
            cwd=ROOT,
            check=True,
        )
        executable = "agent-rust.exe" if sys.platform.startswith("win") else "agent-rust"
        source = ROOT / "agent-rust" / "target" / "release" / executable
        if not source.is_file():
            raise RuntimeError(f"Rust build completed but backend binary was not found: {source}")
        target_dir = Path(self.build_lib) / "agent" / "bin"
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target_dir / executable)


cmdclass = {"build_py": build_py}

if _bdist_wheel is not None:

    class bdist_wheel(_bdist_wheel):
        """Mark wheels as platform-specific because they contain agent-rust."""

        def finalize_options(self) -> None:
            super().finalize_options()
            self.root_is_pure = False

        def get_tag(self) -> tuple[str, str, str]:
            # The package contains no CPython extension. Its bundled Rust binary is
            # Linux-platform-specific, but works with every supported Python 3.
            _python, _abi, platform = super().get_tag()
            return "py3", "none", platform

    cmdclass["bdist_wheel"] = bdist_wheel


setup(cmdclass=cmdclass)
