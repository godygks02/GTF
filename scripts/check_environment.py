"""Perform a lightweight check of the local GTF research environment."""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

REQUIRED_MODULES = ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "yaml", "gtf")


def main() -> int:
    """Print environment information and return nonzero when a requirement is missing."""
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Project: {Path.cwd()}")

    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
            print(f"[missing] {module_name}")
        else:
            version = getattr(module, "__version__", "version not exposed")
            print(f"[ok] {module_name}: {version}")

    if missing:
        print("Environment check failed. Missing: " + ", ".join(missing))
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
