from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def portable_path(path: str | Path, base_directory: str | Path) -> str:
    """Serialize a path relative to the file that will contain it when possible."""
    target = Path(path).expanduser().resolve()
    base = Path(base_directory).expanduser().resolve()
    try:
        return Path(os.path.relpath(target, base)).as_posix()
    except ValueError:
        # Different Windows drives cannot be represented with a relative path.
        return str(target)


def resolve_portable_path(value: str | Path, base_directory: str | Path) -> Path:
    """Resolve both new relative paths and legacy absolute paths."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_directory).expanduser().resolve() / path).resolve()


def portable_data(value: Any, base_directory: str | Path) -> Any:
    """Recursively replace Path objects and absolute path strings for JSON output."""
    if isinstance(value, Path):
        return portable_path(value, base_directory)
    if isinstance(value, str) and Path(value).expanduser().is_absolute():
        return portable_path(value, base_directory)
    if isinstance(value, tuple):
        return [portable_data(item, base_directory) for item in value]
    if isinstance(value, list):
        return [portable_data(item, base_directory) for item in value]
    if isinstance(value, dict):
        return {str(key): portable_data(item, base_directory) for key, item in value.items()}
    return value
