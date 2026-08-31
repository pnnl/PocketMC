from __future__ import annotations

from pathlib import Path
import shutil


PYTHON_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = PYTHON_ROOT / "assets"
DEFAULTS_DIR = ASSETS_ROOT / "defaults"
EXAMPLE_DIR = ASSETS_ROOT / "example"

DEFAULT_ITEMS = (
    "amber14sb_parmbsc1.ff",
    "residuetypes.dat",
    "WAT.itp",
    "co",
    "chk.mdp",
    "steep.mdp",
    "em.mdp",
    "COM.gro",
)
EXAMPLE_ITEMS = ("topol.top", "init.gro", "config.example.toml")


def default_asset_path(name: str) -> Path:
    return (DEFAULTS_DIR / name).resolve()


def example_asset_path(name: str) -> Path:
    return (EXAMPLE_DIR / name).resolve()


def resolve_project_input(
    project_root: Path,
    value: str,
    *,
    default_name: str | None = None,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    candidate = (project_root / path).resolve()
    if candidate.exists():
        return candidate

    bundled_candidate = (DEFAULTS_DIR / path).resolve()
    if bundled_candidate.exists():
        return bundled_candidate

    if default_name and path.as_posix() == default_name:
        fallback = default_asset_path(default_name)
        if fallback.exists():
            return fallback
    return candidate


def export_default_assets(target_dir: str | Path, *, force: bool = False) -> Path:
    target = Path(target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    _copy_items(DEFAULTS_DIR, DEFAULT_ITEMS, target, force=force)
    return target


def export_example_case(target_dir: str | Path, *, force: bool = False) -> Path:
    target = Path(target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    _copy_items(EXAMPLE_DIR, EXAMPLE_ITEMS, target, force=force)
    return target


def _copy_items(source_root: Path, names: tuple[str, ...], target_dir: Path, *, force: bool) -> None:
    conflicts = [target_dir / name for name in names if (target_dir / name).exists()]
    if conflicts and not force:
        joined = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"Refusing to overwrite existing paths without --force: {joined}")

    for name in names:
        source = source_root / name
        destination = target_dir / name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=force)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
