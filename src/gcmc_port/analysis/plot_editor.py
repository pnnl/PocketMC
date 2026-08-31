from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from . import plot_template
from .plotting import render_aggregate_pose_plots, render_result_plots
from .models import ANALYSIS_CACHE_VERSION
from .pose import POSE_HYDRATION_CACHE_VERSION, pose_hydration_fingerprint


STYLE_FILE = "plot_style.json"


@dataclass(frozen=True, slots=True)
class PlotTarget:
    key: str
    label: str
    description: str
    paths: tuple[Path, ...]
    run_directory: Path | None = None
    aggregate: bool = False


def _script_style(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "STYLE" for target in node.targets):
                value = ast.literal_eval(node.value)
                return dict(value) if isinstance(value, dict) else {}
    except (OSError, SyntaxError, ValueError, TypeError):
        return {}
    return {}


def load_plot_style(result_root: str | Path) -> dict[str, Any]:
    root = Path(result_root).expanduser().resolve()
    style = dict(plot_template.STYLE)
    style.update(_script_style(root / "plot_results.py"))
    override = root / STYLE_FILE
    if override.exists():
        try:
            payload = json.loads(override.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                style.update(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return style


def save_plot_style(result_root: str | Path, style: dict[str, Any]) -> Path:
    root = Path(result_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / STYLE_FILE
    serializable = {key: value for key, value in style.items() if not key.startswith("_")}
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return path


def _density_outputs(run: Path, *, pose: bool, three_d: bool) -> tuple[Path, ...]:
    if pose:
        paths: list[Path] = []
        for cluster in sorted((run / "poses").glob("cluster_*")):
            for frame in ("pocket-frame", "substrate-frame"):
                if not (cluster / frame / "density_maps.npz").exists():
                    continue
                names = [f"{frame}_3d.png"] if three_d else [f"{frame}_{plane}.png" for plane in ("xy", "xz", "yz")]
                paths.extend(cluster / "plots" / name for name in names)
        return tuple(paths)
    if not (run / "density" / "density_maps.npz").exists():
        return ()
    names = ["density_3d.png"] if three_d else [f"density_{plane}.png" for plane in ("xy", "xz", "yz")]
    return tuple(run / "plots" / name for name in names)


def discover_plot_targets(result_root: str | Path) -> list[PlotTarget]:
    root = Path(result_root).expanduser().resolve()
    targets: list[PlotTarget] = []
    runs = sorted(path for path in root.iterdir() if path.is_dir() and (path / "tables" / "summary.json").exists()) if root.exists() else []
    for run in runs:
        summary_paths = tuple(
            sorted(
                path for path in (run / "plots").glob("*.png")
                if not path.name.startswith("density_")
            )
        )
        if summary_paths:
            targets.append(PlotTarget(f"{run.name}:summary", f"{run.name}: time/lifetime/path plots", f"{len(summary_paths)} current PNG file(s)", summary_paths, run))
        pose_summary = tuple(sorted((run / "poses" / "plots").glob("*.png")))
        if pose_summary:
            targets.append(PlotTarget(f"{run.name}:pose-summary", f"{run.name}: pose summary plots", f"{len(pose_summary)} current PNG file(s)", pose_summary, run))
        for key, label, pose, three_d in (
            ("density-2d", "2D density heatmaps", False, False),
            ("density-3d", "3D density isosurfaces", False, True),
            ("pose-density-2d", "cluster 2D hydration heatmaps", True, False),
            ("pose-density-3d", "cluster 3D hydration isosurfaces", True, True),
        ):
            paths = _density_outputs(run, pose=pose, three_d=three_d)
            if paths:
                targets.append(PlotTarget(f"{run.name}:{key}", f"{run.name}: {label}", f"{len(paths)} plot(s); regenerated from saved NPZ data", paths, run))

    pose_root = root / "aggregate" / "pose-groups"
    if pose_root.exists():
        two_d: list[Path] = []
        three_d: list[Path] = []
        summary: list[Path] = []
        for path in pose_root.glob("**/*.mean_density.npz"):
            two_d.append(path.with_suffix(".png"))
            three_d.append(path.with_name(path.stem + "_3d.png"))
        two_d.extend(path.with_suffix(".png") for path in pose_root.glob("**/difference.*.npz"))
        summary.extend(pose_root.glob("*/system_cluster_populations.png"))
        if summary:
            targets.append(PlotTarget("aggregate:pose-summary", "Aggregate pose populations", f"{len(summary)} plot(s)", tuple(sorted(summary)), aggregate=True))
        if two_d:
            targets.append(PlotTarget("aggregate:pose-density-2d", "Aggregate 2D hydration comparisons", f"{len(two_d)} plot(s)", tuple(sorted(two_d)), aggregate=True))
        if three_d:
            targets.append(PlotTarget("aggregate:pose-density-3d", "Aggregate 3D hydration isosurfaces", f"{len(three_d)} plot(s)", tuple(sorted(three_d)), aggregate=True))
    return targets


def stale_pose_hydration_runs(result_root: str | Path) -> list[str]:
    """Identify pose-density grids that require the cavity-frame/PBC repair, not just a replot."""
    root = Path(result_root).expanduser().resolve()
    current_fingerprints: dict[str, str] = {}
    try:
        root_payload = json.loads((root / "analysis_manifest.json").read_text(encoding="utf-8"))
        config_value = root_payload.get("config")
        if config_value:
            config_path = Path(str(config_value)).expanduser()
            if not config_path.is_absolute():
                config_path = (root / config_path).resolve()
            from .config import load_analysis_config

            config = load_analysis_config(config_path)
            current_fingerprints = {
                dataset.run_id: pose_hydration_fingerprint(config, dataset)
                for dataset in config.datasets
                if dataset.kind == "md"
            }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        current_fingerprints = {}
    stale: list[str] = []
    for pose_root in sorted(root.glob("*/poses")):
        if not any(pose_root.glob("cluster_*/pocket-frame/density_maps.npz")):
            continue
        manifest = pose_root / "pose_manifest.json"
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            version = int(payload.get("pose_hydration_cache_version", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            payload = {}
            version = 0
        current = current_fingerprints.get(pose_root.parent.name)
        saved = str(payload.get("fingerprint", ""))
        if version < POSE_HYDRATION_CACHE_VERSION or (current is not None and current != saved):
            stale.append(pose_root.parent.name)
    return stale


def stale_analysis_runs(result_root: str | Path, *, kinds: set[str] | None = None) -> list[str]:
    """Identify generic case tables/grids produced with stale frame semantics."""
    root = Path(result_root).expanduser().resolve()
    current_fingerprints: dict[str, str] = {}
    root_manifest = root / "analysis_manifest.json"
    try:
        root_payload = json.loads(root_manifest.read_text(encoding="utf-8"))
        config_value = root_payload.get("config")
        if config_value:
            config_path = Path(str(config_value)).expanduser()
            if not config_path.is_absolute():
                config_path = (root / config_path).resolve()
            from .config import load_analysis_config
            from .models import config_for_dataset
            from .runner import _fingerprint

            config = load_analysis_config(config_path)
            for dataset in config.datasets:
                current_fingerprints[dataset.run_id] = _fingerprint(
                    config_for_dataset(config, dataset), dataset
                )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        # Version checking still protects older results when the original
        # config has intentionally been moved or removed.
        current_fingerprints = {}
    stale: list[str] = []
    runs = sorted(path for path in root.iterdir() if path.is_dir()) if root.exists() else []
    for run in runs:
        summary = run / "tables" / "summary.json"
        manifest = run / "analysis_manifest.json"
        if not summary.exists() or not manifest.exists():
            continue
        try:
            kind = str(json.loads(summary.read_text(encoding="utf-8")).get("kind", ""))
            if kind not in {"md", "pocketmc"} or (kinds is not None and kind not in kinds):
                continue
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            version = int(payload.get("analysis_cache_version", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            payload = {}
            version = 0
        current_fingerprint = current_fingerprints.get(run.name)
        saved_fingerprint = str(payload.get("fingerprint", ""))
        if (
            version != ANALYSIS_CACHE_VERSION
            or (current_fingerprint is not None and saved_fingerprint != current_fingerprint)
        ):
            stale.append(run.name)
    return stale


def stale_md_analysis_runs(result_root: str | Path) -> list[str]:
    """Backward-compatible MD-only view of stale generic case results."""
    return stale_analysis_runs(result_root, kinds={"md"})


def render_plot_targets(
    result_root: str | Path,
    style: dict[str, Any],
    targets: Iterable[PlotTarget],
) -> list[Path]:
    root = Path(result_root).expanduser().resolve()
    selected = list(targets)
    pose_density_selected = any(
        ":pose-density-" in target.key or target.key.startswith("aggregate:pose-density-")
        for target in selected
    )
    generic_case_selected = any(
        target.run_directory is not None
        and ":pose-" not in target.key
        for target in selected
    )
    stale_cases = stale_analysis_runs(root) if generic_case_selected else []
    if stale_cases:
        raise RuntimeError(
            "Saved case data predate the current coordinate/membership semantics for: "
            + ", ".join(stale_cases)
            + ". Plot-only regeneration cannot repair the saved tables or density grids. "
            "Run the same analysis config once without --force; pose-training and cluster caches remain reusable."
        )
    stale_runs = stale_pose_hydration_runs(root) if pose_density_selected else []
    if stale_runs:
        manifest = root / "analysis_manifest.json"
        config_hint = ""
        if manifest.exists():
            try:
                config = json.loads(manifest.read_text(encoding="utf-8")).get("config")
                if config:
                    config_hint = f' Rerun once with: python analyses.py run -c "{config}"'
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        raise RuntimeError(
            "Saved pose-hydration grids predate the cavity-frame/PBC repair for: "
            + ", ".join(stale_runs)
            + ". Plot-only regeneration cannot correct molecules selected in the wrong cavity frame or restore "
            "density values that fell outside the substrate grid."
            + config_hint
            + " Valid pose-training and cluster-model caches will be reused; --force is not required."
        )
    selected_paths = {str(path.resolve()) for target in selected for path in target.paths}
    scoped_style = dict(style)
    scoped_style["_selected_plot_paths"] = sorted(selected_paths)
    run_directories = sorted({target.run_directory for target in selected if target.run_directory is not None})
    for run_directory in run_directories:
        render_result_plots(run_directory, scoped_style)
    if any(target.aggregate for target in selected):
        render_aggregate_pose_plots(root, scoped_style)
    return sorted(path for target in selected for path in target.paths if path.exists())


def apply_style_assignments(style: dict[str, Any], assignments: Iterable[str]) -> dict[str, Any]:
    updated = dict(style)
    for assignment in assignments:
        key, separator, raw = assignment.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Plot setting must be KEY=JSON_VALUE: {assignment}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        key = key.strip()
        if key.startswith("axis_limits."):
            axis_key = key.split(".", 1)[1]
            limits = dict(updated.get("axis_limits", {}))
            limits[axis_key] = value
            updated["axis_limits"] = limits
        else:
            updated[key] = value
    return updated
