from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..pathing import portable_data, portable_path, resolve_portable_path
from .cache import cache_directory, read_analysis_cache_metadata

from .density import _write_cube, _write_projection_csv
from .geometry import apply_transform, kabsch_transform
from .plot_editor import load_plot_style
from .plotting import (
    _apply_global_style,
    _density_isosurfaces,
    _draw_substrate_2d,
    _format_axis,
    _overlay,
    _plt,
    _save,
)


GRAND_ALIGNMENT_SCHEMA_VERSION = 5
GRAND_PLOT_SCRIPT = "plot_grand_aligned.py"
GRAND_CAVITY_OVERLAY = "cavity_overlay.npz"
_GRAND_SOURCE_BEGIN = "# __GCMC_PORT_GENERATED_SOURCE_BEGIN__"
_GRAND_SOURCE_END = "# __GCMC_PORT_GENERATED_SOURCE_END__"


@dataclass(frozen=True, slots=True)
class SavedDensityMap:
    path: Path
    overlay: Path
    kind: str


@dataclass(frozen=True, slots=True)
class CompletedAnalysisRoot:
    root: Path
    manifest: Path
    status: str
    completed_runs: tuple[str, ...]
    maps: tuple[SavedDensityMap, ...]


@dataclass(frozen=True, slots=True)
class AlignmentPlan:
    analysis: CompletedAnalysisRoot
    density_map: SavedDensityMap
    transform: tuple[np.ndarray, np.ndarray, np.ndarray]
    matched_atoms: tuple[str, ...]
    fit_rmsd_a: float
    source_axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    source_bin_a: float


@dataclass(frozen=True, slots=True)
class GrandAlignmentResult:
    output_root: Path
    manifest: Path
    aligned_maps: tuple[Path, ...]
    plots: tuple[Path, ...]
    skipped_maps: tuple[dict[str, str], ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExistingGrandAlignment:
    root: Path
    manifest: Path
    schema_version: int
    status: str
    aligned_map_count: int
    stale_reasons: tuple[str, ...]

    @property
    def stale(self) -> bool:
        return bool(self.stale_reasons)


def _map_kind(path: Path) -> str:
    if path.name.startswith("difference."):
        return "aggregate-difference"
    if path.name.endswith(".mean_density.npz"):
        return "aggregate-mean"
    if path.parent.name == "pocket-frame":
        return "pose-pocket"
    if path.parent.name == "substrate-frame":
        return "pose-substrate"
    return "generic"


def discover_saved_density_maps(root: str | Path) -> list[SavedDensityMap]:
    analysis_root = Path(root).expanduser().resolve()
    candidates: set[Path] = set()
    candidates.update(analysis_root.glob("*/density/density_maps.npz"))
    candidates.update(analysis_root.glob("*/poses/cluster_*/pocket-frame/density_maps.npz"))
    candidates.update(analysis_root.glob("*/poses/cluster_*/substrate-frame/density_maps.npz"))
    candidates.update(analysis_root.glob("aggregate/pose-groups/**/*.mean_density.npz"))
    candidates.update(analysis_root.glob("aggregate/pose-groups/**/difference.*.npz"))
    output: list[SavedDensityMap] = []
    for path in sorted(candidates):
        overlay = path.parent / "substrate_overlay.npz"
        if path.is_file() and overlay.is_file():
            output.append(SavedDensityMap(path.resolve(), overlay.resolve(), _map_kind(path)))
    return output


def _completed_analysis_from_manifest(manifest: Path) -> CompletedAnalysisRoot | None:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    completed = payload.get("completed_runs")
    # Per-run manifests share the filename but do not contain completed_runs.
    if not isinstance(completed, list) or not completed:
        return None
    status = str(payload.get("status", ""))
    if status not in {"complete", "partial"}:
        return None
    maps = tuple(discover_saved_density_maps(manifest.parent))
    if not maps:
        return None
    return CompletedAnalysisRoot(
        root=manifest.parent.resolve(),
        manifest=manifest.resolve(),
        status=status,
        completed_runs=tuple(str(item) for item in completed),
        maps=maps,
    )


def discover_completed_analysis_roots(
    scan_root: str | Path,
    *,
    max_depth: int = 5,
) -> list[CompletedAnalysisRoot]:
    root = Path(scan_root).expanduser().resolve()
    if not root.exists():
        return []
    manifests: list[Path] = []
    for directory, child_names, file_names in os.walk(root):
        current = Path(directory)
        depth = len(current.relative_to(root).parts)
        child_names[:] = [
            name
            for name in child_names
            if depth < max_depth
            and name not in {".git", "__pycache__"}
            and not name.lower().startswith("grand-aligned")
        ]
        if "analysis_manifest.json" in file_names:
            manifests.append(current / "analysis_manifest.json")
    found: dict[Path, CompletedAnalysisRoot] = {}
    for manifest in manifests:
        analysis = _completed_analysis_from_manifest(manifest)
        if analysis is not None:
            found[analysis.root] = analysis
    return sorted(found.values(), key=lambda item: str(item.root).lower())


def load_completed_analysis_root(root: str | Path) -> CompletedAnalysisRoot:
    path = Path(root).expanduser().resolve()
    manifest = path / "analysis_manifest.json"
    analysis = _completed_analysis_from_manifest(manifest)
    if analysis is None:
        raise ValueError(f"No completed analysis with saved density/substrate NPZ data was found at {path}")
    return analysis


def _grand_manifest_info(manifest: Path) -> ExistingGrandAlignment | None:
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        version = int(payload.get("schema_version", 0))
        records = payload.get("aligned_maps", [])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(records, list) or not records:
        return None
    reasons: list[str] = []
    if version < GRAND_ALIGNMENT_SCHEMA_VERSION:
        reasons.append(
            f"manifest schema {version} predates the current display/replot schema "
            f"{GRAND_ALIGNMENT_SCHEMA_VERSION}"
        )
    if not (manifest.parent / GRAND_PLOT_SCRIPT).is_file():
        reasons.append(f"missing editable {GRAND_PLOT_SCRIPT}")
    if not payload.get("display_bounds_A"):
        reasons.append("missing density-content display bounds")
    if version >= GRAND_ALIGNMENT_SCHEMA_VERSION and not payload.get("cavity_overlay_summary"):
        reasons.append("missing cavity-boundary repair summary")
    return ExistingGrandAlignment(
        root=manifest.parent.resolve(),
        manifest=manifest.resolve(),
        schema_version=version,
        status=str(payload.get("status", "unknown")),
        aligned_map_count=len(records),
        stale_reasons=tuple(reasons),
    )


def discover_grand_alignment_outputs(
    scan_root: str | Path,
    *,
    max_depth: int = 3,
) -> list[ExistingGrandAlignment]:
    root = Path(scan_root).expanduser().resolve()
    if not root.exists():
        return []
    manifests: list[Path] = []
    for directory, child_names, file_names in os.walk(root):
        current = Path(directory)
        depth = len(current.relative_to(root).parts)
        child_names[:] = [
            name
            for name in child_names
            if depth < max_depth and name not in {".git", "__pycache__"}
        ]
        if "grand_alignment_manifest.json" in file_names:
            manifests.append(current / "grand_alignment_manifest.json")
    found = [item for manifest in manifests if (item := _grand_manifest_info(manifest)) is not None]
    return sorted(found, key=lambda item: item.manifest.stat().st_mtime_ns, reverse=True)


def _load_overlay(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    positions = np.asarray(payload.get("positions_A", []), dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError(f"Invalid substrate positions in {path}")
    for field in ("atom_names", "resnames"):
        if field not in payload or len(payload[field]) != len(positions):
            raise ValueError(f"Missing or inconsistent {field} in {path}")
    return payload


def substrate_coverage(analyses: Iterable[CompletedAnalysisRoot]) -> dict[str, int]:
    roots = list(analyses)
    coverage: dict[str, int] = {}
    for analysis in roots:
        present: set[str] = set()
        for density_map in analysis.maps:
            try:
                overlay = _load_overlay(density_map.overlay)
            except (OSError, ValueError):
                continue
            present.update(str(value).strip().upper() for value in overlay["resnames"] if str(value).strip())
        for resname in present:
            coverage[resname] = coverage.get(resname, 0) + 1
    return dict(sorted(coverage.items()))


def common_substrates(analyses: Iterable[CompletedAnalysisRoot]) -> list[str]:
    roots = list(analyses)
    coverage = substrate_coverage(roots)
    common = [name for name, count in coverage.items() if count == len(roots)]
    return sorted(common, key=lambda name: (name != "OPP", name))


def default_fixed_substrates(analyses: Iterable[CompletedAnalysisRoot]) -> tuple[str, ...]:
    common = common_substrates(analyses)
    if not common:
        raise ValueError("The selected analyses have no common substrate residue name in their saved overlays")
    if "OPP" in common:
        return ("OPP",)
    starts_opp = [name for name in common if name.startswith("OPP")]
    return (starts_opp[0] if starts_opp else common[0],)


def _selected_heavy_atoms(
    overlay: dict[str, np.ndarray],
    substrates: tuple[str, ...],
    *,
    context: Path,
) -> dict[tuple[str, str], np.ndarray]:
    positions = np.asarray(overlay["positions_A"], dtype=float)
    names = np.asarray(overlay["atom_names"], dtype=str)
    resnames = np.char.upper(np.asarray(overlay["resnames"], dtype=str))
    elements = np.char.upper(
        np.asarray(overlay.get("elements", np.asarray([""] * len(positions))), dtype=str)
    )
    resids = np.asarray(overlay.get("resids", np.zeros(len(positions), dtype=int)), dtype=int)
    output: dict[tuple[str, str], np.ndarray] = {}
    for substrate in substrates:
        selected = np.flatnonzero(resnames == substrate)
        if selected.size == 0:
            raise ValueError(f"{context}: substrate {substrate} is absent")
        distinct_resids = sorted({int(resids[index]) for index in selected})
        if len(distinct_resids) > 1:
            raise ValueError(
                f"{context}: substrate {substrate} occurs as multiple residues {distinct_resids}; "
                "a unique fixed substrate is required"
            )
        seen_names: set[str] = set()
        for index in selected:
            atom_name = str(names[index]).strip().upper()
            element = str(elements[index]).strip().upper()
            if element == "H" or atom_name.startswith("H"):
                continue
            if atom_name in seen_names:
                raise ValueError(f"{context}: duplicate atom name {substrate}:{atom_name}")
            seen_names.add(atom_name)
            output[(substrate, atom_name)] = positions[index]
    return output


def _alignment_transform(
    mobile_overlay: dict[str, np.ndarray],
    target_overlay: dict[str, np.ndarray],
    substrates: tuple[str, ...],
    *,
    mobile_context: Path,
    target_context: Path,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[str, ...], float]:
    mobile = _selected_heavy_atoms(mobile_overlay, substrates, context=mobile_context)
    target = _selected_heavy_atoms(target_overlay, substrates, context=target_context)
    keys = tuple(sorted(set(mobile) & set(target)))
    if len(keys) < 3:
        raise ValueError(
            f"{mobile_context}: only {len(keys)} matching heavy atom(s) were found for "
            + ", ".join(substrates)
        )
    mobile_positions = np.asarray([mobile[key] for key in keys], dtype=float)
    target_positions = np.asarray([target[key] for key in keys], dtype=float)
    if np.linalg.matrix_rank(mobile_positions - mobile_positions.mean(axis=0)) < 2:
        raise ValueError(f"{mobile_context}: fixed-substrate matching atoms are collinear")
    transform = kabsch_transform(mobile_positions, target_positions)
    fitted = apply_transform(mobile_positions, transform)
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target_positions) ** 2, axis=1))))
    labels = tuple(f"{resname}:{name}" for resname, name in keys)
    return transform, labels, rmsd


def _axes_and_bin(path: Path) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], float]:
    with np.load(path) as data:
        axes = tuple(np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz")
        if any(axis.ndim != 1 or axis.size == 0 for axis in axes):
            raise ValueError(f"Invalid density axes in {path}")
        if "bin_A" in data.files:
            bin_a = float(data["bin_A"])
        else:
            steps = [float(np.median(np.diff(axis))) for axis in axes if axis.size > 1]
            bin_a = float(np.median(steps)) if steps else 1.0
    if not math.isfinite(bin_a) or bin_a <= 0.0:
        raise ValueError(f"Invalid density bin spacing in {path}: {bin_a}")
    return axes, bin_a


def _reference_map(analysis: CompletedAnalysisRoot, substrates: tuple[str, ...]) -> SavedDensityMap:
    priority = {
        "generic": 0,
        "pose-pocket": 1,
        "aggregate-mean": 2,
        "pose-substrate": 3,
        "aggregate-difference": 4,
    }
    failures: list[str] = []
    for density_map in sorted(analysis.maps, key=lambda item: (priority.get(item.kind, 99), str(item.path))):
        try:
            overlay = _load_overlay(density_map.overlay)
            _selected_heavy_atoms(overlay, substrates, context=density_map.overlay)
            return density_map
        except (OSError, ValueError) as exc:
            failures.append(str(exc))
    details = failures[0] if failures else "no substrate overlays were found"
    raise ValueError(f"Reference analysis {analysis.root} cannot use {', '.join(substrates)}: {details}")


def _mask_spacing_a(points_a: np.ndarray, configured_meta: Any) -> float:
    if configured_meta:
        try:
            meta_path = Path(str(configured_meta)).expanduser()
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            spacing = float(payload.get("dx", 0.0)) * 10.0
            if math.isfinite(spacing) and spacing > 0.0:
                return spacing
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    # A rigidly rotated voxel lattice still has the original nearest-neighbor
    # distance, so this fallback also works for trajectory-carried masks.
    if len(points_a) > 1:
        try:
            from scipy.spatial import cKDTree

            distances, _indices = cKDTree(points_a).query(points_a, k=2)
            nearest = np.asarray(distances, dtype=float)[:, 1]
            nearest = nearest[np.isfinite(nearest) & (nearest > 1.0e-8)]
            if nearest.size:
                return float(np.median(nearest))
        except (ValueError, TypeError):
            pass
    return 1.0


def _mask_boundary_points(points_a: np.ndarray, spacing_a: float) -> np.ndarray:
    if len(points_a) < 7:
        return points_a.copy()
    try:
        from scipy.spatial import cKDTree

        counts = np.asarray(
            cKDTree(points_a).query_ball_point(
                points_a,
                r=max(spacing_a * 1.10, 1.0e-6),
                return_length=True,
            ),
            dtype=int,
        )
        boundary = points_a[counts < 7]
        return boundary if boundary.size else points_a.copy()
    except (TypeError, ValueError):
        return points_a.copy()


def _normalized_cavity_descriptor(
    raw: dict[str, Any],
    cavity: dict[str, Any],
    run_id: str,
) -> dict[str, np.ndarray] | None:
    mode = str(np.asarray(raw.get("mode", cavity.get("mode", ""))).item()).strip().lower()
    if mode == "mask" or (not mode and np.asarray(raw.get("points_A", [])).size):
        points_a = np.asarray(raw.get("points_A", []), dtype=float).reshape((-1, 3))
        if not points_a.size:
            return None
        spacing_a = float(raw.get("voxel_spacing_A", 0.0))
        if not math.isfinite(spacing_a) or spacing_a <= 0.0:
            spacing_a = _mask_spacing_a(points_a, cavity.get("meta"))
        boundary = np.asarray(raw.get("boundary_points_A", []), dtype=float).reshape((-1, 3))
        return {
            "mode": np.asarray("mask"),
            "points_A": points_a,
            "boundary_points_A": boundary if boundary.size else _mask_boundary_points(points_a, spacing_a),
            "voxel_spacing_A": np.asarray(spacing_a),
            "source_run": np.asarray(run_id),
        }
    center = np.asarray(raw.get("center_A", []), dtype=float).reshape((-1,))
    radius_a = float(raw.get("radius_A", float(cavity.get("radius_nm", 0.60)) * 10.0))
    if center.size != 3 or not math.isfinite(radius_a) or radius_a <= 0.0:
        return None
    return {
        "mode": np.asarray("sphere"),
        "center_A": center,
        "radius_A": np.asarray(radius_a),
        "source_run": np.asarray(run_id),
    }


def _vmd_mask_points_a(path: Path) -> np.ndarray:
    points: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            points.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except ValueError:
            fields = line.split()
            if len(fields) >= 9:
                points.append(tuple(float(value) for value in fields[6:9]))
    return np.asarray(points, dtype=float).reshape((-1, 3))


def _sphere_center_from_substrate(
    overlay_path: Path,
    cavity: dict[str, Any],
) -> np.ndarray | None:
    anchor = str(cavity.get("anchor", "")).strip().upper()
    match = re.fullmatch(r"(-?\d+)([A-Z][A-Z0-9_]*)", anchor)
    if match is None:
        return None
    resid = int(match.group(1))
    resname = match.group(2)
    overlay = _load_overlay(overlay_path)
    positions = np.asarray(overlay["positions_A"], dtype=float)
    resids = np.asarray(overlay.get("resids", np.zeros(len(positions))), dtype=int)
    resnames = np.char.upper(np.asarray(overlay["resnames"], dtype=str))
    atom_names = np.char.upper(np.asarray(overlay["atom_names"], dtype=str))
    selected = (resids == resid) & (resnames == resname)
    configured_names = {
        str(value).strip().upper() for value in cavity.get("anchor_atoms", []) if str(value).strip()
    }
    if configured_names:
        selected &= np.asarray([name in configured_names for name in atom_names], dtype=bool)
    points = positions[selected]
    return points.mean(axis=0) if len(points) else None


def _saved_cavity_descriptor(
    run_root: Path,
    cavity: dict[str, Any],
    run_id: str,
    overlay_path: Path | None,
) -> dict[str, np.ndarray] | None:
    # New analyses save this companion beside density_maps.npz, so Grand
    # alignment does not depend on private frame-cache records for cavity plotting.
    saved = run_root / "density" / GRAND_CAVITY_OVERLAY
    if saved.is_file():
        try:
            with np.load(saved) as data:
                raw = {name: np.asarray(data[name]) for name in data.files}
            descriptor = _normalized_cavity_descriptor(raw, cavity, run_id)
            if descriptor is not None:
                descriptor["geometry_source"] = np.asarray(str(saved))
                return descriptor
        except (OSError, ValueError, TypeError, KeyError):
            pass

    # Older VMD-enabled analyses persisted the exact canonical mask as PDB and
    # the exact sphere center/radius as a graphics command.
    mask_pdb = run_root / "vmd" / "cavity_mask_points.pdb"
    if mask_pdb.is_file():
        try:
            points_a = _vmd_mask_points_a(mask_pdb)
            descriptor = _normalized_cavity_descriptor(
                {"mode": "mask", "points_A": points_a}, cavity, run_id
            )
            if descriptor is not None:
                descriptor["geometry_source"] = np.asarray(str(mask_pdb))
                return descriptor
        except (OSError, ValueError, TypeError):
            pass
    session = run_root / "vmd" / "session.vmd.tcl"
    if session.is_file():
        try:
            source = session.read_text(encoding="utf-8", errors="replace")
            sphere = re.search(
                r"graphics\s+\$base_mol\s+sphere\s+\{([^}]+)\}\s+radius\s+([-+0-9.eE]+)",
                source,
            )
            if sphere is not None:
                center = np.asarray([float(value) for value in sphere.group(1).split()], dtype=float)
                descriptor = _normalized_cavity_descriptor(
                    {"mode": "sphere", "center_A": center, "radius_A": float(sphere.group(2))},
                    cavity,
                    run_id,
                )
                if descriptor is not None:
                    descriptor["geometry_source"] = np.asarray(str(session))
                    return descriptor
        except (OSError, ValueError, TypeError):
            pass

    # A legacy static mask with no declared source transform was, by the
    # analysis reader's contract, already in the analysis-reference frame.
    # This is therefore a safe cache/VMD-independent fallback.
    if str(cavity.get("mode", "")).strip().lower() == "mask" and cavity.get("mask"):
        try:
            meta_payload: dict[str, Any] = {}
            if cavity.get("meta"):
                meta_path = Path(str(cavity["meta"])).expanduser()
                if meta_path.is_file():
                    meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            declared_source = cavity.get("build_source") or meta_payload.get("source_gro")
            if not declared_source:
                from gcmc_port.cavity import load_voxel_mask
                from .masking import mask_from_first_trajectory_frame

                mask = load_voxel_mask(
                    Path(str(cavity["mask"])).expanduser(),
                    Path(str(cavity["meta"])).expanduser() if cavity.get("meta") else None,
                    membership_padding=float(cavity.get("membership_padding_nm", 0.02)),
                )
                trajectory = (
                    Path(str(cavity["mask_trajectory"])).expanduser()
                    if cavity.get("mask_trajectory")
                    else None
                )
                mask = mask_from_first_trajectory_frame(mask, trajectory)
                descriptor = _normalized_cavity_descriptor(
                    {
                        "mode": "mask",
                        "points_A": np.asarray(mask.points, dtype=float) * 10.0,
                        "voxel_spacing_A": float(mask.dx) * 10.0,
                    },
                    cavity,
                    run_id,
                )
                if descriptor is not None:
                    descriptor["geometry_source"] = np.asarray(str(cavity["mask"]))
                    return descriptor
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            pass

    # Sphere anchors are also exactly recoverable from the already-imaged
    # substrate overlay when the configured anchor residue was plotted.
    if str(cavity.get("mode", "")).strip().lower() == "sphere" and overlay_path is not None:
        try:
            center = _sphere_center_from_substrate(overlay_path, cavity)
            if center is not None:
                descriptor = _normalized_cavity_descriptor(
                    {"mode": "sphere", "center_A": center}, cavity, run_id
                )
                if descriptor is not None:
                    descriptor["geometry_source"] = np.asarray(str(overlay_path))
                    return descriptor
        except (OSError, ValueError, TypeError):
            pass
    return None


def _resolved_cavity_settings(cavity: Any, run_root: Path) -> dict[str, Any]:
    resolved = dict(cavity) if isinstance(cavity, dict) else {}
    for key in ("mask", "meta", "mask_trajectory", "points", "nearby_residues", "build_source", "build_output_prefix"):
        value = resolved.get(key)
        if value:
            resolved[key] = str(resolve_portable_path(str(value), run_root))
    return resolved


def _cached_cavity_descriptor(
    analysis: CompletedAnalysisRoot,
    run_id: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], Path] | None:
    run_root = analysis.root / run_id
    cache = cache_directory(run_root)
    density_map = run_root / "density" / "density_maps.npz"
    overlay = run_root / "density" / "substrate_overlay.npz"
    manifest = run_root / "analysis_manifest.json"
    if not manifest.is_file():
        return None
    source_overlay: dict[str, np.ndarray] | None = None
    source_context = overlay if overlay.is_file() else cache
    if overlay.is_file():
        try:
            source_overlay = _load_overlay(overlay)
        except (OSError, ValueError):
            source_overlay = None
    try:
        settings = json.loads(manifest.read_text(encoding="utf-8")).get("settings", {})
        cavity = _resolved_cavity_settings(
            settings.get("cavity", {}) if isinstance(settings, dict) else {},
            run_root,
        )
        descriptor: dict[str, np.ndarray] | None = None
        if cache.is_dir():
            cached = read_analysis_cache_metadata(run_root)
            metadata = cached.get("result_metadata") if isinstance(cached, dict) else None
            if isinstance(metadata, dict):
                cached_overlay = metadata.get("substrate_overlay")
                if source_overlay is None and isinstance(cached_overlay, dict):
                    candidate = {
                        key: np.asarray(value)
                        for key, value in cached_overlay.items()
                        if key in {
                            "positions_A", "atom_names", "elements", "resnames", "resids",
                            "atom_indices_0based", "coordinate_frame",
                        }
                    }
                    positions = np.asarray(candidate.get("positions_A", []), dtype=float)
                    if positions.ndim == 2 and positions.shape[1:] == (3,):
                        source_overlay = candidate
                        source_context = cache
                mask_points = np.asarray(
                    metadata.get("cavity_mask_points_nm", []), dtype=float
                ).reshape((-1, 3))
                center = np.asarray(metadata.get("cavity_center_nm", []), dtype=float).reshape((-1,))
                raw: dict[str, Any] = {"mode": cavity.get("mode", "")}
                if mask_points.size:
                    raw["points_A"] = mask_points * 10.0
                if center.size == 3:
                    raw["center_A"] = center * 10.0
                descriptor = _normalized_cavity_descriptor(raw, cavity, run_id)
                if descriptor is not None:
                    descriptor["geometry_source"] = np.asarray(str(cache))
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        EOFError,
        ImportError,
    ):
        descriptor = None
        try:
            settings = json.loads(manifest.read_text(encoding="utf-8")).get("settings", {})
            cavity = _resolved_cavity_settings(
                settings.get("cavity", {}) if isinstance(settings, dict) else {},
                run_root,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            cavity = {}
    if descriptor is None:
        descriptor = _saved_cavity_descriptor(
            run_root,
            cavity,
            run_id,
            overlay if overlay.is_file() else None,
        )
    if descriptor is None or source_overlay is None:
        return None
    descriptor["source_density_map"] = np.asarray(
        str(density_map.resolve()) if density_map.is_file() else "not-generated"
    )
    return descriptor, source_overlay, source_context.resolve()


def _cavity_descriptor_for_map(
    analysis: CompletedAnalysisRoot,
    density_map: SavedDensityMap,
    substrates: tuple[str, ...],
) -> dict[str, np.ndarray] | None:
    try:
        relative = density_map.path.relative_to(analysis.root)
        preferred_run = relative.parts[0] if relative.parts and relative.parts[0] != "aggregate" else None
    except ValueError:
        preferred_run = None
    run_ids = list(analysis.completed_runs)
    if preferred_run in run_ids:
        run_ids.remove(preferred_run)
        run_ids.insert(0, preferred_run)
    target_overlay = _load_overlay(density_map.overlay)
    for run_id in run_ids:
        found = _cached_cavity_descriptor(analysis, run_id)
        if found is None:
            continue
        descriptor, source_overlay, source_context = found
        try:
            local_transform, _matched, _rmsd = _alignment_transform(
                source_overlay,
                target_overlay,
                substrates,
                mobile_context=source_context,
                target_context=density_map.overlay,
            )
        except (OSError, ValueError):
            continue
        mapped = dict(descriptor)
        for key in ("points_A", "boundary_points_A"):
            if key in mapped:
                mapped[key] = apply_transform(np.asarray(mapped[key], dtype=float), local_transform)
        if "center_A" in mapped:
            mapped["center_A"] = apply_transform(
                np.asarray(mapped["center_A"], dtype=float).reshape((1, 3)), local_transform
            )[0]
        mapped["source_geometry_frame"] = np.asarray(str(source_context))
        return mapped
    return None


def _write_aligned_cavity_overlay(
    analysis: CompletedAnalysisRoot,
    density_map: SavedDensityMap,
    transform: tuple[np.ndarray, np.ndarray, np.ndarray],
    substrates: tuple[str, ...],
    destination: Path,
) -> dict[str, Any]:
    descriptor = _cavity_descriptor_for_map(analysis, density_map, substrates)
    if descriptor is None:
        destination.unlink(missing_ok=True)
        return {
            "status": "unavailable",
            "reason": (
                "No compatible cavity_overlay.npz, analysis cache, VMD cavity geometry, "
                "or recoverable sphere anchor was found with a generic substrate overlay."
            ),
        }
    aligned = dict(descriptor)
    for key in ("points_A", "boundary_points_A"):
        if key in aligned:
            aligned[key] = apply_transform(np.asarray(aligned[key], dtype=float), transform)
    if "center_A" in aligned:
        aligned["center_A"] = apply_transform(
            np.asarray(aligned["center_A"], dtype=float).reshape((1, 3)), transform
        )[0]
    aligned.update(
        {
            "coordinate_frame": np.asarray("grand-aligned"),
            "grand_alignment_schema_version": np.asarray(GRAND_ALIGNMENT_SCHEMA_VERSION),
            "grand_alignment_fixed_substrates": np.asarray(substrates, dtype="U16"),
        }
    )
    for key in ("geometry_source", "source_density_map", "source_geometry_frame"):
        if key not in aligned:
            continue
        value = str(np.asarray(aligned[key]).item())
        if value and value != "not-generated":
            aligned[key] = np.asarray(portable_path(value, destination.parent))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **aligned)
    return {
        "status": "available",
        "mode": str(np.asarray(aligned["mode"]).item()),
        "output": portable_path(destination, destination.parent),
        "source_run": str(np.asarray(aligned["source_run"]).item()),
    }


def _grid_corners(axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            (x, y, z)
            for x in (float(axes[0][0]), float(axes[0][-1]))
            for y in (float(axes[1][0]), float(axes[1][-1]))
            for z in (float(axes[2][0]), float(axes[2][-1]))
        ],
        dtype=float,
    )


def _common_axes(plans: list[AlignmentPlan], spacing_a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = np.vstack(
        [apply_transform(_grid_corners(plan.source_axes), plan.transform) for plan in plans]
    )
    low = np.floor(transformed.min(axis=0) / spacing_a) * spacing_a
    high = np.ceil(transformed.max(axis=0) / spacing_a) * spacing_a
    counts = np.maximum(2, np.rint((high - low) / spacing_a).astype(int) + 1)
    axes = tuple(low[index] + np.arange(int(counts[index])) * spacing_a for index in range(3))
    voxel_count = int(np.prod([len(axis) for axis in axes]))
    if voxel_count > 50_000_000:
        raise ValueError(
            f"The common grid would contain {voxel_count:,} voxels. "
            "Choose a larger grand-alignment grid spacing."
        )
    return axes  # type: ignore[return-value]


def _inverse_points(
    target_points: np.ndarray,
    transform: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    rotation, mobile_center, target_center = transform
    return (target_points - target_center) @ rotation.T + mobile_center


def _regrid_volume(
    values: np.ndarray,
    source_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    transform: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    method = "linear" if all(axis.size >= 2 for axis in source_axes) else "nearest"
    interpolator = RegularGridInterpolator(
        source_axes,
        np.asarray(values, dtype=float),
        method=method,
        bounds_error=False,
        fill_value=0.0,
    )
    output = np.empty(tuple(len(axis) for axis in target_axes), dtype=float)
    yz = np.stack(np.meshgrid(target_axes[1], target_axes[2], indexing="ij"), axis=-1).reshape((-1, 2))
    # Work in x slabs so a large but valid common grid does not require a second
    # full-volume coordinate array.
    slab_width = max(1, min(16, len(target_axes[0])))
    for start in range(0, len(target_axes[0]), slab_width):
        stop = min(start + slab_width, len(target_axes[0]))
        x_values = target_axes[0][start:stop]
        points = np.empty((len(x_values) * len(yz), 3), dtype=float)
        points[:, 0] = np.repeat(x_values, len(yz))
        points[:, 1:] = np.tile(yz, (len(x_values), 1))
        mobile_points = _inverse_points(points, transform)
        output[start:stop] = interpolator(mobile_points).reshape((len(x_values), len(target_axes[1]), len(target_axes[2])))
    return output


def _primary_key(payload: dict[str, np.ndarray]) -> str:
    if "rho" in payload and np.asarray(payload["rho"]).ndim == 3:
        return "rho"
    if "rho_difference" in payload and np.asarray(payload["rho_difference"]).ndim == 3:
        return "rho_difference"
    candidates = [name for name, value in payload.items() if np.asarray(value).ndim == 3]
    if not candidates:
        raise ValueError("The NPZ contains no 3D density array")
    return candidates[0]


def _volume_element(axes: tuple[np.ndarray, np.ndarray, np.ndarray], fallback: float) -> float:
    steps = [float(np.median(np.diff(axis))) if axis.size > 1 else fallback for axis in axes]
    return float(np.prod(steps))


def _write_aligned_map(
    plan: AlignmentPlan,
    destination: Path,
    target_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing_a: float,
    reference_map: SavedDensityMap,
    substrates: tuple[str, ...],
) -> dict[str, Any]:
    with np.load(plan.density_map.path) as data:
        source = {name: np.asarray(data[name]) for name in data.files}
    source_shape = tuple(len(axis) for axis in plan.source_axes)
    payload: dict[str, np.ndarray] = {}
    transformed_keys: list[str] = []
    for name, value in source.items():
        array = np.asarray(value)
        if array.ndim == 3 and array.shape == source_shape:
            payload[name] = _regrid_volume(array, plan.source_axes, target_axes, plan.transform)
            transformed_keys.append(name)
        elif name not in {"x_A", "y_A", "z_A", "xy_projection", "xz_projection", "yz_projection", "bin_A"}:
            payload[name] = array
    payload.update(
        {
            "x_A": target_axes[0],
            "y_A": target_axes[1],
            "z_A": target_axes[2],
            "bin_A": np.asarray(spacing_a),
            "grand_alignment_schema_version": np.asarray(GRAND_ALIGNMENT_SCHEMA_VERSION),
            "grand_alignment_fixed_substrates": np.asarray(substrates, dtype="U16"),
        }
    )
    primary = _primary_key(payload)
    rho = np.asarray(payload[primary], dtype=float)
    payload["xy_projection"] = rho.sum(axis=2).T * spacing_a
    payload["xz_projection"] = rho.sum(axis=1).T * spacing_a
    payload["yz_projection"] = rho.sum(axis=0).T * spacing_a
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)

    source_overlay = _load_overlay(plan.density_map.overlay)
    aligned_overlay = dict(source_overlay)
    aligned_overlay["positions_A"] = apply_transform(source_overlay["positions_A"], plan.transform)
    aligned_overlay["coordinate_frame"] = np.asarray("grand-aligned")
    aligned_overlay["grand_alignment_fixed_substrates"] = np.asarray(substrates, dtype="U16")
    np.savez_compressed(destination.parent / "substrate_overlay.npz", **aligned_overlay)
    cavity_overlay = _write_aligned_cavity_overlay(
        plan.analysis,
        plan.density_map,
        plan.transform,
        substrates,
        destination.parent / GRAND_CAVITY_OVERLAY,
    )

    for plane, horizontal, vertical, image in (
        ("xy", target_axes[0], target_axes[1], payload["xy_projection"]),
        ("xz", target_axes[0], target_axes[2], payload["xz_projection"]),
        ("yz", target_axes[1], target_axes[2], payload["yz_projection"]),
    ):
        _write_projection_csv(destination.parent / f"{plane}_projection.csv", horizontal, vertical, image)
    if destination.name == "density_maps.npz" and "rho_probability" in payload:
        _write_cube(destination.parent / "density.cube", np.asarray(payload["rho_probability"], dtype=float), target_axes, spacing_a)

    source_primary = np.asarray(source[_primary_key(source)], dtype=float)
    source_integral = float(source_primary.sum() * _volume_element(plan.source_axes, plan.source_bin_a))
    target_integral = float(rho.sum() * spacing_a ** 3)
    relative_drift = (
        float((target_integral - source_integral) / abs(source_integral))
        if abs(source_integral) > 1.0e-30
        else 0.0
    )
    metadata = {
        "schema_version": GRAND_ALIGNMENT_SCHEMA_VERSION,
        "source_map": portable_path(plan.density_map.path, destination.parent),
        "source_analysis": portable_path(plan.analysis.root, destination.parent),
        "reference_map": portable_path(reference_map.path, destination.parent),
        "map_kind": plan.density_map.kind,
        "fixed_substrates": list(substrates),
        "matched_atoms": list(plan.matched_atoms),
        "matched_atom_count": len(plan.matched_atoms),
        "fit_rmsd_A": plan.fit_rmsd_a,
        "rotation_row_vector_convention": plan.transform[0].tolist(),
        "mobile_center_A": plan.transform[1].tolist(),
        "target_center_A": plan.transform[2].tolist(),
        "transformed_density_keys": transformed_keys,
        "source_primary_integral": source_integral,
        "aligned_primary_integral": target_integral,
        "relative_integral_drift": relative_drift,
        "interpolation": "linear Cartesian regridding; zero outside the source grid",
        "cavity_overlay": cavity_overlay,
    }
    meta_path = destination.parent / "grand_alignment.meta.json"
    meta_path.write_text(
        json.dumps(portable_data(metadata, meta_path.parent), indent=2) + "\n",
        encoding="utf-8",
    )
    source_meta = plan.density_map.path.parent / "density_maps.meta.json"
    if source_meta.is_file() and destination.name == "density_maps.npz":
        try:
            density_meta = json.loads(source_meta.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            density_meta = {}
        density_meta["coordinate_frame"] = "grand-aligned"
        density_meta["grand_alignment"] = metadata
        (destination.parent / "density_maps.meta.json").write_text(
            json.dumps(portable_data(density_meta, destination.parent), indent=2) + "\n", encoding="utf-8"
        )
    return metadata


def _safe_label(path: Path, scan_root: Path, used: set[str]) -> str:
    try:
        relative = path.relative_to(scan_root)
        raw = "__".join(relative.parts)
    except ValueError:
        raw = path.name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "analysis"
    label = base
    serial = 2
    while label.lower() in used:
        label = f"{base}_{serial}"
        serial += 1
    used.add(label.lower())
    return label


def _plot_group(kind: str) -> str:
    if kind == "generic":
        return "generic"
    if kind == "aggregate-difference":
        return "difference"
    if kind == "aggregate-mean":
        return "aggregate-mean"
    return "pose"


def _limits_from_bounds(low: np.ndarray, high: np.ndarray) -> list[float]:
    return [
        float(low[0]), float(high[0]),
        float(low[1]), float(high[1]),
        float(low[2]), float(high[2]),
    ]


def _full_grid_limits(path: Path) -> list[float]:
    with np.load(path) as data:
        axes = [np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz"]
    return [
        float(axes[0][0]), float(axes[0][-1]),
        float(axes[1][0]), float(axes[1][-1]),
        float(axes[2][0]), float(axes[2][-1]),
    ]


def _cavity_overlay(path: Path) -> dict[str, np.ndarray] | None:
    overlay_path = path.parent / GRAND_CAVITY_OVERLAY
    if not overlay_path.is_file():
        return None
    try:
        with np.load(overlay_path) as data:
            payload = {name: np.asarray(data[name]) for name in data.files}
        mode = str(np.asarray(payload.get("mode", "")).item()).strip().lower()
        if mode == "mask":
            points = np.asarray(payload.get("points_A", []), dtype=float).reshape((-1, 3))
            if not points.size:
                return None
        elif mode == "sphere":
            center = np.asarray(payload.get("center_A", []), dtype=float).reshape((-1,))
            radius = float(payload.get("radius_A", 0.0))
            if center.size != 3 or not math.isfinite(radius) or radius <= 0.0:
                return None
        else:
            return None
        return payload
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _cavity_bounds(payload: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    mode = str(np.asarray(payload.get("mode", "")).item()).strip().lower()
    if mode == "mask":
        points = np.asarray(payload.get("points_A", []), dtype=float).reshape((-1, 3))
        if points.size:
            half_voxel = max(float(payload.get("voxel_spacing_A", 0.0)), 0.0) * 0.5
            return points.min(axis=0) - half_voxel, points.max(axis=0) + half_voxel
    if mode == "sphere":
        center = np.asarray(payload.get("center_A", []), dtype=float).reshape((-1,))
        radius = float(payload.get("radius_A", 0.0))
        if center.size == 3 and radius > 0.0:
            return center - radius, center + radius
    return None


def _nice_upper_and_step(maximum: float, *, target_intervals: int = 7) -> tuple[float, float]:
    value = float(maximum)
    if not math.isfinite(value) or value <= 0.0:
        return 1.0, 0.2
    exponent = int(math.floor(math.log10(value)))
    candidates: list[tuple[tuple[float, float, float], float, float]] = []
    for power in range(exponent - 3, exponent + 2):
        for multiplier in (1.0, 2.0, 5.0):
            step = multiplier * (10.0 ** power)
            intervals = max(1, int(math.ceil(value / step - 1.0e-12)))
            upper = intervals * step
            outside = max(0, 4 - intervals) + max(0, intervals - 8)
            score = (
                float(outside * 20 + abs(intervals - target_intervals)),
                float(-intervals),
                step,
            )
            candidates.append((score, upper, step))
    _score, upper, step = min(candidates, key=lambda item: item[0])
    return float(upper), float(step)


def _square_plane_limits(limits: Iterable[float]) -> list[float]:
    values = [float(value) for value in limits]
    if len(values) != 4:
        raise ValueError("A 2D axis limit requires [xmin, xmax, ymin, ymax]")
    x_mid = (values[0] + values[1]) * 0.5
    y_mid = (values[2] + values[3]) * 0.5
    span = max(values[1] - values[0], values[3] - values[2], 1.0e-6)
    return [x_mid - span * 0.5, x_mid + span * 0.5, y_mid - span * 0.5, y_mid + span * 0.5]


def _content_limits(
    records: list[tuple[Path, str]],
    style: dict[str, Any],
) -> dict[str, list[float]]:
    percentages = [
        float(value)
        for value in style.get("density_3d_isosurface_levels_percent", [8.0, 25.0, 50.0])
        if 0.0 < float(value) < 100.0
    ]
    cutoff_fraction = min(percentages, default=8.0) / 100.0
    lows: dict[str, list[np.ndarray]] = {}
    highs: dict[str, list[np.ndarray]] = {}
    fallback_by_group: dict[str, list[float]] = {}
    for path, kind in records:
        group = _plot_group(kind)
        fallback_by_group.setdefault(group, _full_grid_limits(path))
        with np.load(path) as data:
            payload = {name: np.asarray(data[name]) for name in data.files}
        rho = np.asarray(payload[_primary_key(payload)], dtype=float)
        axes = [np.asarray(payload[f"{name}_A"], dtype=float) for name in "xyz"]
        support = np.abs(rho) if kind == "aggregate-difference" else rho
        maximum = float(np.nanmax(support)) if support.size else 0.0
        if math.isfinite(maximum) and maximum > 0.0:
            selected = np.isfinite(support) & (support >= maximum * cutoff_fraction)
            locations = np.where(selected)
            if locations[0].size:
                low = np.asarray(
                    [axes[index][int(np.min(locations[index]))] for index in range(3)],
                    dtype=float,
                )
                high = np.asarray(
                    [axes[index][int(np.max(locations[index]))] for index in range(3)],
                    dtype=float,
                )
                lows.setdefault(group, []).append(low)
                highs.setdefault(group, []).append(high)
        overlay = _overlay(path)
        if overlay is not None and bool(style.get("substrate_overlay", True)):
            positions = np.asarray(overlay.get("positions_A", []), dtype=float).reshape((-1, 3))
            if positions.size:
                if not bool(style.get("substrate_show_hydrogens", False)):
                    elements = np.char.upper(
                        np.asarray(overlay.get("elements", [""] * len(positions)), dtype=str)
                    )
                    atom_names = np.char.upper(
                        np.asarray(overlay.get("atom_names", [""] * len(positions)), dtype=str)
                    )
                    keep = (elements != "H") & ~np.char.startswith(atom_names, "H")
                    positions = positions[keep]
                if positions.size:
                    lows.setdefault(group, []).append(positions.min(axis=0))
                    highs.setdefault(group, []).append(positions.max(axis=0))
        cavity = _cavity_overlay(path)
        if cavity is not None and bool(style.get("cavity_boundary", True)):
            cavity_limits = _cavity_bounds(cavity)
            if cavity_limits is not None:
                lows.setdefault(group, []).append(cavity_limits[0])
                highs.setdefault(group, []).append(cavity_limits[1])
    padding_fraction = float(style.get("grand_axis_padding_fraction", 0.10))
    padding_a = float(style.get("grand_axis_padding_A", 1.0))
    if padding_fraction < 0.0 or padding_a < 0.0:
        raise ValueError("Grand-axis padding values cannot be negative")
    limits: dict[str, list[float]] = {}
    for group, fallback in fallback_by_group.items():
        if group not in lows:
            limits[group] = fallback
            continue
        low = np.min(np.vstack(lows[group]), axis=0)
        high = np.max(np.vstack(highs[group]), axis=0)
        span = high - low
        # At least one grid cell/Angstrom of visible room prevents a planar or
        # nearly point-like density from collapsing an axis.
        padding = np.maximum(span * padding_fraction, padding_a)
        limits[group] = _limits_from_bounds(low - padding, high + padding)
    return limits


def _effective_grand_style(
    records: list[tuple[Path, str]],
    style: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    effective = dict(style)
    mode = str(effective.get("grand_axis_mode", "auto-content")).strip().lower()
    scope = str(effective.get("grand_axis_scope", "per-map")).strip().lower()
    if scope not in {"per-map", "shared-group"}:
        raise ValueError("grand_axis_scope must be per-map or shared-group")
    groups = sorted({_plot_group(kind) for _path, kind in records})
    limits_by_map: dict[str, list[float]] = {}
    if mode == "auto-content":
        limits_by_group = _content_limits(records, effective)
        if scope == "per-map":
            for path, kind in records:
                local = _content_limits([(path, kind)], effective)
                limits_by_map[str(path.resolve())] = local[_plot_group(kind)]
    elif mode == "manual":
        manual = effective.get("density_3d_axis_limits")
        if not manual or len(manual) != 6:
            raise ValueError(
                "grand_axis_mode='manual' requires density_3d_axis_limits with six values"
            )
        limits_by_group = {group: [float(value) for value in manual] for group in groups}
    elif mode == "full-grid":
        limits_by_group = {}
        for group in groups:
            paths = [path for path, kind in records if _plot_group(kind) == group]
            raw = [_full_grid_limits(path) for path in paths]
            limits_by_group[group] = [
                min(item[0] for item in raw), max(item[1] for item in raw),
                min(item[2] for item in raw), max(item[3] for item in raw),
                min(item[4] for item in raw), max(item[5] for item in raw),
            ]
    else:
        raise ValueError("grand_axis_mode must be auto-content, manual, or full-grid")
    effective["_grand_axis_limits_by_group"] = limits_by_group
    effective["_grand_axis_limits_by_map"] = limits_by_map
    return effective, limits_by_group


def _draw_cavity_2d(
    ax: Any,
    cavity: dict[str, np.ndarray] | None,
    horizontal: str,
    vertical: str,
    style: dict[str, Any],
) -> None:
    if cavity is None or not bool(style.get("cavity_boundary", True)):
        return
    indices = {name: index for index, name in enumerate("xyz")}
    h_index = indices[horizontal]
    v_index = indices[vertical]
    color = str(style.get("cavity_boundary_color", "#00E5FF"))
    linewidth = float(style.get("cavity_boundary_line_width", 2.2))
    alpha = float(style.get("cavity_boundary_alpha", 0.95))
    linestyle = str(style.get("cavity_boundary_line_style", "--"))
    mode = str(np.asarray(cavity.get("mode", "")).item()).strip().lower()
    if mode == "sphere":
        from matplotlib.patches import Circle

        center = np.asarray(cavity["center_A"], dtype=float)
        radius = float(cavity["radius_A"])
        ax.add_patch(
            Circle(
                (center[h_index], center[v_index]),
                radius,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                zorder=8,
            )
        )
        return
    points = np.asarray(cavity.get("points_A", []), dtype=float).reshape((-1, 3))
    if not points.size:
        return
    projected = points[:, [h_index, v_index]]
    spacing = max(float(cavity.get("voxel_spacing_A", 1.0)) * 0.75, 1.0e-3)
    spans = np.maximum(np.ptp(projected, axis=0), spacing)
    # Bound the raster size for very large masks while keeping the outline tied
    # to the saved voxel scale whenever practical.
    spacing = max(spacing, float(np.max(spans)) / 600.0)
    low = projected.min(axis=0) - spacing
    high = projected.max(axis=0) + spacing
    counts = np.maximum(3, np.ceil((high - low) / spacing).astype(int) + 1)
    x_edges = np.linspace(low[0], high[0], int(counts[0]) + 1)
    y_edges = np.linspace(low[1], high[1], int(counts[1]) + 1)
    occupancy, _x, _y = np.histogram2d(projected[:, 0], projected[:, 1], bins=(x_edges, y_edges))
    try:
        from scipy.ndimage import binary_dilation

        occupied = binary_dilation(occupancy > 0.0, iterations=1)
    except ImportError:
        occupied = occupancy > 0.0
    occupied = np.pad(occupied, 1, constant_values=False)
    x_centers = (x_edges[:-1] + x_edges[1:]) * 0.5
    y_centers = (y_edges[:-1] + y_edges[1:]) * 0.5
    x_centers = np.concatenate(([x_centers[0] - spacing], x_centers, [x_centers[-1] + spacing]))
    y_centers = np.concatenate(([y_centers[0] - spacing], y_centers, [y_centers[-1] + spacing]))
    ax.contour(
        x_centers,
        y_centers,
        occupied.T.astype(float),
        levels=[0.5],
        colors=[color],
        linewidths=[linewidth],
        linestyles=[linestyle],
        alpha=alpha,
        zorder=8,
    )


def _draw_cavity_3d(
    ax: Any,
    cavity: dict[str, np.ndarray] | None,
    style: dict[str, Any],
) -> None:
    if cavity is None or not bool(style.get("cavity_boundary", True)):
        return
    color = str(style.get("cavity_boundary_color", "#00E5FF"))
    alpha = float(style.get("cavity_boundary_alpha_3d", 0.68))
    linewidth = float(style.get("cavity_boundary_line_width_3d", 0.75))
    mode = str(np.asarray(cavity.get("mode", "")).item()).strip().lower()
    if mode == "sphere":
        center = np.asarray(cavity["center_A"], dtype=float)
        radius = float(cavity["radius_A"])
        u = np.linspace(0.0, 2.0 * math.pi, 36)
        v = np.linspace(0.0, math.pi, 18)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(
            x, y, z,
            rstride=2,
            cstride=2,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=7,
        )
        return
    points = np.asarray(
        cavity.get("boundary_points_A", cavity.get("points_A", [])), dtype=float
    ).reshape((-1, 3))
    if not points.size:
        return
    maximum = max(100, int(style.get("cavity_boundary_3d_max_points", 20000)))
    if len(points) > maximum:
        points = points[:: int(math.ceil(len(points) / maximum))]
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=float(style.get("cavity_boundary_point_size_3d", 5.0)),
        c=color,
        marker=".",
        alpha=alpha,
        linewidths=0.0,
        depthshade=False,
        zorder=7,
    )


def _set_square_box(ax: Any) -> None:
    try:
        ax.set_box_aspect(1.0)
    except (AttributeError, TypeError):
        pass


def _format_colorbar(colorbar: Any, *, limit: float, step: float, difference: bool) -> None:
    from matplotlib.ticker import FormatStrFormatter

    intervals = max(1, int(round(limit / step)))
    ticks = (
        np.arange(-intervals, intervals + 1, dtype=float) * step
        if difference
        else np.arange(intervals + 1, dtype=float) * step
    )
    colorbar.set_ticks(ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def _render_one_map(
    path: Path,
    kind: str,
    style: dict[str, Any],
    vmax: float | None,
) -> list[Path]:
    with np.load(path) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    primary = _primary_key(payload)
    rho = np.asarray(payload[primary], dtype=float)
    axes = {name: np.asarray(payload[f"{name}_A"], dtype=float) for name in "xyz"}
    bin_a = float(payload["bin_A"])
    overlay = _overlay(path)
    cavity = _cavity_overlay(path)
    projections = {
        "xy": rho.sum(axis=2).T * bin_a,
        "xz": rho.sum(axis=1).T * bin_a,
        "yz": rho.sum(axis=0).T * bin_a,
    }
    plt = _plt()
    outputs: list[Path] = []
    difference = kind == "aggregate-difference"
    local_style = dict(style)
    raw_maximum = (
        max(float(np.nanmax(np.abs(image))) for image in projections.values())
        if difference
        else max(float(np.nanmax(image)) for image in projections.values())
    )
    scale_source = float(vmax) if vmax is not None else raw_maximum
    color_limit, color_step = _nice_upper_and_step(scale_source)
    if not difference:
        local_style["density_vmin"] = float(local_style.get("density_vmin", 0.0) or 0.0)
        local_style["density_vmax"] = color_limit
    map_limits = style.get("_grand_axis_limits_by_map", {}).get(str(path.resolve()))
    display_limits = map_limits or style.get("_grand_axis_limits_by_group", {}).get(
        _plot_group(kind)
    )
    if display_limits:
        local_style["density_3d_axis_limits"] = display_limits
        automatic_2d = {
            "xy": display_limits[0:2] + display_limits[2:4],
            "xz": display_limits[0:2] + display_limits[4:6],
            "yz": display_limits[2:4] + display_limits[4:6],
        }
        manual_2d = dict(local_style.get("axis_limits", {}))
        combined_2d = dict(automatic_2d)
        combined_2d.update({f"pose_{plane}": values for plane, values in automatic_2d.items()})
        combined_2d.update(manual_2d)
        if bool(local_style.get("grand_square_2d_panels", True)):
            for key in ("xy", "xz", "yz", "pose_xy", "pose_xz", "pose_yz"):
                if key in combined_2d:
                    combined_2d[key] = _square_plane_limits(combined_2d[key])
        local_style["axis_limits"] = combined_2d
    if kind in {"generic", "pose-pocket", "pose-substrate"}:
        plot_dir = path.parent.parent / "plots"
        if kind == "generic":
            prefixes = {plane: plot_dir / f"density_{plane}.png" for plane in projections}
        else:
            prefixes = {
                plane: plot_dir / f"{path.parent.name}_{plane}.png"
                for plane in projections
            }
        for plane, horizontal, vertical in (("xy", "x", "y"), ("xz", "x", "z"), ("yz", "y", "z")):
            output = prefixes[plane]
            fig, ax = plt.subplots(figsize=local_style["density_figure_size"])
            shown = ax.imshow(
                projections[plane], origin="lower", aspect="equal", cmap=local_style["density_cmap"],
                vmin=local_style.get("density_vmin"), vmax=local_style.get("density_vmax"),
                extent=(axes[horizontal][0], axes[horizontal][-1], axes[vertical][0], axes[vertical][-1]),
            )
            ax.set_xlabel(f"{horizontal.upper()} (A)")
            ax.set_ylabel(f"{vertical.upper()} (A)")
            ax.set_title(f"Grand-aligned {plane.upper()} density projection")
            colorbar = fig.colorbar(shown, ax=ax, pad=0.04, label="Projected density")
            _format_colorbar(colorbar, limit=color_limit, step=color_step, difference=False)
            _draw_substrate_2d(ax, overlay, horizontal, vertical, local_style)
            _draw_cavity_2d(ax, cavity, horizontal, vertical, local_style)
            _format_axis(ax, local_style, plane if kind == "generic" else f"pose_{plane}")
            _set_square_box(ax)
            _save(fig, output, local_style)
            outputs.append(output)
    else:
        output = path.with_suffix(".png")
        fig = plt.figure(
            figsize=(local_style["figure_size"][0] * 1.85, local_style["figure_size"][1])
        )
        grid = fig.add_gridspec(
            1, 4,
            width_ratios=(1.0, 1.0, 1.0, 0.055),
            wspace=float(local_style.get("grand_2d_panel_spacing", 0.36)),
        )
        plot_axes = np.asarray([fig.add_subplot(grid[0, index]) for index in range(3)])
        colorbar_axis = fig.add_subplot(grid[0, 3])
        shown: Any = None
        for axis, (plane, horizontal, vertical) in zip(
            plot_axes,
            (("xy", "x", "y"), ("xz", "x", "z"), ("yz", "y", "z")),
        ):
            shown = axis.imshow(
                projections[plane], origin="lower", aspect="equal",
                cmap=local_style.get("difference_cmap", "coolwarm") if difference else local_style["density_cmap"],
                vmin=-color_limit if difference else local_style.get("density_vmin"),
                vmax=color_limit if difference else local_style.get("density_vmax"),
                extent=(axes[horizontal][0], axes[horizontal][-1], axes[vertical][0], axes[vertical][-1]),
            )
            axis.set_title(plane.upper())
            axis.set_xlabel(f"{horizontal.upper()} (A)")
            axis.set_ylabel(f"{vertical.upper()} (A)")
            _draw_substrate_2d(axis, overlay, horizontal, vertical, local_style)
            _draw_cavity_2d(axis, cavity, horizontal, vertical, local_style)
            _format_axis(axis, local_style, f"pose_{plane}")
            _set_square_box(axis)
        colorbar = fig.colorbar(
            shown,
            cax=colorbar_axis,
            label="Density difference" if difference else "Projected density",
        )
        _format_colorbar(
            colorbar,
            limit=color_limit,
            step=color_step,
            difference=difference,
        )
        fig.suptitle(f"Grand-aligned {path.stem}")
        _save(fig, output, local_style)
        outputs.append(output)

    if not difference:
        if kind == "generic":
            output_3d = path.parent.parent / "plots" / "density_3d.png"
        elif kind in {"pose-pocket", "pose-substrate"}:
            output_3d = path.parent.parent / "plots" / f"{path.parent.name}_3d.png"
        else:
            output_3d = path.with_name(path.stem + "_3d.png")
        fig = plt.figure(figsize=local_style["density_figure_size"])
        ax = fig.add_subplot(111, projection="3d")
        _density_isosurfaces(fig, ax, rho, axes, local_style, "Density", overlay)
        _draw_cavity_3d(ax, cavity, local_style)
        ax.set_xlabel("X (A)")
        ax.set_ylabel("Y (A)")
        ax.set_zlabel("Z (A)")
        ax.set_title("Grand-aligned 3D density isosurfaces")
        _save(fig, output_3d, local_style)
        outputs.append(output_3d)
    return outputs


def _render_aligned_maps(
    records: list[tuple[Path, str]],
    style: dict[str, Any],
) -> list[Path]:
    style, _limits = _effective_grand_style(records, style)
    _apply_global_style(style)
    maxima: dict[str, float] = {}
    for path, kind in records:
        with np.load(path) as data:
            payload = {name: np.asarray(data[name]) for name in data.files}
        rho = np.asarray(payload[_primary_key(payload)], dtype=float)
        bin_a = float(payload["bin_A"])
        images = (
            rho.sum(axis=2).T * bin_a,
            rho.sum(axis=1).T * bin_a,
            rho.sum(axis=0).T * bin_a,
        )
        maximum = (
            max(float(np.nanmax(np.abs(image))) for image in images)
            if kind == "aggregate-difference"
            else max(float(np.nanmax(image)) for image in images)
        )
        group = _plot_group(kind)
        maxima[group] = max(maxima.get(group, 0.0), maximum)
    outputs: list[Path] = []
    shared_scale = bool(style.get("grand_shared_color_scale", False))
    for path, kind in records:
        configured = style.get("density_vmax")
        vmax = (
            maxima.get("difference") if kind == "aggregate-difference" and shared_scale
            else None if kind == "aggregate-difference"
            else float(configured)
            if configured is not None
            else maxima.get(_plot_group(kind)) if shared_scale
            else None
        )
        outputs.extend(_render_one_map(path, kind, style, vmax))
    return outputs


def _common_parent(paths: list[Path]) -> Path:
    if not paths:
        return Path.cwd().resolve()
    common = Path(paths[0])
    for path in paths[1:]:
        while common != common.parent and path != common and common not in path.parents:
            common = common.parent
    return common


def _ensure_grand_plot_style_options(path: Path, defaults: dict[str, Any]) -> None:
    """Expose newly added plot controls in an older generated editable script."""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return
    assignment: ast.Assign | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "STYLE" for target in node.targets
        ):
            assignment = node
            break
    if (
        assignment is None
        or not isinstance(assignment.value, ast.Dict)
        or assignment.value.end_lineno is None
        or assignment.value.end_lineno <= assignment.value.lineno
    ):
        return
    existing = {
        str(key.value)
        for key in assignment.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    missing = [(key, value) for key, value in defaults.items() if key not in existing]
    if not missing:
        return
    lines = source.splitlines(keepends=True)
    insertion = assignment.value.end_lineno - 1
    block = ["    # Options added by Grand-alignment plot repair\n"]
    block.extend(f"    {key!r}: {value!r},\n" for key, value in missing)
    lines[insertion:insertion] = block
    path.write_text("".join(lines), encoding="utf-8")


def _copy_grand_plot_script(
    result_root: Path,
    *,
    reset: bool = False,
    style_overrides: dict[str, Any] | None = None,
) -> Path:
    from . import grand_plot_template

    target = result_root / GRAND_PLOT_SCRIPT
    created = reset or not target.exists()
    source = (
        Path(grand_plot_template.__file__).read_text(encoding="utf-8")
        if created
        else target.read_text(encoding="utf-8")
    )
    if created or style_overrides:
        for key, value in (style_overrides or {}).items():
            source = re.sub(
                rf'(?m)^(\s*"{re.escape(key)}"\s*:\s*).*$' ,
                lambda match, replacement=repr(value): f"{match.group(1)}{replacement},",
                source,
            )
        target.write_text(source, encoding="utf-8")
    _ensure_grand_plot_style_options(target, dict(grand_plot_template.STYLE))
    _install_grand_plot_source_bootstrap(target)
    return target


def _grand_plot_source_bootstrap() -> str:
    return "\n".join(
        (
            _GRAND_SOURCE_BEGIN,
            "_GENERATED_PACKAGE_SOURCE = None",
            "if _GENERATED_PACKAGE_SOURCE:",
            "    _generated_source = Path(_GENERATED_PACKAGE_SOURCE).expanduser()",
            '    if (_generated_source / "gcmc_port" / "analysis" / "grand_alignment.py").exists():',
            "        if str(_generated_source) not in sys.path:",
            "            sys.path.insert(0, str(_generated_source))",
            _GRAND_SOURCE_END,
        )
    )


def _install_grand_plot_source_bootstrap(path: Path) -> None:
    """Remove legacy checkout paths while retaining an installed-package fallback.

    Existing scripts are patched in place so user-edited STYLE values and any
    other customizations are retained during a plot-only repair.
    """
    source = path.read_text(encoding="utf-8")
    block = _grand_plot_source_bootstrap()
    pattern = re.compile(
        rf"(?ms)^{re.escape(_GRAND_SOURCE_BEGIN)}$.*?^{re.escape(_GRAND_SOURCE_END)}$"
    )
    if pattern.search(source):
        updated = pattern.sub(lambda _match: block, source, count=1)
    else:
        anchor = "# Source-checkout fallback; installed users resolve the package normally."
        if anchor not in source:
            raise ValueError(f"Could not locate the import bootstrap in editable plot script: {path}")
        updated = source.replace(anchor, block + "\n\n" + anchor, 1)
    if updated != source:
        path.write_text(updated, encoding="utf-8")


def _style_from_python(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "STYLE"
                for target in node.targets
            ):
                value = ast.literal_eval(node.value)
                return dict(value) if isinstance(value, dict) else {}
    except (OSError, SyntaxError, ValueError, TypeError):
        return {}
    return {}


def load_grand_plot_style(result_root: str | Path) -> dict[str, Any]:
    from . import grand_plot_template

    root = Path(result_root).expanduser().resolve()
    style = dict(grand_plot_template.STYLE)
    style.update(_style_from_python(root / GRAND_PLOT_SCRIPT))
    return style


def _aligned_records_from_manifest(result_root: Path) -> list[tuple[Path, str]]:
    manifest = result_root / "grand_alignment_manifest.json"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read grand-alignment manifest {manifest}: {exc}") from exc
    records: list[tuple[Path, str]] = []
    for item in payload.get("aligned_maps", []):
        if not isinstance(item, dict) or not item.get("output"):
            continue
        path = Path(str(item["output"])).expanduser()
        if not path.is_absolute():
            path = result_root / path
        if path.is_file():
            records.append((path.resolve(), str(item.get("map_kind", _map_kind(path)))))
    if records:
        return records
    # Relocated v1 result trees can contain stale absolute paths in the root
    # manifest. Per-map metadata remains beside each NPZ and is relocatable.
    for meta in sorted(result_root.glob("**/grand_alignment.meta.json")):
        try:
            item = json.loads(meta.read_text(encoding="utf-8"))
            source_name = Path(str(item["source_map"])).name
            path = meta.parent / source_name
            if path.is_file():
                records.append((path.resolve(), str(item.get("map_kind", _map_kind(path)))))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    if not records:
        raise ValueError(f"No aligned NPZ maps listed by {manifest} could be found")
    return records


def _ensure_grand_cavity_overlays(records: list[tuple[Path, str]]) -> list[str]:
    warnings: list[str] = []
    for aligned_path, kind in records:
        cavity_path = aligned_path.parent / GRAND_CAVITY_OVERLAY
        if _cavity_overlay(aligned_path) is not None:
            continue
        metadata_path = aligned_path.parent / "grand_alignment.meta.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            analysis = load_completed_analysis_root(
                resolve_portable_path(str(metadata["source_analysis"]), metadata_path.parent)
            )
            source_path = resolve_portable_path(str(metadata["source_map"]), metadata_path.parent)
            source_map = SavedDensityMap(
                source_path,
                (source_path.parent / "substrate_overlay.npz").resolve(),
                str(metadata.get("map_kind", kind)),
            )
            if not source_map.path.is_file() or not source_map.overlay.is_file():
                raise FileNotFoundError(f"source map/overlay is no longer available: {source_map.path}")
            transform = (
                np.asarray(metadata["rotation_row_vector_convention"], dtype=float),
                np.asarray(metadata["mobile_center_A"], dtype=float),
                np.asarray(metadata["target_center_A"], dtype=float),
            )
            substrates = tuple(str(value).strip().upper() for value in metadata["fixed_substrates"])
            info = _write_aligned_cavity_overlay(
                analysis,
                source_map,
                transform,
                substrates,
                cavity_path,
            )
            metadata["schema_version"] = GRAND_ALIGNMENT_SCHEMA_VERSION
            metadata["cavity_overlay"] = info
            metadata_path.write_text(
                json.dumps(portable_data(metadata, metadata_path.parent), indent=2) + "\n",
                encoding="utf-8",
            )
            if info.get("status") != "available":
                warnings.append(f"{aligned_path.name}: {info.get('reason', 'cavity overlay unavailable')}")
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            warnings.append(f"{aligned_path.name}: cavity boundary could not be repaired ({exc})")
    return warnings


def _cavity_overlay_summary(records: list[tuple[Path, str]]) -> dict[str, int]:
    available = sum(1 for path, _kind in records if _cavity_overlay(path) is not None)
    return {"available": available, "unavailable": len(records) - available}


def _write_grand_plot_style(result_root: Path, style: dict[str, Any]) -> Path:
    path = result_root / "plot_style.json"
    serializable = {key: value for key, value in style.items() if not key.startswith("_")}
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")
    return path


def replot_grand_alignment(
    result_root: str | Path,
    style: dict[str, Any] | None = None,
) -> list[Path]:
    root = Path(result_root).expanduser().resolve()
    editable_script = root / GRAND_PLOT_SCRIPT
    if editable_script.is_file():
        from . import grand_plot_template

        _ensure_grand_plot_style_options(editable_script, dict(grand_plot_template.STYLE))
    records = _aligned_records_from_manifest(root)
    cavity_warnings = _ensure_grand_cavity_overlays(records)
    chosen_style = load_grand_plot_style(root) if style is None else dict(style)
    effective, display_bounds = _effective_grand_style(records, chosen_style)
    plots = _render_aligned_maps(records, effective)
    manifest = root / "grand_alignment_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = GRAND_ALIGNMENT_SCHEMA_VERSION
    payload["display_bounds_A"] = display_bounds
    payload["display_bounds_by_map_A"] = {
        portable_path(path, root): limits
        for path, limits in effective.get("_grand_axis_limits_by_map", {}).items()
    }
    payload["plot_script"] = portable_path(root / GRAND_PLOT_SCRIPT, root)
    payload["plots"] = [portable_path(path, root) for path in plots]
    payload["cavity_overlay_warnings"] = cavity_warnings
    payload["cavity_overlay_summary"] = _cavity_overlay_summary(records)
    payload["last_replot_utc"] = datetime.now(timezone.utc).isoformat()
    manifest.write_text(
        json.dumps(portable_data(payload, root), indent=2) + "\n",
        encoding="utf-8",
    )
    _write_grand_plot_style(root, chosen_style)
    for warning in cavity_warnings:
        print(f"[grand-alignment cavity warning] {warning}", flush=True)
    return plots


def repair_grand_alignment_output(result_root: str | Path) -> list[Path]:
    """Upgrade old plot metadata/scripts and redraw from aligned NPZ data only."""
    root = Path(result_root).expanduser().resolve()
    if not (root / "grand_alignment_manifest.json").is_file():
        raise FileNotFoundError(f"Grand-alignment manifest not found under {root}")
    _copy_grand_plot_script(
        root,
        reset=False,
        style_overrides={
            "grand_shared_color_scale": False,
            "density_vmin": 0.0,
            "density_vmax": None,
            "grand_axis_scope": "per-map",
            "cavity_boundary": True,
        },
    )
    style = load_grand_plot_style(root)
    # Old v1 outputs saved full interpolation-grid limits. The new script's
    # content-aware mode deliberately replaces those computed display limits.
    style["grand_axis_mode"] = "auto-content"
    style["grand_axis_scope"] = "per-map"
    style["grand_shared_color_scale"] = False
    style["density_vmin"] = 0.0
    style["density_vmax"] = None
    style["grand_square_2d_panels"] = True
    style["cavity_boundary"] = True
    style["density_3d_axis_limits"] = None
    style["axis_limits"] = {
        key: value
        for key, value in dict(style.get("axis_limits", {})).items()
        if key not in {"xy", "xz", "yz", "pose_xy", "pose_xz", "pose_yz"}
    }
    plots = replot_grand_alignment(root, style)
    manifest = root / "grand_alignment_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["last_repair_utc"] = datetime.now(timezone.utc).isoformat()
    payload["repair_scope"] = (
        "Plot-only upgrade: editable script, per-map color/axes, equal square 2D panels, "
        "recoverable cavity boundaries, and PNG files. "
        "Aligned density NPZ data and source analyses were unchanged."
    )
    manifest.write_text(
        json.dumps(portable_data(payload, root), indent=2) + "\n",
        encoding="utf-8",
    )
    return plots


def grand_align_analysis_roots(
    analysis_roots: Iterable[str | Path | CompletedAnalysisRoot],
    output_root: str | Path,
    *,
    fixed_substrates: Iterable[str],
    reference_root: str | Path | CompletedAnalysisRoot | None = None,
    spacing_a: float | None = None,
    elevation: float = 30.0,
    azimuth: float = -60.0,
    roll: float = 0.0,
    render_plots: bool = True,
) -> GrandAlignmentResult:
    analyses = [
        item if isinstance(item, CompletedAnalysisRoot) else load_completed_analysis_root(item)
        for item in analysis_roots
    ]
    unique = {item.root: item for item in analyses}
    analyses = sorted(unique.values(), key=lambda item: str(item.root).lower())
    if len(analyses) < 2:
        raise ValueError("Grand alignment requires at least two completed analysis roots")
    substrates = tuple(dict.fromkeys(str(item).strip().upper() for item in fixed_substrates if str(item).strip()))
    if not substrates:
        substrates = default_fixed_substrates(analyses)
    coverage = substrate_coverage(analyses)
    absent = [name for name in substrates if coverage.get(name, 0) != len(analyses)]
    if absent:
        raise ValueError("Fixed substrate(s) are not present in every selected analysis: " + ", ".join(absent))

    if reference_root is None:
        reference = analyses[0]
    elif isinstance(reference_root, CompletedAnalysisRoot):
        reference = reference_root
    else:
        requested = Path(reference_root).expanduser().resolve()
        try:
            reference = unique[requested]
        except KeyError as exc:
            raise ValueError(f"Reference root is not one of the selected analyses: {requested}") from exc
    reference_density = _reference_map(reference, substrates)
    target_overlay = _load_overlay(reference_density.overlay)

    output = Path(output_root).expanduser().resolve()
    for analysis in analyses:
        if output == analysis.root or analysis.root in output.parents:
            raise ValueError(f"Grand-aligned output must not be written inside a source analysis root: {analysis.root}")
    if output.exists():
        raise FileExistsError(f"Grand-aligned output already exists: {output}")

    plans: list[AlignmentPlan] = []
    skipped: list[dict[str, str]] = []
    for analysis in analyses:
        aligned_count = 0
        for density_map in analysis.maps:
            try:
                mobile_overlay = _load_overlay(density_map.overlay)
                transform, matched, rmsd = _alignment_transform(
                    mobile_overlay,
                    target_overlay,
                    substrates,
                    mobile_context=density_map.overlay,
                    target_context=reference_density.overlay,
                )
                axes, bin_a = _axes_and_bin(density_map.path)
                plans.append(AlignmentPlan(analysis, density_map, transform, matched, rmsd, axes, bin_a))
                aligned_count += 1
            except (OSError, ValueError) as exc:
                skipped.append({"source": portable_path(density_map.path, output.parent), "reason": str(exc)})
        if aligned_count == 0:
            raise ValueError(
                f"No saved map in {analysis.root} could be aligned on {', '.join(substrates)}"
            )
    if not plans:
        raise ValueError("No compatible saved density maps were found")
    resolved_spacing = float(spacing_a) if spacing_a is not None else max(plan.source_bin_a for plan in plans)
    if not math.isfinite(resolved_spacing) or resolved_spacing <= 0.0:
        raise ValueError("Grand-alignment grid spacing must be positive")
    target_axes = _common_axes(plans, resolved_spacing)

    scan_root = _common_parent([analysis.root for analysis in analyses])
    labels: dict[Path, str] = {}
    used_labels: set[str] = set()
    for analysis in analyses:
        labels[analysis.root] = _safe_label(analysis.root, scan_root, used_labels)

    output.mkdir(parents=True)
    aligned_paths: list[Path] = []
    render_records: list[tuple[Path, str]] = []
    map_records: list[dict[str, Any]] = []
    for plan in plans:
        relative = plan.density_map.path.relative_to(plan.analysis.root)
        destination = output / labels[plan.analysis.root] / relative
        metadata = _write_aligned_map(
            plan, destination, target_axes, resolved_spacing, reference_density, substrates
        )
        aligned_paths.append(destination)
        render_records.append((destination, plan.density_map.kind))
        map_records.append({"output": portable_path(destination, output), **metadata})

    style = load_grand_plot_style(output)
    # Carry familiar color/font choices from the selected reference analysis,
    # while the generated grand script remains the authoritative easy editor.
    reference_style = load_plot_style(reference.root)
    for key in tuple(style):
        if key in reference_style and key not in {
            "axis_limits", "density_3d_axis_limits", "density_3d_elev",
            "density_3d_azim", "density_3d_roll",
        }:
            style[key] = reference_style[key]
    style.update(
        {
            "grand_axis_mode": "auto-content",
            "grand_axis_scope": "per-map",
            "grand_shared_color_scale": False,
            "density_vmin": 0.0,
            "density_vmax": None,
            "grand_square_2d_panels": True,
            "cavity_boundary": True,
            "density_3d_axis_limits": None,
            "axis_limits": {},
            "density_3d_elev": float(elevation),
            "density_3d_azim": float(azimuth),
            "density_3d_roll": float(roll),
        }
    )
    plot_script = _copy_grand_plot_script(
        output,
        style_overrides=style,
    )
    _write_grand_plot_style(output, style)
    _effective_style, display_bounds = _effective_grand_style(render_records, style)
    plots = _render_aligned_maps(render_records, style) if render_plots else []

    warnings: list[str] = []
    high_rmsd = [item for item in map_records if float(item["fit_rmsd_A"]) > 1.0]
    if high_rmsd:
        warnings.append(
            f"{len(high_rmsd)} map(s) have fixed-substrate fit RMSD above 1.0 A; "
            "inspect their grand_alignment.meta.json before structural comparison."
        )
    high_drift = [
        item
        for item in map_records
        if abs(float(item["relative_integral_drift"])) > 0.05
    ]
    if high_drift:
        warnings.append(
            f"{len(high_drift)} map(s) changed primary density integral by more than 5% during interpolation; "
            "consider a finer common grid."
        )
    missing_cavity = [
        item
        for item in map_records
        if dict(item.get("cavity_overlay", {})).get("status") != "available"
    ]
    if missing_cavity:
        warnings.append(
            f"{len(missing_cavity)} map(s) could not receive a cavity boundary because a compatible "
            "saved cavity geometry and source substrate coordinate frame were unavailable."
        )

    manifest_payload = {
        "schema_version": GRAND_ALIGNMENT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scientific_scope": (
            "Plot/data-coordinate post-processing only. Source trajectories, cavity memberships, "
            "lifetimes, occupancies, and event tables were not recalculated or modified."
        ),
        "reference_analysis": portable_path(reference.root, output),
        "reference_map": portable_path(reference_density.path, output),
        "fixed_substrates": list(substrates),
        "grid_spacing_A": resolved_spacing,
        "common_axes_A": {name: [float(axis[0]), float(axis[-1]), len(axis)] for name, axis in zip("xyz", target_axes)},
        "display_bounds_A": display_bounds,
        "display_bounds_by_map_A": {
            portable_path(path, output): limits
            for path, limits in _effective_style.get("_grand_axis_limits_by_map", {}).items()
        },
        "cavity_overlay_summary": _cavity_overlay_summary(render_records),
        "view": {"elevation": float(elevation), "azimuth": float(azimuth), "roll": float(roll)},
        "source_analyses": [
            {
                "root": portable_path(analysis.root, output),
                "label": labels[analysis.root],
                "status": analysis.status,
                "completed_runs": list(analysis.completed_runs),
                "saved_map_count": len(analysis.maps),
            }
            for analysis in analyses
        ],
        "aligned_maps": map_records,
        "skipped_maps": skipped,
        "warnings": warnings,
        "plots": [portable_path(path, output) for path in plots],
        "plot_script": portable_path(plot_script, output),
    }
    manifest = output / "grand_alignment_manifest.json"
    manifest.write_text(
        json.dumps(portable_data(manifest_payload, output), indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.txt").write_text(
        "Grand-aligned density results\n\n"
        "Every saved 3D density was rigidly aligned using the selected fixed substrate, "
        "interpolated onto one common Cartesian grid, and projected again for 2D plots.\n"
        "The original analysis directories were read only. Lifetime, occupancy, and event results "
        "were not copied or recalculated. See grand_alignment_manifest.json and each "
        "grand_alignment.meta.json for transforms, RMSD, and density-integral drift.\n"
        f"When source caches are available, {GRAND_CAVITY_OVERLAY} stores the same rigidly aligned "
        "mask boundary or sphere used by the analysis.\n"
        f"Edit {GRAND_PLOT_SCRIPT} and run it to change colors, opacity, axes, camera, and fonts "
        "without rerunning analysis or alignment.\n",
        encoding="utf-8",
    )
    return GrandAlignmentResult(
        output,
        manifest,
        tuple(aligned_paths),
        tuple(plots),
        tuple(skipped),
        tuple(warnings),
    )
