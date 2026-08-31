from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.interpolate import RegularGridInterpolator

from gcmc_port.cavity import VoxelMask, load_voxel_mask
from gcmc_port.pathing import portable_data

from .anchors import select_mda_anchor
from .density import _write_cube, _write_projection_csv, hpd_thresholds
from .geometry import _cell_vectors, apply_inverse_transform, apply_transform, kabsch_transform, minimum_image
from .models import AnalysisConfig, DatasetSpec, PoseFeatureSpec, config_for_dataset
from .masking import mask_dependency_paths, mask_from_first_trajectory_frame, resolve_mask_source_path
from .residue_mapping import AA3_TO_1, _global_pairs
from .tables import write_tsv


POSE_CACHE_VERSION = 2
POSE_HYDRATION_CACHE_VERSION = 5


@dataclass(slots=True)
class PoseStageResult:
    outputs_by_run: dict[str, list[Path]]
    metadata_by_run: dict[str, dict[str, Any]]
    warnings_by_run: dict[str, list[str]]
    aggregate_outputs: list[Path]
    failures: list[dict[str, str]]


def _element_from_name(name: str) -> str:
    """Best-effort element label for topology formats that omit elements."""
    cleaned = "".join(character for character in str(name).strip() if character.isalpha())
    if not cleaned:
        return "C"
    upper = cleaned.upper()
    if upper.startswith(("CL", "BR")):
        return upper[:2].title()
    return upper[0]


def _atom_elements(atoms: Any) -> np.ndarray:
    try:
        values = [str(value).strip() for value in atoms.elements]
    except Exception:
        values = []
    if len(values) != atoms.n_atoms or any(not value for value in values):
        values = [_element_from_name(str(atom.name)) for atom in atoms]
    return np.asarray(values, dtype="U4")


def _write_substrate_overlay(path: Path, atoms: Any, positions: np.ndarray, coordinate_frame: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions_A=np.asarray(positions, dtype=float),
        atom_names=np.asarray([str(atom.name) for atom in atoms], dtype="U16"),
        elements=_atom_elements(atoms),
        resnames=np.asarray([str(atom.resname) for atom in atoms], dtype="U16"),
        resids=np.asarray([int(atom.resid) for atom in atoms], dtype=int),
        coordinate_frame=np.asarray(coordinate_frame),
    )
    return path


def _fingerprint(
    settings: Any,
    paths: Iterable[Path | None],
    *,
    version: int = POSE_CACHE_VERSION,
) -> str:
    digest = hashlib.sha256()
    digest.update(str(version).encode())
    digest.update(repr(settings).encode())
    for path in paths:
        if path is None:
            continue
        digest.update(str(path).encode())
        if path.exists():
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def uniform_frame_indices(eligible: Iterable[int], maximum: int) -> np.ndarray:
    """Choose an exact, deterministic number of uniformly spaced indices including both endpoints."""
    values = np.asarray(list(eligible), dtype=int)
    if values.size == 0:
        return values
    if maximum <= 0 or values.size <= maximum:
        return values
    if maximum == 1:
        return values[[0]]
    positions = np.floor(np.arange(maximum, dtype=float) * (values.size - 1) / (maximum - 1) + 1e-12).astype(int)
    return values[positions]


def _require_mda():
    try:
        import MDAnalysis as mda
    except Exception as exc:
        raise RuntimeError(f"MDAnalysis is required for pose analysis: {exc}") from exc
    return mda


def _stage_root(config: AnalysisConfig) -> Path:
    return config.output.root / ".pose-stage"


def _group_id(dataset: DatasetSpec) -> str:
    return dataset.comparison_group or "default"


def _training_paths(config: AnalysisConfig, dataset: DatasetSpec) -> tuple[Path, Path]:
    root = _stage_root(config) / dataset.run_id
    return root / "training.npz", root / "training.meta.json"


def _model_paths(config: AnalysisConfig, group: str) -> tuple[Path, Path]:
    root = _stage_root(config) / "models"
    return root / f"{group}.npz", root / f"{group}.json"


def _selection(dataset: DatasetSpec, config: AnalysisConfig) -> str:
    selection = dataset.substrate_selection or config.substrate.selection
    if not selection:
        raise ValueError(f"{dataset.run_id}: substrate selection is empty")
    return selection


def _pocket_selection(dataset: DatasetSpec, config: AnalysisConfig) -> str:
    return dataset.pocket_selection or config.pose.pocket_selection or config.cavity.align_selection or "protein and backbone"


def _whole_positions(universe: Any) -> np.ndarray:
    try:
        return np.asarray(universe.atoms.unwrap(compound="fragments", reference="cog", inplace=False), dtype=float)
    except Exception:
        return np.asarray(universe.atoms.positions, dtype=float).copy()


def _reference_for(config: AnalysisConfig, dataset: DatasetSpec) -> Path:
    if config.pose.reference is not None:
        return config.pose.reference
    group = _group_id(dataset)
    candidates = [item for item in config.datasets if item.kind == "md" and _group_id(item) == group]
    first = candidates[0] if candidates else dataset
    reference = first.reference or first.topology
    if reference is None:
        raise ValueError(f"{dataset.run_id}: no pose reference or topology is available")
    return reference


def _alignment_indices_and_reference(
    config: AnalysisConfig,
    dataset: DatasetSpec,
    universe: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mda = _require_mda()
    path = _reference_for(config, dataset)
    reference = mda.Universe(str(path))
    mobile = universe.select_atoms(_pocket_selection(dataset, config))
    reference_selection = config.pose.reference_pocket_selection or config.pose.pocket_selection or _pocket_selection(dataset, config)
    target = reference.select_atoms(reference_selection)
    if mobile.n_atoms >= 3 and mobile.n_atoms == target.n_atoms and list(mobile.names) == list(target.names):
        mobile_indices = np.asarray(mobile.indices, dtype=int)
        target_indices = np.asarray(target.indices, dtype=int)
        method = "direct-selection atom-name match"
        mapped_residues = len(set(int(value) for value in target.resindices))
    else:
        simulation_residues = list(universe.select_atoms("protein").residues)
        reference_residues = list(reference.select_atoms("protein").residues)
        simulation_sequence = "".join(AA3_TO_1.get(str(item.resname).upper(), "X") for item in simulation_residues)
        reference_sequence = "".join(AA3_TO_1.get(str(item.resname).upper(), "X") for item in reference_residues)
        pairs, identity = _global_pairs(simulation_sequence, reference_sequence)
        target_atom_indices = set(int(value) for value in target.indices)
        mobile_allowed = set(int(value) for value in mobile.indices)
        mobile_list: list[int] = []
        target_list: list[int] = []
        mapped_residues = 0
        for simulation_index, reference_index in pairs:
            sim_residue = simulation_residues[simulation_index]
            ref_residue = reference_residues[reference_index]
            ref_atoms = [atom for atom in ref_residue.atoms if int(atom.index) in target_atom_indices and str(atom.name).upper() in {"N", "CA", "C", "O"}]
            matched_this_residue = False
            for ref_atom in ref_atoms:
                candidates = [
                    atom for atom in sim_residue.atoms
                    if str(atom.name).upper() == str(ref_atom.name).upper()
                    and (not mobile_allowed or int(atom.index) in mobile_allowed)
                ]
                if not candidates:
                    continue
                mobile_list.append(int(candidates[0].index))
                target_list.append(int(ref_atom.index))
                matched_this_residue = True
            mapped_residues += int(matched_this_residue)
        if len(mobile_list) < 3:
            close = getattr(reference.trajectory, "close", None)
            if close is not None:
                close()
            raise ValueError(
                f"{dataset.run_id}: pocket fit could not match at least three canonical backbone atoms; "
                "set case.pocket_selection and pose.reference_pocket_selection explicitly"
            )
        mobile_indices = np.asarray(mobile_list, dtype=int)
        target_indices = np.asarray(target_list, dtype=int)
        method = f"protein sequence mapping (identity={identity:.4f})"
    positions = _whole_positions(reference)[target_indices]
    close = getattr(reference.trajectory, "close", None)
    if close is not None:
        close()
    return mobile_indices, positions, {
        "method": method,
        "fit_atom_count": int(mobile_indices.size),
        "mapped_residue_count": int(mapped_residues),
        "reference": str(path),
        "reference_selection": reference_selection,
        "mobile_selection": _pocket_selection(dataset, config),
    }


def _eligible_indices(universe: Any, config: AnalysisConfig) -> tuple[list[int], dict[int, float]]:
    eligible: list[int] = []
    times: dict[int, float] = {}
    opts = config.analysis
    for sequential, ts in enumerate(universe.trajectory):
        time_ps = float(getattr(ts, "time", sequential))
        if opts.start_ps is not None and time_ps < opts.start_ps - 1e-9:
            continue
        if opts.stop_ps is not None and time_ps > opts.stop_ps + 1e-9:
            break
        if sequential % opts.stride:
            continue
        frame = int(getattr(ts, "frame", sequential))
        eligible.append(frame)
        times[frame] = time_ps
    return eligible, times


def _point(group: Any, positions: np.ndarray, mode: str) -> np.ndarray:
    if group.n_atoms == 0:
        raise ValueError("pose feature selection matched no atoms")
    coords = positions[group.indices]
    key = mode.lower().strip()
    if key in {"atom", "first"}:
        return coords[0]
    if key == "cog":
        return coords.mean(axis=0)
    if key == "com":
        masses = np.asarray(group.masses, dtype=float)
        if masses.size != coords.shape[0] or not np.all(np.isfinite(masses)) or masses.sum() <= 0:
            raise ValueError("COM feature requested but positive masses are unavailable")
        return np.average(coords, axis=0, weights=masses)
    try:
        charges = np.asarray(group.charges, dtype=float)
    except Exception as exc:
        raise ValueError(f"{key} feature requires topology charges") from exc
    if key in {"positive", "positive_charge_center"}:
        weights = np.maximum(charges, 0.0)
    elif key in {"negative", "negative_charge_center"}:
        weights = np.maximum(-charges, 0.0)
    else:
        raise ValueError(f"unsupported pose point mode: {mode}")
    if weights.sum() <= 1e-12:
        raise ValueError(f"{key} feature selection contains no charge of the requested sign")
    return np.average(coords, axis=0, weights=weights)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first, second = a - b, c - b
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    b0 = -(b - a)
    b1 = c - b
    b2 = d - c
    norm = np.linalg.norm(b1)
    if norm <= 1e-12:
        return float("nan")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _ring_pucker(coords: np.ndarray) -> tuple[float, float, float]:
    if coords.shape != (6, 3):
        raise ValueError("ring_pucker requires a selection containing exactly six ordered atoms")
    centered = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    z = centered @ vh[-1]
    indices = np.arange(6, dtype=float)
    q2c = math.sqrt(1.0 / 3.0) * float(np.sum(z * np.cos(2.0 * math.pi * indices / 3.0)))
    q2s = -math.sqrt(1.0 / 3.0) * float(np.sum(z * np.sin(2.0 * math.pi * indices / 3.0)))
    q3 = math.sqrt(1.0 / 6.0) * float(np.sum(((-1.0) ** indices) * z))
    q = math.sqrt(q2c * q2c + q2s * q2s + q3 * q3)
    theta = math.degrees(math.acos(np.clip(q3 / q, -1.0, 1.0))) if q > 1e-12 else 0.0
    phi = math.degrees(math.atan2(q2s, q2c)) % 360.0
    return q, theta, phi


def _descriptor_values(universe: Any, positions: np.ndarray, feature: PoseFeatureSpec) -> list[float]:
    groups = [universe.select_atoms(selection) for selection in feature.selections]
    modes = list(feature.point_modes) + ["cog"] * max(0, len(groups) - len(feature.point_modes))
    kind = feature.kind
    if kind == "ring_pucker":
        if len(groups) != 1:
            raise ValueError(f"pose feature {feature.name}: ring_pucker needs one selection")
        return list(_ring_pucker(positions[groups[0].indices]))
    points = [_point(group, positions, modes[index]) for index, group in enumerate(groups)]
    if kind in {"distance", "end_to_end"} and len(points) == 2:
        return [float(np.linalg.norm(points[1] - points[0]))]
    if kind == "angle" and len(points) == 3:
        return [_angle(*points)]
    if kind == "dihedral" and len(points) == 4:
        return [_dihedral(*points)]
    if kind in {"orientation", "pocket_axis_orientation"} and len(points) == 2:
        vector = points[1] - points[0]
        norm = np.linalg.norm(vector)
        return [float(np.degrees(np.arccos(np.clip(vector[2] / norm, -1.0, 1.0)))) if norm > 1e-12 else float("nan")]
    raise ValueError(f"pose feature {feature.name}: {kind} received an invalid number of selections")


def _feature_vector(
    universe: Any,
    positions: np.ndarray,
    substrate_indices: np.ndarray,
    features: tuple[PoseFeatureSpec, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    substrate = np.asarray(positions[substrate_indices], dtype=float)
    coordinate = substrate.reshape(-1) / math.sqrt(max(1, substrate.shape[0]))
    descriptors: list[float] = []
    names: list[str] = []
    for feature in features:
        values = _descriptor_values(universe, positions, feature)
        for offset, value in enumerate(values):
            descriptors.append(float(value))
            names.append(feature.name if len(values) == 1 else f"{feature.name}_{offset + 1}")
    return coordinate, np.asarray(descriptors, dtype=float), names


def _descriptor_weights(features: tuple[PoseFeatureSpec, ...]) -> np.ndarray:
    values: list[float] = []
    for feature in features:
        values.extend([feature.weight] * (3 if feature.kind == "ring_pucker" else 1))
    return np.asarray(values, dtype=float)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> Path:
    return write_tsv(path, rows)


def extract_pose_training(config: AnalysisConfig, dataset: DatasetSpec, *, force: bool = False) -> list[Path]:
    config = config_for_dataset(config, dataset)
    if dataset.kind != "md":
        return []
    cache_npz, cache_meta = _training_paths(config, dataset)
    run_pose = config.output.root / dataset.run_id / "poses"
    xtc_path = run_pose / "cluster_training.xtc"
    table_path = run_pose / "cluster_training_frames.tsv"
    fingerprint = _fingerprint(
        (config.analysis.start_ps, config.analysis.stop_ps, config.analysis.stride, config.substrate, config.pose, dataset.substrate_selection, dataset.pocket_selection),
        (dataset.topology, dataset.trajectory, dataset.reference, config.pose.reference),
    )
    if not force and cache_npz.exists() and cache_meta.exists() and table_path.exists() and (xtc_path.exists() or not config.pose.write_trajectory):
        try:
            if json.loads(cache_meta.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
                print(f"[cache] {dataset.run_id}: reusing pose-training features.", flush=True)
                return [cache_npz, cache_meta, table_path, *([xtc_path] if xtc_path.exists() else [])]
        except Exception:
            pass
    if dataset.topology is None:
        raise ValueError(f"{dataset.run_id}: pose analysis requires a topology")
    mda = _require_mda()
    universe = mda.Universe(str(dataset.topology), str(dataset.trajectory))
    pocket_indices, reference_positions, alignment_metadata = _alignment_indices_and_reference(config, dataset, universe)
    substrate_all = universe.select_atoms(_selection(dataset, config))
    substrate = substrate_all.select_atoms(config.substrate.heavy_selection)
    if substrate.n_atoms == 0:
        raise ValueError(f"{dataset.run_id}: heavy substrate selection matched no atoms")
    fit = universe.select_atoms(config.substrate.fit_selection) if config.substrate.fit_selection else substrate
    if fit.n_atoms < 3:
        raise ValueError(f"{dataset.run_id}: substrate fit requires at least three atoms")
    eligible, times = _eligible_indices(universe, config)
    selected = uniform_frame_indices(eligible, config.pose.max_frames_per_trajectory)
    if selected.size == 0:
        raise ValueError(f"{dataset.run_id}: no frames matched the pose sampling window")
    run_pose.mkdir(parents=True, exist_ok=True)
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if config.pose.write_trajectory:
        writer = mda.Writer(str(xtc_path), universe.atoms.n_atoms)
    coordinates: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    substrate_coords: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    descriptor_names: list[str] = []
    selected_set = set(int(value) for value in selected)
    for sequential, ts in enumerate(universe.trajectory):
        source_frame = int(getattr(ts, "frame", sequential))
        if source_frame not in selected_set:
            continue
        whole = _whole_positions(universe)
        transform = kabsch_transform(whole[pocket_indices], reference_positions)
        aligned = apply_transform(whole, transform)
        coordinate, descriptor, names = _feature_vector(universe, aligned, substrate.indices, config.pose.features)
        if not np.all(np.isfinite(coordinate)) or not np.all(np.isfinite(descriptor)):
            raise ValueError(f"{dataset.run_id}: non-finite pose feature at frame {source_frame}")
        coordinates.append(coordinate)
        descriptors.append(descriptor)
        substrate_coords.append(aligned[fit.indices])
        descriptor_names = names
        sample_index = len(rows)
        rows.append(
            {
                "run_id": dataset.run_id,
                "system_id": dataset.system_id or dataset.run_id,
                "comparison_group": _group_id(dataset),
                "replica": dataset.replica,
                "sweep": dataset.sweep,
                "sample_index": sample_index,
                "source_frame": source_frame,
                "time_ps": times[source_frame],
            }
        )
        if writer is not None:
            universe.atoms.positions = aligned
            writer.write(universe.atoms)
    if writer is not None:
        writer.close()
    _write_rows(table_path, rows)
    descriptor_array = np.vstack(descriptors) if descriptor_names else np.empty((len(rows), 0), dtype=float)
    signature = np.asarray([f"{atom.resname}:{atom.name}" for atom in substrate], dtype="U64")
    fit_signature = np.asarray([f"{atom.resname}:{atom.name}" for atom in fit], dtype="U64")
    np.savez_compressed(
        cache_npz,
        coordinates=np.vstack(coordinates),
        descriptors=descriptor_array,
        substrate_coords=np.stack(substrate_coords),
        source_frames=np.asarray([row["source_frame"] for row in rows], dtype=int),
        times_ps=np.asarray([row["time_ps"] for row in rows], dtype=float),
        atom_signature=signature,
        fit_signature=fit_signature,
        descriptor_names=np.asarray(descriptor_names, dtype="U128"),
        descriptor_weights=_descriptor_weights(config.pose.features),
    )
    metadata = {
        "schema_version": POSE_CACHE_VERSION,
        "run_id": dataset.run_id,
        "system_id": dataset.system_id or dataset.run_id,
        "comparison_group": _group_id(dataset),
        "replica": dataset.replica,
        "eligible_frames": len(eligible),
        "training_frames": int(selected.size),
        "sampling_fraction": float(selected.size / max(len(eligible), 1)),
        "max_frames_per_trajectory": config.pose.max_frames_per_trajectory,
        "substrate_selection": _selection(dataset, config),
        "pocket_selection": _pocket_selection(dataset, config),
        "reference": str(_reference_for(config, dataset)),
        "alignment": alignment_metadata,
        "fingerprint": fingerprint,
    }
    cache_meta.write_text(
        json.dumps(portable_data(metadata, cache_meta.parent), indent=2) + "\n",
        encoding="utf-8",
    )
    close = getattr(universe.trajectory, "close", None)
    if close is not None:
        close()
    return [cache_npz, cache_meta, table_path, *([xtc_path] if xtc_path.exists() else [])]


def _training_data(config: AnalysisConfig, dataset: DatasetSpec) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    npz, meta = _training_paths(config, dataset)
    if not npz.exists() or not meta.exists():
        raise FileNotFoundError(f"pose training stage is missing for {dataset.run_id}")
    with np.load(npz) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return arrays, json.loads(meta.read_text(encoding="utf-8"))


def _weighted_kmeans(
    values: np.ndarray,
    weights: np.ndarray,
    clusters: int,
    *,
    seed: int,
    restarts: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if values.shape[0] < clusters:
        raise ValueError(f"pose clustering requested k={clusters} with only {values.shape[0]} training frames")
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        centers = [values[int(rng.choice(values.shape[0], p=weights))]]
        while len(centers) < clusters:
            distances = np.min(
                np.stack([np.sum((values - center) ** 2, axis=1) for center in centers]), axis=0
            )
            probabilities = weights * np.maximum(distances, 1e-30)
            probabilities /= probabilities.sum()
            centers.append(values[int(rng.choice(values.shape[0], p=probabilities))])
        centroid = np.stack(centers)
        labels = np.zeros(values.shape[0], dtype=int)
        for _ in range(300):
            distance = np.sum((values[:, None, :] - centroid[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distance, axis=1)
            updated = centroid.copy()
            for cluster in range(clusters):
                mask = new_labels == cluster
                if np.any(mask):
                    updated[cluster] = np.average(values[mask], axis=0, weights=weights[mask])
                else:
                    updated[cluster] = values[int(np.argmax(np.min(distance, axis=1) * weights))]
            if np.array_equal(labels, new_labels) and np.allclose(updated, centroid, atol=1e-10):
                labels, centroid = new_labels, updated
                break
            labels, centroid = new_labels, updated
        inertia = float(np.sum(weights * np.min(np.sum((values[:, None, :] - centroid[None, :, :]) ** 2, axis=2), axis=1)))
        if best is None or inertia < best[2]:
            best = centroid, labels, inertia
    assert best is not None
    centroid, labels, inertia = best
    populations = np.asarray([weights[labels == index].sum() for index in range(clusters)])
    order = np.argsort(-populations, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(clusters)
    return centroid[order], inverse[labels], inertia


def fit_pose_models(config: AnalysisConfig, *, force: bool = False) -> list[Path]:
    outputs: list[Path] = []
    groups: dict[str, list[DatasetSpec]] = defaultdict(list)
    for dataset in config.datasets:
        if dataset.kind == "md":
            groups[_group_id(dataset)].append(dataset)
    for group_name, datasets in groups.items():
        missing = [dataset.run_id for dataset in datasets if not _training_paths(config, dataset)[0].exists()]
        datasets = [dataset for dataset in datasets if dataset.run_id not in set(missing)]
        if not datasets:
            raise FileNotFoundError(f"comparison group {group_name}: no successful pose training caches are available")
        model_npz, model_json = _model_paths(config, group_name)
        training_paths = [_training_paths(config, dataset)[0] for dataset in datasets]
        model_fingerprint = _fingerprint((config.pose.clusters, config.pose.seed, config.pose.restarts, group_name), training_paths)
        if not force and model_npz.exists() and model_json.exists():
            try:
                if json.loads(model_json.read_text(encoding="utf-8")).get("fingerprint") == model_fingerprint:
                    print(f"[cache] comparison group {group_name}: reusing pooled pose-cluster model.", flush=True)
                    outputs.extend([model_npz, model_json])
                    continue
            except Exception:
                pass
        arrays_and_meta = [_training_data(config, dataset) for dataset in datasets]
        signatures = [tuple(item[0]["atom_signature"].tolist()) for item in arrays_and_meta]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"comparison group {group_name}: substrate heavy-atom signatures differ across cases")
        fit_signatures = [tuple(item[0]["fit_signature"].tolist()) for item in arrays_and_meta]
        if any(signature != fit_signatures[0] for signature in fit_signatures[1:]):
            raise ValueError(f"comparison group {group_name}: substrate fit atom signatures differ across cases")
        descriptor_names = [tuple(item[0]["descriptor_names"].tolist()) for item in arrays_and_meta]
        if any(names != descriptor_names[0] for names in descriptor_names[1:]):
            raise ValueError(f"comparison group {group_name}: pose descriptor definitions differ across cases")
        descriptor_weights = arrays_and_meta[0][0]["descriptor_weights"]
        if any(not np.array_equal(item[0]["descriptor_weights"], descriptor_weights) for item in arrays_and_meta[1:]):
            raise ValueError(f"comparison group {group_name}: pose descriptor weights differ across cases")
        coordinates = np.vstack([item[0]["coordinates"] for item in arrays_and_meta])
        descriptors = np.vstack([item[0]["descriptors"] for item in arrays_and_meta])
        descriptor_mean = descriptors.mean(axis=0) if descriptors.shape[1] else np.empty(0)
        descriptor_scale = descriptors.std(axis=0) if descriptors.shape[1] else np.empty(0)
        if descriptor_scale.size:
            descriptor_scale[descriptor_scale < 1e-12] = 1.0
            scaled_descriptors = (descriptors - descriptor_mean) / descriptor_scale * descriptor_weights
            features = np.hstack([coordinates, scaled_descriptors])
        else:
            features = coordinates
        systems: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        offset = 0
        for dataset, (arrays, _meta) in zip(datasets, arrays_and_meta):
            count = arrays["coordinates"].shape[0]
            systems[dataset.system_id or dataset.run_id][dataset.replica or dataset.run_id].extend((offset + i, i) for i in range(count))
            offset += count
        weights = np.zeros(features.shape[0], dtype=float)
        for replicas in systems.values():
            for records in replicas.values():
                value = 1.0 / len(systems) / len(replicas) / len(records)
                for global_index, _ in records:
                    weights[global_index] = value
        centroids, labels, inertia = _weighted_kmeans(
            features, weights, config.pose.clusters, seed=config.pose.seed, restarts=config.pose.restarts
        )
        weighted_mean = np.average(features, axis=0, weights=weights)
        centered = features - weighted_mean
        _, _, vh = np.linalg.svd(centered * np.sqrt(weights[:, None]), full_matrices=False)
        components = vh[: min(2, vh.shape[0])]
        representatives: list[dict[str, Any]] = []
        all_substrate = np.concatenate([item[0]["substrate_coords"] for item in arrays_and_meta], axis=0)
        global_frames = np.concatenate([item[0]["source_frames"] for item in arrays_and_meta])
        run_labels = np.concatenate(
            [np.asarray([dataset.run_id] * item[0]["coordinates"].shape[0], dtype="U128") for dataset, item in zip(datasets, arrays_and_meta)]
        )
        for cluster in range(config.pose.clusters):
            candidates = np.flatnonzero(labels == cluster)
            distances = np.sum((features[candidates] - centroids[cluster]) ** 2, axis=1)
            best = int(candidates[int(np.argmin(distances))])
            representatives.append(
                {"cluster": cluster + 1, "run_id": str(run_labels[best]), "source_frame": int(global_frames[best]), "index": best}
            )
        model_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            model_npz,
            centroids=centroids,
            descriptor_mean=descriptor_mean,
            descriptor_scale=descriptor_scale,
            pca_mean=weighted_mean,
            pca_components=components,
            representative_substrate_coords=np.stack([all_substrate[item["index"]] for item in representatives]),
            atom_signature=np.asarray(signatures[0], dtype="U64"),
            fit_signature=np.asarray(fit_signatures[0], dtype="U64"),
            descriptor_names=np.asarray(descriptor_names[0], dtype="U128"),
            descriptor_weights=descriptor_weights,
        )
        model_json.write_text(
            json.dumps(
                {
                    "schema_version": POSE_CACHE_VERSION,
                    "comparison_group": group_name,
                    "clusters": config.pose.clusters,
                    "seed": config.pose.seed,
                    "restarts": config.pose.restarts,
                    "training_frames": int(features.shape[0]),
                    "system_weighting": "equal system; equal replica within system",
                    "inertia": inertia,
                    "representatives": [{key: value for key, value in item.items() if key != "index"} for item in representatives],
                    "fingerprint": model_fingerprint,
                    "excluded_missing_training_cases": missing,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        outputs.extend([model_npz, model_json])
    return outputs


def _load_model(config: AnalysisConfig, dataset: DatasetSpec) -> dict[str, np.ndarray]:
    path, _meta = _model_paths(config, _group_id(dataset))
    if not path.exists():
        raise FileNotFoundError(f"pose cluster model is missing for group {_group_id(dataset)}")
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _scaled_feature(coordinate: np.ndarray, descriptor: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    if descriptor.size:
        descriptor = (descriptor - model["descriptor_mean"]) / model["descriptor_scale"] * model["descriptor_weights"]
        return np.concatenate([coordinate, descriptor])
    return coordinate


def _select_anchor(universe: Any, config: AnalysisConfig) -> Any:
    group, resolution = select_mda_anchor(
        universe,
        config.cavity.anchor,
        config.cavity.anchor_atoms,
        context="pose cavity",
    )
    if resolution.warning:
        print(f"[anchor] {resolution.warning}", flush=True)
    return group


def _mask_source_path(config: AnalysisConfig, dataset: DatasetSpec, mask: VoxelMask) -> Path | None:
    """Resolve the structure whose coordinate frame was used to build a voxel mask."""
    return resolve_mask_source_path(
        mask=mask,
        mask_path=config.cavity.mask,
        meta_path=config.cavity.meta,
        build_source=config.cavity.build_source,
        run_dir=dataset.run_dir,
        config_dir=config.config_path.parent,
    )


def _mask_with_coordinates(mask: VoxelMask, points_nm: np.ndarray, reference_nm: np.ndarray) -> VoxelMask:
    return VoxelMask(
        points=np.asarray(points_nm, dtype=float),
        dx=mask.dx,
        reference_point=tuple(float(value) for value in reference_nm),
        effective_volume=mask.effective_volume,
        membership_padding=mask.membership_padding,
        probe_radius=mask.probe_radius,
        source_gro=mask.source_gro,
        exclude_residues=mask.exclude_residues,
    )


def _canonicalize_pose_mask(
    config: AnalysisConfig,
    dataset: DatasetSpec,
    trajectory_universe: Any,
    mask: VoxelMask,
    representative_targets: np.ndarray,
) -> tuple[VoxelMask, dict[str, Any]]:
    """Rigid-fit a GCMC mask from its build structure into the canonical pose frame.

    This mirrors xyz2cube's cavity-reference fit.  A final lattice-image choice keeps
    the fitted mask in the same triclinic periodic image as the pooled substrate model.
    """
    # An explicitly supplied mask trajectory is already expressed in the
    # corresponding trajectory frame and therefore keeps its historical
    # precedence over the static mask/source metadata.
    explicit_mask_frame = config.cavity.mask_trajectory is not None
    if explicit_mask_frame:
        mask = mask_from_first_trajectory_frame(mask, config.cavity.mask_trajectory)
    source_path = None if explicit_mask_frame else _mask_source_path(config, dataset, mask)
    declared_source = (
        str(config.cavity.build_source)
        if config.cavity.build_source is not None
        else mask.source_gro
    )
    if not explicit_mask_frame and declared_source and source_path is None:
        raise FileNotFoundError(
            f"{dataset.run_id}: the cavity mask source structure could not be resolved: {declared_source}. "
            "The mask cannot be rigidly aligned to the canonical pose frame without that structure."
        )
    points_a = np.asarray(mask.points, dtype=float) * 10.0
    reference_a = np.asarray(mask.reference_point, dtype=float) * 10.0
    fit_metadata: dict[str, Any]
    cell_a: np.ndarray | None = None
    transform: tuple[np.ndarray, np.ndarray, np.ndarray]

    if source_path is not None:
        mda = _require_mda()
        source = mda.Universe(str(source_path))
        source_indices, target_positions, source_alignment = _alignment_indices_and_reference(
            config, dataset, source
        )
        source_positions = _whole_positions(source)
        transform = kabsch_transform(source_positions[source_indices], target_positions)
        fitted_source = apply_transform(source_positions[source_indices], transform)
        fit_rmsd_a = float(
            np.sqrt(np.mean(np.sum((fitted_source - target_positions) ** 2, axis=1)))
        )
        source_dimensions = getattr(source.trajectory.ts, "dimensions", None)

        # Mask files and their source structure can themselves use different
        # periodic images after GROMACS centering.  Select the mask image nearest
        # the substrate/anchor in the source structure before the rigid fit.
        source_substrate = source.select_atoms(_selection(dataset, config))
        if source_substrate.n_atoms:
            source_center = source_positions[source_substrate.indices].mean(axis=0)
        else:
            source_anchor = _select_anchor(source, config)
            source_center = source_positions[source_anchor.indices].mean(axis=0)
        nearest_reference = minimum_image(reference_a, source_center, source_dimensions)
        source_image_shift = nearest_reference - reference_a
        points_a += source_image_shift
        reference_a += source_image_shift

        raw_cell = _cell_vectors(source_dimensions) if source_dimensions is not None else None
        if raw_cell is not None:
            cell_a = raw_cell @ transform[0]
        fit_metadata = {
            "method": "mask source-structure pocket Kabsch fit",
            "source": str(source_path),
            "source_image_shift_A": source_image_shift.tolist(),
            "fit_rmsd_A": fit_rmsd_a,
            "alignment": source_alignment,
        }
        close = getattr(source.trajectory, "close", None)
        if close is not None:
            close()
    else:
        # Legacy masks without source metadata retain the historical assumption,
        # but still receive a triclinic image correction below.
        points_a = np.asarray(mask.points, dtype=float) * 10.0
        reference_a = np.asarray(mask.reference_point, dtype=float) * 10.0
        trajectory_universe.trajectory[0]
        initial = _whole_positions(trajectory_universe)
        mobile_indices, target_positions, _trajectory_alignment = _alignment_indices_and_reference(
            config, dataset, trajectory_universe
        )
        transform = kabsch_transform(initial[mobile_indices], target_positions)
        dimensions = getattr(trajectory_universe.trajectory.ts, "dimensions", None)
        raw_cell = _cell_vectors(dimensions) if dimensions is not None else None
        if raw_cell is not None:
            cell_a = raw_cell @ transform[0]
        fit_metadata = {
            "method": (
                "explicit first mask-trajectory frame pocket Kabsch fit"
                if explicit_mask_frame
                else "legacy first-trajectory-frame pocket Kabsch fallback"
            ),
            "source": None,
            "source_image_shift_A": [0.0, 0.0, 0.0],
        }

    canonical_points_a = apply_transform(points_a, transform)
    canonical_reference_a = apply_transform(reference_a.reshape(1, 3), transform)[0]
    model_center = np.asarray(representative_targets, dtype=float).mean(axis=(0, 1))
    nearest_canonical_reference = minimum_image(canonical_reference_a, model_center, cell_a)
    canonical_image_shift = nearest_canonical_reference - canonical_reference_a
    canonical_points_a += canonical_image_shift
    canonical_reference_a += canonical_image_shift
    fit_metadata.update(
        {
            "canonical_image_shift_A": canonical_image_shift.tolist(),
            "canonical_reference_A": canonical_reference_a.tolist(),
            "substrate_model_center_A": model_center.tolist(),
            "reference_to_substrate_distance_A": float(np.linalg.norm(canonical_reference_a - model_center)),
        }
    )
    return _mask_with_coordinates(mask, canonical_points_a / 10.0, canonical_reference_a / 10.0), fit_metadata


def _molecule_point(residue: Any, positions: np.ndarray, config: AnalysisConfig) -> np.ndarray:
    atoms = residue.atoms
    if config.molecule.point_mode == "atom":
        for name in config.molecule.atom_names:
            selected = atoms.select_atoms(f"name {name}")
            if selected.n_atoms:
                return positions[int(selected.indices[0])]
        raise ValueError(f"{residue.resid}{residue.resname}: no representative molecule atom matched")
    selected = atoms
    if config.molecule.atom_names:
        named = atoms.select_atoms("name " + " ".join(config.molecule.atom_names))
        if named.n_atoms:
            selected = named
    if config.molecule.point_mode == "cog":
        return positions[selected.indices].mean(axis=0)
    masses = np.asarray(selected.masses, dtype=float)
    if masses.size != selected.n_atoms or not np.all(np.isfinite(masses)) or masses.sum() <= 0:
        raise ValueError("molecule COM requested but positive masses are unavailable")
    return np.average(positions[selected.indices], axis=0, weights=masses)


def _grid(center: np.ndarray, half_width: float, bin_a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low = np.asarray(center, dtype=float) - half_width
    high = np.asarray(center, dtype=float) + half_width
    counts = np.maximum(1, np.ceil((high - low) / bin_a).astype(int))
    axes = tuple(low[index] + (np.arange(counts[index]) + 0.5) * bin_a for index in range(3))
    return low, axes[0], axes[1], axes[2]


def _deposit(histogram: np.ndarray, point: np.ndarray, low: np.ndarray, bin_a: float) -> bool:
    index = np.floor((point - low) / bin_a).astype(int)
    if np.all(index >= 0) and np.all(index < np.asarray(histogram.shape)):
        histogram[tuple(index)] += 1.0
        return True
    return False


def _deposit_with_growth(
    histogram: np.ndarray,
    point: np.ndarray,
    low: np.ndarray,
    bin_a: float,
    *,
    margin_a: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deposit a point without silently losing substrate-frame density outside the initial grid."""
    index = np.floor((point - low) / bin_a).astype(int)
    shape = np.asarray(histogram.shape, dtype=int)
    if np.any(index < 0) or np.any(index >= shape):
        margin = max(1, int(math.ceil(margin_a / bin_a)))
        before = np.maximum(0, -index) + (index < 0) * margin
        after = np.maximum(0, index - shape + 1) + (index >= shape) * margin
        histogram = np.pad(
            histogram,
            tuple((int(before[axis]), int(after[axis])) for axis in range(3)),
            mode="constant",
        )
        low = np.asarray(low, dtype=float) - before * bin_a
        index += before
    histogram[tuple(index)] += 1.0
    return histogram, low


def _grid_half_width(
    center: np.ndarray,
    support_points: np.ndarray,
    reach_a: float,
) -> float:
    """Return a rotation-safe cubic half width enclosing support points plus a reach radius."""
    points = np.asarray(support_points, dtype=float).reshape((-1, 3))
    radial_extent = float(np.max(np.linalg.norm(points - np.asarray(center, dtype=float), axis=1))) if points.size else 0.0
    return max(float(reach_a) + radial_extent, 0.5)


def _write_density(
    directory: Path,
    histogram: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: AnalysisConfig,
    cluster_frames: int,
    all_frames: int,
    *,
    coordinate_frame: str,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    opts = config.analysis
    smoothed = gaussian_filter(
        histogram,
        sigma=opts.density_sigma_a / opts.density_bin_a,
        mode="constant",
        truncate=opts.density_cutoff_sigma,
    )
    volume = opts.density_bin_a ** 3
    conditional = smoothed / max(cluster_frames * volume, 1e-30)
    overall = smoothed / max(all_frames * volume, 1e-30)
    probability = smoothed / max(float(smoothed.sum()) * volume, 1e-30)
    projections = {
        "xy": conditional.sum(axis=2).T * opts.density_bin_a,
        "xz": conditional.sum(axis=1).T * opts.density_bin_a,
        "yz": conditional.sum(axis=0).T * opts.density_bin_a,
    }
    npz = directory / "density_maps.npz"
    np.savez_compressed(
        npz,
        rho=conditional,
        rho_conditional=conditional,
        rho_overall=overall,
        rho_probability=probability,
        x_A=axes[0], y_A=axes[1], z_A=axes[2],
        xy_projection=projections["xy"], xz_projection=projections["xz"], yz_projection=projections["yz"],
        bin_A=np.asarray(opts.density_bin_a),
        cluster_frame_count=np.asarray(cluster_frames), all_frame_count=np.asarray(all_frames),
    )
    cube = directory / "density.cube"
    _write_cube(cube, probability, axes, opts.density_bin_a)
    projection_outputs: list[Path] = []
    for name, horizontal, vertical in (("xy", axes[0], axes[1]), ("xz", axes[0], axes[2]), ("yz", axes[1], axes[2])):
        path = directory / f"{name}_projection.csv"
        _write_projection_csv(path, horizontal, vertical, projections[name])
        projection_outputs.append(path)
    meta = directory / "density_maps.meta.json"
    meta.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "coordinate_frame": coordinate_frame,
                "cluster_frame_count": cluster_frames,
                "all_frame_count": all_frames,
                "conditional_occupancy_integral": float(conditional.sum() * volume),
                "overall_contribution_integral": float(overall.sum() * volume),
                "probability_integral": float(probability.sum() * volume),
                "bin_A": opts.density_bin_a,
                "sigma_A": opts.density_sigma_a,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [npz, cube, *projection_outputs, meta]


def _site_rows(
    rho: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    bin_a: float,
    config: AnalysisConfig,
) -> list[dict[str, float | int]]:
    threshold = hpd_thresholds(rho, (config.pose.site_hpd,))[config.pose.site_hpd]
    radius_cells = max(1, int(math.ceil(config.pose.site_min_separation_a / bin_a)))
    local = maximum_filter(rho, size=2 * radius_cells + 1, mode="constant")
    candidates = np.argwhere((rho == local) & (rho >= threshold) & (rho > 0))
    ranked = sorted(candidates, key=lambda item: float(rho[tuple(item)]), reverse=True)[: config.pose.site_max_count]
    integral_radius2 = config.pose.site_integral_radius_a ** 2
    grids = np.meshgrid(*axes, indexing="ij")
    rows: list[dict[str, float | int]] = []
    for serial, index in enumerate(ranked, start=1):
        point = np.asarray([axes[axis][int(index[axis])] for axis in range(3)])
        distance2 = sum((grid - point[axis]) ** 2 for axis, grid in enumerate(grids))
        occupancy = float(rho[distance2 <= integral_radius2].sum() * bin_a ** 3)
        rows.append(
            {
                "site": serial, "x_A": float(point[0]), "y_A": float(point[1]), "z_A": float(point[2]),
                "integrated_occupancy": occupancy, "peak_density": float(rho[tuple(index)]),
            }
        )
    return rows


def _hydration_sites(path: Path, density_npz: Path, config: AnalysisConfig) -> list[Path]:
    with np.load(density_npz) as data:
        rho = np.asarray(data["rho_conditional"], dtype=float)
        axes = tuple(np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz")
        bin_a = float(data["bin_A"])
    rows = _site_rows(rho, axes, bin_a, config)
    lines = ["REMARK Cluster-conditioned hydration density sites"]
    for row in rows:
        lines.append(
            f"HETATM{int(row['site']):5d}  OW  HYS H{int(row['site']):4d}    "
            f"{float(row['x_A']):8.3f}{float(row['y_A']):8.3f}{float(row['z_A']):8.3f}"
            f"{min(float(row['integrated_occupancy']), 99.99):6.2f}{float(row['peak_density']):6.2f}           O"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table = write_tsv(path.with_suffix(".tsv"), rows)
    return [path, table]


def _write_pose_pdb(mda: Any, path: Path, universe: Any, positions: np.ndarray, indices: np.ndarray) -> None:
    universe.atoms.positions = positions
    path.parent.mkdir(parents=True, exist_ok=True)
    with mda.Writer(str(path), n_atoms=len(indices)) as writer:
        writer.write(universe.atoms[indices])


def _ensure_cached_substrate_overlays(config: AnalysisConfig, dataset: DatasetSpec) -> list[Path]:
    """Add lightweight plotting overlays to old, otherwise-valid pose caches."""
    if dataset.topology is None:
        return []
    mda = _require_mda()
    model = _load_model(config, dataset)
    outputs: list[Path] = []
    for cluster in range(model["representative_substrate_coords"].shape[0]):
        cluster_dir = config.output.root / dataset.run_id / "poses" / f"cluster_{cluster + 1:02d}"
        structure = cluster_dir / "representative_structure.pdb"
        if not structure.exists():
            continue
        pocket_overlay = cluster_dir / "pocket-frame" / "substrate_overlay.npz"
        substrate_overlay = cluster_dir / "substrate-frame" / "substrate_overlay.npz"
        if pocket_overlay.exists() and substrate_overlay.exists():
            outputs.extend([pocket_overlay, substrate_overlay])
            continue
        universe = mda.Universe(str(structure))
        substrate_all = universe.select_atoms(_selection(dataset, config))
        if substrate_all.n_atoms == 0:
            expected_resnames = sorted(
                {
                    str(value).split(":", 1)[0]
                    for value in model["atom_signature"].tolist()
                    if str(value).split(":", 1)[0]
                }
            )
            if expected_resnames:
                substrate_all = universe.select_atoms("resname " + " ".join(expected_resnames))
        if substrate_all.n_atoms == 0:
            close = getattr(universe.trajectory, "close", None)
            if close is not None:
                close()
            continue
        fit = universe.select_atoms(config.substrate.fit_selection) if config.substrate.fit_selection else substrate_all.select_atoms(config.substrate.heavy_selection)
        target = np.asarray(model["representative_substrate_coords"][cluster], dtype=float)
        if fit.n_atoms != target.shape[0] or fit.n_atoms < 3:
            close = getattr(universe.trajectory, "close", None)
            if close is not None:
                close()
            continue
        pocket_positions = np.asarray(substrate_all.positions, dtype=float).copy()
        transform = kabsch_transform(np.asarray(fit.positions, dtype=float), target)
        substrate_positions = apply_transform(pocket_positions, transform)
        outputs.extend(
            [
                _write_substrate_overlay(pocket_overlay, substrate_all, pocket_positions, "canonical-pocket"),
                _write_substrate_overlay(substrate_overlay, substrate_all, substrate_positions, "cluster-substrate"),
            ]
        )
        close = getattr(universe.trajectory, "close", None)
        if close is not None:
            close()
    return outputs


def _write_cluster_vmd(cluster_dir: Path, dataset: DatasetSpec, cluster: int) -> list[Path]:
    structure = cluster_dir / "representative_structure.pdb"
    water = cluster_dir / "representative_with_waters.pdb"
    sites = cluster_dir / "hydration_sites.pdb"
    cube = cluster_dir / "pocket-frame" / "density.cube"
    density_npz = cluster_dir / "pocket-frame" / "density_maps.npz"
    tcl = cluster_dir / "cluster_session.vmd.tcl"
    render = cluster_dir / "render_headless.vmd.tcl"
    def q(path: Path) -> str:
        return "{" + path.as_posix().replace("{", "\\{").replace("}", "\\}") + "}"
    thresholds: dict[float, float] = {}
    if density_npz.exists():
        with np.load(density_npz) as data:
            thresholds = hpd_thresholds(np.asarray(data["rho_probability"], dtype=float))
    lines = [
        f"mol new {q(structure)} waitfor all", "set base [molinfo top]", "mol delrep 0 $base",
        "mol representation NewCartoon", "mol color Structure", "mol selection protein", "mol addrep $base",
        f"if {{[file exists {q(water)}]}} {{ mol new {q(water)} waitfor all }}",
        f"if {{[file exists {q(sites)}]}} {{ mol new {q(sites)} waitfor all; mol representation VDW 0.35 12; mol color ColorID 7 }}",
        f"if {{[file exists {q(cube)}]}} {{ mol new {q(cube)} type cube waitfor all; set density_mol [molinfo top]; mol delrep 0 $density_mol }}",
        "color Display Background white", "axes location Off", "display resetview",
        f"puts \"Loaded {dataset.run_id} pose cluster {cluster}. Coordinates already share the canonical pocket frame.\"",
    ]
    for (probability, level), color in zip(sorted(thresholds.items(), reverse=True), (8, 0, 1, 7)):
        lines.extend(
            [
                f"# HPD {probability:.0%}", f"mol representation Isosurface {level:.10g} 0 0 0 1 1",
                f"mol color ColorID {color}", "mol material Transparent", "mol selection all", "mol addrep $density_mol",
            ]
        )
    tcl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    render.write_text(
        f"source {q(tcl)}\ndisplay projection Orthographic\nrender TachyonInternal {q(cluster_dir / 'canonical_snapshot.tga')}\nquit\n",
        encoding="utf-8",
    )
    return [tcl, render]


def pose_hydration_fingerprint(config: AnalysisConfig, dataset: DatasetSpec) -> str:
    """Fingerprint every input that determines a case hydration grid."""
    config = config_for_dataset(config, dataset)
    model_path, _ = _model_paths(config, _group_id(dataset))
    mask_source_path: Path | None = None
    if config.cavity.mode == "mask":
        assert config.cavity.mask is not None
        configured_mask = load_voxel_mask(
            config.cavity.mask,
            config.cavity.meta,
            membership_padding=config.cavity.membership_padding_nm,
        )
        if config.cavity.mask_trajectory is None:
            mask_source_path = _mask_source_path(config, dataset, configured_mask)
    mask_dependencies = mask_dependency_paths(
        mask_path=config.cavity.mask,
        meta_path=config.cavity.meta,
        build_source=config.cavity.build_source,
        run_dir=dataset.run_dir,
        config_dir=config.config_path.parent,
        membership_padding=config.cavity.membership_padding_nm,
    )
    return _fingerprint(
        (config.molecule, config.cavity, config.analysis, config.substrate, config.pose, dataset),
        (
            dataset.topology,
            dataset.trajectory,
            dataset.reference,
            model_path,
            config.cavity.mask,
            config.cavity.meta,
            config.cavity.mask_trajectory,
            mask_source_path,
            *mask_dependencies,
        ),
        version=POSE_HYDRATION_CACHE_VERSION,
    )


def hydrate_pose_case(config: AnalysisConfig, dataset: DatasetSpec, *, force: bool = False) -> list[Path]:
    config = config_for_dataset(config, dataset)
    if dataset.kind != "md":
        return []
    configured_mask: VoxelMask | None = None
    if config.cavity.mode == "mask":
        assert config.cavity.mask is not None
        configured_mask = load_voxel_mask(
            config.cavity.mask,
            config.cavity.meta,
            membership_padding=config.cavity.membership_padding_nm,
        )
    done = config.output.root / dataset.run_id / "poses" / "pose_manifest.json"
    hydration_fingerprint = pose_hydration_fingerprint(config, dataset)
    if done.exists() and not force:
        payload = json.loads(done.read_text(encoding="utf-8"))
        if payload.get("fingerprint") == hydration_fingerprint:
            print(f"[cache] {dataset.run_id}: reusing pose-hydration outputs.", flush=True)
            cached = [Path(item) for item in payload.get("outputs", []) if Path(item).exists()]
            cached.extend(_ensure_cached_substrate_overlays(config, dataset))
            return list(dict.fromkeys(cached))
    if dataset.topology is None:
        raise ValueError(f"{dataset.run_id}: pose hydration requires a topology")
    mda = _require_mda()
    universe = mda.Universe(str(dataset.topology), str(dataset.trajectory))
    model = _load_model(config, dataset)
    representative_targets = model["representative_substrate_coords"]
    pocket_indices, reference_positions, alignment_metadata = _alignment_indices_and_reference(config, dataset, universe)
    substrate_all = universe.select_atoms(_selection(dataset, config))
    substrate = substrate_all.select_atoms(config.substrate.heavy_selection)
    fit = universe.select_atoms(config.substrate.fit_selection) if config.substrate.fit_selection else substrate
    signature = tuple(f"{atom.resname}:{atom.name}" for atom in substrate)
    if signature != tuple(model["atom_signature"].tolist()):
        raise ValueError(f"{dataset.run_id}: substrate atom signature differs from pooled cluster model")
    if tuple(f"{atom.resname}:{atom.name}" for atom in fit) != tuple(model["fit_signature"].tolist()):
        raise ValueError(f"{dataset.run_id}: substrate fit atom signature differs from pooled cluster model")
    tracked_residues = [residue for residue in universe.residues if str(residue.resname).upper() in config.molecule.resnames]
    if not tracked_residues:
        raise ValueError(f"{dataset.run_id}: no tracked molecule residues matched {config.molecule.resnames}")
    mask = None
    mask_alignment_metadata: dict[str, Any] | None = None
    if config.cavity.mode == "mask":
        assert configured_mask is not None
        mask = configured_mask
        mask, mask_alignment_metadata = _canonicalize_pose_mask(
            config,
            dataset,
            universe,
            mask,
            representative_targets,
        )
    anchor = None if mask is not None else _select_anchor(universe, config)
    eligible, _times = _eligible_indices(universe, config)
    eligible_set = set(eligible)
    training_arrays, _training_meta = _training_data(config, dataset)
    training_frame_set = set(int(value) for value in training_arrays["source_frames"])
    k = model["centroids"].shape[0]
    pose_root = config.output.root / dataset.run_id / "poses"
    cluster_writers: list[Any | None] = []
    cluster_training_paths: list[Path] = []
    for cluster in range(k):
        path = pose_root / f"cluster_{cluster + 1:02d}" / "cluster_training.xtc"
        if config.pose.write_trajectory:
            path.parent.mkdir(parents=True, exist_ok=True)
            cluster_writers.append(mda.Writer(str(path), universe.atoms.n_atoms))
            cluster_training_paths.append(path)
        else:
            cluster_writers.append(None)
    bin_a = config.analysis.density_bin_a
    padding = config.analysis.density_cutoff_sigma * config.analysis.density_sigma_a
    # The pooled model fixes one canonical center/orientation across homologs. Masks may
    # still require different extents, which aggregate_pose later regrids safely.
    center = representative_targets.mean(axis=(0, 1))
    if mask is not None:
        mask_points = np.asarray(mask.points, dtype=float) * 10.0
        half_width = _grid_half_width(center, mask_points, padding)
    else:
        half_width = _grid_half_width(
            center,
            representative_targets.reshape((-1, 3)),
            config.cavity.radius_nm * 10.0 + config.cavity.membership_padding_nm * 10.0 + padding,
        )
    low, x, y, z = _grid(center, half_width, bin_a)
    axes = (x, y, z)
    shape = tuple(len(axis) for axis in axes)
    pocket_hist = np.zeros((k, *shape), dtype=float)
    substrate_lows: list[np.ndarray] = []
    substrate_axes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    substrate_hist: list[np.ndarray] = []
    for cluster in range(k):
        target_center = representative_targets[cluster].mean(axis=0)
        if mask is not None:
            substrate_half_width = _grid_half_width(target_center, mask_points, padding)
        else:
            substrate_half_width = _grid_half_width(
                target_center,
                representative_targets[cluster],
                config.cavity.radius_nm * 10.0 + config.cavity.membership_padding_nm * 10.0 + padding,
            )
        sub_low, sx, sy, sz = _grid(target_center, substrate_half_width, bin_a)
        substrate_lows.append(sub_low)
        substrate_axes.append((sx, sy, sz))
        substrate_hist.append(np.zeros((len(sx), len(sy), len(sz)), dtype=float))
    cluster_frames = np.zeros(k, dtype=int)
    transitions = np.zeros((k, k), dtype=int)
    assignment_rows: list[dict[str, Any]] = []
    best_distance = np.full(k, np.inf)
    best_positions: list[np.ndarray | None] = [None] * k
    best_water_indices: list[np.ndarray] = [np.empty(0, dtype=int) for _ in range(k)]
    previous_cluster: int | None = None
    for sequential, ts in enumerate(universe.trajectory):
        frame = int(getattr(ts, "frame", sequential))
        if frame not in eligible_set:
            continue
        whole = _whole_positions(universe)
        transform = kabsch_transform(whole[pocket_indices], reference_positions)
        aligned = apply_transform(whole, transform)
        coordinate, descriptor, _names = _feature_vector(universe, aligned, substrate.indices, config.pose.features)
        feature = _scaled_feature(coordinate, descriptor, model)
        distances = np.sum((model["centroids"] - feature) ** 2, axis=1)
        cluster = int(np.argmin(distances))
        cluster_frames[cluster] += 1
        if previous_cluster is not None:
            transitions[previous_cluster, cluster] += 1
        previous_cluster = cluster
        pc = (feature - model["pca_mean"]) @ model["pca_components"].T
        row: dict[str, Any] = {
            "run_id": dataset.run_id, "system_id": dataset.system_id or dataset.run_id,
            "comparison_group": _group_id(dataset), "replica": dataset.replica, "sweep": dataset.sweep,
            "frame": frame, "time_ps": float(getattr(ts, "time", sequential)), "cluster": cluster + 1,
            "distance_to_centroid": float(distances[cluster]),
            "pc1": float(pc[0]) if pc.size else 0.0, "pc2": float(pc[1]) if pc.size > 1 else 0.0,
        }
        for name, value in zip(model["descriptor_names"].tolist(), descriptor):
            row[str(name)] = float(value)
        assignment_rows.append(row)
        if frame in training_frame_set and cluster_writers[cluster] is not None:
            universe.atoms.positions = aligned
            cluster_writers[cluster].write(universe.atoms)
        substrate_transform = kabsch_transform(aligned[fit.indices], representative_targets[cluster])
        output_positions = aligned.copy()
        inside_atom_indices: list[int] = []
        if mask is not None:
            anchor_point = np.asarray(mask.reference_point, dtype=float) * 10.0
            raw_anchor_point = apply_inverse_transform(anchor_point.reshape(1, 3), transform)[0]
        else:
            assert anchor is not None
            raw_anchor_point = whole[anchor.indices].mean(axis=0)
            anchor_point = apply_transform(raw_anchor_point.reshape(1, 3), transform)[0]
        dimensions = getattr(ts, "dimensions", None)
        box_a = None if dimensions is None else np.asarray(dimensions, dtype=float)
        for residue in tracked_residues:
            raw_point = _molecule_point(residue, whole, config)
            imaged_point = minimum_image(raw_point, raw_anchor_point, box_a)
            image_shift = imaged_point - raw_point
            point = apply_transform(imaged_point.reshape(1, 3), transform)[0]
            inside = mask.contains_point(tuple(point / 10.0)) if mask is not None else np.linalg.norm(point - anchor_point) <= (config.cavity.radius_nm + config.cavity.membership_padding_nm) * 10.0
            if not inside:
                continue
            if not _deposit(pocket_hist[cluster], point, low, bin_a):
                raise RuntimeError(
                    f"{dataset.run_id}: a cavity molecule fell outside the canonical pocket grid; "
                    "the saved density would be incomplete"
                )
            substrate_point = apply_transform(point.reshape(1, 3), substrate_transform)[0]
            substrate_hist[cluster], substrate_lows[cluster] = _deposit_with_growth(
                substrate_hist[cluster],
                substrate_point,
                substrate_lows[cluster],
                bin_a,
                margin_a=padding,
            )
            output_positions[residue.atoms.indices] = apply_transform(
                whole[residue.atoms.indices] + image_shift,
                transform,
            )
            inside_atom_indices.extend(int(index) for index in residue.atoms.indices)
        if distances[cluster] < best_distance[cluster]:
            best_distance[cluster] = distances[cluster]
            best_positions[cluster] = output_positions
            best_water_indices[cluster] = np.asarray(inside_atom_indices, dtype=int)
    if not assignment_rows:
        raise ValueError(f"{dataset.run_id}: no frames were available for pose assignment")
    for writer in cluster_writers:
        if writer is not None:
            writer.close()
    outputs: list[Path] = [path for path in cluster_training_paths if path.exists()]
    assignments = _write_rows(pose_root / "pose_assignments.tsv", assignment_rows)
    outputs.append(assignments)
    summary_rows = []
    for cluster in range(k):
        summary_rows.append(
            {
                "run_id": dataset.run_id, "system_id": dataset.system_id or dataset.run_id,
                "comparison_group": _group_id(dataset), "replica": dataset.replica, "sweep": dataset.sweep,
                "cluster": cluster + 1, "frame_count": int(cluster_frames[cluster]),
                "population": float(cluster_frames[cluster] / len(assignment_rows)),
                "representative_distance": float(best_distance[cluster]),
            }
        )
    outputs.append(_write_rows(pose_root / "cluster_summary.tsv", summary_rows))
    transition_rows = [
        {"run_id": dataset.run_id, "from_cluster": first + 1, "to_cluster": second + 1, "count": int(transitions[first, second])}
        for first in range(k) for second in range(k)
    ]
    outputs.append(_write_rows(pose_root / "cluster_transitions.tsv", transition_rows))
    protein_indices = universe.select_atoms("protein").indices
    substrate_indices_all = substrate_all.indices
    substrate_axes = [
        tuple(
            substrate_lows[cluster][axis]
            + (np.arange(substrate_hist[cluster].shape[axis]) + 0.5) * bin_a
            for axis in range(3)
        )
        for cluster in range(k)
    ]
    for cluster in range(k):
        cluster_dir = pose_root / f"cluster_{cluster + 1:02d}"
        density_outputs = _write_density(
            cluster_dir / "pocket-frame", pocket_hist[cluster], axes, config,
            int(cluster_frames[cluster]), len(assignment_rows), coordinate_frame="canonical-pocket",
        )
        density_outputs.extend(
            _write_density(
                cluster_dir / "substrate-frame", substrate_hist[cluster], substrate_axes[cluster], config,
                int(cluster_frames[cluster]), len(assignment_rows), coordinate_frame="cluster-substrate",
            )
        )
        outputs.extend(density_outputs)
        outputs.extend(_hydration_sites(cluster_dir / "hydration_sites.pdb", cluster_dir / "pocket-frame" / "density_maps.npz", config))
        positions = best_positions[cluster]
        if positions is not None:
            structure_indices = np.unique(np.concatenate([protein_indices, substrate_indices_all]))
            with_water = np.unique(np.concatenate([structure_indices, best_water_indices[cluster]]))
            structure_path = cluster_dir / "representative_structure.pdb"
            water_path = cluster_dir / "representative_with_waters.pdb"
            _write_pose_pdb(mda, structure_path, universe, positions, structure_indices)
            _write_pose_pdb(mda, water_path, universe, positions, with_water)
            outputs.extend([structure_path, water_path])
            substrate_transform = kabsch_transform(positions[fit.indices], representative_targets[cluster])
            outputs.extend(
                [
                    _write_substrate_overlay(
                        cluster_dir / "pocket-frame" / "substrate_overlay.npz",
                        substrate_all,
                        positions[substrate_indices_all],
                        "canonical-pocket",
                    ),
                    _write_substrate_overlay(
                        cluster_dir / "substrate-frame" / "substrate_overlay.npz",
                        substrate_all,
                        apply_transform(positions[substrate_indices_all], substrate_transform),
                        "cluster-substrate",
                    ),
                ]
            )
            outputs.extend(_write_cluster_vmd(cluster_dir, dataset, cluster + 1))
        else:
            absent = cluster_dir / "cluster_absent.txt"
            absent.write_text("No analyzed frames from this run were assigned to the common cluster.\n", encoding="utf-8")
            outputs.append(absent)
    manifest = {
        "schema_version": 1, "status": "complete", "run_id": dataset.run_id,
        "pose_hydration_cache_version": POSE_HYDRATION_CACHE_VERSION,
        "comparison_group": _group_id(dataset), "cluster_count": k,
        "analyzed_frames": len(assignment_rows), "cluster_frames": cluster_frames.tolist(),
        "outputs": [str(path) for path in outputs],
        "fingerprint": hydration_fingerprint,
        "alignment": alignment_metadata,
        "mask_alignment": mask_alignment_metadata,
    }
    done.write_text(
        json.dumps(portable_data(manifest, done.parent), indent=2) + "\n",
        encoding="utf-8",
    )
    outputs.append(done)
    close = getattr(universe.trajectory, "close", None)
    if close is not None:
        close()
    return outputs


def _regrid_density_groups(
    entries: dict[str, list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]]],
    bin_a: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Interpolate aligned case grids onto one union grid and preserve each integral."""
    flat = [item for values in entries.values() for item in values]
    if not flat:
        raise ValueError("no density grids were supplied")
    lows = np.min(np.asarray([[axis[0] - 0.5 * bin_a for axis in axes] for _rho, axes in flat]), axis=0)
    highs = np.max(np.asarray([[axis[-1] + 0.5 * bin_a for axis in axes] for _rho, axes in flat]), axis=0)
    lows = np.floor(lows / bin_a + 1e-9) * bin_a
    highs = np.ceil(highs / bin_a - 1e-9) * bin_a
    counts = np.maximum(1, np.ceil((highs - lows) / bin_a - 1e-9).astype(int))
    target_axes = tuple(lows[index] + (np.arange(int(counts[index])) + 0.5) * bin_a for index in range(3))

    def regrid(rho: np.ndarray, axes: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        if rho.shape == tuple(len(axis) for axis in target_axes) and all(
            np.allclose(source, target, atol=1e-7, rtol=0.0) for source, target in zip(axes, target_axes)
        ):
            return np.asarray(rho, dtype=float)
        interpolator = RegularGridInterpolator(axes, rho, bounds_error=False, fill_value=0.0)
        output = np.empty(tuple(len(axis) for axis in target_axes), dtype=float)
        yy, zz = np.meshgrid(target_axes[1], target_axes[2], indexing="ij")
        for index, x_value in enumerate(target_axes[0]):
            points = np.column_stack((np.full(yy.size, x_value), yy.ravel(), zz.ravel()))
            output[index] = interpolator(points).reshape(yy.shape)
        source_sum = float(np.nansum(rho))
        target_sum = float(np.nansum(output))
        if source_sum > 0.0 and target_sum > 0.0:
            output *= source_sum / target_sum
        return output

    replicas: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    for system, values in entries.items():
        stacked = np.stack([regrid(rho, axes) for rho, axes in values])
        replicas[system] = stacked
        means[system] = stacked.mean(axis=0)
    return means, replicas, target_axes


def _write_common_model_overlay(config: AnalysisConfig, dataset: DatasetSpec, cluster: int, directory: Path) -> Path:
    model = _load_model(config, dataset)
    coordinates = np.asarray(model["representative_substrate_coords"][cluster - 1], dtype=float)
    signatures = [str(value) for value in model["fit_signature"].tolist()]
    resnames: list[str] = []
    names: list[str] = []
    for signature in signatures:
        resname, separator, name = signature.partition(":")
        resnames.append(resname if separator else "SUB")
        names.append(name if separator else signature)
    path = directory / "substrate_overlay.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions_A=coordinates,
        atom_names=np.asarray(names, dtype="U16"),
        elements=np.asarray([_element_from_name(name) for name in names], dtype="U4"),
        resnames=np.asarray(resnames, dtype="U16"),
        resids=np.zeros(len(names), dtype=int),
        coordinate_frame=np.asarray("common-cluster-representative"),
    )
    return path


def aggregate_pose(config: AnalysisConfig) -> list[Path]:
    root = config.output.root / "aggregate" / "pose-groups"
    outputs: list[Path] = []
    groups: dict[str, list[DatasetSpec]] = defaultdict(list)
    for dataset in config.datasets:
        if dataset.kind == "md":
            groups[_group_id(dataset)].append(dataset)
    for group, datasets in groups.items():
        group_dir = root / group
        summaries: list[dict[str, Any]] = []
        for dataset in datasets:
            path = config.output.root / dataset.run_id / "poses" / "cluster_summary.tsv"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", newline="") as handle:
                summaries.extend(dict(row) for row in csv.DictReader(handle, delimiter="\t"))
        if not summaries:
            continue
        outputs.append(write_tsv(group_dir / "cluster_populations.tsv", summaries))
        by_system: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in summaries:
            by_system[(str(row["system_id"]), int(row["cluster"]))].append(float(row["population"]))
        aggregate_rows = []
        for (system, cluster), values in sorted(by_system.items()):
            array = np.asarray(values, dtype=float)
            sem = float(array.std(ddof=1) / math.sqrt(array.size)) if array.size >= 3 else float("nan")
            aggregate_rows.append(
                {
                    "comparison_group": group, "system_id": system, "cluster": cluster,
                    "replica_count": int(array.size), "mean_population": float(array.mean()),
                    "ci95_low": float(array.mean() - 1.96 * sem) if array.size >= 3 else "",
                    "ci95_high": float(array.mean() + 1.96 * sem) if array.size >= 3 else "",
                    "ci_note": "replica-based" if array.size >= 3 else "fewer than 3 independent replicas; CI omitted",
                }
            )
        outputs.append(write_tsv(group_dir / "system_cluster_summary.tsv", aggregate_rows))
        systems = sorted({dataset.system_id or dataset.run_id for dataset in datasets})
        for cluster in range(1, config.pose.clusters + 1):
            density_entries: dict[str, list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = defaultdict(list)
            for system in systems:
                for dataset in datasets:
                    if (dataset.system_id or dataset.run_id) != system:
                        continue
                    path = config.output.root / dataset.run_id / "poses" / f"cluster_{cluster:02d}" / "pocket-frame" / "density_maps.npz"
                    if not path.exists():
                        continue
                    with np.load(path) as data:
                        if int(data["cluster_frame_count"]) == 0:
                            continue
                        density_entries[system].append(
                            (
                                np.asarray(data["rho_conditional"], dtype=float),
                                tuple(np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz"),
                            )
                        )
            if not density_entries:
                continue
            system_density, system_replicas, common_axes = _regrid_density_groups(
                density_entries, config.analysis.density_bin_a
            )
            axes = dict(zip("xyz", common_axes))
            cluster_dir = group_dir / f"cluster_{cluster:02d}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            outputs.append(_write_common_model_overlay(config, datasets[0], cluster, cluster_dir))
            for system, rho in system_density.items():
                path = cluster_dir / f"{system}.mean_density.npz"
                replicas = system_replicas[system]
                if replicas.shape[0] >= 3:
                    ci = 1.96 * replicas.std(axis=0, ddof=1) / math.sqrt(replicas.shape[0])
                else:
                    ci = np.full_like(rho, np.nan)
                np.savez_compressed(
                    path, rho=rho, rho_ci95_low=rho - ci, rho_ci95_high=rho + ci,
                    replica_count=np.asarray(replicas.shape[0]),
                    bin_A=np.asarray(config.analysis.density_bin_a),
                    x_A=axes["x"], y_A=axes["y"], z_A=axes["z"],
                )
                outputs.append(path)
            common_density = np.mean(np.stack(list(system_density.values())), axis=0)
            common_site_rows = _site_rows(common_density, (axes["x"], axes["y"], axes["z"]), config.analysis.density_bin_a, config)
            site_comparison_rows: list[dict[str, Any]] = []
            grids = np.meshgrid(axes["x"], axes["y"], axes["z"], indexing="ij")
            for row in common_site_rows:
                point = np.asarray([row["x_A"], row["y_A"], row["z_A"]], dtype=float)
                distance2 = sum((grid - point[index]) ** 2 for index, grid in enumerate(grids))
                selected = distance2 <= config.pose.site_integral_radius_a ** 2
                for system, rho in system_density.items():
                    site_comparison_rows.append(
                        {
                            "comparison_group": group, "cluster": cluster, "site": row["site"],
                            "x_A": row["x_A"], "y_A": row["y_A"], "z_A": row["z_A"],
                            "system_id": system,
                            "integrated_occupancy": float(rho[selected].sum() * config.analysis.density_bin_a ** 3),
                        }
                    )
            outputs.append(write_tsv(cluster_dir / "common_hydration_site_comparison.tsv", site_comparison_rows))
            for first_index, first in enumerate(systems):
                for second in systems[first_index + 1:]:
                    if first not in system_density or second not in system_density:
                        continue
                    difference = system_density[second] - system_density[first]
                    first_replicas, second_replicas = system_replicas[first], system_replicas[second]
                    if first_replicas.shape[0] >= 3 and second_replicas.shape[0] >= 3:
                        first_sem = first_replicas.std(axis=0, ddof=1) / math.sqrt(first_replicas.shape[0])
                        second_sem = second_replicas.std(axis=0, ddof=1) / math.sqrt(second_replicas.shape[0])
                        difference_ci = 1.96 * np.sqrt(first_sem * first_sem + second_sem * second_sem)
                    else:
                        difference_ci = np.full_like(difference, np.nan)
                    path = cluster_dir / f"difference.{second}-minus-{first}.npz"
                    np.savez_compressed(
                        path, rho_difference=difference,
                        rho_difference_ci95_low=difference - difference_ci,
                        rho_difference_ci95_high=difference + difference_ci,
                        bin_A=np.asarray(config.analysis.density_bin_a),
                        x_A=axes["x"], y_A=axes["y"], z_A=axes["z"],
                    )
                    outputs.append(path)
            # Substrate-frame maps use the same global representative for a cluster, so their grids are also comparable.
            substrate_dir = cluster_dir / "substrate-frame"
            substrate_dir.mkdir(parents=True, exist_ok=True)
            substrate_entries: dict[str, list[tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]]] = defaultdict(list)
            for system in systems:
                for dataset in datasets:
                    if (dataset.system_id or dataset.run_id) != system:
                        continue
                    path = config.output.root / dataset.run_id / "poses" / f"cluster_{cluster:02d}" / "substrate-frame" / "density_maps.npz"
                    if not path.exists():
                        continue
                    with np.load(path) as data:
                        if int(data["cluster_frame_count"]) == 0:
                            continue
                        substrate_entries[system].append(
                            (
                                np.asarray(data["rho_conditional"], dtype=float),
                                tuple(np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz"),
                            )
                        )
            if substrate_entries:
                substrate_systems, substrate_replicas, common_substrate_axes = _regrid_density_groups(
                    substrate_entries, config.analysis.density_bin_a
                )
                substrate_axes = dict(zip("xyz", common_substrate_axes))
                outputs.append(_write_common_model_overlay(config, datasets[0], cluster, substrate_dir))
                for system, rho in substrate_systems.items():
                    replicas = substrate_replicas[system]
                    ci = 1.96 * replicas.std(axis=0, ddof=1) / math.sqrt(replicas.shape[0]) if replicas.shape[0] >= 3 else np.full_like(rho, np.nan)
                    path = substrate_dir / f"{system}.mean_density.npz"
                    np.savez_compressed(
                        path, rho=rho, rho_ci95_low=rho - ci, rho_ci95_high=rho + ci,
                        replica_count=np.asarray(replicas.shape[0]), bin_A=np.asarray(config.analysis.density_bin_a),
                        x_A=substrate_axes["x"], y_A=substrate_axes["y"], z_A=substrate_axes["z"],
                    )
                    outputs.append(path)
                for first_index, first in enumerate(systems):
                    for second in systems[first_index + 1:]:
                        if first not in substrate_systems or second not in substrate_systems:
                            continue
                        difference = substrate_systems[second] - substrate_systems[first]
                        first_replicas, second_replicas = substrate_replicas[first], substrate_replicas[second]
                        if first_replicas.shape[0] >= 3 and second_replicas.shape[0] >= 3:
                            first_sem = first_replicas.std(axis=0, ddof=1) / math.sqrt(first_replicas.shape[0])
                            second_sem = second_replicas.std(axis=0, ddof=1) / math.sqrt(second_replicas.shape[0])
                            difference_ci = 1.96 * np.sqrt(first_sem * first_sem + second_sem * second_sem)
                        else:
                            difference_ci = np.full_like(difference, np.nan)
                        path = substrate_dir / f"difference.{second}-minus-{first}.npz"
                        np.savez_compressed(
                            path, rho_difference=difference,
                            rho_difference_ci95_low=difference - difference_ci,
                            rho_difference_ci95_high=difference + difference_ci,
                            bin_A=np.asarray(config.analysis.density_bin_a),
                            x_A=substrate_axes["x"], y_A=substrate_axes["y"], z_A=substrate_axes["z"],
                        )
                        outputs.append(path)
    return outputs


def run_pose_stage(
    config: AnalysisConfig,
    phase: str = "all",
    *,
    case_index: int | None = None,
    force: bool = False,
) -> PoseStageResult:
    outputs_by_run: dict[str, list[Path]] = defaultdict(list)
    metadata_by_run: dict[str, dict[str, Any]] = defaultdict(dict)
    warnings_by_run: dict[str, list[str]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    aggregate_outputs: list[Path] = []
    wants_hydration = bool(set(config.analysis.tasks) & {"pose-hydration", "compare-hydration"})
    wants_comparison = "compare-hydration" in config.analysis.tasks
    md_datasets = [item for item in config.datasets if item.kind == "md"]
    selected = md_datasets
    if case_index is not None:
        if case_index < 0 or case_index >= len(md_datasets):
            raise ValueError(f"pose stage case index out of range: {case_index}")
        selected = [md_datasets[case_index]]

    def capture(dataset: DatasetSpec, operation: Any, failed_phase: str) -> None:
        try:
            outputs_by_run[dataset.run_id].extend(operation())
        except Exception as exc:
            failures.append({"run_id": dataset.run_id, "phase": failed_phase, "error": f"{type(exc).__name__}: {exc}"})
            warnings_by_run[dataset.run_id].append(f"pose_{failed_phase}_failed: {exc}")
            failure_path = config.output.root / dataset.run_id / "poses" / f"pose_{failed_phase}_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1, "status": "failed", "run_id": dataset.run_id,
                        "phase": failed_phase, "error": f"{type(exc).__name__}: {exc}",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            outputs_by_run[dataset.run_id].append(failure_path)

    if phase in {"all", "features"}:
        for dataset in selected:
            capture(dataset, lambda dataset=dataset: extract_pose_training(config, dataset, force=force), "features")
        if phase == "features":
            return PoseStageResult(dict(outputs_by_run), dict(metadata_by_run), dict(warnings_by_run), [], failures)
    if phase in {"all", "cluster"}:
        try:
            aggregate_outputs.extend(fit_pose_models(config, force=force))
        except Exception as exc:
            failures.append({"run_id": "aggregate", "phase": "cluster", "error": f"{type(exc).__name__}: {exc}"})
            failure_path = _stage_root(config) / "pose_cluster_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failures[-1], indent=2) + "\n", encoding="utf-8")
            aggregate_outputs.append(failure_path)
            if phase == "cluster":
                return PoseStageResult(dict(outputs_by_run), dict(metadata_by_run), dict(warnings_by_run), aggregate_outputs, failures)
    if phase in {"all", "hydrate"} and wants_hydration and not any(item["phase"] == "cluster" for item in failures):
        for dataset in selected:
            capture(dataset, lambda dataset=dataset: hydrate_pose_case(config, dataset, force=force), "hydrate")
    if phase in {"all", "finalize"} and wants_comparison:
        try:
            aggregate_outputs.extend(aggregate_pose(config))
        except Exception as exc:
            failures.append({"run_id": "aggregate", "phase": "finalize", "error": f"{type(exc).__name__}: {exc}"})
            failure_path = _stage_root(config) / "pose_finalize_failure.json"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failures[-1], indent=2) + "\n", encoding="utf-8")
            aggregate_outputs.append(failure_path)
    for dataset in selected:
        training_meta = _training_paths(config, dataset)[1]
        hydration_meta = config.output.root / dataset.run_id / "poses" / "pose_manifest.json"
        payload: dict[str, Any] = {}
        if training_meta.exists():
            payload["training"] = json.loads(training_meta.read_text(encoding="utf-8"))
        if hydration_meta.exists():
            payload["hydration"] = json.loads(hydration_meta.read_text(encoding="utf-8"))
        if payload:
            metadata_by_run[dataset.run_id] = payload
    return PoseStageResult(dict(outputs_by_run), dict(metadata_by_run), dict(warnings_by_run), aggregate_outputs, failures)
