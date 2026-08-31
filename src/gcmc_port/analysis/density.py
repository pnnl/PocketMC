from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from .models import AnalysisConfig, RunResult


# Retained from the existing xyz2cube.py/C implementation for numeric regression.
BOHR_RADIUS_A = 0.529121


def _points_a(result: RunResult) -> np.ndarray:
    inside = [item.point_nm for frame in result.frames for item in frame.molecules if item.inside]
    # Never turn a failed/empty cavity selection into a convincing bulk-water
    # density map.  An empty selection is represented by an explicit zero grid.
    return np.asarray(inside, dtype=float).reshape((-1, 3)) * 10.0


def _density_support_a(config: AnalysisConfig, result: RunResult, points: np.ndarray) -> np.ndarray:
    if points.size:
        return points
    mask_points = np.asarray(result.metadata.get("cavity_mask_points_nm", []), dtype=float)
    if mask_points.size:
        return mask_points.reshape((-1, 3)) * 10.0
    center = np.asarray(result.metadata.get("cavity_center_nm", []), dtype=float)
    if center.size == 3:
        radius_a = max(float(config.cavity.radius_nm) * 10.0, float(config.analysis.density_bin_a))
        offsets = np.asarray(
            [
                (-radius_a, 0.0, 0.0), (radius_a, 0.0, 0.0),
                (0.0, -radius_a, 0.0), (0.0, radius_a, 0.0),
                (0.0, 0.0, -radius_a), (0.0, 0.0, radius_a),
            ],
            dtype=float,
        )
        return center.reshape(1, 3) * 10.0 + offsets
    overlay = result.metadata.get("substrate_overlay", {})
    overlay_points = np.asarray(overlay.get("positions_A", []), dtype=float) if isinstance(overlay, dict) else np.zeros((0, 3))
    if overlay_points.size:
        return overlay_points.reshape((-1, 3))
    raise ValueError("No inside-cavity points or cavity coordinates are available for density generation")


def _write_substrate_overlay(path: Path, result: RunResult) -> Path | None:
    overlay = result.metadata.get("substrate_overlay")
    if not isinstance(overlay, dict):
        path.unlink(missing_ok=True)
        return None
    positions = np.asarray(overlay.get("positions_A", []), dtype=float)
    if positions.size == 0:
        path.unlink(missing_ok=True)
        return None
    np.savez_compressed(
        path,
        positions_A=positions.reshape((-1, 3)),
        atom_names=np.asarray(overlay.get("atom_names", []), dtype="U16"),
        elements=np.asarray(overlay.get("elements", []), dtype="U4"),
        resnames=np.asarray(overlay.get("resnames", []), dtype="U16"),
        resids=np.asarray(overlay.get("resids", []), dtype=int),
        atom_indices_0based=np.asarray(overlay.get("atom_indices_0based", []), dtype=int),
        coordinate_frame=np.asarray(str(overlay.get("coordinate_frame", "analysis-reference"))),
    )
    return path


def _write_cavity_overlay(path: Path, config: AnalysisConfig, result: RunResult) -> Path | None:
    """Persist the exact analysis-frame cavity independently of the private analysis cache."""
    if config.cavity.mode == "mask":
        points = np.asarray(result.metadata.get("cavity_mask_points_nm", []), dtype=float).reshape((-1, 3))
        if not points.size:
            path.unlink(missing_ok=True)
            return None
        spacing_nm = 0.0
        if config.cavity.mask is not None:
            try:
                from gcmc_port.cavity import load_voxel_mask

                spacing_nm = float(
                    load_voxel_mask(
                        config.cavity.mask,
                        config.cavity.meta,
                        membership_padding=config.cavity.membership_padding_nm,
                    ).dx
                )
            except (OSError, ValueError, TypeError):
                spacing_nm = 0.0
        if spacing_nm <= 0.0 and len(points) > 1:
            try:
                from scipy.spatial import cKDTree

                distances, _indices = cKDTree(points).query(points, k=2)
                nearest = np.asarray(distances, dtype=float)[:, 1]
                nearest = nearest[np.isfinite(nearest) & (nearest > 1.0e-8)]
                spacing_nm = float(np.median(nearest)) if nearest.size else 0.0
            except (TypeError, ValueError):
                spacing_nm = 0.0
        np.savez_compressed(
            path,
            mode=np.asarray("mask"),
            points_A=points * 10.0,
            voxel_spacing_A=np.asarray(max(spacing_nm * 10.0, 1.0e-6)),
            coordinate_frame=np.asarray("analysis-reference"),
            source_run=np.asarray(result.dataset.run_id),
        )
        return path
    center = np.asarray(result.metadata.get("cavity_center_nm", []), dtype=float).reshape((-1,))
    if center.size != 3:
        path.unlink(missing_ok=True)
        return None
    np.savez_compressed(
        path,
        mode=np.asarray("sphere"),
        center_A=center * 10.0,
        radius_A=np.asarray(float(config.cavity.radius_nm) * 10.0),
        coordinate_frame=np.asarray("analysis-reference"),
        source_run=np.asarray(result.dataset.run_id),
    )
    return path


def _axes(points: np.ndarray, bin_a: float, pad_a: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = points.min(axis=0) - pad_a
    high = points.max(axis=0) + pad_a
    counts = np.maximum(1, np.floor((high - low) / bin_a).astype(int) + 1)
    return tuple(low[index] + (np.arange(int(counts[index])) + 0.5) * bin_a for index in range(3))  # type: ignore[return-value]


def legacy_gaussian_density(
    points: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    sigma_a: float,
    cutoff_sigma: float,
) -> np.ndarray:
    """Deposit the same spherical, cell-center Gaussian kernel as xyz2cube.py."""
    rho = np.zeros(tuple(len(axis) for axis in axes), dtype=float)
    cutoff_a = cutoff_sigma * sigma_a
    cutoff2 = cutoff_a * cutoff_a
    inverse_two_sigma2 = 1.0 / (2.0 * sigma_a * sigma_a)
    for point in points:
        selections = [np.flatnonzero(np.abs(axis - point[index]) <= cutoff_a) for index, axis in enumerate(axes)]
        if any(indices.size == 0 for indices in selections):
            continue
        dx = axes[0][selections[0]][:, None, None] - point[0]
        dy = axes[1][selections[1]][None, :, None] - point[1]
        dz = axes[2][selections[2]][None, None, :] - point[2]
        distance2 = dx * dx + dy * dy + dz * dz
        values = np.exp(-distance2 * inverse_two_sigma2)
        values[distance2 > cutoff2] = 0.0
        rho[np.ix_(selections[0], selections[1], selections[2])] += values
    return rho


def _write_projection_csv(path: Path, horizontal: np.ndarray, vertical: np.ndarray, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["vertical_A/horizontal_A", *[f"{value:.8f}" for value in horizontal]])
        for coordinate, row in zip(vertical, image):
            writer.writerow([f"{coordinate:.8f}", *[f"{value:.12e}" for value in row]])


def _write_cube(path: Path, rho: np.ndarray, axes: tuple[np.ndarray, np.ndarray, np.ndarray], bin_a: float) -> None:
    origin_a = [float(axis[0] - 0.5 * bin_a) for axis in axes]
    to_bohr = 1.0 / BOHR_RADIUS_A
    lines = [
        "PocketMC molecule density",
        "Values are probability density in Angstrom^-3",
        f"{1:5d}{origin_a[0] * to_bohr:13.6f}{origin_a[1] * to_bohr:13.6f}{origin_a[2] * to_bohr:13.6f}",
        f"{rho.shape[0]:5d}{bin_a * to_bohr:13.6f}{0.0:13.6f}{0.0:13.6f}",
        f"{rho.shape[1]:5d}{0.0:13.6f}{bin_a * to_bohr:13.6f}{0.0:13.6f}",
        f"{rho.shape[2]:5d}{0.0:13.6f}{0.0:13.6f}{bin_a * to_bohr:13.6f}",
        f"{1:5d}{0.0:13.6f}{origin_a[0] * to_bohr:13.6f}{origin_a[1] * to_bohr:13.6f}{origin_a[2] * to_bohr:13.6f}",
    ]
    values = rho.reshape(-1)
    for start in range(0, values.size, 6):
        lines.append("".join(f"{float(value):13.5e}" for value in values[start : start + 6]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hpd_thresholds(rho: np.ndarray, probabilities: tuple[float, ...] = (0.90, 0.70, 0.50, 0.30)) -> dict[float, float]:
    values = np.sort(np.asarray(rho, dtype=float).reshape(-1))[::-1]
    positive = values[values > 0]
    if positive.size == 0:
        return {probability: 0.0 for probability in probabilities}
    cumulative = np.cumsum(positive)
    total = float(cumulative[-1])
    output: dict[float, float] = {}
    for probability in probabilities:
        index = int(np.searchsorted(cumulative, probability * total, side="left"))
        output[probability] = float(positive[min(index, positive.size - 1)])
    return output


def build_density(config: AnalysisConfig, result: RunResult, density_dir: Path) -> list[Path]:
    density_dir.mkdir(parents=True, exist_ok=True)
    points = _points_a(result)
    opts = config.analysis
    support = _density_support_a(config, result, points)
    axes = _axes(support, opts.density_bin_a, opts.density_cutoff_sigma * opts.density_sigma_a)
    smoothed = legacy_gaussian_density(points, axes, opts.density_sigma_a, opts.density_cutoff_sigma)
    volume_a3 = opts.density_bin_a ** 3
    probability = smoothed / max(float(smoothed.sum()) * volume_a3, 1e-30)
    occupancy = probability * (float(points.shape[0]) / max(float(len(result.frames)), 1.0))
    selected = occupancy if opts.density_quantity == "occupancy" else probability
    projections = {
        "xy": selected.sum(axis=2).T * opts.density_bin_a,
        "xz": selected.sum(axis=1).T * opts.density_bin_a,
        "yz": selected.sum(axis=0).T * opts.density_bin_a,
    }
    npz_path = density_dir / "density_maps.npz"
    npz_payload = {
        "rho": selected,
        "rho_probability": probability,
        "rho_occupancy": occupancy,
        "x_A": axes[0],
        "y_A": axes[1],
        "z_A": axes[2],
        "bin_A": np.asarray(opts.density_bin_a),
        "xy_projection": projections["xy"],
        "xz_projection": projections["xz"],
        "yz_projection": projections["yz"],
    }
    np.savez_compressed(npz_path, **npz_payload)
    raw_npz_path = density_dir / "density_maps.raw.npz"
    np.savez(raw_npz_path, **npz_payload)
    outputs = [npz_path, raw_npz_path]
    overlay_path = _write_substrate_overlay(density_dir / "substrate_overlay.npz", result)
    if overlay_path is not None:
        outputs.append(overlay_path)
    cavity_overlay_path = _write_cavity_overlay(density_dir / "cavity_overlay.npz", config, result)
    if cavity_overlay_path is not None:
        outputs.append(cavity_overlay_path)
    plane_axes = {"xy": (axes[0], axes[1]), "xz": (axes[0], axes[2]), "yz": (axes[1], axes[2])}
    for plane, image in projections.items():
        csv_path = density_dir / f"{plane}_projection.csv"
        _write_projection_csv(csv_path, plane_axes[plane][0], plane_axes[plane][1], image)
        outputs.append(csv_path)
    cube_path = density_dir / "density.cube"
    _write_cube(cube_path, probability, axes, opts.density_bin_a)
    outputs.append(cube_path)
    thresholds = hpd_thresholds(probability)
    metadata = {
        "schema_version": 1,
        "run_id": result.dataset.run_id,
        "kind": result.dataset.kind,
        "quantity_shown": opts.density_quantity,
        "units": {"probability": "A^-3", "occupancy": "molecules A^-3 per analyzed frame"},
        "point_count": int(points.shape[0]),
        "empty_inside_selection": bool(points.shape[0] == 0),
        "frame_count": len(result.frames),
        "bin_A": opts.density_bin_a,
        "sigma_A": opts.density_sigma_a,
        "grid_shape": list(probability.shape),
        "probability_integral": float(probability.sum() * volume_a3),
        "mean_occupancy_integral": float(occupancy.sum() * volume_a3),
        "hpd_thresholds": {str(key): value for key, value in thresholds.items()},
        "scientific_scope": (
            "Accepted-state occurrence density; not an equilibrium probability density."
            if result.dataset.kind == "pocketmc"
            else "Frame-normalized MD occupancy density."
        ),
    }
    meta_path = density_dir / "density_maps.meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    outputs.append(meta_path)
    result.metadata["density"] = metadata
    return outputs
