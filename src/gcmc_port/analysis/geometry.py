from __future__ import annotations

import math
from itertools import product
import re
from typing import Any, Iterable

import numpy as np


def kabsch_transform(mobile: Any, target: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_array = np.asarray(mobile, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if mobile_array.shape != target_array.shape or mobile_array.ndim != 2 or mobile_array.shape[1] != 3:
        raise ValueError("Alignment selections must have matching (N, 3) coordinates")
    if mobile_array.shape[0] < 3:
        raise ValueError("At least three atoms are required for rigid alignment")
    mobile_center = mobile_array.mean(axis=0)
    target_center = target_array.mean(axis=0)
    covariance = (mobile_array - mobile_center).T @ (target_array - target_center)
    u, _singular, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, mobile_center, target_center


def apply_transform(points: Any, transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if transform is None:
        return array.copy()
    rotation, mobile_center, target_center = transform
    return (array - mobile_center) @ rotation + target_center


def apply_inverse_transform(points: Any, transform: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Map aligned/reference-frame coordinates back into the current mobile frame."""
    array = np.asarray(points, dtype=float)
    if transform is None:
        return array.copy()
    rotation, mobile_center, target_center = transform
    return (array - target_center) @ rotation.T + mobile_center


def minimum_image(point: Any, anchor: Any, box: Any | None) -> np.ndarray:
    output = np.asarray(point, dtype=float).copy()
    return minimum_image_points(output.reshape(1, 3), anchor, box)[0]


def minimum_image_points(points: Any, anchor: Any, box: Any | None) -> np.ndarray:
    """Place Cartesian points in their nearest lattice images around *anchor*.

    The neighboring-lattice search is required for skewed triclinic cells; a
    component-wise fractional wrap is only guaranteed to be nearest for an
    orthorhombic box.
    """
    output = np.asarray(points, dtype=float).copy()
    if output.size == 0:
        return output.reshape((-1, 3))
    if output.ndim != 2 or output.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    reference = np.asarray(anchor, dtype=float)
    if box is None:
        return output
    cell = _cell_vectors(box)
    if cell is None:
        return output
    delta = output - reference
    try:
        fractional = delta @ np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return output
    # Component-wise rounding is only exact for orthorhombic cells.  For a
    # skewed cell, inspect the neighboring lattice translations and select the
    # image with the shortest Cartesian displacement.
    nearest_integer = np.round(fractional).astype(int)
    offsets = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=int)
    candidates = nearest_integer[:, None, :] + offsets[None, :, :]
    displacements = (fractional[:, None, :] - candidates) @ cell
    distances2 = np.einsum("nij,nij->ni", displacements, displacements)
    nearest = np.argmin(distances2, axis=1)
    return reference + displacements[np.arange(output.shape[0]), nearest]


def _cell_vectors(box: Any) -> np.ndarray | None:
    """Return row-wise triclinic cell vectors from MDAnalysis or matrix box data."""
    values = np.asarray(box, dtype=float)
    if values.shape == (3, 3):
        cell = values.copy()
    else:
        flat = values.reshape(-1)
        if flat.size < 3:
            return None
        lengths = flat[:3]
        if np.any(lengths <= 0) or not np.all(np.isfinite(lengths)):
            return None
        if flat.size < 6 or not np.all(np.isfinite(flat[3:6])):
            cell = np.diag(lengths)
        else:
            alpha, beta, gamma = np.deg2rad(flat[3:6])
            sin_gamma = float(np.sin(gamma))
            if abs(sin_gamma) < 1.0e-12:
                return None
            a, b, c = (float(value) for value in lengths)
            bx = b * float(np.cos(gamma))
            by = b * sin_gamma
            cx = c * float(np.cos(beta))
            cy = c * (float(np.cos(alpha)) - float(np.cos(beta)) * float(np.cos(gamma))) / sin_gamma
            cz2 = c * c - cx * cx - cy * cy
            if cz2 < -1.0e-8:
                return None
            cell = np.asarray([[a, 0.0, 0.0], [bx, by, 0.0], [cx, cy, math.sqrt(max(cz2, 0.0))]])
    if not np.all(np.isfinite(cell)) or abs(float(np.linalg.det(cell))) < 1.0e-12:
        return None
    return cell


def parse_residue_token(token: str) -> tuple[int | None, str | None]:
    text = token.strip()
    match = re.fullmatch(r"(\d+)([A-Za-z0-9_+\-]+)", text)
    if match:
        return int(match.group(1)), match.group(2).upper()
    match = re.fullmatch(r"([A-Za-z0-9_+\-]+):(\d+)", text)
    if match:
        return int(match.group(2)), match.group(1).upper()
    if text.isdigit():
        return int(text), None
    return None, text.upper() if text else None


def safe_mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")
