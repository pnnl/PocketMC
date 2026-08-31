from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from gcmc_port.cavity import VoxelMask, infer_mask_meta_path
from gcmc_port.gro import parse_atom_line


def mask_from_first_trajectory_frame(mask: VoxelMask, trajectory: Path | None) -> VoxelMask:
    if trajectory is None:
        return mask
    lines = trajectory.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Mask trajectory has no complete GRO frame: {trajectory}")
    try:
        count = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid mask trajectory GRO atom count: {trajectory}") from exc
    if len(lines) < count + 3:
        raise ValueError(f"Incomplete first mask trajectory frame: {trajectory}")
    points = np.asarray(
        [(atom.x, atom.y, atom.z) for atom in (parse_atom_line(line) for line in lines[2 : 2 + count])],
        dtype=float,
    )
    if points.shape[0] != mask.point_count:
        raise ValueError(
            f"Mask trajectory point count ({points.shape[0]}) does not match mask ({mask.point_count}): {trajectory}"
        )
    return VoxelMask(
        points=points,
        dx=mask.dx,
        reference_point=tuple(float(value) for value in points.mean(axis=0)),
        effective_volume=mask.effective_volume,
        membership_padding=mask.membership_padding,
        probe_radius=mask.probe_radius,
        source_gro=mask.source_gro,
        exclude_residues=mask.exclude_residues,
    )


def resolved_mask_meta_path(mask_path: Path | None, meta_path: Path | None) -> Path | None:
    if meta_path is not None:
        return meta_path.expanduser().resolve()
    if mask_path is None:
        return None
    return infer_mask_meta_path(mask_path).expanduser().resolve()


def resolve_mask_source_path(
    *,
    mask: VoxelMask,
    mask_path: Path | None,
    meta_path: Path | None,
    build_source: Path | None,
    run_dir: Path,
    config_dir: Path,
) -> Path | None:
    """Resolve a mask's coordinate source, including moved HPC directory trees."""
    raw_values: list[str | Path] = []
    # An explicit analysis override is authoritative.  Metadata is consulted
    # only when the user did not supply one.
    if build_source is not None:
        raw_values.append(build_source)
    elif mask.source_gro:
        raw_values.append(mask.source_gro)
    resolved_meta = resolved_mask_meta_path(mask_path, meta_path)
    roots = [
        resolved_meta.parent if resolved_meta is not None else None,
        mask_path.parent if mask_path is not None else None,
        run_dir,
        config_dir,
    ]
    checked: set[Path] = set()
    for raw in raw_values:
        path = Path(raw).expanduser()
        if path.is_absolute() and path.exists():
            return path.resolve()
        candidates = [] if path.is_absolute() else [root / path for root in roots if root is not None]
        # A result tree is often copied between workstations and HPC storage.
        # If an old absolute path no longer exists, retry its basename beside
        # the mask, case, and analysis config.
        if path.is_absolute() and not path.exists():
            candidates.extend(root / path.name for root in roots if root is not None)
        found: list[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in checked:
                continue
            checked.add(resolved)
            if resolved.exists():
                found.append(resolved)
        if path.is_absolute() and len(found) > 1:
            raise ValueError(
                "Ambiguous relocated cavity mask source "
                f"{path}: " + ", ".join(str(item) for item in found)
            )
        if found:
            return found[0]
    return None


def mask_dependency_paths(
    *,
    mask_path: Path | None,
    meta_path: Path | None,
    build_source: Path | None,
    run_dir: Path,
    config_dir: Path,
    membership_padding: float,
) -> tuple[Path, ...]:
    """Return implicit mask inputs that must participate in cache fingerprints."""
    paths: list[Path] = []
    resolved_meta = resolved_mask_meta_path(mask_path, meta_path)
    if resolved_meta is not None:
        paths.append(resolved_meta)
    if build_source is not None:
        paths.append(build_source.expanduser().resolve())
    if mask_path is None or not mask_path.exists():
        return tuple(dict.fromkeys(paths))
    try:
        from gcmc_port.cavity import load_voxel_mask

        mask = load_voxel_mask(mask_path, meta_path, membership_padding=membership_padding)
        source = resolve_mask_source_path(
            mask=mask,
            mask_path=mask_path,
            meta_path=meta_path,
            build_source=build_source,
            run_dir=run_dir,
            config_dir=config_dir,
        )
        if source is not None:
            paths.append(source)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return tuple(dict.fromkeys(path.resolve() for path in paths))
