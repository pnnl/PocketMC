from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from gcmc_port.cavity import VoxelMask, load_voxel_mask

from .anchors import select_mda_anchor
from .geometry import apply_inverse_transform, apply_transform, kabsch_transform, minimum_image, minimum_image_points
from .masking import mask_from_first_trajectory_frame, resolve_mask_source_path
from .models import AnalysisConfig, DatasetSpec, FrameRecord, MoleculeFrame, RunResult
from .residue_mapping import resolve_residue_mapping


def _require_mdanalysis():
    try:
        import MDAnalysis as mda
    except Exception as exc:
        raise RuntimeError(f"MDAnalysis is required for MD trajectory analysis: {exc}") from exc
    return mda


def _length_to_nm(universe: Any) -> float:
    units = getattr(universe.trajectory, "units", {}) or {}
    unit = str(units.get("length", "nm") or "nm").lower()
    if unit in {"a", "angstrom", "angstroem", "ångström"}:
        return 0.1
    return 1.0


def _residue_uid(residue: Any) -> str:
    segid = str(getattr(residue, "segid", "")).strip()
    prefix = f"{segid}:" if segid else ""
    return f"{prefix}{int(residue.resid)}{str(residue.resname).upper()}"


def _whole_positions_nm(universe: Any, factor: float) -> np.ndarray:
    try:
        positions = universe.atoms.unwrap(compound="fragments", reference="cog", inplace=False)
    except Exception:
        positions = universe.atoms.positions
    return np.asarray(positions, dtype=float) * factor


def _representative_point_nm(
    residue: Any,
    config: AnalysisConfig,
    factor: float,
    whole_positions_nm: np.ndarray | None = None,
) -> np.ndarray:
    molecule = config.molecule
    if molecule.point_mode == "atom":
        for name in molecule.atom_names:
            selected = residue.atoms.select_atoms(f"name {name}")
            if selected.n_atoms:
                if whole_positions_nm is not None:
                    return np.asarray(whole_positions_nm[int(selected.indices[0])], dtype=float)
                return np.asarray(selected.positions[0], dtype=float) * factor
        raise ValueError(f"{_residue_uid(residue)} has none of the requested representative atoms: {molecule.atom_names}")
    group = residue.atoms
    if molecule.atom_names:
        names = " ".join(molecule.atom_names)
        selected = group.select_atoms(f"name {names}")
        if selected.n_atoms:
            group = selected
    positions = (
        np.asarray(whole_positions_nm[group.indices], dtype=float)
        if whole_positions_nm is not None
        else np.asarray(group.positions, dtype=float) * factor
    )
    if molecule.point_mode == "cog":
        return positions.mean(axis=0)
    try:
        masses = np.asarray(group.masses, dtype=float)
    except Exception as exc:
        raise ValueError(f"COM requested for {_residue_uid(residue)}, but masses are unavailable") from exc
    if masses.size != positions.shape[0] or not np.all(np.isfinite(masses)) or float(masses.sum()) <= 0:
        raise ValueError(f"COM requested for {_residue_uid(residue)}, but valid positive masses are unavailable")
    return np.average(positions, axis=0, weights=masses)


def _select_anchor(universe: Any, token: str, atom_names: tuple[str, ...]) -> Any:
    group, resolution = select_mda_anchor(universe, token, atom_names, context="Cavity")
    if resolution.warning:
        print(f"[anchor] {resolution.warning}", flush=True)
    return group


def _dimensions_nm(universe: Any, factor: float) -> np.ndarray | None:
    dimensions = getattr(universe.trajectory.ts, "dimensions", None)
    if dimensions is None:
        return None
    output = np.asarray(dimensions, dtype=float).copy()
    output[:3] *= factor
    return output


def _load_reference(mda: Any, dataset: DatasetSpec, universe: Any) -> tuple[Any, bool]:
    if dataset.reference is None or not dataset.reference.exists():
        universe.trajectory[0]
        return universe, False
    try:
        try:
            return mda.Universe(str(dataset.reference), convert_units=False), True
        except TypeError:
            return mda.Universe(str(dataset.reference)), True
    except Exception:
        try:
            return mda.Universe(str(dataset.topology), str(dataset.reference), convert_units=False), True
        except TypeError:
            return mda.Universe(str(dataset.topology), str(dataset.reference)), True


def _matching_groups(mobile: Any, target: Any, selection: str) -> tuple[Any, Any]:
    mobile_group = mobile.select_atoms(selection)
    target_group = target.select_atoms(selection)
    if mobile_group.n_atoms < 3 or target_group.n_atoms < 3:
        raise ValueError(f"Alignment selection matched fewer than three atoms: {selection}")
    if mobile_group.n_atoms != target_group.n_atoms or list(mobile_group.names) != list(target_group.names):
        raise ValueError(f"Reference and trajectory alignment selections do not match: {selection}")
    return mobile_group, target_group


def _anchor_center_nm(universe: Any, config: AnalysisConfig, positions_nm: np.ndarray, box_nm: Any) -> np.ndarray:
    anchor = _select_anchor(universe, config.cavity.anchor, config.cavity.anchor_atoms)
    points = minimum_image_points(positions_nm[anchor.indices], positions_nm[int(anchor.indices[0])], box_nm)
    return points.mean(axis=0)


def _compact_fit_positions(points_nm: Any, box_nm: Any) -> np.ndarray:
    """Return one internally consistent periodic image for a fit selection.

    GRO/reference files do not always carry enough bond information for
    ``AtomGroup.unwrap``.  Imaging every selected atom around the first one
    keeps a local pocket/anchor selection whole before Kabsch fitting.  The
    same operation is applied to mobile and target selections.
    """
    points = np.asarray(points_nm, dtype=float)
    if points.size == 0:
        return points.reshape((-1, 3))
    return minimum_image_points(points, points[0], box_nm)


def _fit_positions_nm(
    points_nm: Any,
    box_nm: Any,
    *,
    compact: bool,
    anchor_nm: Any | None = None,
) -> np.ndarray:
    """Prepare fit coordinates without changing an already-whole global shape.

    Local selections may straddle a periodic boundary and therefore need to be
    compacted.  Global protein selections come from ``_whole_positions_nm`` and
    must only be translated as a rigid body: atom-wise minimum imaging can fold
    a protein whose diameter exceeds half a box length.  When an anchor is
    supplied, move the prepared selection by one lattice translation so its
    center shares the anchor's periodic image.
    """
    points = np.asarray(points_nm, dtype=float)
    prepared = _compact_fit_positions(points, box_nm) if compact else points.copy()
    if prepared.size == 0 or anchor_nm is None:
        return prepared.reshape((-1, 3))
    center = prepared.mean(axis=0)
    image_center = minimum_image(center, np.asarray(anchor_nm, dtype=float), box_nm)
    return prepared + (image_center - center)


def _fallback_alignment_groups(mobile: Any, target: Any) -> tuple[Any, Any, str]:
    for selection, description in (
        ("protein and backbone", "fallback protein backbone"),
        ("protein and not name H*", "fallback all protein heavy atoms"),
    ):
        try:
            first, second = _matching_groups(mobile, target, selection)
            return first, second, description
        except ValueError:
            continue
    raise ValueError("Alignment requires at least three matching protein atoms")


def _matching_atom_indices(
    source: Any,
    target: Any,
    source_indices: Any,
    target_selection: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Match a source subset to target atoms without relying on global indices.

    Mask build structures and MD references commonly contain different water
    counts.  Their protein atom indices therefore need not be globally
    interchangeable even though the pocket atoms themselves are identical.
    """
    source_atoms = source.atoms[np.asarray(source_indices, dtype=int)]
    target_atoms = target.select_atoms(target_selection)

    def key(atom: Any, *, segmented: bool) -> tuple[Any, ...]:
        identity: tuple[Any, ...] = (
            int(atom.resid),
            str(atom.resname).upper(),
            str(atom.name).upper(),
        )
        if not segmented:
            return identity
        return (
            str(getattr(atom, "segid", "")).strip(),
            str(getattr(atom, "chainID", "")).strip(),
            *identity,
        )

    for segmented in (True, False):
        if not segmented:
            source_counts: dict[tuple[Any, ...], int] = {}
            target_counts: dict[tuple[Any, ...], int] = {}
            for atom in source_atoms:
                plain = key(atom, segmented=False)
                source_counts[plain] = source_counts.get(plain, 0) + 1
            for atom in target_atoms:
                plain = key(atom, segmented=False)
                target_counts[plain] = target_counts.get(plain, 0) + 1
        available: dict[tuple[Any, ...], list[int]] = {}
        for atom in target_atoms:
            atom_key = key(atom, segmented=segmented)
            if not segmented and target_counts.get(atom_key, 0) != 1:
                continue
            available.setdefault(atom_key, []).append(int(atom.index))
        matched_source: list[int] = []
        matched_target: list[int] = []
        for atom in source_atoms:
            atom_key = key(atom, segmented=segmented)
            if not segmented and source_counts.get(atom_key, 0) != 1:
                continue
            candidates = available.get(atom_key, [])
            if not candidates:
                continue
            matched_source.append(int(atom.index))
            matched_target.append(candidates.pop(0))
        if len(matched_source) >= 3:
            return np.asarray(matched_source, dtype=int), np.asarray(matched_target, dtype=int)
    return np.zeros(0, dtype=int), np.zeros(0, dtype=int)


def _canonicalize_mask(
    mda: Any,
    config: AnalysisConfig,
    dataset: DatasetSpec,
    universe: Any,
    reference: Any,
    mask: VoxelMask,
) -> tuple[VoxelMask, dict[str, Any]]:
    """Map a mask from its declared build structure into the analysis reference frame."""
    explicit_mask_frame = config.cavity.mask_trajectory is not None
    source: Any | None = None
    close_source = False
    source_path: Path | None = None
    if explicit_mask_frame:
        mask = mask_from_first_trajectory_frame(mask, config.cavity.mask_trajectory)
    else:
        source_path = resolve_mask_source_path(
            mask=mask,
            mask_path=config.cavity.mask,
            meta_path=config.cavity.meta,
            build_source=config.cavity.build_source,
            run_dir=dataset.run_dir,
            config_dir=config.config_path.parent,
        )
        if source_path is not None:
            try:
                try:
                    source = mda.Universe(str(source_path), convert_units=False)
                except TypeError:
                    source = mda.Universe(str(source_path))
            except Exception as exc:
                raise ValueError(f"{dataset.run_id}: could not load cavity mask source {source_path}: {exc}") from exc
            close_source = True
        elif mask.source_gro or config.cavity.build_source is not None:
            declared = (
                str(config.cavity.build_source)
                if config.cavity.build_source is not None
                else mask.source_gro
            )
            raise FileNotFoundError(
                f"{dataset.run_id}: the cavity mask source structure could not be resolved: {declared}"
            )

    reference_factor = _length_to_nm(reference)
    reference.trajectory[0]
    reference_all = _whole_positions_nm(reference, reference_factor)
    reference_box = _dimensions_nm(reference, reference_factor)
    points = np.asarray(mask.points, dtype=float).copy()
    mask_reference = np.asarray(mask.reference_point, dtype=float).copy()
    metadata: dict[str, Any]

    if source is None:
        # A legacy mask with no source declaration was normally generated from
        # previous.gro itself.  Do not repeat the trajectory->reference fit on
        # coordinates that are already in the reference frame.
        transform = None
        metadata = {
            "method": (
                "explicit mask-trajectory frame0 assumed in analysis-reference frame"
                if explicit_mask_frame
                else "mask assumed in analysis-reference frame (no declared source)"
            ),
            "source": str(config.cavity.mask_trajectory) if explicit_mask_frame else None,
            "source_image_shift_nm": [0.0, 0.0, 0.0],
            "fit_rmsd_nm": 0.0,
        }
    else:
        source_factor = _length_to_nm(source)
        source.trajectory[0]
        source_all = _whole_positions_nm(source, source_factor)
        source_box = _dimensions_nm(source, source_factor)
        try:
            source_anchor = _anchor_center_nm(source, config, source_all, source_box)
        except Exception:
            source_anchor = mask_reference
        nearest_reference = minimum_image(mask_reference, source_anchor, source_box)
        source_shift = nearest_reference - mask_reference
        points += source_shift
        mask_reference += source_shift

        if config.cavity.align_selection:
            source_group, reference_group = _matching_groups(source, reference, config.cavity.align_selection)
            description = f"custom selection: {config.cavity.align_selection}"
            compact_fit = False
        else:
            heavy = source.select_atoms("protein and not name H*")
            local_indices: np.ndarray = np.zeros(0, dtype=int)
            if heavy.n_atoms:
                heavy_positions = minimum_image_points(source_all[heavy.indices], mask_reference, source_box)
                local_indices = np.asarray(heavy.indices[np.linalg.norm(heavy_positions - mask_reference, axis=1) <= 1.0], dtype=int)
                matched_source, matched_reference = _matching_atom_indices(
                    source,
                    reference,
                    local_indices,
                    "protein and not name H*",
                )
                if matched_source.size >= 3:
                    source_group = source.atoms[matched_source]
                    reference_group = reference.atoms[matched_reference]
                    description = "automatic local protein heavy atoms within 1.0 nm of cavity mask"
                    compact_fit = True
                else:
                    source_group, reference_group, description = _fallback_alignment_groups(source, reference)
                    compact_fit = False
            else:
                source_group, reference_group, description = _fallback_alignment_groups(source, reference)
                compact_fit = False
        # Keep the fit selection and mask in the *same* source lattice image.
        # Otherwise a subsequent rotation turns their lattice-vector offset
        # into a non-lattice Cartesian offset that cannot be repaired in the
        # target box.
        source_fit_unshifted = _fit_positions_nm(
            source_all[source_group.indices], source_box, compact=compact_fit
        )
        source_fit = _fit_positions_nm(
            source_fit_unshifted,
            source_box,
            compact=False,
            anchor_nm=mask_reference,
        )
        fit_image_shift = source_fit.mean(axis=0) - source_fit_unshifted.mean(axis=0)
        target_fit = _fit_positions_nm(
            reference_all[reference_group.indices], reference_box, compact=compact_fit
        )
        transform = kabsch_transform(source_fit, target_fit)
        fitted = apply_transform(source_fit, transform)
        metadata = {
            "method": (
                "explicit mask-trajectory frame0 to analysis-reference Kabsch fit"
                if explicit_mask_frame
                else "mask source-structure to analysis-reference Kabsch fit"
            ),
            "source": "trajectory frame 0" if explicit_mask_frame else str(source_path),
            "selection": description,
            "source_image_shift_nm": source_shift.tolist(),
            "source_fit_image_shift_nm": fit_image_shift.tolist(),
            "fit_rmsd_nm": float(np.sqrt(np.mean(np.sum((fitted - target_fit) ** 2, axis=1)))),
        }

    canonical_points = apply_transform(points, transform)
    canonical_reference = apply_transform(mask_reference.reshape(1, 3), transform)[0]
    try:
        target_anchor = _anchor_center_nm(reference, config, reference_all, reference_box)
    except Exception:
        target_anchor = canonical_reference
    nearest_canonical = minimum_image(canonical_reference, target_anchor, reference_box)
    canonical_shift = nearest_canonical - canonical_reference
    canonical_points += canonical_shift
    canonical_reference += canonical_shift
    metadata["canonical_image_shift_nm"] = canonical_shift.tolist()
    metadata["canonical_reference_nm"] = canonical_reference.tolist()

    if close_source:
        close = getattr(source.trajectory, "close", None)
        if close is not None:
            close()
    return (
        VoxelMask(
            points=canonical_points,
            dx=mask.dx,
            reference_point=tuple(float(value) for value in canonical_reference),
            effective_volume=mask.effective_volume,
            membership_padding=mask.membership_padding,
            probe_radius=mask.probe_radius,
            source_gro=mask.source_gro,
            exclude_residues=mask.exclude_residues,
        ),
        metadata,
    )


def _analysis_alignment(
    config: AnalysisConfig,
    universe: Any,
    reference: Any,
    reference_all_nm: np.ndarray,
    reference_box_nm: Any,
    mask: VoxelMask | None,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    if config.cavity.align_selection:
        mobile, target = _matching_groups(universe, reference, config.cavity.align_selection)
        return (
            np.asarray(mobile.indices, dtype=int),
            _fit_positions_nm(
                reference_all_nm[target.indices],
                reference_box_nm,
                compact=False,
                anchor_nm=None if mask is None else mask.reference_point,
            ),
            f"custom selection: {config.cavity.align_selection}",
            False,
        )

    if mask is not None:
        heavy = reference.select_atoms("protein and not name H*")
        if heavy.n_atoms:
            positions = minimum_image_points(reference_all_nm[heavy.indices], mask.reference_point, reference_box_nm)
            local = np.asarray(heavy.indices[np.linalg.norm(positions - np.asarray(mask.reference_point), axis=1) <= 1.0], dtype=int)
            matched_reference, matched_mobile = _matching_atom_indices(
                reference,
                universe,
                local,
                "protein and not name H*",
            )
            if matched_reference.size >= 3:
                return (
                    matched_mobile,
                    _fit_positions_nm(
                        reference_all_nm[matched_reference],
                        reference_box_nm,
                        compact=True,
                        anchor_nm=mask.reference_point,
                    ),
                    "automatic local protein heavy atoms within 1.0 nm of cavity mask",
                    True,
                )
        mobile, target, description = _fallback_alignment_groups(universe, reference)
        return (
            np.asarray(mobile.indices, dtype=int),
            _fit_positions_nm(
                reference_all_nm[target.indices],
                reference_box_nm,
                compact=False,
                anchor_nm=mask.reference_point,
            ),
            description,
            False,
        )

    try:
        mobile_anchor = _select_anchor(universe, config.cavity.anchor, config.cavity.anchor_atoms)
        reference_anchor = _select_anchor(reference, config.cavity.anchor, config.cavity.anchor_atoms)
        if (
            mobile_anchor.n_atoms >= 3
            and mobile_anchor.n_atoms == reference_anchor.n_atoms
            and list(mobile_anchor.names) == list(reference_anchor.names)
        ):
            return (
                np.asarray(mobile_anchor.indices, dtype=int),
                _fit_positions_nm(
                    reference_all_nm[reference_anchor.indices], reference_box_nm, compact=True
                ),
                "cavity-anchor atoms (substrate-local sphere frame)",
                True,
            )
    except Exception:
        pass
    mobile, target, description = _fallback_alignment_groups(universe, reference)
    return (
        np.asarray(mobile.indices, dtype=int),
        _fit_positions_nm(reference_all_nm[target.indices], reference_box_nm, compact=False),
        description,
        False,
    )


def _element(atom: Any) -> str:
    try:
        value = str(atom.element).strip()
        if value:
            return value.title()
    except Exception:
        pass
    cleaned = "".join(character for character in str(atom.name) if character.isalpha()).upper()
    return cleaned[:2].title() if cleaned.startswith(("CL", "BR")) else (cleaned[:1] or "C")


def _substrate_overlay(
    config: AnalysisConfig,
    dataset: DatasetSpec,
    reference: Any,
    reference_all_nm: np.ndarray,
    reference_box_nm: Any,
    cavity_center_nm: np.ndarray,
) -> dict[str, Any] | None:
    selection = dataset.substrate_selection or config.substrate.selection
    if not selection:
        return None
    atoms = reference.select_atoms(selection)
    if atoms.n_atoms == 0:
        return None
    positions = np.asarray(reference_all_nm[atoms.indices], dtype=float).copy()
    # Keep each selected residue whole, then choose the image nearest the cavity.
    for residue in atoms.residues:
        selected_indices = [index for index, atom in enumerate(atoms) if int(atom.resindex) == int(residue.resindex)]
        if not selected_indices:
            continue
        local = positions[selected_indices]
        local = minimum_image_points(local, local[0], reference_box_nm)
        center = local.mean(axis=0)
        shift = minimum_image(center, cavity_center_nm, reference_box_nm) - center
        positions[selected_indices] = local + shift
    return {
        "positions_A": (positions * 10.0).tolist(),
        "atom_names": [str(atom.name) for atom in atoms],
        "elements": [_element(atom) for atom in atoms],
        "resnames": [str(atom.resname) for atom in atoms],
        "resids": [int(atom.resid) for atom in atoms],
        "atom_indices_0based": [int(atom.index) for atom in atoms],
        "selection": selection,
        "coordinate_frame": "analysis-reference",
    }


def read_md_dataset(config: AnalysisConfig, dataset: DatasetSpec) -> RunResult:
    mda = _require_mdanalysis()
    if dataset.topology is None:
        raise ValueError("MD dataset requires a topology")
    try:
        universe = mda.Universe(str(dataset.topology), str(dataset.trajectory), convert_units=False)
    except TypeError:
        universe = mda.Universe(str(dataset.topology), str(dataset.trajectory))
    factor = _length_to_nm(universe)
    residues = [residue for residue in universe.residues if str(residue.resname).upper() in config.molecule.resnames]
    if not residues:
        raise ValueError(f"No residues matched molecule.resnames={config.molecule.resnames}")
    protein = universe.select_atoms(config.cavity.protein_selection)
    if protein.n_atoms == 0:
        raise ValueError(f"protein selection matched no atoms: {config.cavity.protein_selection}")
    residue_mapping = resolve_residue_mapping(mda, universe, config)

    reference, close_reference = _load_reference(mda, dataset, universe)
    reference_factor = _length_to_nm(reference)
    reference.trajectory[0]
    reference_all_nm = _whole_positions_nm(reference, reference_factor)
    reference_box_nm = _dimensions_nm(reference, reference_factor)

    mask = None
    if config.cavity.mode == "mask":
        assert config.cavity.mask is not None
        mask = load_voxel_mask(
            config.cavity.mask,
            config.cavity.meta,
            membership_padding=config.cavity.membership_padding_nm,
        )
        mask, mask_alignment = _canonicalize_mask(mda, config, dataset, universe, reference, mask)
    else:
        mask_alignment = None
    align_indices, reference_positions, alignment_description, compact_alignment = _analysis_alignment(
        config, universe, reference, reference_all_nm, reference_box_nm, mask
    )
    anchor_group = None if mask is not None else _select_anchor(universe, config.cavity.anchor, config.cavity.anchor_atoms)
    if mask is not None:
        reference_cavity_center = np.asarray(mask.reference_point, dtype=float)
    else:
        reference_anchor = _select_anchor(reference, config.cavity.anchor, config.cavity.anchor_atoms)
        anchor_points = minimum_image_points(
            reference_all_nm[reference_anchor.indices], reference_all_nm[int(reference_anchor.indices[0])], reference_box_nm
        )
        reference_cavity_center = anchor_points.mean(axis=0)
    opts = config.analysis
    frames: list[FrameRecord] = []
    cavity_center_nm: tuple[float, float, float] | None = None
    eligible_index = 0
    for sequential_index, ts in enumerate(universe.trajectory):
        time_ps = float(getattr(ts, "time", sequential_index))
        if opts.start_ps is not None and time_ps < opts.start_ps - 1e-9:
            continue
        if opts.stop_ps is not None and time_ps > opts.stop_ps + 1e-9:
            break
        if eligible_index % opts.stride:
            eligible_index += 1
            continue
        eligible_index += 1
        box_nm = _dimensions_nm(universe, factor)
        whole_positions_nm = _whole_positions_nm(universe, factor)
        mobile_align = _fit_positions_nm(
            whole_positions_nm[align_indices], box_nm, compact=compact_alignment
        )
        transform = kabsch_transform(mobile_align, reference_positions)
        if mask is not None:
            anchor_nm = np.asarray(mask.reference_point, dtype=float)
            raw_anchor_nm = apply_inverse_transform(anchor_nm.reshape(1, 3), transform)[0]
        else:
            assert anchor_group is not None
            raw_anchor_points = minimum_image_points(
                whole_positions_nm[anchor_group.indices], whole_positions_nm[int(anchor_group.indices[0])], box_nm
            )
            raw_anchor_nm = raw_anchor_points.mean(axis=0)
            anchor_nm = apply_transform(raw_anchor_nm.reshape(1, 3), transform)[0]
        raw_protein_positions = minimum_image_points(whole_positions_nm[protein.indices], raw_anchor_nm, box_nm)
        protein_positions = apply_transform(raw_protein_positions, transform)
        protein_labels = np.asarray(
            [
                residue_mapping.display_by_resindex.get(
                    int(atom.resindex), f"{int(atom.resid)}{str(atom.resname).upper()}"
                )
                for atom in protein
            ],
            dtype=object,
        )
        protein_sim_labels = np.asarray(
            [
                residue_mapping.simulation_by_resindex.get(
                    int(atom.resindex), f"{int(atom.resid)}{str(atom.resname).upper()}"
                )
                for atom in protein
            ],
            dtype=object,
        )
        protein_homolog_labels = np.asarray(
            [residue_mapping.homolog_by_resindex.get(int(atom.resindex), "") for atom in protein], dtype=object
        )
        protein_tree = cKDTree(protein_positions)
        if cavity_center_nm is None:
            cavity_center_nm = tuple(float(value) for value in anchor_nm)
        molecule_rows: list[MoleculeFrame] = []
        for residue in residues:
            raw_point = _representative_point_nm(residue, config, factor, whole_positions_nm)
            raw_point = minimum_image(raw_point, raw_anchor_nm, box_nm)
            point = apply_transform(raw_point.reshape(1, 3), transform)[0]
            if mask is not None:
                inside = mask.contains_point(tuple(float(value) for value in point))
            else:
                inside = float(np.linalg.norm(point - anchor_nm)) <= (
                    config.cavity.radius_nm + config.cavity.membership_padding_nm
                )
            distance, hit = protein_tree.query(point, k=1)
            molecule_rows.append(
                MoleculeFrame(
                    uid=_residue_uid(residue),
                    resid=int(residue.resid),
                    resname=str(residue.resname).upper(),
                    point_nm=tuple(float(value) for value in point),
                    inside=bool(inside),
                    nearest_residue=str(protein_labels[int(hit)]),
                    nearest_distance_nm=float(distance),
                    nearest_residue_sim=str(protein_sim_labels[int(hit)]),
                    nearest_residue_homolog=str(protein_homolog_labels[int(hit)]),
                )
            )
        frames.append(
            FrameRecord(
                frame=int(getattr(ts, "frame", sequential_index)),
                time_ps=time_ps,
                molecules=tuple(molecule_rows),
                occupancy=sum(1 for item in molecule_rows if item.inside),
            )
        )
    if not frames:
        raise ValueError("No MD trajectory frames matched the requested window/stride")
    result = RunResult(dataset=dataset, frames=frames)
    if cavity_center_nm is not None:
        result.metadata["cavity_center_nm"] = cavity_center_nm
    result.metadata["residue_mapping"] = residue_mapping.metadata
    result.metadata["alignment"] = {
        "description": alignment_description,
        "atom_indices_0based": align_indices.tolist(),
        "reference": str(dataset.reference) if dataset.reference is not None else "trajectory frame 0",
    }
    if mask is not None:
        result.metadata["mask_alignment"] = mask_alignment
        result.metadata["cavity_mask_points_nm"] = np.asarray(mask.points, dtype=float).tolist()
        result.metadata["cavity_mask_reference_nm"] = list(mask.reference_point)
    overlay = _substrate_overlay(
        config, dataset, reference, reference_all_nm, reference_box_nm, reference_cavity_center
    )
    if overlay is not None:
        result.metadata["substrate_overlay"] = overlay
    if sum(frame.occupancy for frame in frames) == 0:
        result.warnings.append(
            "No tracked molecules were inside the configured cavity in any analyzed frame; density is written as zero rather than using bulk molecules."
        )
    if close_reference:
        close = getattr(reference.trajectory, "close", None)
        if close is not None:
            close()
    close_trajectory = getattr(universe.trajectory, "close", None)
    if close_trajectory is not None:
        close_trajectory()
    return result
