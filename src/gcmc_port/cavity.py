from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import sqrt
from pathlib import Path
import random
import re

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .clash import atom_radius_nm
from .gro import Atom, GroStructure, WATER_NAMES, coordinates_center, parse_gro, renumber_atoms, write_gro
from .pathing import portable_path, resolve_portable_path
CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)
DEFAULT_PROBE_RADIUS_NM = 0.10


@dataclass(frozen=True, slots=True)
class ResidueSpec:
    resid: int | None = None
    resname: str | None = None

    def label(self) -> str:
        if self.resid is not None and self.resname:
            return f"{self.resid}{self.resname}"
        if self.resid is not None:
            return str(self.resid)
        if self.resname:
            return self.resname
        return "unknown"


@dataclass(slots=True)
class NearbyResidue:
    resid: int
    resname: str
    distance: float


@dataclass(slots=True)
class CandidatePocket:
    label: str
    seed_point: tuple[float, float, float]
    reference_point: tuple[float, float, float]
    points: np.ndarray
    clearances: np.ndarray
    nearby_residues: list[NearbyResidue]

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    def effective_volume(self, dx: float) -> float:
        return float(self.point_count) * dx ** 3


@dataclass(slots=True)
class VoxelMask:
    points: np.ndarray
    dx: float
    reference_point: tuple[float, float, float]
    effective_volume: float
    membership_padding: float = 0.02
    probe_radius: float | None = None
    source_gro: str | None = None
    exclude_residues: tuple[str, ...] = ()
    _tree: cKDTree | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.size == 0:
            points = points.reshape(0, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Voxel mask points must be an (N, 3) array")
        self.points = points
        self._tree = cKDTree(self.points) if self.points.shape[0] > 0 else None

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def membership_radius(self) -> float:
        return (sqrt(3.0) * self.dx * 0.5) + self.membership_padding

    def contains_point(self, point: tuple[float, float, float]) -> bool:
        if self._tree is None:
            return False
        distance, _ = self._tree.query(np.asarray(point, dtype=float), k=1)
        return bool(distance <= self.membership_radius)

    def contains_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or self._tree is None:
            return np.zeros(0, dtype=bool)
        distances, _ = self._tree.query(points, k=1)
        return distances <= self.membership_radius

    def sample_point(self, rng: random.Random, jitter: float) -> tuple[float, float, float]:
        if self.point_count == 0:
            raise RuntimeError("Voxel mask has no accessible points")
        base = self.points[rng.randrange(self.points.shape[0])]
        if jitter > 0:
            offset = np.array(
                [
                    (rng.random() * 2.0 - 1.0) * jitter * 0.5,
                    (rng.random() * 2.0 - 1.0) * jitter * 0.5,
                    (rng.random() * 2.0 - 1.0) * jitter * 0.5,
                ],
                dtype=float,
            )
            candidate = base + offset
            if self.contains_point((float(candidate[0]), float(candidate[1]), float(candidate[2]))):
                base = candidate
        return float(base[0]), float(base[1]), float(base[2])

    def translated(self, shift: tuple[float, float, float]) -> "VoxelMask":
        delta = np.asarray(shift, dtype=float)
        reference = tuple(float(value) for value in (np.asarray(self.reference_point, dtype=float) + delta))
        return VoxelMask(
            points=self.points + delta,
            dx=self.dx,
            reference_point=reference,
            effective_volume=self.effective_volume,
            membership_padding=self.membership_padding,
            probe_radius=self.probe_radius,
            source_gro=self.source_gro,
            exclude_residues=self.exclude_residues,
        )

    def with_membership_padding(self, membership_padding: float) -> "VoxelMask":
        return VoxelMask(
            points=np.array(self.points, copy=True),
            dx=self.dx,
            reference_point=self.reference_point,
            effective_volume=self.effective_volume,
            membership_padding=membership_padding,
            probe_radius=self.probe_radius,
            source_gro=self.source_gro,
            exclude_residues=self.exclude_residues,
        )


def parse_residue_spec(token: str) -> ResidueSpec:
    value = token.strip()
    if not value:
        raise ValueError("Residue token must not be empty")

    match = re.fullmatch(r"(\d+)([A-Za-z0-9_+\-]+)", value)
    if match:
        return ResidueSpec(resid=int(match.group(1)), resname=match.group(2).upper())

    match = re.fullmatch(r"([A-Za-z0-9_+\-]+):(\d+)", value)
    if match:
        return ResidueSpec(resid=int(match.group(2)), resname=match.group(1).upper())

    match = re.fullmatch(r"(\d+):([A-Za-z0-9_+\-]+)", value)
    if match:
        return ResidueSpec(resid=int(match.group(1)), resname=match.group(2).upper())

    if value.isdigit():
        return ResidueSpec(resid=int(value), resname=None)

    return ResidueSpec(resid=None, resname=value.upper())


def infer_mask_meta_path(mask_path: str | Path) -> Path:
    path = Path(mask_path)
    if path.name.endswith("_mask.dat"):
        return path.with_name(path.name[:-9] + ".meta.json")
    return path.with_suffix(path.suffix + ".meta.json")


def load_voxel_mask(
    mask_path: str | Path,
    meta_path: str | Path | None = None,
    *,
    membership_padding: float = 0.02,
) -> VoxelMask:
    mask = Path(mask_path)
    points = _read_mask_points(mask)
    metadata: dict[str, object] = {}
    resolved_meta = Path(meta_path) if meta_path else infer_mask_meta_path(mask)
    if resolved_meta.exists():
        metadata = json.loads(resolved_meta.read_text(encoding="utf-8"))

    dx = float(metadata.get("dx", _infer_dx(points)))
    ref_raw = metadata.get("reference_point")
    if isinstance(ref_raw, (list, tuple)) and len(ref_raw) == 3:
        reference_point = (float(ref_raw[0]), float(ref_raw[1]), float(ref_raw[2]))
    else:
        reference_point = tuple(float(value) for value in points.mean(axis=0))
    effective_volume = float(metadata.get("effective_volume", float(points.shape[0]) * dx ** 3))
    probe_raw = metadata.get("probe_radius")
    probe_radius = float(probe_raw) if probe_raw is not None else None
    source_gro_raw = metadata.get("source_gro")
    source_gro = str(resolve_portable_path(str(source_gro_raw), resolved_meta.parent)) if source_gro_raw else None
    exclude_raw = metadata.get("exclude_residues", [])
    exclude_residues = tuple(str(value) for value in exclude_raw) if isinstance(exclude_raw, list) else ()
    return VoxelMask(
        points=points,
        dx=dx,
        reference_point=reference_point,
        effective_volume=effective_volume,
        membership_padding=membership_padding,
        probe_radius=probe_radius,
        source_gro=source_gro,
        exclude_residues=exclude_residues,
    )


def water_residue_ids_in_mask(gro_or_structure: str | Path | GroStructure, mask: VoxelMask) -> list[int]:
    return molecule_residue_ids_in_mask(gro_or_structure, mask, "SOL")


def molecule_residue_ids_in_mask(
    gro_or_structure: str | Path | GroStructure,
    mask: VoxelMask,
    resname: str,
) -> list[int]:
    structure = parse_gro(gro_or_structure) if isinstance(gro_or_structure, (str, Path)) else gro_or_structure
    target_resname = resname.upper()
    water_like = target_resname in WATER_NAMES
    residue_atoms: dict[int, list[Atom]] = {}
    ordered_resids: list[int] = []
    for atom in structure.atoms:
        atom_resname = atom.resname.upper()
        if water_like:
            if atom_resname not in WATER_NAMES:
                continue
        elif atom_resname != target_resname:
            continue
        if atom.resid not in residue_atoms:
            ordered_resids.append(atom.resid)
            residue_atoms[atom.resid] = []
        residue_atoms[atom.resid].append(atom)

    cavity_resids: list[int] = []
    for resid in ordered_resids:
        atoms = residue_atoms[resid]
        if water_like:
            marker = next((atom for atom in atoms if atom.atomname.upper() in {"OW", "O"}), atoms[0])
            point = (marker.x, marker.y, marker.z)
        else:
            point = coordinates_center(atoms)
        if mask.contains_point(point):
            cavity_resids.append(resid)
    return cavity_resids


def remove_waters_in_mask(input_gro: str | Path, output_gro: str | Path, mask: VoxelMask) -> int:
    structure = parse_gro(input_gro)
    remove_resids = set(water_residue_ids_in_mask(structure, mask))
    kept_atoms = [
        atom
        for atom in structure.atoms
        if not (atom.resname in WATER_NAMES and atom.resid in remove_resids)
    ]
    renumbered = renumber_atoms(GroStructure(title=structure.title, atoms=kept_atoms, box_line=structure.box_line))
    write_gro(output_gro, renumbered)
    return len(remove_resids)


def clip_voxel_mask(
    mask: VoxelMask,
    gro_or_structure: str | Path | GroStructure,
    *,
    probe_radius: float | None = None,
    exclude_residues: list[str] | None = None,
) -> VoxelMask:
    structure = parse_gro(gro_or_structure) if isinstance(gro_or_structure, (str, Path)) else gro_or_structure
    exclude_specs = [parse_residue_spec(token) for token in exclude_residues or []]
    clip_atoms = _non_water_atoms(structure, exclude_specs)
    threshold = probe_radius if probe_radius is not None else (mask.probe_radius if mask.probe_radius is not None else DEFAULT_PROBE_RADIUS_NM)
    if mask.point_count == 0 or not clip_atoms:
        return VoxelMask(
            points=np.array(mask.points, copy=True),
            dx=mask.dx,
            reference_point=mask.reference_point,
            effective_volume=float(mask.point_count) * mask.dx ** 3,
            membership_padding=mask.membership_padding,
            probe_radius=threshold,
            source_gro=mask.source_gro,
            exclude_residues=mask.exclude_residues,
        )

    clearance = _surface_clearance(mask.points, clip_atoms)
    kept_points = mask.points[clearance >= threshold]
    if kept_points.shape[0] > 1:
        kept_points = _connected_component_near_reference(kept_points, mask.dx, mask.reference_point)
    if kept_points.shape[0] == 0:
        reference_point = mask.reference_point
    else:
        distances = np.linalg.norm(kept_points - np.asarray(mask.reference_point, dtype=float), axis=1)
        reference = kept_points[int(np.argmin(distances))]
        reference_point = (float(reference[0]), float(reference[1]), float(reference[2]))
    return VoxelMask(
        points=kept_points,
        dx=mask.dx,
        reference_point=reference_point,
        effective_volume=float(kept_points.shape[0]) * mask.dx ** 3,
        membership_padding=mask.membership_padding,
        probe_radius=threshold,
        source_gro=mask.source_gro,
        exclude_residues=mask.exclude_residues,
    )


def align_voxel_mask_to_structure(
    mask: VoxelMask,
    target_gro_or_structure: str | Path | GroStructure,
    *,
    source_gro_or_structure: str | Path | GroStructure | None = None,
) -> tuple[VoxelMask, tuple[float, float, float]]:
    target = parse_gro(target_gro_or_structure) if isinstance(target_gro_or_structure, (str, Path)) else target_gro_or_structure
    source_input: str | Path | GroStructure | None = source_gro_or_structure
    if source_input is None and mask.source_gro:
        source_path = Path(mask.source_gro)
        if source_path.exists():
            source_input = source_path
    if source_input is None:
        return mask.with_membership_padding(mask.membership_padding), (0.0, 0.0, 0.0)

    source = parse_gro(source_input) if isinstance(source_input, (str, Path)) else source_input
    specs = [parse_residue_spec(token) for token in mask.exclude_residues]
    source_center = _alignment_center(source, specs)
    target_center = _alignment_center(target, specs)
    if source_center is None or target_center is None:
        return mask.with_membership_padding(mask.membership_padding), (0.0, 0.0, 0.0)

    shift = tuple(float(target_center[idx] - source_center[idx]) for idx in range(3))
    if np.linalg.norm(np.asarray(shift, dtype=float)) <= 1.0e-9:
        return mask.with_membership_padding(mask.membership_padding), shift
    return mask.translated(shift), shift


def build_cavity_from_structure(
    gro_path: str | Path,
    *,
    outprefix: str | Path,
    mode: str,
    dx: float,
    probe_radius: float,
    search_radius: float,
    seed_point: tuple[float, float, float] | None = None,
    seed_residue: str | None = None,
    seed_atoms: list[str] | None = None,
    exclude_residues: list[str] | None = None,
    nearby_cutoff: float = 0.45,
    min_peak_clearance: float = 0.20,
    candidate_limit: int = 5,
    min_points: int = 20,
) -> list[Path]:
    structure = parse_gro(gro_path)
    exclude_specs = [parse_residue_spec(token) for token in exclude_residues or []]
    excluded_atoms = _matching_atoms(structure, exclude_specs) if exclude_specs else []
    discovery_atoms = _non_water_atoms(structure, exclude_specs)
    clipping_atoms = discovery_atoms
    if not discovery_atoms:
        raise ValueError("No non-water atoms available for cavity discovery")

    atom_names = [name.upper() for name in (seed_atoms or [])]
    outputs: list[Path] = []
    outprefix_path = Path(outprefix).resolve()

    if mode == "seeded":
        seed = _resolve_seed_point(structure, seed_point, seed_residue, atom_names, exclude_specs)
        pocket = _build_local_pocket(
            structure=structure,
            discovery_atoms=discovery_atoms,
            clipping_atoms=clipping_atoms,
            seed_point=seed,
            dx=dx,
            probe_radius=probe_radius,
            search_radius=search_radius,
            nearby_cutoff=nearby_cutoff,
            min_points=min_points,
            label="selected",
        )
        if pocket is None:
            raise RuntimeError("Could not identify a cavity around the requested seed")
        outputs.extend(
            _write_pocket_outputs(
                pocket,
                outprefix_path,
                dx=dx,
                probe_radius=probe_radius,
                search_radius=search_radius,
                source_gro=Path(gro_path).resolve(),
                mode=mode,
                exclude_residues=[spec.label() for spec in exclude_specs],
            )
        )
        return outputs

    seeds = _discover_seed_points(
        discovery_atoms=discovery_atoms,
        dx=dx,
        probe_radius=probe_radius,
        search_radius=search_radius,
        min_peak_clearance=min_peak_clearance,
        limit=max(candidate_limit * 3, candidate_limit),
        guide_point=coordinates_center(excluded_atoms) if excluded_atoms else None,
        guide_radius=_auto_guide_radius(search_radius),
    )
    candidates: list[CandidatePocket] = []
    for index, candidate_seed in enumerate(seeds, start=1):
        pocket = _build_local_pocket(
            structure=structure,
            discovery_atoms=discovery_atoms,
            clipping_atoms=clipping_atoms,
            seed_point=tuple(float(value) for value in candidate_seed),
            dx=dx,
            probe_radius=probe_radius,
            search_radius=search_radius,
            nearby_cutoff=nearby_cutoff,
            min_points=min_points,
            label=f"candidate{index:02d}",
        )
        if pocket is None:
            continue
        if not pocket.nearby_residues:
            continue
        if any(_points_overlap(pocket.points, existing.points, dx) >= 0.50 for existing in candidates):
            continue
        candidates.append(pocket)
        if len(candidates) >= candidate_limit:
            break

    if not candidates:
        raise RuntimeError("Auto cavity search did not find any candidate pockets")

    summary_path = outprefix_path.with_name(f"{outprefix_path.name}_candidates.tsv")
    summary_lines = ["id\tpoints\tveff_nm3\tseed_x\tseed_y\tseed_z\tref_x\tref_y\tref_z\tclearance_max_nm\tnearby"]
    for index, pocket in enumerate(candidates, start=1):
        nearby = ",".join(f"{item.resid}{item.resname}" for item in pocket.nearby_residues[:8])
        summary_lines.append(
            "\t".join(
                [
                    f"candidate{index:02d}",
                    str(pocket.point_count),
                    f"{pocket.effective_volume(dx):.6f}",
                    f"{pocket.seed_point[0]:.6f}",
                    f"{pocket.seed_point[1]:.6f}",
                    f"{pocket.seed_point[2]:.6f}",
                    f"{pocket.reference_point[0]:.6f}",
                    f"{pocket.reference_point[1]:.6f}",
                    f"{pocket.reference_point[2]:.6f}",
                    f"{float(np.max(pocket.clearances)):.6f}",
                    nearby,
                ]
            )
        )
        candidate_prefix = outprefix_path.with_name(f"{outprefix_path.name}_candidate{index:02d}")
        outputs.extend(
            _write_pocket_outputs(
                pocket,
                candidate_prefix,
                dx=dx,
                probe_radius=probe_radius,
                search_radius=search_radius,
                source_gro=Path(gro_path).resolve(),
                mode=mode,
                exclude_residues=[spec.label() for spec in exclude_specs],
            )
        )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    outputs.insert(0, summary_path)
    return outputs


def _resolve_seed_point(
    structure: GroStructure,
    explicit_point: tuple[float, float, float] | None,
    seed_residue: str | None,
    atom_names: list[str],
    exclude_specs: list[ResidueSpec],
) -> tuple[float, float, float]:
    if explicit_point is not None:
        return explicit_point
    if seed_residue:
        return _residue_center(structure, parse_residue_spec(seed_residue), atom_names)
    if exclude_specs:
        atoms = _matching_atoms(structure, exclude_specs)
        if atoms:
            return coordinates_center(atoms)
    raise ValueError("Seeded cavity mode requires --seed-point, --seed-residue, or --exclude-residue")


def _non_water_atoms(structure: GroStructure, exclude_specs: list[ResidueSpec]) -> list[Atom]:
    return [
        atom
        for atom in structure.atoms
        if atom.resname not in WATER_NAMES and not _matches_specs(atom, exclude_specs)
    ]


def _matching_atoms(structure: GroStructure, specs: list[ResidueSpec]) -> list[Atom]:
    return [atom for atom in structure.atoms if _matches_specs(atom, specs)]


def _matches_specs(atom: Atom, specs: list[ResidueSpec]) -> bool:
    for spec in specs:
        resid_match = spec.resid is None or atom.resid == spec.resid
        resname_match = spec.resname is None or atom.resname.upper() == spec.resname
        if resid_match and resname_match:
            return True
    return False


def _residue_center(structure: GroStructure, spec: ResidueSpec, atom_names: list[str]) -> tuple[float, float, float]:
    residue_atoms = [
        atom
        for atom in structure.atoms
        if (spec.resid is None or atom.resid == spec.resid)
        and (spec.resname is None or atom.resname.upper() == spec.resname)
    ]
    if atom_names:
        selected = [atom for atom in residue_atoms if atom.atomname.upper() in set(atom_names)]
        residue_atoms = selected or residue_atoms
    if not residue_atoms:
        raise ValueError(f"Could not locate residue {spec.label()} in structure")
    return coordinates_center(residue_atoms)


def _alignment_center(structure: GroStructure, specs: list[ResidueSpec]) -> tuple[float, float, float] | None:
    atoms = _non_water_atoms(structure, specs)
    if atoms:
        return coordinates_center(atoms)
    if specs:
        atoms = _matching_atoms(structure, specs)
        if atoms:
            return coordinates_center(atoms)
    atoms = _non_water_atoms(structure, [])
    if atoms:
        return coordinates_center(atoms)
    return None


def _build_local_pocket(
    *,
    structure: GroStructure,
    discovery_atoms: list[Atom],
    clipping_atoms: list[Atom],
    seed_point: tuple[float, float, float],
    dx: float,
    probe_radius: float,
    search_radius: float,
    nearby_cutoff: float,
    min_points: int,
    label: str,
) -> CandidatePocket | None:
    axes = _grid_axes(seed_point, search_radius, dx)
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()], axis=1)
    shape = mesh[0].shape
    seed = np.asarray(seed_point, dtype=float)
    in_sphere = np.sum((points - seed) ** 2, axis=1) <= (search_radius ** 2 + 1.0e-12)
    if not np.any(in_sphere):
        return None

    discovery_clearance = np.full(points.shape[0], -np.inf, dtype=float)
    discovery_clearance[in_sphere] = _surface_clearance(points[in_sphere], discovery_atoms)
    free = discovery_clearance >= probe_radius
    if not np.any(free):
        return None

    full_clearance = np.full(points.shape[0], -np.inf, dtype=float)
    full_clearance[free] = _surface_clearance(points[free], clipping_atoms)
    clipped = free & (full_clearance >= probe_radius)
    if not np.any(clipped):
        return None

    clipped_labels, _ = ndimage.label(clipped.reshape(shape), structure=CONNECTIVITY)
    clipped_indices = np.flatnonzero(clipped)
    nearest_clipped = clipped_indices[int(np.argmin(np.linalg.norm(points[clipped] - seed, axis=1)))]
    component_label = clipped_labels.ravel()[nearest_clipped]
    final_mask = clipped_labels.ravel() == component_label
    if int(np.count_nonzero(final_mask)) < min_points:
        return None

    pocket_points = points[final_mask]
    pocket_clearance = full_clearance[final_mask]
    ref_index = int(np.argmin(np.linalg.norm(pocket_points - seed, axis=1)))
    reference_point = tuple(float(value) for value in pocket_points[ref_index])
    nearby = _nearby_residues(structure, pocket_points, nearby_cutoff)
    return CandidatePocket(
        label=label,
        seed_point=seed_point,
        reference_point=reference_point,
        points=pocket_points,
        clearances=pocket_clearance,
        nearby_residues=nearby,
    )


def _discover_seed_points(
    *,
    discovery_atoms: list[Atom],
    dx: float,
    probe_radius: float,
    search_radius: float,
    min_peak_clearance: float,
    limit: int,
    guide_point: tuple[float, float, float] | None = None,
    guide_radius: float | None = None,
) -> list[np.ndarray]:
    coordinates = np.asarray([[atom.x, atom.y, atom.z] for atom in discovery_atoms], dtype=float)
    margin = max(search_radius, 0.30)
    mins = coordinates.min(axis=0) - margin
    maxs = coordinates.max(axis=0) + margin
    axes = [np.arange(mins[idx], maxs[idx] + dx * 0.5, dx) for idx in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([mesh[0].ravel(), mesh[1].ravel(), mesh[2].ravel()], axis=1)
    shape = mesh[0].shape

    clearance = _surface_clearance(points, discovery_atoms)
    clearance_grid = clearance.reshape(shape)
    free_grid = clearance_grid >= probe_radius
    maxima = ndimage.maximum_filter(clearance_grid, size=3, mode="constant", cval=-np.inf)
    peak_grid = free_grid & (clearance_grid >= min_peak_clearance) & (clearance_grid == maxima)
    peak_points = points[peak_grid.ravel()]
    peak_clearances = clearance_grid.ravel()[peak_grid.ravel()]
    if peak_points.size == 0:
        peak_points = points[free_grid.ravel()]
        peak_clearances = clearance_grid.ravel()[free_grid.ravel()]
    atom_tree = cKDTree(coordinates)
    contact_radius = max(search_radius + probe_radius + dx, dx * 3.0)
    contact_counts = np.asarray(atom_tree.query_ball_point(peak_points, r=contact_radius, return_length=True), dtype=int)

    guide_distances = np.zeros(peak_points.shape[0], dtype=float)
    candidate_mask = np.ones(peak_points.shape[0], dtype=bool)
    if np.any(contact_counts > 0):
        candidate_mask &= contact_counts > 0

    if guide_point is not None:
        guide = np.asarray(guide_point, dtype=float)
        guide_distances = np.linalg.norm(peak_points - guide, axis=1)
        radius = guide_radius if guide_radius is not None else _auto_guide_radius(search_radius)
        guided_mask = guide_distances <= radius
        if np.any(candidate_mask & guided_mask):
            candidate_mask &= guided_mask
        elif np.any(guided_mask):
            candidate_mask = guided_mask

    if not np.any(candidate_mask):
        candidate_mask = np.ones(peak_points.shape[0], dtype=bool)

    peak_points = peak_points[candidate_mask]
    peak_clearances = peak_clearances[candidate_mask]
    contact_counts = contact_counts[candidate_mask]
    guide_distances = guide_distances[candidate_mask]
    if guide_point is not None:
        order = np.lexsort((-peak_clearances, guide_distances, -contact_counts))
    else:
        order = np.lexsort((-peak_clearances, -contact_counts))

    selected: list[np.ndarray] = []
    min_separation = max(search_radius * 0.5, dx * 2.0)
    for idx in order:
        point = peak_points[idx]
        if any(np.linalg.norm(point - existing) < min_separation for existing in selected):
            continue
        selected.append(point)
        if len(selected) >= limit:
            break
    return selected


def _auto_guide_radius(search_radius: float) -> float:
    return max(search_radius * 1.5, search_radius + 0.30)


def _connected_component_near_reference(
    points: np.ndarray,
    dx: float,
    reference_point: tuple[float, float, float],
) -> np.ndarray:
    if points.shape[0] <= 1:
        return points
    tree = cKDTree(points)
    seed_index = int(np.argmin(np.linalg.norm(points - np.asarray(reference_point, dtype=float), axis=1)))
    threshold = (sqrt(3.0) * dx) + 1.0e-6
    stack = [seed_index]
    visited = np.zeros(points.shape[0], dtype=bool)
    visited[seed_index] = True
    while stack:
        current = stack.pop()
        for neighbor in tree.query_ball_point(points[current], r=threshold):
            if not visited[neighbor]:
                visited[neighbor] = True
                stack.append(neighbor)
    return points[visited]


def _surface_clearance(points: np.ndarray, atoms: list[Atom]) -> np.ndarray:
    if not atoms:
        return np.full(points.shape[0], 1.0e6, dtype=float)
    centers = np.asarray([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)
    radii = np.asarray([atom_radius_nm(atom.atomname) for atom in atoms], dtype=float)
    k = min(8, centers.shape[0])
    distances, indices = cKDTree(centers).query(points, k=k)
    if k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]
    return np.min(distances - radii[indices], axis=1)


def _grid_axes(center: tuple[float, float, float], radius: float, dx: float) -> list[np.ndarray]:
    return [np.arange(center[idx] - radius, center[idx] + radius + dx * 0.5, dx) for idx in range(3)]


def _nearby_residues(structure: GroStructure, points: np.ndarray, cutoff: float) -> list[NearbyResidue]:
    atoms = [atom for atom in structure.atoms if atom.resname not in WATER_NAMES]
    if not atoms:
        return []
    coords = np.asarray([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)
    distances, _ = cKDTree(points).query(coords, k=1)
    by_residue: dict[tuple[int, str], float] = {}
    for atom, distance in zip(atoms, distances, strict=True):
        if distance > cutoff:
            continue
        key = (atom.resid, atom.resname)
        if key not in by_residue or distance < by_residue[key]:
            by_residue[key] = float(distance)
    nearby = [NearbyResidue(resid=resid, resname=resname, distance=dist) for (resid, resname), dist in by_residue.items()]
    nearby.sort(key=lambda item: (item.distance, item.resid, item.resname))
    return nearby


def _write_pocket_outputs(
    pocket: CandidatePocket,
    outprefix: Path,
    *,
    dx: float,
    probe_radius: float,
    search_radius: float,
    source_gro: Path,
    mode: str,
    exclude_residues: list[str],
) -> list[Path]:
    mask_path = outprefix.with_name(f"{outprefix.name}_mask.dat")
    pdb_path = outprefix.with_name(f"{outprefix.name}_points.pdb")
    meta_path = outprefix.with_suffix(".meta.json")
    nearby_path = outprefix.with_name(f"{outprefix.name}_nearby_residues.tsv")

    mask_path.write_text(
        "\n".join(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}" for point in pocket.points) + "\n",
        encoding="utf-8",
    )

    pdb_lines = []
    for serial, point in enumerate(pocket.points, start=1):
        resid = ((serial - 1) % 9999) + 1
        pdb_lines.append(
            f"HETATM{serial:5d}  HE  CAV A{resid:4d}    {point[0]*10:8.3f}{point[1]*10:8.3f}{point[2]*10:8.3f}  1.00  0.00          He"
        )
    pdb_lines.append("END")
    pdb_path.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")

    metadata = {
        "version": 2,
        "build_mode": mode,
        "dx": dx,
        "probe_radius": probe_radius,
        "search_radius": search_radius,
        "point_count": pocket.point_count,
        "effective_volume": pocket.effective_volume(dx),
        "seed_point": list(pocket.seed_point),
        "reference_point": list(pocket.reference_point),
        "clearance_min": float(np.min(pocket.clearances)),
        "clearance_max": float(np.max(pocket.clearances)),
        "clearance_mean": float(np.mean(pocket.clearances)),
        "source_gro": portable_path(source_gro, meta_path.parent),
        "exclude_residues": exclude_residues,
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    nearby_lines = ["resid\tresname\tmin_distance_nm"]
    nearby_lines.extend(f"{item.resid}\t{item.resname}\t{item.distance:.6f}" for item in pocket.nearby_residues)
    nearby_path.write_text("\n".join(nearby_lines) + "\n", encoding="utf-8")
    return [mask_path, pdb_path, meta_path, nearby_path]


def _read_mask_points(path: Path) -> np.ndarray:
    rows: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
    if not rows:
        raise ValueError(f"Mask file {path} is empty")
    return np.asarray(rows, dtype=float)


def _infer_dx(points: np.ndarray) -> float:
    candidates: list[float] = []
    for axis in range(3):
        values = np.unique(np.round(points[:, axis], 8))
        if values.size < 2:
            continue
        diffs = np.diff(values)
        positive = diffs[diffs > 1.0e-6]
        if positive.size > 0:
            candidates.append(float(np.min(positive)))
    if not candidates:
        return 0.1
    return min(candidates)


def _points_overlap(points_a: np.ndarray, points_b: np.ndarray, dx: float) -> float:
    if points_a.size == 0 or points_b.size == 0:
        return 0.0
    tree = cKDTree(points_b)
    matches = tree.query_ball_point(points_a, r=dx * 0.6)
    overlap = sum(1 for item in matches if item)
    return overlap / max(1, min(points_a.shape[0], points_b.shape[0]))
