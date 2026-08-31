from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cavity import parse_residue_spec
from .gro import Atom, GroStructure, parse_gro, coordinates_center


SKIP_RESIDUE_NAMES = {"SOL", "WAT", "HOH", "NA", "CL"}


@dataclass(frozen=True, slots=True)
class StructureResidue:
    resid: int
    resname: str
    token: str
    atom_names: tuple[str, ...]


def normalize_residue_tokens(tokens: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        value = str(token).strip()
        if not value:
            continue
        label = parse_residue_spec(value).label()
        if label in seen:
            continue
        seen.add(label)
        normalized.append(label)
    return normalized


def collect_structure_residues(gro_or_structure: str | Path | GroStructure) -> list[StructureResidue]:
    structure = parse_gro(gro_or_structure) if isinstance(gro_or_structure, (str, Path)) else gro_or_structure
    ordered_keys: list[tuple[int, str]] = []
    atoms_by_key: dict[tuple[int, str], list[Atom]] = {}
    for atom in structure.atoms:
        key = (atom.resid, atom.resname.upper())
        if key not in atoms_by_key:
            ordered_keys.append(key)
            atoms_by_key[key] = []
        atoms_by_key[key].append(atom)

    residues: list[StructureResidue] = []
    for resid, resname in ordered_keys:
        atom_names = tuple(atom.atomname for atom in atoms_by_key[(resid, resname)])
        residues.append(StructureResidue(resid=resid, resname=resname, token=f"{resid}{resname}", atom_names=atom_names))
    return residues


def resolve_reference_center(
    gro_or_structure: str | Path | GroStructure,
    *,
    residue_tokens: list[str] | None = None,
    reference_mode: str = "atoms",
    center_atoms: list[str] | None = None,
    fallback_resid: int = 0,
    fallback_resname: str = "",
) -> tuple[float, float, float]:
    structure = parse_gro(gro_or_structure) if isinstance(gro_or_structure, (str, Path)) else gro_or_structure
    tokens = normalize_residue_tokens(list(residue_tokens or []))
    if not tokens and fallback_resid and fallback_resname:
        tokens = [f"{fallback_resid}{fallback_resname}"]
    if not tokens:
        raise ValueError("No anchor residues were configured")

    specs = [parse_residue_spec(token) for token in tokens]
    matched: list[Atom] = []
    ordered_groups: list[list[Atom]] = []
    for spec in specs:
        atoms = [
            atom
            for atom in structure.atoms
            if (spec.resid is None or atom.resid == spec.resid)
            and (spec.resname is None or atom.resname.upper() == spec.resname)
        ]
        if not atoms:
            raise ValueError(f"Could not locate residue {spec.label()} in structure")
        ordered_groups.append(atoms)
        matched.extend(atoms)

    mode = str(reference_mode).strip().lower() or "atoms"
    if mode not in {"atoms", "com"}:
        raise ValueError(f"Unsupported reference mode: {reference_mode!r}")
    if mode == "atoms":
        if len(ordered_groups) != 1:
            raise ValueError("Atom-based reference mode requires exactly one residue")
        chosen_atoms = ordered_groups[0]
        if center_atoms:
            atom_filter = {name.upper() for name in center_atoms if str(name).strip()}
            filtered = [atom for atom in chosen_atoms if atom.atomname.upper() in atom_filter]
            if filtered:
                chosen_atoms = filtered
        return coordinates_center(chosen_atoms)
    return coordinates_center(matched)
