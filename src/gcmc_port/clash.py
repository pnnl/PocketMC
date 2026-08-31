from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from .gro import Atom, GroStructure, parse_gro


ELEMENT_RADII_NM = {
    "H": 0.120,
    "C": 0.170,
    "N": 0.155,
    "O": 0.152,
    "S": 0.180,
    "P": 0.180,
    "F": 0.147,
    "CL": 0.175,
    "BR": 0.185,
    "I": 0.198,
}
HEAVY_ELEMENTS = frozenset(element for element in ELEMENT_RADII_NM if element != "H")
CLASH_SCALE = 0.85


@dataclass(frozen=True, slots=True)
class ClashDetail:
    moved_atom: Atom
    other_atom: Atom
    distance: float
    cutoff: float


def infer_element(atom_name: str) -> str:
    letters = "".join(char for char in atom_name if char.isalpha()).upper()
    if letters.startswith("CL"):
        return "CL"
    if letters.startswith("BR"):
        return "BR"
    if letters.startswith("H"):
        return "H"
    if letters.startswith("N"):
        return "N"
    if letters.startswith("O"):
        return "O"
    if letters.startswith("S"):
        return "S"
    if letters.startswith("P"):
        return "P"
    if letters.startswith("F"):
        return "F"
    if letters.startswith("I"):
        return "I"
    return "C"


def atom_radius_nm(atom_name: str) -> float:
    return ELEMENT_RADII_NM[infer_element(atom_name)]


def is_heavy_atom(atom: Atom) -> bool:
    return infer_element(atom.atomname) in HEAVY_ELEMENTS


def pair_clash_cutoff(atom_a: Atom, atom_b: Atom, minimum_cutoff: float) -> float:
    radius_sum = atom_radius_nm(atom_a.atomname) + atom_radius_nm(atom_b.atomname)
    return max(minimum_cutoff, CLASH_SCALE * radius_sum)


def find_heavy_atom_clash(moved_atoms: list[Atom], other_atoms: list[Atom], minimum_cutoff: float) -> ClashDetail | None:
    moved_heavy = [atom for atom in moved_atoms if is_heavy_atom(atom)]
    other_heavy = [atom for atom in other_atoms if is_heavy_atom(atom)]
    if not moved_heavy or not other_heavy:
        return None

    for moved_atom in moved_heavy:
        for other_atom in other_heavy:
            cutoff = pair_clash_cutoff(moved_atom, other_atom, minimum_cutoff)
            distance = sqrt(
                (moved_atom.x - other_atom.x) ** 2
                + (moved_atom.y - other_atom.y) ** 2
                + (moved_atom.z - other_atom.z) ** 2
            )
            if distance >= cutoff:
                continue
            # Callers only need to reject on any hard clash.  Returning
            # immediately avoids scanning a million-atom environment after a
            # collision has already been established.
            return ClashDetail(moved_atom=moved_atom, other_atom=other_atom, distance=distance, cutoff=cutoff)
    return None


def residue_clash_in_structure(
    gro_or_structure: str | Path | GroStructure,
    resid: int,
    minimum_cutoff: float,
) -> ClashDetail | None:
    structure = parse_gro(gro_or_structure) if isinstance(gro_or_structure, (str, Path)) else gro_or_structure
    moved_atoms = [atom for atom in structure.atoms if atom.resid == resid]
    other_atoms = [atom for atom in structure.atoms if atom.resid != resid]
    return find_heavy_atom_clash(moved_atoms, other_atoms, minimum_cutoff)


def describe_clash(detail: ClashDetail) -> str:
    moved = detail.moved_atom
    other = detail.other_atom
    return (
        f"{moved.resname}{moved.resid}:{moved.atomname} vs "
        f"{other.resname}{other.resid}:{other.atomname} "
        f"(d={detail.distance:.3f} nm < cutoff={detail.cutoff:.3f} nm)"
    )
