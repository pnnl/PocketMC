from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


WATER_NAMES = {"SOL", "WAT", "HOH"}
GRO_INDEX_MODULUS = 100_000
GRO_NAME_WIDTH = 5
GRO_COORDINATE_WIDTH = 8


def _format_gro_index(value: int, *, field: str) -> str:
    """Format an index without allowing it to overflow a GRO fixed-width field."""
    if value < 0:
        if value < -9_999:
            raise ValueError(f"{field}={value} does not fit the GRO 5-character integer field")
        encoded = value
    else:
        # This is also what GROMACS does for systems with 100,000+ atoms.
        # GRO serials are labels; GROMACS derives the real atom index from the
        # coordinate-line order rather than from this wrapped value.
        encoded = value % GRO_INDEX_MODULUS
    return f"{encoded:5d}"


def _format_gro_name(value: str, *, field: str, align_left: bool) -> str:
    if len(value) > GRO_NAME_WIDTH:
        raise ValueError(f"{field}={value!r} exceeds the GRO 5-character field")
    return f"{value:<5}" if align_left else f"{value:>5}"


def _format_gro_coordinate(value: float, *, field: str) -> str:
    if not isfinite(value):
        raise ValueError(f"{field} coordinate must be finite, got {value!r}")
    encoded = f"{value:8.3f}"
    if len(encoded) != GRO_COORDINATE_WIDTH:
        raise ValueError(f"{field}={value!r} does not fit the GRO 8.3 coordinate field")
    return encoded


@dataclass(slots=True)
class Atom:
    resid: int
    resname: str
    atomname: str
    atomnr: int
    x: float
    y: float
    z: float

    def line(self) -> str:
        return "".join(
            (
                _format_gro_index(self.resid, field="residue number"),
                _format_gro_name(self.resname, field="residue name", align_left=True),
                _format_gro_name(self.atomname, field="atom name", align_left=False),
                _format_gro_index(self.atomnr, field="atom number"),
                _format_gro_coordinate(self.x, field="x"),
                _format_gro_coordinate(self.y, field="y"),
                _format_gro_coordinate(self.z, field="z"),
            )
        )


@dataclass(slots=True)
class GroStructure:
    title: str
    atoms: list[Atom]
    box_line: str

    def body_lines(self) -> list[str]:
        return [atom.line() for atom in self.atoms]


def contiguous_residue_groups(structure: GroStructure) -> list[list[Atom]]:
    """Return residue occurrences in file order without merging repeated labels."""
    groups: list[list[Atom]] = []
    previous_key: tuple[int, str] | None = None
    for atom in structure.atoms:
        key = (atom.resid, atom.resname)
        if key != previous_key:
            groups.append([])
            previous_key = key
        groups[-1].append(atom)
    return groups


def parse_atom_line(line: str) -> Atom:
    if len(line) < 44:
        raise ValueError(f"GRO atom line is too short ({len(line)} characters; expected at least 44)")

    # The standard coordinate fields are 8.3, but GROMACS can also emit GRO
    # coordinates at another precision.  In that case each coordinate field is
    # n+5 characters wide; the distance between decimal points reveals n+5.
    coordinate_text = line[20:]
    decimal_positions = [index for index, char in enumerate(coordinate_text) if char == "."]
    coordinate_width = (
        decimal_positions[1] - decimal_positions[0]
        if len(decimal_positions) >= 2
        else GRO_COORDINATE_WIDTH
    )
    if coordinate_width < 5:
        raise ValueError(f"Invalid GRO coordinate field width {coordinate_width}")
    coordinate_end = 20 + 3 * coordinate_width
    if len(line) < coordinate_end:
        raise ValueError(
            f"GRO atom line is too short for three {coordinate_width}-character coordinate fields"
        )

    return Atom(
        resid=int(line[0:5]),
        resname=line[5:10].strip(),
        atomname=line[10:15].strip(),
        atomnr=int(line[15:20]),
        x=float(line[20 : 20 + coordinate_width]),
        y=float(line[20 + coordinate_width : 20 + 2 * coordinate_width]),
        z=float(line[20 + 2 * coordinate_width : coordinate_end]),
    )


def parse_gro(path: str | Path) -> GroStructure:
    gro_path = Path(path)
    lines = gro_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid GRO file: {gro_path}")
    try:
        natoms = int(lines[1].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in GRO file {gro_path}: {lines[1]!r}") from exc
    if natoms < 0:
        raise ValueError(f"Invalid negative atom count in GRO file {gro_path}: {natoms}")
    if len(lines) < natoms + 3:
        raise ValueError(
            f"Invalid GRO file {gro_path}: header declares {natoms} atoms, "
            f"but the file does not contain all atom lines and a box line"
        )

    atom_lines = lines[2 : 2 + natoms]
    atoms: list[Atom] = []
    residue_offset = 0
    previous_raw_resid: int | None = None
    for atom_index, line in enumerate(atom_lines, start=1):
        try:
            atom = parse_atom_line(line)
        except ValueError as exc:
            file_line = atom_index + 2
            raise ValueError(f"Invalid GRO atom record at {gro_path}:{file_line}: {exc}") from exc

        raw_resid = atom.resid
        if (
            previous_raw_resid is not None
            and raw_resid < previous_raw_resid
            and previous_raw_resid - raw_resid > GRO_INDEX_MODULUS // 2
        ):
            residue_offset += GRO_INDEX_MODULUS
        atom.resid = raw_resid + residue_offset
        # GROMACS uses coordinate-line order as the actual 1-based atom index;
        # the five-character atom-number field is only a wrapping display label.
        atom.atomnr = atom_index
        atoms.append(atom)
        previous_raw_resid = raw_resid

    box_line = lines[2 + natoms]
    if not box_line.strip():
        raise ValueError(f"Invalid GRO file {gro_path}: box line is empty")
    return GroStructure(title=lines[0], atoms=atoms, box_line=box_line)


def write_gro(path: str | Path, structure: GroStructure) -> None:
    gro_path = Path(path)
    lines = [structure.title, f"{len(structure.atoms):5d}"]
    lines.extend(atom.line() for atom in structure.atoms)
    lines.append(structure.box_line)
    gro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_raw_lines(path: str | Path, lines: list[str]) -> None:
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def raw_atom_lines(structure: GroStructure) -> list[str]:
    return [atom.line() for atom in structure.atoms]


def find_first_resid_by_name(structure: GroStructure, resname: str) -> int | None:
    for atom in structure.atoms:
        if atom.resname == resname:
            return atom.resid
    return None


def residue_atoms(structure: GroStructure, resid: int, resname: str | None = None) -> list[Atom]:
    atoms = [atom for atom in structure.atoms if atom.resid == resid]
    if resname is None:
        return atoms
    return [atom for atom in atoms if atom.resname == resname]


def last_residue_atoms(structure: GroStructure) -> list[Atom]:
    if not structure.atoms:
        return []
    last = structure.atoms[-1]
    atoms: list[Atom] = []
    for atom in reversed(structure.atoms):
        if atom.resid != last.resid or atom.resname != last.resname:
            break
        atoms.append(atom)
    atoms.reverse()
    return atoms


def renumber_atoms(structure: GroStructure) -> GroStructure:
    new_atoms: list[Atom] = []
    for index, atom in enumerate(structure.atoms, start=1):
        new_atoms.append(
            Atom(
                resid=atom.resid,
                resname=atom.resname,
                atomname=atom.atomname,
                atomnr=index,
                x=atom.x,
                y=atom.y,
                z=atom.z,
            )
        )
    return GroStructure(title=structure.title, atoms=new_atoms, box_line=structure.box_line)


def count_water_molecules(structure: GroStructure) -> int:
    return sum(1 for atoms in contiguous_residue_groups(structure) if atoms[0].resname in WATER_NAMES)


def inserted_residue_ids(structure: GroStructure, orig_atom_count: int, gas_name: str) -> list[int]:
    water_alias = gas_name in WATER_NAMES
    residue_ids: list[int] = []
    seen: set[int] = set()
    for atom in structure.atoms:
        target = atom.resname in WATER_NAMES if water_alias else atom.resname == gas_name
        if target and atom.atomnr > orig_atom_count and atom.resid not in seen:
            residue_ids.append(atom.resid)
            seen.add(atom.resid)
    return residue_ids


def coordinates_center(atoms: list[Atom]) -> tuple[float, float, float]:
    if not atoms:
        raise ValueError("Cannot compute center of empty atom list")
    sx = sum(atom.x for atom in atoms)
    sy = sum(atom.y for atom in atoms)
    sz = sum(atom.z for atom in atoms)
    count = float(len(atoms))
    return sx / count, sy / count, sz / count
