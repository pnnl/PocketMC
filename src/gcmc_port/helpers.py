from __future__ import annotations

from math import sqrt
from pathlib import Path

from .gro import (
    Atom,
    GroStructure,
    WATER_NAMES,
    contiguous_residue_groups,
    coordinates_center,
    parse_gro,
    write_gro,
    write_raw_lines,
)
from .topology import read_lines, write_lines


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def get_alcove_center(
    gro_path: str | Path,
    target_resid: int = 0,
    target_resname: str = "ATC",
    center_atoms: list[str] | None = None,
) -> tuple[float, float, float]:
    structure = parse_gro(gro_path)
    atoms_filter = set(center_atoms or [])
    if not target_resid:
        for atom in structure.atoms:
            if atom.resname == target_resname:
                target_resid = atom.resid
                break
    if not target_resid:
        raise ValueError(f"Could not find residue {target_resname} in {gro_path}")

    residue = [atom for atom in structure.atoms if atom.resid == target_resid and atom.resname == target_resname]
    if atoms_filter:
        residue = [atom for atom in residue if atom.atomname in atoms_filter] or residue
    if not residue:
        raise ValueError(f"Could not determine alcove center for {target_resid} {target_resname}")
    return coordinates_center(residue)


def get_alcove_residues(
    gro_path: str | Path,
    rmax: float,
    center: tuple[float, float, float],
    rfree: float,
    *,
    out_dir: str | Path,
) -> None:
    structure = parse_gro(gro_path)
    radius2 = (rfree * rmax) ** 2
    x_lines: list[str] = []
    x_nowat: list[str] = []
    x_solid: list[str] = []
    selected_atom_indices: list[str] = []
    for atom in structure.atoms:
        d2 = (atom.x - center[0]) ** 2 + (atom.y - center[1]) ** 2 + (atom.z - center[2]) ** 2
        if d2 >= radius2:
            continue
        line = atom.line()
        x_lines.append(line)
        selected_atom_indices.append(str(atom.atomnr))
        if atom.resname not in WATER_NAMES:
            x_nowat.append(line)
            if atom.resname != "ATC":
                x_solid.append(line)

    out_path = Path(out_dir)
    write_raw_lines(out_path / "x.gro", x_lines)
    write_raw_lines(out_path / "x_nowat.gro", x_nowat)
    write_raw_lines(out_path / "x_solid.gro", x_solid)
    # GRO display serials wrap at 100,000.  Keep the real, line-order atom
    # indices in a sidecar so position restraints remain correct for large
    # systems.
    write_raw_lines(out_path / "x.2", selected_atom_indices)


def remove_waters_near_centroid(
    input_gro: str | Path,
    output_gro: str | Path,
    center: tuple[float, float, float],
    radius: float,
) -> None:
    structure = parse_gro(input_gro)
    kept_atoms: list[Atom] = []
    for residue_atoms in contiguous_residue_groups(structure):
        is_water = residue_atoms[0].resname in WATER_NAMES
        delete = is_water and any(
            _distance((atom.x, atom.y, atom.z), center) <= radius for atom in residue_atoms
        )
        if not delete:
            kept_atoms.extend(residue_atoms)

    write_gro(output_gro, GroStructure(title=structure.title, atoms=kept_atoms, box_line=structure.box_line))


def alcove_remove_initial_waters(
    input_gro: str | Path,
    output_gro: str | Path,
    radius: float,
    center: tuple[float, float, float],
) -> None:
    structure = parse_gro(input_gro)
    kept_atoms: list[Atom] = []
    for residue_atoms in contiguous_residue_groups(structure):
        if residue_atoms[0].resname not in WATER_NAMES:
            kept_atoms.extend(residue_atoms)
            continue
        cx, cy, cz = coordinates_center(residue_atoms)
        if _distance((cx, cy, cz), center) > radius:
            kept_atoms.extend(residue_atoms)

    write_gro(output_gro, GroStructure(title=structure.title, atoms=kept_atoms, box_line=structure.box_line))


def downsize_gro_legacy(
    previous_gro: str | Path,
    center: tuple[float, float, float],
    acs: int,
    output_gro: str | Path,
    *,
    water_cutoff: float = 2.5,
) -> None:
    structure = parse_gro(previous_gro)
    acs_prefix = str(acs)
    new_atoms: list[Atom] = []
    atomnr = 0
    for atom in structure.atoms:
        include = False
        if not atom.resname.startswith("SOL") and str(atom.resid).startswith(acs_prefix):
            include = True
        elif atom.resname == "SOL":
            distance = _distance((atom.x, atom.y, atom.z), center)
            if distance < water_cutoff:
                include = True
        if include:
            atomnr += 1
            new_atoms.append(
                Atom(atom.resid, atom.resname, atom.atomname, atomnr, atom.x, atom.y, atom.z)
            )
    write_gro(output_gro, GroStructure(title=structure.title, atoms=new_atoms, box_line=structure.box_line))


def downsize_top_legacy(previous_top: str | Path, previous_gro: str | Path, acs: int, output_top: str | Path) -> None:
    lines = read_lines(previous_top)
    nwaters = sum(1 for atom in parse_gro(previous_gro).atoms if atom.resname == "SOL" and atom.atomname == "OW")
    res_prefix = str(acs)
    atom_map: dict[int, int] = {}
    new_lines: list[str] = []
    section = ""
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]").strip().lower()
            new_lines.append(line)
            idx += 1
            continue

        if section == "atoms" and stripped and not stripped.startswith(";"):
            fields = line.split()
            if len(fields) >= 6 and fields[2].startswith(res_prefix):
                new_index = len(atom_map) + 1
                atom_map[int(fields[0])] = new_index
                fields[0] = str(new_index)
                fields[5] = str(new_index)
                new_lines.append(" ".join(fields))
            elif not stripped:
                new_lines.append(line)
            idx += 1
            continue

        if section in {"bonds", "pairs", "angles", "dihedrals"} and stripped and not stripped.startswith(";"):
            fields = line.split()
            needed = {"bonds": 2, "pairs": 2, "angles": 3, "dihedrals": 4}[section]
            old = line
            mapped: list[str] = []
            keep = True
            for pos in range(needed):
                atom_id = int(fields[pos])
                if atom_id not in atom_map:
                    keep = False
                    break
                mapped.append(str(atom_map[atom_id]))
            if keep:
                fields[:needed] = mapped
                new_lines.append(" ".join(fields) + f"  ; Old: {old}")
            idx += 1
            continue

        if stripped.startswith("SOL"):
            new_lines.append(f"SOL {nwaters}")
        elif stripped.startswith("NA"):
            new_lines.append(f"; {line}")
        else:
            new_lines.append(line)
        idx += 1

    write_lines(output_top, new_lines)


def build_cavity_mask(
    xgro: str | Path,
    center: tuple[float, float, float],
    rseed: float,
    dx: float,
    rexcl: float,
    outprefix: str | Path,
) -> None:
    if dx <= 0 or rseed <= 0 or rexcl <= 0:
        raise ValueError("dx, rseed, and rexcl must be positive")

    xgro_lines = Path(xgro).read_text(encoding="utf-8").splitlines()
    atoms: list[tuple[float, float, float]] = []
    for line in xgro_lines:
        if not line.strip():
            continue
        try:
            atom = Atom(
                resid=int(line[0:5]),
                resname=line[5:10].strip(),
                atomname=line[10:15].strip(),
                atomnr=int(line[15:20]),
                x=float(line[20:28]),
                y=float(line[28:36]),
                z=float(line[36:44]),
            )
        except ValueError:
            continue
        atoms.append((atom.x, atom.y, atom.z))

    outprefix = Path(outprefix)
    mask_path = outprefix.with_name(f"{outprefix.name}_mask.dat")
    pdb_path = outprefix.with_name(f"{outprefix.name}_points.pdb")
    meta_path = outprefix.with_suffix(".meta")

    x0, y0, z0 = center
    mask_points: list[tuple[float, float, float, float]] = []
    radius2 = rseed * rseed
    x = x0 - rseed
    while x <= x0 + rseed + 1e-12:
        y = y0 - rseed
        while y <= y0 + rseed + 1e-12:
            z = z0 - rseed
            while z <= z0 + rseed + 1e-12:
                dx0 = x - x0
                dy0 = y - y0
                dz0 = z - z0
                if dx0 * dx0 + dy0 * dy0 + dz0 * dz0 <= radius2:
                    min_distance = min((_distance((x, y, z), atom) for atom in atoms), default=1e9)
                    if min_distance >= rexcl:
                        mask_points.append((x, y, z, min_distance))
                z += dx
            y += dx
        x += dx

    mask_path.write_text(
        "\n".join(f"{xv:.8f} {yv:.8f} {zv:.8f}" for xv, yv, zv, _ in mask_points) + ("\n" if mask_points else ""),
        encoding="utf-8",
    )

    pdb_lines: list[str] = []
    for serial, (xv, yv, zv, _) in enumerate(mask_points, start=1):
        resid = ((serial - 1) % 9999) + 1
        pdb_lines.append(
            f"HETATM{serial:5d}  HE  CAV A{resid:4d}    {xv*10:8.3f}{yv*10:8.3f}{zv*10:8.3f}  1.00  0.00          He"
        )
    pdb_lines.append("END")
    pdb_path.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")

    if mask_points:
        xs = [value[0] for value in mask_points]
        ys = [value[1] for value in mask_points]
        zs = [value[2] for value in mask_points]
        ds = [value[3] for value in mask_points]
        meta_lines = [
            f"seed_nm {x0} {y0} {z0}",
            f"Rseed_nm {rseed}",
            f"dx_nm {dx}",
            f"rexcl_nm {rexcl}",
            f"mask_points {len(mask_points)}",
            f"Veff_nm3 {len(mask_points) * dx * dx * dx}",
            f"bbox_nm {min(xs)} {max(xs)} {min(ys)} {max(ys)} {min(zs)} {max(zs)}",
            f"clearance_min_nm {min(ds)}",
            f"clearance_avg_nm {sum(ds) / len(ds)}",
        ]
    else:
        meta_lines = [
            f"seed_nm {x0} {y0} {z0}",
            f"Rseed_nm {rseed}",
            f"dx_nm {dx}",
            f"rexcl_nm {rexcl}",
            "mask_points 0",
            "Veff_nm3 0.0",
        ]
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
