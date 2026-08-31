from __future__ import annotations

from math import log, pi, sin, cos, sqrt
from pathlib import Path
import json
import random
import shutil

from .cavity import VoxelMask
from .clash import find_heavy_atom_clash
from .gro import (
    Atom,
    GroStructure,
    WATER_NAMES,
    coordinates_center,
    inserted_residue_ids,
    last_residue_atoms,
    parse_atom_line,
    parse_gro,
    write_gro,
)
from .helpers import get_alcove_center
from .topology import adjust_molecule_count


def _read_raw_atoms(path: str | Path) -> list[Atom]:
    raw_path = Path(path)
    if not raw_path.exists():
        return []
    atoms: list[Atom] = []
    for line in raw_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.rstrip()
        if not line:
            continue
        try:
            atoms.append(parse_atom_line(line))
        except ValueError:
            continue
    return atoms


def _rotation_matrix(rng: random.Random) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    qx = sqrt(1.0 - u1) * sin(2.0 * pi * u2)
    qy = sqrt(1.0 - u1) * cos(2.0 * pi * u2)
    qz = sqrt(u1) * sin(2.0 * pi * u3)
    qw = sqrt(u1) * cos(2.0 * pi * u3)
    return (
        (1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)),
        (2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)),
        (2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)),
    )


def _rotate_point(
    point: tuple[float, float, float],
    pivot: tuple[float, float, float],
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[float, float, float]:
    x = point[0] - pivot[0]
    y = point[1] - pivot[1]
    z = point[2] - pivot[2]
    tx = pivot[0] + rotation[0][0] * x + rotation[0][1] * y + rotation[0][2] * z
    ty = pivot[1] + rotation[1][0] * x + rotation[1][1] * y + rotation[1][2] * z
    tz = pivot[2] + rotation[2][0] * x + rotation[2][1] * y + rotation[2][2] * z
    return tx, ty, tz


def _copy_top(previous_top: str | Path, current_top: str | Path) -> None:
    shutil.copyfile(previous_top, current_top)


def _select_inserted_residue(
    structure: GroStructure,
    nmol: int,
    gas_name: str,
    orig_atom_count: int,
    rng: random.Random,
) -> int:
    candidates = inserted_residue_ids(structure, orig_atom_count, gas_name)
    if not candidates:
        raise ValueError(f"No inserted {gas_name} residues available")
    if nmol <= 1:
        return candidates[0]
    index = rng.randrange(min(len(candidates), max(1, nmol - 1)))
    return candidates[index]


def _target_residue(atom: Atom, gas_name: str, water_alias: bool) -> bool:
    if water_alias:
        return atom.resname in WATER_NAMES
    return atom.resname == gas_name


def propose_insertion(
    previous_gro: str | Path,
    current_gro: str | Path,
    previous_top: str | Path,
    current_top: str | Path,
    *,
    rvdw: float,
    gas_name: str,
    gas_gro: str | Path,
    rmax: float,
    xyz_path: str | Path,
    out_dir: str | Path,
    anchor_resid: int = 0,
    anchor_resname: str = "",
    center_atoms: list[str] | None = None,
    mask_file: str | Path | None = None,
    mask_model: VoxelMask | None = None,
    mask_dx: float = 0.0,
    seed_point: tuple[float, float, float] | None = None,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    rng = rng or random.Random()
    structure = parse_gro(previous_gro)
    gas_structure = parse_gro(gas_gro)
    gas_center = coordinates_center(gas_structure.atoms)
    water_like = gas_name in WATER_NAMES or gas_structure.atoms[0].resname in WATER_NAMES
    anchor_atom = next(
        (atom for atom in gas_structure.atoms if atom.atomname.upper() in {"OW", "O"}),
        None,
    )
    gas_anchor = (
        (anchor_atom.x, anchor_atom.y, anchor_atom.z)
        if water_like and anchor_atom is not None
        else gas_center
    )

    def placed_coordinates(
        candidate: tuple[float, float, float],
        rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    ) -> list[Atom]:
        placed_atoms: list[Atom] = []
        for gas_atom in gas_structure.atoms:
            rotated = _rotate_point((gas_atom.x, gas_atom.y, gas_atom.z), gas_anchor, rotation)
            placed_atoms.append(
                Atom(
                    resid=0,
                    resname=gas_atom.resname,
                    atomname=gas_atom.atomname,
                    atomnr=gas_atom.atomnr,
                    x=candidate[0] + (rotated[0] - gas_anchor[0]),
                    y=candidate[1] + (rotated[1] - gas_anchor[1]),
                    z=candidate[2] + (rotated[2] - gas_anchor[2]),
                )
            )
        return placed_atoms

    def non_clashing(atoms: list[Atom]) -> bool:
        return find_heavy_atom_clash(atoms, structure.atoms, rvdw) is None

    trial: tuple[float, float, float] | None = None
    trial_atoms: list[Atom] | None = None
    if mask_model is not None:
        candidate = mask_model.sample_point(rng, mask_dx)
        atoms = placed_coordinates(candidate, _rotation_matrix(rng))
        if not non_clashing(atoms):
            raise RuntimeError("Insertion proposal rejected by a heavy-atom clash")
        trial = candidate
        trial_atoms = atoms
    else:
        mask_path = Path(mask_file) if mask_file else None
        points: list[tuple[float, float, float]] = []
        if mask_path and mask_path.exists() and mask_path.stat().st_size > 0:
            for line in mask_path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) >= 3:
                    points.append((float(fields[0]), float(fields[1]), float(fields[2])))
        if points:
            base = rng.choice(points)
            jitter = mask_dx / 2.0
            candidate = (
                base[0] + (rng.random() * 2.0 - 1.0) * jitter,
                base[1] + (rng.random() * 2.0 - 1.0) * jitter,
                base[2] + (rng.random() * 2.0 - 1.0) * jitter,
            )
        else:
            center = seed_point or get_alcove_center(previous_gro, anchor_resid, anchor_resname, center_atoms or [])
            while True:
                dx = rmax * (rng.random() * 2.0 - 1.0)
                dy = rmax * (rng.random() * 2.0 - 1.0)
                dz = rmax * (rng.random() * 2.0 - 1.0)
                if dx * dx + dy * dy + dz * dz <= rmax * rmax:
                    break
            candidate = (center[0] + dx, center[1] + dy, center[2] + dz)
        atoms = placed_coordinates(candidate, _rotation_matrix(rng))
        if not non_clashing(atoms):
            raise RuntimeError("Insertion proposal rejected by a heavy-atom clash")
        trial = candidate
        trial_atoms = atoms

    if trial is None or trial_atoms is None:
        raise RuntimeError("Could not find valid insertion point")

    xyz = Path(xyz_path)
    xyz_line = f"He {trial[0] * 10.0:8.3f} {trial[1] * 10.0:8.3f} {trial[2] * 10.0:8.3f}"
    if not xyz.exists():
        xyz.write_text(f"1\n\n{xyz_line}\n", encoding="utf-8")
    else:
        existing = xyz.read_text(encoding="utf-8").splitlines()
        count = int(existing[0].strip()) if existing else 0
        updated = [str(count + 1)]
        updated.extend(existing[1:])
        updated.append(xyz_line)
        xyz.write_text("\n".join(updated) + "\n", encoding="utf-8")

    new_resid = structure.atoms[-1].resid + 1 if structure.atoms else 1
    next_atomnr = structure.atoms[-1].atomnr if structure.atoms else 0
    new_atoms = list(structure.atoms)
    for placed_atom in trial_atoms:
        next_atomnr += 1
        new_atoms.append(
            Atom(
                resid=new_resid,
                resname=placed_atom.resname,
                atomname=placed_atom.atomname,
                atomnr=next_atomnr,
                x=placed_atom.x,
                y=placed_atom.y,
                z=placed_atom.z,
            )
        )

    write_gro(current_gro, GroStructure(title=structure.title, atoms=new_atoms, box_line=structure.box_line))
    shutil.copyfile(previous_top, current_top)
    adjust_molecule_count(current_top, gas_name, +1)
    return trial


def propose_deletion(
    previous_gro: str | Path,
    current_gro: str | Path,
    previous_top: str | Path,
    current_top: str | Path,
    *,
    nmol: int,
    gas_name: str,
    orig_atom_count: int,
    out_dir: str | Path,
    candidate_resids: list[int] | None = None,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    structure = parse_gro(previous_gro)
    if candidate_resids is None and nmol - 1 <= 0:
        raise ValueError(f"nmol={nmol} implies no inserted molecules to delete")

    water_alias = gas_name in WATER_NAMES
    x_atoms = _read_raw_atoms(Path(out_dir) / "x.gro")
    def collect_candidates(atoms_source: list[Atom]) -> list[int]:
        candidates: list[int] = []
        seen: set[int] = set()
        for atom in atoms_source:
            if candidate_resids is None and atom.atomnr <= orig_atom_count:
                continue
            if not _target_residue(atom, gas_name, water_alias):
                continue
            if atom.resid not in seen:
                candidates.append(atom.resid)
                seen.add(atom.resid)
        return candidates

    candidates = list(candidate_resids or [])
    if not candidates:
        candidates = collect_candidates(x_atoms) if x_atoms else []
    if not candidates:
        candidates = collect_candidates(structure.atoms)
    if not candidates:
        raise ValueError(f"No inserted {gas_name} candidates found to delete")

    resid_to_delete = rng.choice(candidates)
    kept_atoms = [
        atom
        for atom in structure.atoms
        if not (
            atom.resid == resid_to_delete
            and _target_residue(atom, gas_name, water_alias)
            and (candidate_resids is not None or atom.atomnr > orig_atom_count)
        )
    ]
    if len(kept_atoms) == len(structure.atoms):
        raise RuntimeError(f"Deletion target residue {resid_to_delete} had no removable atoms")
    renumbered = []
    for index, atom in enumerate(kept_atoms, start=1):
        renumbered.append(Atom(atom.resid, atom.resname, atom.atomname, index, atom.x, atom.y, atom.z))
    write_gro(current_gro, GroStructure(title=structure.title, atoms=renumbered, box_line=structure.box_line))
    shutil.copyfile(previous_top, current_top)
    adjust_molecule_count(current_top, gas_name, -1)
    return resid_to_delete


def propose_rotation(
    previous_gro: str | Path,
    current_gro: str | Path,
    previous_top: str | Path,
    current_top: str | Path,
    *,
    nmol: int,
    gas_name: str,
    orig_atom_count: int,
    rvdw: float,
    candidate_resids: list[int] | None = None,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    structure = parse_gro(previous_gro)
    if candidate_resids:
        resid = candidate_resids[rng.randrange(len(candidate_resids))]
    else:
        resid = _select_inserted_residue(structure, nmol, gas_name, orig_atom_count, rng)
    rotation = _rotation_matrix(rng)
    residue_atoms = [atom for atom in structure.atoms if atom.resid == resid]
    water_pivot = next((atom for atom in residue_atoms if atom.atomname in {"OW", "O"}), None)
    pivot = (water_pivot.x, water_pivot.y, water_pivot.z) if water_pivot else coordinates_center(residue_atoms)

    rotated_atoms: list[Atom] = []
    for atom in structure.atoms:
        if atom.resid == resid:
            x, y, z = _rotate_point((atom.x, atom.y, atom.z), pivot, rotation)
            rotated_atoms.append(Atom(atom.resid, atom.resname, atom.atomname, atom.atomnr, x, y, z))
        else:
            rotated_atoms.append(atom)
    clash = find_heavy_atom_clash(
        [atom for atom in rotated_atoms if atom.resid == resid],
        [atom for atom in rotated_atoms if atom.resid != resid],
        rvdw,
    )
    if clash is not None:
        raise RuntimeError("Rotated structure creates a heavy-atom clash")
    write_gro(current_gro, GroStructure(title=structure.title, atoms=rotated_atoms, box_line=structure.box_line))
    _copy_top(previous_top, current_top)
    return resid


def propose_translation(
    previous_gro: str | Path,
    current_gro: str | Path,
    previous_top: str | Path,
    current_top: str | Path,
    *,
    nmol: int,
    gas_name: str,
    orig_atom_count: int,
    rvdw: float,
    delta: float = 0.05,
    candidate_resids: list[int] | None = None,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    structure = parse_gro(previous_gro)
    if candidate_resids:
        resid = candidate_resids[rng.randrange(len(candidate_resids))]
    else:
        resid = _select_inserted_residue(structure, nmol, gas_name, orig_atom_count, rng)
    dx = delta * (rng.random() * 2.0 - 1.0)
    dy = delta * (rng.random() * 2.0 - 1.0)
    dz = delta * (rng.random() * 2.0 - 1.0)
    translated_atoms: list[Atom] = []
    for atom in structure.atoms:
        if atom.resid == resid:
            translated_atoms.append(
                Atom(atom.resid, atom.resname, atom.atomname, atom.atomnr, atom.x + dx, atom.y + dy, atom.z + dz)
            )
        else:
            translated_atoms.append(atom)
    clash = find_heavy_atom_clash(
        [atom for atom in translated_atoms if atom.resid == resid],
        [atom for atom in translated_atoms if atom.resid != resid],
        rvdw,
    )
    if clash is not None:
        raise RuntimeError("Translated structure creates a heavy-atom clash")
    write_gro(current_gro, GroStructure(title=structure.title, atoms=translated_atoms, box_line=structure.box_line))
    _copy_top(previous_top, current_top)
    return resid


def write_position_restraints(gro_path: str | Path, kres: float, *, out_dir: str | Path) -> None:
    structure = parse_gro(gro_path)
    out_path = Path(out_dir)
    x_atoms = _read_raw_atoms(out_path / "x.gro")
    na_pro = next((atom.atomnr - 1 for atom in structure.atoms if atom.resname == "SOL"), structure.atoms[-1].atomnr)
    x2_path = out_path / "x.2"
    free_ids: set[int] = set()
    if x2_path.exists():
        for line in x2_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                free_ids.add(int(line.strip()))
            except ValueError:
                continue
    else:
        # Backward compatibility for callers that supply only the legacy
        # raw-atom file.  These serials are unambiguous only below 100,000.
        free_ids = {atom.atomnr for atom in x_atoms}
        x2_path.write_text(
            "\n".join(str(atom.atomnr) for atom in x_atoms) + ("\n" if x_atoms else ""),
            encoding="utf-8",
        )
    lines = ["[ position_restraints ]"]
    for atom_id in range(1, na_pro + 1):
        if atom_id not in free_ids:
            lines.append(f"{atom_id:5d} 1 {kres} {kres} {kres}")
    (out_path / "posre_cavity.itp").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_position_restraints_gas(
    gro_path: str | Path,
    kres: float,
    *,
    rvdw: float,
    rfree: float,
    out_dir: str | Path,
) -> None:
    structure = parse_gro(gro_path)
    trial_atoms = last_residue_atoms(structure)
    if not trial_atoms:
        write_position_restraints(gro_path, kres, out_dir=out_dir)
        return
    pivot = (trial_atoms[0].x, trial_atoms[0].y, trial_atoms[0].z)
    na_pro = next((atom.atomnr - 1 for atom in structure.atoms if atom.resname == "SOL"), structure.atoms[-1].atomnr)
    selected_resids: set[int] = set()
    for atom in structure.atoms:
        if atom.resname == "SOL":
            break
        distance = sqrt((atom.x - pivot[0]) ** 2 + (atom.y - pivot[1]) ** 2 + (atom.z - pivot[2]) ** 2)
        if distance < rvdw * rfree:
            selected_resids.add(atom.resid)

    free_ids = {atom.atomnr for atom in structure.atoms if atom.resid in selected_resids}
    tmp_atoms = [atom for atom in structure.atoms if atom.resid in selected_resids]
    (Path(out_dir) / "tmp.gro").write_text("\n".join(atom.line() for atom in tmp_atoms) + ("\n" if tmp_atoms else ""), encoding="utf-8")
    (Path(out_dir) / "tmp.dat").write_text(
        "\n".join(str(resid) for resid in sorted(selected_resids)) + ("\n" if selected_resids else ""),
        encoding="utf-8",
    )
    lines = ["[ position_restraints ]"]
    for atom_id in range(1, na_pro + 1):
        if atom_id not in free_ids:
            lines.append(f"{atom_id:5d} 1 {kres} {kres} {kres}")
    (Path(out_dir) / "posre_cavity.itp").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trajectory(
    gro_path: str | Path,
    energy: float,
    naccepted: int,
    nmol: int,
    target_nmol: int,
    *,
    trajectory_path: str | Path,
    gas_gro: str | Path | None = None,
    trial: int | None = None,
    move: str = "",
    active_resids: list[int] | tuple[int, ...] | None = None,
    provenance: dict[int, str] | None = None,
    trajectory_meta_path: str | Path | None = None,
) -> None:
    structure = parse_gro(gro_path)
    pad = max(0, target_nmol - nmol) if target_nmol > 0 else 0
    if gas_gro is None:
        template_atoms = [
            Atom(1, "WAT", "OW", 1, 0.0, 0.0, 0.0),
            Atom(1, "WAT", "HW1", 2, 0.0, 0.0, 0.0),
            Atom(1, "WAT", "HW2", 3, 0.0, 0.0, 0.0),
        ]
    else:
        template_atoms = parse_gro(gas_gro).atoms
        if not template_atoms:
            raise ValueError(f"Gas template contains no atoms: {gas_gro}")
    na_out = len(structure.atoms) + len(template_atoms) * pad
    water_atom = next((atom for atom in structure.atoms if atom.resname in WATER_NAMES and atom.atomname in {"OW", "O"}), None)
    if water_atom is None:
        water_atom = structure.atoms[0]

    lines = [f"{naccepted:4d} {energy:.6f}", f"{na_out:5d}"]
    lines.extend(atom.line() for atom in structure.atoms)
    for _dummy_index in range(pad):
        for template_atom in template_atoms:
            lines.append(
                Atom(
                    1,
                    template_atom.resname,
                    template_atom.atomname,
                    template_atom.atomnr,
                    water_atom.x,
                    water_atom.y,
                    water_atom.z,
                ).line()
            )
    lines.append(structure.box_line)

    trajectory = Path(trajectory_path)
    with trajectory.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    if trajectory_meta_path is not None:
        active = set(active_resids or ())
        origins = provenance or {}
        active_resnames = {atom.resname for atom in template_atoms}
        molecules: list[dict[str, object]] = []
        seen: set[tuple[int, str]] = set()
        for atom in structure.atoms:
            key = (atom.resid, atom.resname)
            if atom.resid not in active or atom.resname not in active_resnames or key in seen:
                continue
            seen.add(key)
            origin = origins.get(atom.resid, f"initial:{atom.resid}")
            molecules.append(
                {
                    "uid": origin,
                    "resid": atom.resid,
                    "resname": atom.resname,
                    "provenance": origin,
                }
            )
        record = {
            "schema_version": 1,
            "accepted_state": naccepted,
            "trial": trial,
            "move": move,
            "energy_kj_mol": energy,
            "active_molecule_count": len(molecules),
            "dummy_atom_start": len(structure.atoms) + 1 if pad else None,
            "dummy_atom_count": len(template_atoms) * pad,
            "template_atom_count": len(template_atoms),
            "molecules": molecules,
        }
        meta_path = Path(trajectory_meta_path)
        with meta_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_mask_trajectory(
    mask: VoxelMask,
    gro_path: str | Path,
    *,
    trajectory_path: str | Path,
    accepted: int,
    nmol: int,
    label: str,
) -> None:
    structure = parse_gro(gro_path)
    points = mask.points
    lines = [
        f"{accepted:4d} {label} nmol={nmol} points={mask.point_count} veff={mask.effective_volume:.6f}",
        f"{mask.point_count:5d}",
    ]
    for serial, point in enumerate(points, start=1):
        resid = ((serial - 1) % 99999) + 1
        atomnr = ((serial - 1) % 99999) + 1
        lines.append(Atom(resid, "CAV", "HE", atomnr, float(point[0]), float(point[1]), float(point[2])).line())
    lines.append(structure.box_line)

    trajectory = Path(trajectory_path)
    with trajectory.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def acceptance_probability(
    *,
    de: float,
    temperature: float,
    move: int,
    veff: float,
    v0: float,
    nins: int,
    rng: random.Random,
    gas_constant: float = 0.008314,
) -> int:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if gas_constant <= 0:
        raise ValueError(f"gas_constant must be positive, got {gas_constant}")
    if veff <= 0:
        raise ValueError(f"veff must be positive, got {veff}")
    if v0 <= 0:
        raise ValueError(f"v0 must be positive, got {v0}")
    if nins < 0:
        raise ValueError(f"nins must be non-negative, got {nins}")
    if move not in {1, 2, 3, 4}:
        raise ValueError(f"unsupported move code {move}")

    beta = 1.0 / (gas_constant * temperature)
    pref = 1.0
    qratio = 1.0
    if move == 1:
        pref = (veff / v0) / (nins + 1.0)
        if nins == 0:
            qratio = 0.25
    elif move == 4:
        pref = nins * (v0 / veff)
        if nins == 1:
            qratio = 4.0
    pref *= qratio
    if pref <= 0:
        return 0
    logacc = log(pref) - beta * de
    if logacc >= 0:
        return 2
    u = max(rng.random(), 1e-300)
    return 2 if log(u) < logacc else 0
