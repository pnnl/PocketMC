from __future__ import annotations

from pathlib import Path
import re


WATER_NAMES = {"SOL", "WAT", "HOH"}
INCLUDE_RE = re.compile(r'^\s*#include\s+["<](?P<path>[^">]+)[">]')
SECTION_RE = re.compile(r"^\s*\[\s*(?P<name>[^\]]+?)\s*\]")


def read_lines(path: str | Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def write_lines(path: str | Path, lines: list[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_wat_into_sol(path: str | Path) -> None:
    lines = read_lines(path)
    new_lines: list[str] = []
    in_molecules = False
    wat_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "[ molecules ]":
            in_molecules = True
            new_lines.append(line)
            continue
        if in_molecules and stripped.startswith("["):
            in_molecules = False
        if in_molecules and stripped:
            fields = stripped.split()
            if len(fields) >= 2 and fields[0] == "WAT":
                wat_count += int(fields[1])
                continue
            if len(fields) >= 2 and fields[0] == "SOL":
                new_lines.append(f"SOL {int(fields[1]) + wat_count}")
                wat_count = 0
                continue
        new_lines.append(line)
    write_lines(path, new_lines)


def set_molecule_count(path: str | Path, name: str, count: int) -> None:
    lines = read_lines(path)
    new_lines: list[str] = []
    in_molecules = False
    updated = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "[ molecules ]":
            in_molecules = True
            new_lines.append(line)
            continue
        if in_molecules and stripped.startswith("["):
            in_molecules = False
        if in_molecules and stripped:
            fields = stripped.split()
            if len(fields) >= 2 and fields[0] == name:
                new_lines.append(f"{name} {count}")
                updated = True
                continue
        new_lines.append(line)
    if not updated:
        new_lines.append(f"{name} {count}")
    write_lines(path, new_lines)


def adjust_molecule_count(path: str | Path, gas_name: str, delta: int) -> str:
    lines = read_lines(path)
    new_lines: list[str] = []
    in_molecules = False
    chosen_name = gas_name
    updated = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "[ molecules ]":
            in_molecules = True
            new_lines.append(line)
            continue
        if in_molecules and stripped.startswith("["):
            in_molecules = False
        if in_molecules and stripped:
            fields = stripped.split()
            if len(fields) >= 2:
                mol_name = fields[0]
                if mol_name == chosen_name:
                    count = max(0, int(fields[1]) + delta)
                    new_lines.append(f"{mol_name} {count}")
                    chosen_name = mol_name
                    updated = True
                    continue
        new_lines.append(line)

    if not updated:
        new_lines.append(f"{chosen_name} {max(0, delta)}")
    write_lines(path, new_lines)
    return chosen_name


def ensure_molecule_include(path: str | Path, molecule_itp_name: str, atomtypes_itp_name: str | None = None) -> None:
    include_name = Path(molecule_itp_name).name
    lines = read_lines(path)
    changed = False

    if atomtypes_itp_name is not None:
        atomtypes_name = Path(atomtypes_itp_name).name
        if not _include_basename_present(lines, atomtypes_name):
            atomtypes_block = [
                "; Include inserted molecule atom types",
                f'#include "{atomtypes_name}"',
                "",
            ]
            lines = _insert_atomtypes_include(lines, atomtypes_block)
            changed = True

    if not _include_basename_present(lines, include_name):
        include_block = [
            "; Include inserted molecule topology",
            f'#include "{include_name}"',
            "",
        ]
        lines = _insert_before_sections(lines, {"system"}, include_block)
        changed = True

    if changed:
        write_lines(path, lines)


def _include_basename_present(lines: list[str], include_name: str) -> bool:
    for line in lines:
        match = INCLUDE_RE.match(line)
        if match and Path(match.group("path")).name == include_name:
            return True
    return False


def _section_name(line: str) -> str | None:
    match = SECTION_RE.match(line)
    if not match:
        return None
    return match.group("name").strip().lower()


def _insert_atomtypes_include(lines: list[str], include_block: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        match = INCLUDE_RE.match(line)
        if match and Path(match.group("path")).name.lower() == "forcefield.itp":
            return lines[: index + 1] + include_block + lines[index + 1 :]
    return _insert_before_sections(lines, {"moleculetype", "system"}, include_block)


def _insert_before_sections(lines: list[str], target_sections: set[str], include_block: list[str]) -> list[str]:
    new_lines: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and (_section_name(line) in target_sections):
            new_lines.extend(include_block)
            inserted = True
        new_lines.append(line)
    if not inserted:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.extend(include_block[:-1])
    return new_lines


def insert_posre_include(path: str | Path) -> None:
    lines = read_lines(path)
    include_block = ['#ifdef POSRES_CAVITY', '#include "posre_cavity.itp"', "#endif"]
    if any('#include "posre_cavity.itp"' in line for line in lines):
        return

    marker = "; Include water topology"
    new_lines: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and marker in line:
            new_lines.extend(include_block)
            inserted = True
        new_lines.append(line)
    if not inserted:
        new_lines.extend(include_block)
    write_lines(path, new_lines)
