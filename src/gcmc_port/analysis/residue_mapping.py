from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H", "CYX": "C", "ASH": "D", "GLH": "E",
}


@dataclass(frozen=True, slots=True)
class MappingResult:
    display_by_resindex: dict[int, str]
    simulation_by_resindex: dict[int, str]
    homolog_by_resindex: dict[int, str]
    metadata: dict[str, Any]


def _residue_chain(residue: Any) -> str:
    segid = str(getattr(residue, "segid", "")).strip()
    if segid:
        return segid
    atoms = residue.atoms
    try:
        values = [str(value).strip() for value in atoms.chainIDs if str(value).strip()]
    except Exception:
        values = []
    return values[0] if values else ""


def _residues(universe: Any, selection: str, chain: str = "") -> list[Any]:
    selected = universe.select_atoms(selection).residues
    return [residue for residue in selected if not chain or _residue_chain(residue) == chain]


def _sequence(residues: list[Any]) -> str:
    return "".join(AA3_TO_1.get(str(residue.resname).upper(), "X") for residue in residues)


def _global_pairs(first: str, second: str) -> tuple[list[tuple[int, int]], float]:
    """Needleman-Wunsch mapping with deterministic match/mismatch/gap scores."""
    n, m = len(first), len(second)
    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        scores[i][0] = -2 * i
        trace[i][0] = "U"
    for j in range(1, m + 1):
        scores[0][j] = -2 * j
        trace[0][j] = "L"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = scores[i - 1][j - 1] + (2 if first[i - 1] == second[j - 1] else -1)
            up = scores[i - 1][j] - 2
            left = scores[i][j - 1] - 2
            best = max(diagonal, up, left)
            scores[i][j] = best
            trace[i][j] = "D" if diagonal == best else ("U" if up == best else "L")
    pairs: list[tuple[int, int]] = []
    matches = 0
    i, j = n, m
    while i or j:
        direction = trace[i][j]
        if direction == "D":
            i -= 1
            j -= 1
            pairs.append((i, j))
            matches += int(first[i] == second[j])
        elif direction == "U":
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs, matches / len(pairs) if pairs else 0.0


def _source_mapping(mda: Any, simulation: list[Any], path: Path, chain: str) -> tuple[dict[int, str], float, int]:
    source = mda.Universe(str(path), convert_units=False)
    reference = _residues(source, "protein", chain)
    if not reference:
        raise ValueError(f"No protein residues found in mapping source {path} for chain {chain or '(any)'}")
    pairs, identity = _global_pairs(_sequence(simulation), _sequence(reference))
    mapping = {
        int(simulation[sim_index].resindex): f"{int(reference[ref_index].resid)}{str(reference[ref_index].resname).upper()}"
        for sim_index, ref_index in pairs
    }
    close = getattr(source.trajectory, "close", None)
    if close is not None:
        close()
    return mapping, identity, len(reference)


def resolve_residue_mapping(mda: Any, universe: Any, config: Any) -> MappingResult:
    simulation = _residues(universe, config.cavity.protein_selection)
    simulation_labels = {
        int(residue.resindex): f"{int(residue.resid)}{str(residue.resname).upper()}" for residue in simulation
    }
    display = dict(simulation_labels)
    homolog: dict[int, str] = {}
    metadata: dict[str, Any] = {"simulation_residue_count": len(simulation)}
    if config.analysis.canonical_source is not None:
        canonical, identity, count = _source_mapping(
            mda, simulation, config.analysis.canonical_source, config.analysis.canonical_chain
        )
        display.update(canonical)
        metadata.update(
            {
                "canonical_source": str(config.analysis.canonical_source),
                "canonical_chain": config.analysis.canonical_chain,
                "canonical_residue_count": count,
                "canonical_alignment_identity": identity,
            }
        )
    if config.analysis.homolog_source is not None:
        homolog, identity, count = _source_mapping(
            mda, simulation, config.analysis.homolog_source, config.analysis.homolog_chain
        )
        metadata.update(
            {
                "homolog_source": str(config.analysis.homolog_source),
                "homolog_chain": config.analysis.homolog_chain,
                "homolog_residue_count": count,
                "homolog_alignment_identity": identity,
            }
        )
    return MappingResult(display, simulation_labels, homolog, metadata)
