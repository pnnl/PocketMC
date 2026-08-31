from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from gcmc_port.gro import Atom, GroStructure

from .geometry import parse_residue_token


@dataclass(frozen=True, slots=True)
class AnchorResolution:
    requested: str
    resolved: str
    warning: str = ""


def _residue_token(residue: Any) -> str:
    return f"{int(residue.resid)}{str(residue.resname).upper()}"


def _residue_description(residue: Any) -> str:
    token = _residue_token(residue)
    segid = str(getattr(residue, "segid", "")).strip()
    try:
        resnum = residue.resnum
    except Exception:
        resnum = None
    details = []
    if resnum is not None and int(resnum) != int(residue.resid):
        details.append(f"resnum={int(resnum)}")
    if segid:
        details.append(f"segid={segid}")
    return token + (f" ({', '.join(details)})" if details else "")


def mda_anchor_candidates(universe: Any, token: str) -> list[Any]:
    """Return exact or same-resname residues in deterministic resolution order."""
    resid, resname = parse_residue_token(token)
    residues = list(universe.residues)
    if resname is not None:
        residues = [item for item in residues if str(item.resname).upper() == resname]
    if resid is None:
        return residues
    exact_resid = [item for item in residues if int(item.resid) == resid]
    if exact_resid:
        return exact_resid
    exact_resnum = []
    for item in residues:
        try:
            resnum = item.resnum
        except Exception:
            continue
        if resnum is not None and int(resnum) == resid:
            exact_resnum.append(item)
    if exact_resnum:
        return exact_resnum
    return residues if resname is not None else []


def select_mda_anchor(
    universe: Any,
    token: str,
    atom_names: tuple[str, ...],
    *,
    context: str = "cavity",
) -> tuple[Any, AnchorResolution]:
    """Select one anchor residue, falling back by unique resname across renumbered cases."""
    requested_resid, requested_resname = parse_residue_token(token)
    candidates = mda_anchor_candidates(universe, token)
    exact = [
        item for item in candidates
        if (requested_resid is None or int(item.resid) == requested_resid)
        and (requested_resname is None or str(item.resname).upper() == requested_resname)
    ]
    chosen_pool = exact or candidates
    if not chosen_pool:
        available = sorted(
            {
                _residue_token(item)
                for item in universe.residues
                if requested_resname is None or str(item.resname).upper() == requested_resname
            }
        )
        suffix = f" Available matching residues: {', '.join(available)}." if available else ""
        raise ValueError(f"{context} anchor matched no residue: {token}.{suffix}")
    if len(chosen_pool) > 1:
        choices = ", ".join(_residue_description(item) for item in chosen_pool)
        raise ValueError(
            f"{context} anchor {token!r} is ambiguous in this case: {choices}. "
            "Set this case's cavity_anchor to one exact residue token."
        )
    residue = chosen_pool[0]
    group = residue.atoms
    warning_parts: list[str] = []
    if atom_names:
        named = group.select_atoms("name " + " ".join(atom_names))
        if named.n_atoms:
            group = named
        else:
            warning_parts.append(
                f"configured anchor atoms {', '.join(atom_names)} were absent; using all atoms in {_residue_token(residue)}"
            )
    resolved = _residue_token(residue)
    if resolved != token:
        warning_parts.insert(0, f"anchor {token} resolved by residue name to {resolved}")
    resolution = AnchorResolution(token, resolved, "; ".join(warning_parts))
    return group, resolution


def _gro_residue_groups(structure: GroStructure) -> list[list[Atom]]:
    groups: list[list[Atom]] = []
    for atom in structure.atoms:
        if not groups or (groups[-1][0].resid, groups[-1][0].resname) != (atom.resid, atom.resname):
            groups.append([atom])
        else:
            groups[-1].append(atom)
    return groups


def select_gro_anchor(
    structure: GroStructure,
    token: str,
    atom_names: Iterable[str],
    *,
    context: str = "cavity",
) -> tuple[list[Atom], AnchorResolution]:
    resid, resname = parse_residue_token(token)
    groups = _gro_residue_groups(structure)
    same_name = [group for group in groups if resname is None or group[0].resname.upper() == resname]
    exact = [group for group in same_name if resid is None or group[0].resid == resid]
    chosen_pool = exact or (same_name if resname is not None else [])
    if not chosen_pool:
        raise ValueError(f"{context} anchor matched no residue: {token}")
    if len(chosen_pool) > 1:
        choices = ", ".join(f"{group[0].resid}{group[0].resname}" for group in chosen_pool)
        raise ValueError(
            f"{context} anchor {token!r} is ambiguous in this case: {choices}. "
            "Set this case's cavity_anchor to one exact residue token."
        )
    group = chosen_pool[0]
    wanted = {str(name).upper() for name in atom_names}
    named = [atom for atom in group if atom.atomname.upper() in wanted]
    warning_parts: list[str] = []
    if wanted and not named:
        warning_parts.append(
            f"configured anchor atoms {', '.join(sorted(wanted))} were absent; using all atoms in {group[0].resid}{group[0].resname}"
        )
    selected = named or group
    resolved = f"{group[0].resid}{group[0].resname.upper()}"
    if resolved != token:
        warning_parts.insert(0, f"anchor {token} resolved by residue name to {resolved}")
    return selected, AnchorResolution(token, resolved, "; ".join(warning_parts))
