from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tomllib
from typing import Any, Iterable

from gcmc_port.config import load_config as load_gcmc_config


TRAJECTORY_SUFFIXES = {".xtc", ".trr", ".tng", ".dcd", ".nc", ".ncdf"}
TOPOLOGY_SUFFIXES = {".tpr", ".gro", ".pdb", ".psf", ".prmtop", ".top"}
IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", ".pytest_cache",
    "analysis-results", "analysis-cache", "node_modules", "tmp",
}
POCKETMC_INTERNAL_TRAJECTORIES = {
    "traj.trr", "md.trr", "start.trr", "em.trr", "minim.trr", "confout.trr",
}


@dataclass(frozen=True, slots=True)
class DiscoveredCase:
    case_id: str
    directory: Path
    md_status: str
    pocketmc_status: str
    pocketmc_derived_md: bool
    topology: Path | None
    trajectory: Path | None
    md_alternatives: tuple[str, ...]
    mc_trajectory: Path | None
    mc_log: Path | None
    trajectory_meta: Path | None
    gcmc_configs: tuple[str, ...]
    evidence: tuple[str, ...]
    notes: tuple[str, ...]
    frame_count: int | None = None
    time_start_ps: float | None = None
    time_stop_ps: float | None = None
    cavity_mode: str | None = None
    cavity_mask: Path | None = None
    cavity_meta: Path | None = None
    cavity_points: Path | None = None
    cavity_nearby_residues: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "directory", "topology", "trajectory", "mc_trajectory", "mc_log", "trajectory_meta",
            "cavity_mask", "cavity_meta", "cavity_points", "cavity_nearby_residues",
        ):
            payload[key] = None if payload[key] is None else str(payload[key])
        return payload


def _safe_case_id(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
        raw = "-".join(relative.parts) or path.name
    except ValueError:
        raw = path.name
    safe = "".join(character if character.isalnum() or character in "_.-" else "_" for character in raw)
    return safe.strip(".") or "case"


def _looks_like_gcmc_config(path: Path) -> bool:
    if "example" in path.stem.lower():
        return False
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception:
        return False
    return "paths" in data and "simulation" in data and ("anchor" in data or "cavity" in data)


def _walk_directories(root: Path, max_depth: int) -> Iterable[Path]:
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        yield directory
        if depth >= max_depth:
            continue
        try:
            children = sorted(
                (
                    item for item in directory.iterdir()
                    if item.is_dir() and not item.is_symlink() and item.name not in IGNORED_DIRECTORIES and not item.name.startswith(".")
                ),
                reverse=True,
            )
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in children)


def _topology_for(trajectory: Path, topologies: list[Path]) -> Path | None:
    tprs = [path for path in topologies if path.suffix.lower() == ".tpr"]
    same = [path for path in tprs if path.stem == trajectory.stem]
    if same:
        return same[0]
    named = [path for path in tprs if path.name.lower() in {"md.tpr", "topol.tpr", "production.tpr"}]
    if named:
        return named[0]
    if tprs:
        return tprs[0]
    structures = [path for path in topologies if path.suffix.lower() in {".gro", ".pdb", ".psf", ".prmtop"}]
    preferred = [path for path in structures if path.name.lower() in {"previous.gro", "confout.gro", "topology.pdb", "system.pdb"}]
    return (preferred or structures or [None])[0]


def _deep_metadata(topology: Path | None, trajectory: Path | None) -> tuple[int | None, float | None, float | None, str | None]:
    if topology is None or trajectory is None:
        return None, None, None, None
    try:
        import MDAnalysis as mda
        universe = mda.Universe(str(topology), str(trajectory))
        count = len(universe.trajectory)
        if count:
            universe.trajectory[0]
            start = float(universe.trajectory.ts.time)
            universe.trajectory[-1]
            stop = float(universe.trajectory.ts.time)
        else:
            start = stop = None
        close = getattr(universe.trajectory, "close", None)
        if close is not None:
            close()
        return count, start, stop, None
    except Exception as exc:
        return None, None, None, f"deep inspection failed: {exc}"


def _ancestor_gcmc_configs(directory: Path, root: Path) -> list[Path]:
    output: list[Path] = []
    current = directory
    upper_limit = root
    for _ in range(2):
        if upper_limit.parent == upper_limit:
            break
        upper_limit = upper_limit.parent
    while True:
        try:
            output.extend(
                item for item in current.glob("*.toml")
                if item.is_file() and _looks_like_gcmc_config(item)
            )
        except OSError:
            pass
        if output or current == upper_limit or current.parent == current:
            break
        current = current.parent
    return sorted({item.resolve() for item in output}, key=lambda item: str(item).lower())


def _first_existing(paths: Iterable[Path | None]) -> Path | None:
    return next((path.resolve() for path in paths if path is not None and path.exists()), None)


def _find_named_below(directory: Path, patterns: tuple[str, ...], *, max_depth: int = 3) -> list[Path]:
    found: list[Path] = []
    for child in _walk_directories(directory, max_depth):
        try:
            for pattern in patterns:
                found.extend(item.resolve() for item in child.glob(pattern) if item.is_file())
        except OSError:
            continue
    return sorted(set(found), key=lambda item: (len(item.parts), str(item).lower()))


def _cavity_details(directory: Path, configs: tuple[str, ...]) -> tuple[str | None, Path | None, Path | None, Path | None, Path | None]:
    mode: str | None = None
    configured_mask: Path | None = None
    configured_meta: Path | None = None
    for config_path in configs:
        try:
            gcmc = load_gcmc_config(config_path)
        except Exception:
            continue
        mode = gcmc.cavity.mode
        if gcmc.cavity.mask_file is not None:
            configured_mask = gcmc.cavity.mask_file.resolve()
        if gcmc.cavity.mask_meta is not None:
            configured_meta = gcmc.cavity.mask_meta.resolve()
        if mode:
            break

    masks = _find_named_below(directory, ("*_mask.dat", "cavity_mask.gro"))
    mask = _first_existing((configured_mask, *masks))
    if mode is None and mask is not None:
        mode = "mask"
    if mask is None:
        return mode, None, _first_existing((configured_meta,)), None, None

    if mask.name.endswith("_mask.dat"):
        prefix = mask.with_name(mask.name[:-9])
        inferred_meta = prefix.with_suffix(".meta.json")
        inferred_points = prefix.with_name(prefix.name + "_points.pdb")
        inferred_nearby = prefix.with_name(prefix.name + "_nearby_residues.tsv")
    else:
        prefix = mask.with_suffix("")
        inferred_meta = mask.with_suffix(mask.suffix + ".meta.json")
        inferred_points = prefix.with_name(prefix.name + "_points.pdb")
        inferred_nearby = prefix.with_name(prefix.name + "_nearby_residues.tsv")
    meta = _first_existing((configured_meta, inferred_meta, *_find_named_below(mask.parent, ("*.meta.json",), max_depth=1)))
    points = _first_existing((inferred_points, *_find_named_below(mask.parent, ("*_points.pdb",), max_depth=1)))
    nearby = _first_existing((inferred_nearby, *_find_named_below(mask.parent, ("*_nearby_residues.tsv", "*.tsv"), max_depth=1)))
    return mode, mask, meta, points, nearby


def discover_cases(root: str | Path = ".", *, max_depth: int = 4, deep: bool = False) -> list[DiscoveredCase]:
    scan_root = Path(root).expanduser().resolve()
    if not scan_root.is_dir():
        raise NotADirectoryError(scan_root)
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    output: list[DiscoveredCase] = []
    for directory in _walk_directories(scan_root, max_depth):
        try:
            files = [item for item in directory.iterdir() if item.is_file()]
        except OSError:
            continue
        by_lower = {item.name.lower(): item for item in files}
        mc_log = by_lower.get("mc.log")
        meta = by_lower.get("trajectory.meta.jsonl")
        mc_trajectory = by_lower.get("trajectory.gro")
        local_configs = [item.resolve() for item in files if item.suffix.lower() == ".toml" and _looks_like_gcmc_config(item)]
        config_paths = local_configs or _ancestor_gcmc_configs(directory, scan_root)
        configs = tuple(str(item) for item in config_paths)
        evidence: list[str] = []
        if mc_log is not None:
            evidence.append("mc.log")
        if meta is not None:
            evidence.append("trajectory.meta.jsonl")
        if mc_trajectory is not None:
            evidence.append("trajectory.gro")
        if configs:
            evidence.append("PocketMC/GCMC config")
        strong_count = sum(value is not None for value in (mc_log, meta, mc_trajectory))
        ancestor_status = "not_detected"
        if strong_count == 0 and not configs and directory != scan_root:
            ancestor = directory.parent
            while ancestor == scan_root or scan_root in ancestor.parents:
                try:
                    ancestor_names = {item.name.lower() for item in ancestor.iterdir() if item.is_file()}
                except OSError:
                    ancestor_names = set()
                ancestor_markers = [name for name in ("mc.log", "trajectory.gro", "trajectory.meta.jsonl") if name in ancestor_names]
                if ancestor_markers:
                    evidence.extend(f"ancestor:{name}" for name in ancestor_markers)
                    ancestor_status = "confirmed" if len(ancestor_markers) >= 2 else "probable"
                    break
                if ancestor == scan_root:
                    break
                ancestor = ancestor.parent
        pocketmc_status = "confirmed" if (strong_count >= 2 or configs) else ("probable" if strong_count else "not_detected")
        if pocketmc_status == "not_detected" and ancestor_status != "not_detected":
            pocketmc_status = ancestor_status

        trajectories = [item for item in files if item.suffix.lower() in TRAJECTORY_SUFFIXES]
        gro_candidates = [
            item for item in files
            if item.suffix.lower() == ".gro" and "traj" in item.stem.lower()
            and item.name.lower() != "cavity_trajectory.gro"
            and not (item.name.lower() == "trajectory.gro" and pocketmc_status != "not_detected")
        ]
        topologies = [item for item in files if item.suffix.lower() in TOPOLOGY_SUFFIXES]
        strong_md: list[tuple[Path, Path]] = []
        ambiguous: list[tuple[Path, Path]] = []
        for trajectory in sorted(trajectories, key=lambda path: (path.suffix.lower() == ".trr", path.name.lower())):
            topology = _topology_for(trajectory, topologies)
            if topology is None:
                continue
            internal = trajectory.name.lower() in POCKETMC_INTERNAL_TRAJECTORIES
            if trajectory.suffix.lower() == ".trr" and pocketmc_status != "not_detected" and internal:
                ambiguous.append((topology, trajectory))
            else:
                strong_md.append((topology, trajectory))
        for trajectory in gro_candidates:
            topology = _topology_for(trajectory, [item for item in topologies if item != trajectory])
            if topology is not None:
                ambiguous.append((topology, trajectory))
        selected = strong_md[0] if strong_md else (ambiguous[0] if ambiguous else None)
        md_status = "confirmed" if strong_md else ("ambiguous" if ambiguous else "not_detected")
        notes: list[str] = []
        if ambiguous:
            notes.append("PocketMC minimization-like TRR requires explicit physical-MD confirmation")
        alternatives = tuple(f"{topology.name} + {trajectory.name}" for topology, trajectory in (strong_md + ambiguous)[1:])
        if selected is None and mc_trajectory is None:
            continue
        count = start = stop = None
        if deep and selected is not None:
            count, start, stop, warning = _deep_metadata(selected[0], selected[1])
            if warning:
                notes.append(warning)
            elif count is not None and count < 2:
                md_status = "ambiguous"
                notes.append("trajectory contains fewer than two frames")
            elif count is not None and count >= 2 and selected[1].suffix.lower() == ".gro" and pocketmc_status == "not_detected":
                md_status = "confirmed"
        cavity_mode, cavity_mask, cavity_meta, cavity_points, cavity_nearby = _cavity_details(directory, configs)
        output.append(
            DiscoveredCase(
                case_id=_safe_case_id(directory, scan_root),
                directory=directory.resolve(),
                md_status=md_status,
                pocketmc_status=pocketmc_status,
                pocketmc_derived_md=selected is not None and pocketmc_status in {"confirmed", "probable"},
                topology=None if selected is None else selected[0].resolve(),
                trajectory=None if selected is None else selected[1].resolve(),
                md_alternatives=alternatives,
                mc_trajectory=None if mc_trajectory is None else mc_trajectory.resolve(),
                mc_log=None if mc_log is None else mc_log.resolve(),
                trajectory_meta=None if meta is None else meta.resolve(),
                gcmc_configs=configs,
                evidence=tuple(evidence),
                notes=tuple(notes),
                frame_count=count,
                time_start_ps=start,
                time_stop_ps=stop,
                cavity_mode=cavity_mode,
                cavity_mask=cavity_mask,
                cavity_meta=cavity_meta,
                cavity_points=cavity_points,
                cavity_nearby_residues=cavity_nearby,
            )
        )
    output.sort(key=lambda item: str(item.directory).lower())
    return output


def cases_json(cases: list[DiscoveredCase]) -> str:
    return json.dumps([item.to_dict() for item in cases], indent=2, ensure_ascii=False)


def format_cases(cases: list[DiscoveredCase]) -> str:
    if not cases:
        return "No analyzable cases were detected."
    headers = ("#", "Case", "Physical MD", "PocketMC", "Cavity", "Recommended input", "Directory / evidence")
    rows: list[tuple[str, ...]] = []
    for index, item in enumerate(cases, start=1):
        recommended = ""
        if item.topology and item.trajectory:
            recommended = f"{item.topology.name} + {item.trajectory.name}"
        elif item.mc_trajectory:
            recommended = item.mc_trajectory.name
        rows.append(
            (
                str(index), item.case_id, item.md_status, item.pocketmc_status,
                item.cavity_mode or "not detected", recommended,
                f"{item.directory} | {', '.join(item.evidence) or 'none'}",
            )
        )
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(len(headers))]
    line = "  ".join(headers[index].ljust(widths[index]) for index in range(len(headers)))
    divider = "  ".join("-" * width for width in widths)
    body = ["  ".join(row[index].ljust(widths[index]) for index in range(len(headers))) for row in rows]
    return "\n".join([line, divider, *body])
