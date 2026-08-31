from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import sys
from typing import Iterable

from ..pathing import portable_path
from .anchors import mda_anchor_candidates
from .cavity_setup import prepare_analysis_cavities
from .config import load_analysis_config, validate_analysis_config
from .discovery import DiscoveredCase, discover_cases, format_cases
from .grand_alignment import (
    CompletedAnalysisRoot,
    ExistingGrandAlignment,
    common_substrates,
    default_fixed_substrates,
    discover_completed_analysis_roots,
    discover_grand_alignment_outputs,
    grand_align_analysis_roots,
    load_grand_plot_style,
    repair_grand_alignment_output,
    replot_grand_alignment,
    substrate_coverage,
)
from .models import ANALYSIS_CACHE_VERSION, AnalysisConfig, RunResult, config_for_dataset, expand_tasks
from .pose import run_pose_stage
from .plot_editor import (
    discover_plot_targets,
    load_plot_style,
    render_plot_targets,
    save_plot_style,
    stale_analysis_runs,
    stale_pose_hydration_runs,
)
from .slurm import render_analysis_launchers


ANSI_RESET = "\033[0m"
ANSI = {
    "cyan": "\033[96m",
    "blue": "\033[94m",
    "white": "\033[97m",
    "yellow": "\033[93m",
    "green": "\033[92m",
    "red": "\033[91m",
    "dim": "\033[2m",
    "bold": "\033[1m",
}


@dataclass(frozen=True, slots=True)
class Choice:
    key: str
    label: str
    description: str


@dataclass(frozen=True, slots=True)
class CavityBundle:
    directory: Path
    mask: Path | None
    meta: Path | None
    points: Path | None
    nearby: Path | None
    build_source: Path | None = None
    build_output_prefix: Path | None = None

    @property
    def deferred(self) -> bool:
        return self.build_source is not None and self.build_output_prefix is not None


@dataclass(frozen=True, slots=True)
class ExistingAnalysis:
    manifest: Path
    config_path: Path
    output_root: Path
    failures: tuple[dict[str, str], ...]
    modified_ns: int
    status: str = "partial"


CHOICE_DETAILS = {
    "md": ("Physical MD", "Analyze a physical MD topology/trajectory pair."),
    "pocketmc": ("PocketMC states", "Analyze accepted MC states; these are not physical-time dynamics."),
    "single": ("Single run", "Configure one topology/trajectory input pair."),
    "gcmc": ("GCMC expansion", "Expand replica/sweep inputs from an existing GCMC TOML."),
    "water": ("Water", "Track SOL/WAT/HOH oxygen sites."),
    "co": ("CO", "Track the bundled COM carbon-monoxide residue."),
    "custom": ("Manual selection", "Provide residue names and representative atoms yourself."),
    "sphere": ("Sphere", "Use a radius around an anchor residue/atom center."),
    "mask": ("Voxel mask", "Use GCMC cavity *_mask.dat and *.meta.json outputs."),
    "atom": ("Selected atom", "Use the configured atom name as the molecule position."),
    "cog": ("Center of geometry", "Use the unweighted center of selected atoms."),
    "com": ("Center of mass", "Use the mass-weighted center of selected atoms."),
    "yes": ("Yes", "Continue with this option."),
    "no": ("No", "Do not enable this option."),
    "grand-align": (
        "Grand-align saved maps",
        "Reuse saved 3D NPZ data, fix a chosen substrate, and redraw every map in one common frame.",
    ),
    "new-setup": ("Start a new setup", "Ignore completed outputs and continue to case discovery."),
    "exit": ("Exit", "Leave completed analyses and their files unchanged."),
    "repair-grand": (
        "Repair and replot",
        "Upgrade old grand plot metadata/script and redraw PNGs from aligned NPZ data only.",
    ),
    "replot-grand": (
        "Replot saved grand maps",
        "Use plot_grand_aligned.py settings and redraw from aligned NPZ data only.",
    ),
    "new-grand": (
        "Create another grand alignment",
        "Select completed analysis batches and write a separate new aligned result.",
    ),
}

TASK_CHOICES = (
    Choice("lifetime", "MD lifetimes", "Cavity visits, occupancy, censoring, and residence-time summaries."),
    Choice("paths", "MD paths", "Entry/exit/resident molecule paths and nearby-residue contacts."),
    Choice("mc-states", "MC accepted states", "Accepted-state occupancy and MC move statistics (not physical time)."),
    Choice("density", "Density maps", "3D density grids, projections, and cube/NPZ outputs."),
    Choice("plots", "Plots", "Render per-case and aggregate PNG summaries."),
    Choice("vmd", "VMD session", "Write VMD scripts; the wizard does not launch the GUI."),
    Choice("pose-clusters", "Pose clusters", "Build common substrate-pose clusters from physical MD."),
    Choice("pose-hydration", "Pose hydration", "Calculate cluster-conditioned hydration density."),
    Choice("compare-hydration", "Hydration comparison", "Compare pose-conditioned hydration across groups."),
)


def _supports_color() -> bool:
    return not os.environ.get("NO_COLOR") and sys.stdout.isatty()


def _style(text: str, *styles: str) -> str:
    if not _supports_color():
        return text
    return "".join(ANSI[item] for item in styles) + text + ANSI_RESET


def _banner() -> None:
    print()
    print(_style("=" * 78, "cyan"))
    print(_style(" PocketMC Integrated Analysis Wizard ", "bold", "white"))
    print(_style(" Discover MD results first, preserve per-case cavity inputs, and write runnable jobs.", "dim", "cyan"))
    print(_style("=" * 78, "cyan"))


def _step(number: int, title: str, description: str) -> None:
    print()
    print(_style("-" * 78, "blue"))
    print(f"{_style(f'[Step {number}]', 'bold', 'blue')} {_style(title, 'bold', 'white')}")
    print(_style(description, "cyan"))


def _table(title: str, rows: Iterable[tuple[str, str, str]]) -> None:
    print()
    print(_style(title, "bold", "white"))
    print(_style("-" * max(28, min(78, len(title))), "cyan"))
    for number, label, description in rows:
        print(f"  {_style(str(number).rjust(2), 'bold', 'cyan')}  {_style(label, 'bold', 'white')}  {_style(description, 'dim')}")


def _note(message: str) -> None:
    print(_style(f"  {message}", "yellow"))


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(_style(f"{prompt}{suffix}: ", "green")).strip()
    return answer or default


def _choice(prompt: str, choices: tuple[str, ...], default: str) -> str:
    rows = []
    for index, key in enumerate(choices, start=1):
        label, description = CHOICE_DETAILS.get(key, (key.replace("-", " ").title(), f"Select {key}."))
        rows.append((str(index), label, description))
    _table(prompt, rows)
    default_number = choices.index(default) + 1
    while True:
        raw = _ask("Choose a number", str(default_number)).lower()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:  # Backward-compatible with existing scripted input.
            return raw
        print(_style("Please enter a valid choice number.", "red"))


def _yes_no(prompt: str, *, default: bool) -> bool:
    selected = _choice(prompt, ("yes", "no"), "yes" if default else "no")
    return selected == "yes"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _q(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _array(values: Iterable[str]) -> str:
    return "[" + ", ".join(_q(item) for item in values) + "]"


_GENERATED_PATH_KEYS = {
    "topology", "trajectory", "reference", "mc_log", "trajectory_meta",
    "cavity_mask", "cavity_meta", "cavity_points", "cavity_nearby_residues",
    "cavity_mask_trajectory", "cavity_build_source", "cavity_build_output_prefix",
    "canonical_source", "homolog_source", "root",
}


def _portable_config_sections(sections: list[list[str]], config_directory: Path) -> list[list[str]]:
    """Make generated TOML relocatable without changing selections or other strings."""
    converted: list[list[str]] = []
    for section in sections:
        converted_section: list[str] = []
        for line in section:
            key_text, separator, value_text = line.partition("=")
            key = key_text.strip()
            if separator and key in _GENERATED_PATH_KEYS:
                try:
                    value = json.loads(value_text.strip())
                except (ValueError, TypeError, json.JSONDecodeError):
                    value = None
                if isinstance(value, str) and value and Path(value).expanduser().is_absolute():
                    line = f"{key_text}= {_q(portable_path(value, config_directory))}"
            converted_section.append(line)
        converted.append(converted_section)
    return converted


def _select_numbers(prompt: str, count: int, *, default: str = "1", allow_all: bool = True) -> list[int]:
    while True:
        raw = _ask(prompt, default).strip()
        if allow_all and raw.lower() in {"a", "all", "0"}:
            return list(range(1, count + 1))
        values: list[int] = []
        valid = True
        for token in _csv(raw):
            if not token.isdigit() or not 1 <= int(token) <= count:
                valid = False
                break
            number = int(token)
            if number not in values:
                values.append(number)
        if valid and values:
            return values
        print(_style(f"Enter one or more numbers from 1-{count}" + (", or A/0 for all." if allow_all else "."), "red"))


def _task_selection(has_md: bool, has_mc: bool) -> list[str]:
    allowed = []
    for item in TASK_CHOICES:
        if item.key in {"lifetime", "paths", "pose-clusters", "pose-hydration", "compare-hydration"} and not has_md:
            continue
        if item.key == "mc-states" and not has_mc:
            continue
        allowed.append(item)
    rows = [(str(index), item.label, item.description) for index, item in enumerate(allowed, start=1)]
    rows.insert(0, ("A/0", "All applicable", "Select every task supported by the chosen MD/MC cases."))
    _table("Analysis tasks (multiple selections allowed)", rows)
    while True:
        raw = _ask("Choose task numbers (comma-separated; A/0 = all)", "A").strip()
        if raw.lower() in {"a", "all", "0"}:
            return ["all"]
        selected: list[str] = []
        valid = True
        for token in _csv(raw):
            lowered = token.lower().replace("_", "-")
            if token.isdigit() and 1 <= int(token) <= len(allowed):
                key = allowed[int(token) - 1].key
            elif lowered in {item.key for item in allowed}:  # Legacy/text input.
                key = lowered
            else:
                valid = False
                break
            if key not in selected:
                selected.append(key)
        if valid and selected:
            return selected
        print(_style("Choose valid task numbers, or A/0 for all.", "red"))


def _substrate_candidates(topology: Path | None) -> list[str]:
    if topology is None or not topology.exists():
        return []
    try:
        import MDAnalysis as mda

        universe = mda.Universe(str(topology))
        excluded = {"SOL", "WAT", "HOH", "NA", "CL", "K", "CA", "MG", "ZN"}
        protein = {int(residue.ix) for residue in universe.select_atoms("protein").residues}
        values = [
            f"{int(residue.resid)}{str(residue.resname).upper()}"
            for residue in universe.residues
            if int(residue.ix) not in protein and str(residue.resname).upper() not in excluded
        ]
        close = getattr(universe.trajectory, "close", None)
        if close is not None:
            close()
        return values
    except Exception:
        return []


def _choose_anchor_request(candidates: list[str]) -> str:
    if not candidates:
        return _ask("Anchor residue", "800ATC")
    _table(
        "Anchor candidates from the first physical-MD topology",
        [
            (str(index), token, "Detected non-protein residue")
            for index, token in enumerate(candidates, start=1)
        ]
        + [("M", "Manual residue token", "Enter a token such as 800ATC or a residue name such as ATC.")],
    )
    raw = _ask("Choose an anchor number or enter a residue token", "1")
    if raw.isdigit() and 1 <= int(raw) <= len(candidates):
        return candidates[int(raw) - 1]
    if raw.lower() in {"m", "manual"}:
        return _ask("Manual anchor residue", candidates[0])
    return raw


def _mda_case_anchor_candidates(case: DiscoveredCase, requested: str) -> tuple[list[str], bool]:
    if case.topology is None or not case.topology.exists():
        return [], False
    try:
        import MDAnalysis as mda

        universe = mda.Universe(str(case.topology))
        tokens = list(
            dict.fromkeys(
                f"{int(residue.resid)}{str(residue.resname).upper()}"
                for residue in mda_anchor_candidates(universe, requested)
            )
        )
        close = getattr(universe.trajectory, "close", None)
        if close is not None:
            close()
        return tokens, True
    except Exception:
        return [], False


def _resolve_case_anchors(cases: list[DiscoveredCase], requested: str) -> dict[str, str]:
    """Resolve one logical residue name to exact per-case residue numbers."""
    output: dict[str, str] = {}
    for case in cases:
        candidates, inspected = _mda_case_anchor_candidates(case, requested)
        if len(candidates) == 1:
            output[case.case_id] = candidates[0]
            if candidates[0] != requested:
                print(
                    _style(
                        f"  {case.case_id}: anchor {requested} mapped by residue name to {candidates[0]}",
                        "green",
                    )
                )
            continue
        if len(candidates) > 1:
            _table(
                f"Multiple anchor residues match {requested!r} in {case.case_id}",
                [(str(index), token, "Matching residue; choose the intended cavity anchor") for index, token in enumerate(candidates, start=1)]
                + [("M", "Manual residue token", "Enter the exact residue number and name.")],
            )
            raw = _ask("Choose the case-specific anchor number", "1")
            if raw.isdigit() and 1 <= int(raw) <= len(candidates):
                output[case.case_id] = candidates[int(raw) - 1]
            elif raw.lower() in {"m", "manual"}:
                output[case.case_id] = _ask(f"Exact anchor residue for {case.case_id}")
            else:
                output[case.case_id] = raw
            continue
        if inspected:
            alternatives = _substrate_candidates(case.topology)
            _table(
                f"Anchor {requested!r} was not found in {case.case_id}",
                [(str(index), token, "Available non-protein residue") for index, token in enumerate(alternatives, start=1)]
                + [("M", "Manual residue token", "Enter the exact residue token used by this topology.")],
            )
            raw = _ask("Choose a replacement anchor number", "1" if alternatives else "M")
            if raw.isdigit() and 1 <= int(raw) <= len(alternatives):
                output[case.case_id] = alternatives[int(raw) - 1]
            else:
                output[case.case_id] = _ask(f"Exact anchor residue for {case.case_id}", requested)
            continue
        # Keep the requested token when a lightweight wizard inspection cannot open the topology.
        # The validation command performs the authoritative check before any expensive stage.
        output[case.case_id] = requested
    return output


def _selection_from_tokens(value: str) -> str:
    if value.lower().startswith("selection="):
        return value.split("=", 1)[1].strip()
    terms = []
    for token in _csv(value):
        match = re.fullmatch(r"(\d+)([A-Za-z][A-Za-z0-9]*)", token.strip())
        if not match:
            raise ValueError(
                f"Invalid substrate residue token: {token}; use for example 800ATC or selection=<MDAnalysis selection>"
            )
        terms.append(f"(resid {int(match.group(1))} and resname {match.group(2).upper()})")
    if not terms:
        raise ValueError("At least one substrate residue is required")
    return " or ".join(terms)


def _select_substrate(candidates: list[str], default_tokens: str) -> str:
    if candidates:
        rows = [(str(index), token, "Detected non-protein residue") for index, token in enumerate(candidates, start=1)]
        rows.append(("A/0", "All detected", "Select all listed substrate residues."))
        rows.append(("M", "Manual selection", "Enter residue tokens or an MDAnalysis selection."))
        _table("Detected substrate candidates (multiple selections allowed)", rows)
    raw = _ask(
        "Choose substrate numbers, residue tokens, or selection=<MDAnalysis selection>",
        "1" if candidates else default_tokens,
    )
    if candidates and raw.lower() in {"a", "all", "0"}:
        return _selection_from_tokens(",".join(candidates))
    if candidates and raw.lower() in {"m", "manual"}:
        return _selection_from_tokens(_ask("Manual substrate selection", default_tokens))
    if candidates and all(token.strip().isdigit() for token in raw.split(",")):
        numbers = _select_numbers_from_text(raw, len(candidates))
        return _selection_from_tokens(",".join(candidates[number - 1] for number in numbers))
    return _selection_from_tokens(raw)


def _select_numbers_from_text(raw: str, count: int) -> list[int]:
    values: list[int] = []
    for token in _csv(raw):
        if not token.isdigit() or not 1 <= int(token) <= count:
            raise ValueError(f"Selection number is out of range: {token}")
        if int(token) not in values:
            values.append(int(token))
    return values


def _frame_count(topology: Path | None, trajectory: Path | None) -> int | None:
    if topology is None or trajectory is None:
        return None
    try:
        import MDAnalysis as mda

        universe = mda.Universe(str(topology), str(trajectory))
        count = len(universe.trajectory)
        close = getattr(universe.trajectory, "close", None)
        if close is not None:
            close()
        return count
    except Exception:
        return None


def _prefix_from_mask(mask: Path) -> Path:
    if mask.name.endswith("_mask.dat"):
        return mask.with_name(mask.name[:-9])
    if mask.name == "cavity_mask.gro":
        return mask.with_name("cavity")
    return mask.with_suffix("")


def _bundle_from_mask(mask: Path) -> CavityBundle:
    mask = mask.expanduser().resolve()
    prefix = _prefix_from_mask(mask)
    meta_candidates = [prefix.with_suffix(".meta.json"), mask.with_suffix(mask.suffix + ".meta.json")]
    points_candidate = prefix.with_name(prefix.name + "_points.pdb")
    nearby_candidate = prefix.with_name(prefix.name + "_nearby_residues.tsv")
    meta = next((item for item in meta_candidates if item.exists()), None)
    if meta is None:
        meta = next(iter(sorted(mask.parent.glob("*.meta.json"))), None)
    points = points_candidate if points_candidate.exists() else next(iter(sorted(mask.parent.glob("*_points.pdb"))), None)
    nearby = nearby_candidate if nearby_candidate.exists() else next(iter(sorted(mask.parent.glob("*_nearby_residues.tsv"))), None)
    return CavityBundle(mask.parent, mask if mask.exists() else None, meta, points, nearby)


def _mask_candidates(case: DiscoveredCase, scan_root: Path) -> list[Path]:
    output: list[Path] = []
    if case.cavity_mask is not None and case.cavity_mask.exists():
        output.append(case.cavity_mask.resolve())
    try:
        output.extend(item.resolve() for item in case.directory.rglob("*_mask.dat") if item.is_file())
        output.extend(item.resolve() for item in case.directory.rglob("cavity_mask.gro") if item.is_file())
    except OSError:
        pass
    if not output:
        current = case.directory.parent
        upper_limit = scan_root
        for _ in range(2):
            if upper_limit.parent == upper_limit:
                break
            upper_limit = upper_limit.parent
        while True:
            try:
                output.extend(item.resolve() for item in current.glob("*_mask.dat") if item.is_file())
                output.extend(item.resolve() for item in current.glob("cavity_mask.gro") if item.is_file())
            except OSError:
                pass
            if output or current == upper_limit or current.parent == current:
                break
            current = current.parent
    unique = sorted(set(output), key=lambda item: (len(item.parts), str(item).lower()))
    return unique


def _initial_bundle(case: DiscoveredCase, scan_root: Path) -> CavityBundle:
    candidates = _mask_candidates(case, scan_root)
    if candidates:
        bundle = _bundle_from_mask(candidates[0])
        return CavityBundle(
            directory=bundle.directory,
            mask=bundle.mask,
            meta=case.cavity_meta if case.cavity_meta and case.cavity_meta.exists() else bundle.meta,
            points=case.cavity_points if case.cavity_points and case.cavity_points.exists() else bundle.points,
            nearby=case.cavity_nearby_residues if case.cavity_nearby_residues and case.cavity_nearby_residues.exists() else bundle.nearby,
        )
    return CavityBundle(case.directory, None, None, None, None)


def _bundle_from_user_path(raw: str) -> CavityBundle:
    path = Path(raw).expanduser().resolve()
    if path.is_file():
        if path.name.endswith("_mask.dat") or path.name == "cavity_mask.gro":
            return _bundle_from_mask(path)
        raise ValueError("Provide a cavity directory or the *_mask.dat file")
    if not path.is_dir():
        raise FileNotFoundError(path)
    masks = sorted(path.glob("*_mask.dat")) + sorted(path.glob("cavity_mask.gro"))
    if not masks:
        return CavityBundle(path, None, None, None, None)
    if len(masks) > 1:
        _table(
            f"Cavity masks in {path}",
            [(str(index), item.name, str(item)) for index, item in enumerate(masks, start=1)],
        )
        number = _select_numbers("Choose one mask number", len(masks), allow_all=False)[0]
        return _bundle_from_mask(masks[number - 1])
    return _bundle_from_mask(masks[0])


def _bundle_description(bundle: CavityBundle) -> str:
    state = "deferred build" if bundle.deferred else ("ready" if bundle.mask else "MASK MISSING")
    return (
        f"{state}; dir={bundle.directory}; mask={bundle.mask or 'missing'}; meta={bundle.meta or 'missing'}; "
        f"points={bundle.points or 'missing'}; nearby={bundle.nearby or 'missing'}"
    )


def _review_existing_bundles(
    cases: list[DiscoveredCase], scan_root: Path, modes: dict[str, str]
) -> dict[str, CavityBundle]:
    relevant = [case for case in cases if modes[case.case_id] == "mask"]
    bundles = {case.case_id: _initial_bundle(case, scan_root) for case in relevant}
    if not relevant:
        return bundles
    while True:
        _table(
            "Per-case GCMC cavity files (press a case number to edit its path)",
            [
                (str(index), case.case_id, _bundle_description(bundles[case.case_id]))
                for index, case in enumerate(relevant, start=1)
            ],
        )
        missing = [case for case in relevant if bundles[case.case_id].mask is None]
        if missing:
            _note("A mask is required for each listed case. Choose its number and provide a directory or *_mask.dat path.")
            default = str(relevant.index(missing[0]) + 1)
        else:
            _note("Press Enter if these files are correct, or enter a case number to replace that case's cavity path.")
            default = ""
        raw = _ask("Confirm or edit cavity files", default)
        if not raw and not missing:
            return bundles
        if not raw.isdigit() or not 1 <= int(raw) <= len(relevant):
            print(_style("Enter a case number from the table, or press Enter when all masks are correct.", "red"))
            continue
        case = relevant[int(raw) - 1]
        while True:
            try:
                bundles[case.case_id] = _bundle_from_user_path(
                    _ask(f"Cavity directory or *_mask.dat path for {case.case_id}")
                )
                break
            except (FileNotFoundError, ValueError) as exc:
                print(_style(str(exc), "red"))


def _gro_candidates(case: DiscoveredCase) -> list[Path]:
    preferred = []
    for name in ("init.gro", "previous.gro", "confout.gro"):
        candidate = case.directory / name
        if candidate.exists():
            preferred.append(candidate.resolve())
    if case.topology is not None and case.topology.suffix.lower() == ".gro" and case.topology.exists():
        preferred.append(case.topology.resolve())
    try:
        preferred.extend(item.resolve() for item in case.directory.glob("*.gro") if item.name != "trajectory.gro")
    except OSError:
        pass
    return list(dict.fromkeys(preferred))


def _deferred_bundle(case: DiscoveredCase) -> CavityBundle:
    candidates = _gro_candidates(case)
    rows = [(str(index), item.name, str(item)) for index, item in enumerate(candidates, start=1)]
    rows.append(("0", "Manual path", "Enter a GRO structure path."))
    _table(f"Source structure for deferred cavity build: {case.case_id}", rows)
    while True:
        raw = _ask("Choose a GRO source number", "1" if candidates else "0")
        if raw == "0":
            source = Path(_ask("Path to source GRO")).expanduser().resolve()
        elif raw.isdigit() and 1 <= int(raw) <= len(candidates):
            source = candidates[int(raw) - 1]
        else:
            print(_style("Choose a valid source number.", "red"))
            continue
        if source.exists() and source.suffix.lower() == ".gro":
            break
        print(_style(f"A readable GRO structure is required: {source}", "red"))
    prefix = case.directory / "cavity"
    return CavityBundle(
        directory=prefix.parent,
        mask=prefix.with_name(prefix.name + "_mask.dat"),
        meta=prefix.with_suffix(".meta.json"),
        points=prefix.with_name(prefix.name + "_points.pdb"),
        nearby=prefix.with_name(prefix.name + "_nearby_residues.tsv"),
        build_source=source,
        build_output_prefix=prefix.resolve(),
    )


def _detected_cavity_mode(case: DiscoveredCase) -> str | None:
    if case.pocketmc_status in {"confirmed", "probable"} and case.cavity_mode in {"sphere", "mask"}:
        return case.cavity_mode
    if case.pocketmc_status in {"confirmed", "probable"} and case.cavity_mask is not None:
        return "mask"
    if case.pocketmc_status in {"confirmed", "probable"}:
        # A completed MC case with no mask/config evidence follows the legacy sphere workflow.
        return "sphere"
    return None


def _choose_cavity_modes(cases: list[DiscoveredCase]) -> dict[str, str]:
    detected = {case.case_id: _detected_cavity_mode(case) for case in cases}
    known = {value for value in detected.values() if value is not None}
    if detected:
        _table(
            "Cavity mode inferred from each selected MC run",
            [
                (str(index), case.case_id, f"detected={detected[case.case_id] or 'none (default sphere)'}")
                for index, case in enumerate(cases, start=1)
            ],
        )
    if len(known) > 1:
        choices = ("detected", "sphere", "mask")
        CHOICE_DETAILS["detected"] = ("Use each detected mode", "Keep the sphere/mask definition inferred for each case.")
        selection = _choice("Cavity definition", choices, "detected")
        modes = {case.case_id: (detected[case.case_id] or "sphere") for case in cases} if selection == "detected" else {case.case_id: selection for case in cases}
    else:
        default = next(iter(known), "sphere")
        selection = _choice("Cavity definition", ("sphere", "mask"), default)
        modes = {case.case_id: selection for case in cases}
    changed = [case.case_id for case in cases if detected[case.case_id] and modes[case.case_id] != detected[case.case_id]]
    if changed:
        _note("The selected mode differs from the MC setup for: " + ", ".join(changed))
        if not _yes_no("Continue with the cavity-mode override", default=False):
            return _choose_cavity_modes(cases)
    return modes


def _case_lines(
    case: DiscoveredCase,
    kind: str,
    cavity_mode: str,
    cavity: CavityBundle | None,
    *,
    anchor: str,
    anchor_atoms: list[str],
    radius_nm: float,
) -> list[str]:
    run_id = case.case_id
    replica_match = re.fullmatch(r"(?:replica[-_])?(\d+)", case.directory.name, re.I)
    sweep_match = re.fullmatch(r"\d+(?:\.\d+)?", case.directory.parent.name)
    if replica_match and sweep_match:
        system_default = case.directory.parent.parent.name
    elif replica_match:
        system_default = case.directory.parent.name
    else:
        system_default = case.directory.name
    lines = [
        "[[case]]", f"id = {_q(run_id)}", f"kind = {_q(kind)}", f"system_id = {_q(system_default)}",
        'comparison_group = "default"', f"run_dir = {_q(case.directory)}",
        f"pocketmc_status = {_q(case.pocketmc_status)}", f"pocketmc_evidence = {_array(case.evidence)}",
    ]
    if replica_match:
        lines.append(f"replica = {_q(replica_match.group(1))}")
    if replica_match and sweep_match:
        lines.append(f"sweep = {_q(case.directory.parent.name)}")
    if kind == "md":
        if case.topology is None or case.trajectory is None:
            raise ValueError(f"{case.case_id} has no confirmed physical-MD topology/trajectory pair")
        lines.extend([f"topology = {_q(case.topology)}", f"trajectory = {_q(case.trajectory)}"])
        reference = case.directory / "previous.gro"
        if reference.exists():
            lines.append(f"reference = {_q(reference)}")
    else:
        if case.mc_trajectory is None:
            raise ValueError(f"{case.case_id} has no PocketMC trajectory.gro")
        lines.append(f"trajectory = {_q(case.mc_trajectory)}")
        if case.mc_log is not None:
            lines.append(f"mc_log = {_q(case.mc_log)}")
        if case.trajectory_meta is not None:
            lines.append(f"trajectory_meta = {_q(case.trajectory_meta)}")
        previous = case.directory / "previous.gro"
        if previous.exists():
            lines.append(f"topology = {_q(previous)}")
    lines.extend(
        [
            f"cavity_mode = {_q(cavity_mode)}",
            f"cavity_anchor = {_q(anchor)}",
            f"cavity_anchor_atoms = {_array(anchor_atoms)}",
            f"cavity_radius_nm = {radius_nm}",
        ]
    )
    if cavity_mode == "mask" and cavity is not None:
        if cavity.mask is not None:
            lines.append(f"cavity_mask = {_q(cavity.mask)}")
        if cavity.meta is not None:
            lines.append(f"cavity_meta = {_q(cavity.meta)}")
        if cavity.points is not None:
            lines.append(f"cavity_points = {_q(cavity.points)}")
        if cavity.nearby is not None:
            lines.append(f"cavity_nearby_residues = {_q(cavity.nearby)}")
        mask_trajectory = case.directory / "cavity_trajectory.gro"
        if kind == "pocketmc" and mask_trajectory.exists():
            lines.append(f"cavity_mask_trajectory = {_q(mask_trajectory)}")
        if cavity.deferred:
            lines.extend(
                [
                    "cavity_build_enabled = true",
                    f"cavity_build_source = {_q(cavity.build_source or '')}",
                    f"cavity_build_output_prefix = {_q(cavity.build_output_prefix or '')}",
                    'cavity_build_mode = "seeded"',
                    f"cavity_build_exclude_residues = {_array([anchor])}",
                    "cavity_build_dx = 0.075",
                    "cavity_build_probe_radius = 0.10",
                    "cavity_build_search_radius = 0.90",
                    "cavity_build_nearby_cutoff = 0.45",
                    "cavity_build_min_points = 20",
                ]
            )
    return lines


def _failed_analyses(root: Path) -> list[ExistingAnalysis]:
    manifests = set(root.glob("*/analysis_manifest.json"))
    manifests.update(root.glob("analysis-results*/analysis_manifest.json"))
    found: list[ExistingAnalysis] = []
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            failures = payload.get("failures", [])
            config_value = payload.get("config")
            if payload.get("status") != "partial" and not failures:
                continue
            if not config_value or not isinstance(failures, list):
                continue
            config_path = Path(str(config_value)).expanduser()
            if not config_path.is_absolute():
                config_path = (manifest.parent / config_path).resolve()
            if not config_path.exists():
                continue
            found.append(
                ExistingAnalysis(
                    manifest=manifest.resolve(),
                    config_path=config_path.resolve(),
                    output_root=manifest.parent.resolve(),
                    failures=tuple(dict(item) for item in failures if isinstance(item, dict)),
                    modified_ns=manifest.stat().st_mtime_ns,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda item: item.modified_ns, reverse=True)


def _completed_analyses(root: Path) -> list[ExistingAnalysis]:
    manifests = set(root.glob("*/analysis_manifest.json"))
    manifests.update(root.glob("analysis-results*/analysis_manifest.json"))
    found: list[ExistingAnalysis] = []
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            if payload.get("status") != "complete" or payload.get("failures"):
                continue
            config_value = payload.get("config")
            if not config_value:
                continue
            config_path = Path(str(config_value)).expanduser()
            if not config_path.is_absolute():
                config_path = (manifest.parent / config_path).resolve()
            if not config_path.exists():
                continue
            found.append(
                ExistingAnalysis(
                    manifest=manifest.resolve(),
                    config_path=config_path.resolve(),
                    output_root=manifest.parent.resolve(),
                    failures=(),
                    modified_ns=manifest.stat().st_mtime_ns,
                    status="complete",
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return sorted(found, key=lambda item: item.modified_ns, reverse=True)


def _choose_failed_analysis(candidates: list[ExistingAnalysis]) -> ExistingAnalysis:
    if len(candidates) == 1:
        return candidates[0]
    _table(
        "Existing completed analyses" if all(item.status == "complete" for item in candidates) else "Existing partial analyses",
        [
            (
                str(index),
                item.output_root.name,
                f"status={item.status}; failures={len(item.failures)}; config={item.config_path}; manifest={item.manifest}",
            )
            for index, item in enumerate(candidates, start=1)
        ],
    )
    number = _select_numbers("Choose the analysis number", len(candidates), allow_all=False)[0]
    return candidates[number - 1]


def _available_grand_output(path: Path) -> Path:
    if not path.exists():
        return path
    serial = 1
    while True:
        candidate = path.with_name(f"{path.name}_{serial}")
        if not candidate.exists():
            return candidate
        serial += 1


def _choose_grand_alignment(
    scan_root: Path,
    candidates: list[ExistingGrandAlignment],
) -> ExistingGrandAlignment:
    if len(candidates) == 1:
        return candidates[0]
    _table(
        "Existing grand-aligned results",
        [
            (
                str(index),
                str(item.root.relative_to(scan_root)) if item.root != scan_root else item.root.name,
                f"maps={item.aligned_map_count}; schema={item.schema_version}; "
                f"{'repair recommended: ' + '; '.join(item.stale_reasons) if item.stale else 'current'}",
            )
            for index, item in enumerate(candidates, start=1)
        ],
    )
    number = _select_numbers(
        "Choose the grand-aligned result number",
        len(candidates),
        allow_all=False,
    )[0]
    return candidates[number - 1]


def _manage_existing_grand_alignment(existing: ExistingGrandAlignment) -> str:
    print()
    print(_style("Existing grand-aligned result detected", "bold", "green"))
    print(f"  Output: {existing.root}")
    print(f"  Saved maps: {existing.aligned_map_count}")
    print(f"  Schema: {existing.schema_version}")
    if existing.stale:
        for reason in existing.stale_reasons:
            _note(f"Repair recommended: {reason}")
        action = _choice(
            "Choose how to handle the old grand-aligned result",
            ("repair-grand", "new-grand", "new-setup", "exit"),
            "repair-grand",
        )
    else:
        action = _choice(
            "Choose how to handle the grand-aligned result",
            ("replot-grand", "new-grand", "new-setup", "exit"),
            "replot-grand",
        )
    if action == "repair-grand":
        print(_style("Repairing plot-only grand-alignment outputs now.", "bold", "cyan"))
        plots = repair_grand_alignment_output(existing.root)
        print(_style(f"Repair complete: regenerated {len(plots)} plot(s).", "bold", "green"))
        print(f"  Editable rerun: python {existing.root / 'plot_grand_aligned.py'}")
        return "done"
    if action == "replot-grand":
        plots = replot_grand_alignment(existing.root, load_grand_plot_style(existing.root))
        print(_style(f"Grand replot complete: regenerated {len(plots)} plot(s).", "bold", "green"))
        print(f"  Editable rerun: python {existing.root / 'plot_grand_aligned.py'}")
        return "done"
    return action


def _run_grand_alignment_wizard(
    scan_root: Path,
    candidates: list[CompletedAnalysisRoot],
) -> None:
    _step(
        1,
        "Select Completed Analyses",
        "Choose independently completed triplicate batches. Source result directories are read only.",
    )
    coverage_by_root = {
        item.root: sorted(substrate_coverage([item]))
        for item in candidates
    }
    _table(
        "Auto-detected completed analysis roots",
        [
            (
                str(index),
                str(item.root.relative_to(scan_root)) if item.root != scan_root else item.root.name,
                f"runs={len(item.completed_runs)}; maps={len(item.maps)}; "
                f"substrates={','.join(coverage_by_root[item.root]) or 'none'}; status={item.status}",
            )
            for index, item in enumerate(candidates, start=1)
        ],
    )
    selected_numbers = _select_numbers(
        "Choose analysis-root numbers (comma-separated; A/0 = all)",
        len(candidates),
        default="A",
        allow_all=True,
    )
    selected = [candidates[index - 1] for index in selected_numbers]
    if len(selected) < 2:
        raise ValueError("Grand alignment requires at least two selected completed analyses")

    _step(
        2,
        "Choose the Fixed Substrate",
        "Heavy atoms of the selected residue name(s) define the common rigid coordinate frame.",
    )
    common = common_substrates(selected)
    if not common:
        raise ValueError("The selected analyses have no common substrate residue name")
    defaults = default_fixed_substrates(selected)
    default_indices = [str(common.index(name) + 1) for name in defaults]
    _table(
        "Substrates present in every selected analysis",
        [
            (str(index), name, "OPP-preferred fixed reference" if name == "OPP" else "Available fixed reference")
            for index, name in enumerate(common, start=1)
        ],
    )
    substrate_numbers = _select_numbers(
        "Choose fixed substrate number(s); multiple selections make one joint fit",
        len(common),
        default=",".join(default_indices),
        allow_all=True,
    )
    substrates = [common[index - 1] for index in substrate_numbers]

    _step(
        3,
        "Reference Frame and Output",
        "Choose the target orientation and write a new standalone result under the current directory.",
    )
    _table(
        "Reference analysis",
        [
            (str(index), item.root.name, str(item.root))
            for index, item in enumerate(selected, start=1)
        ],
    )
    reference_number = _select_numbers(
        "Choose the reference-analysis number",
        len(selected),
        default="1",
        allow_all=False,
    )[0]
    reference = selected[reference_number - 1]
    spacing_text = _ask(
        "Common grid spacing in Angstrom (auto uses the coarsest saved map)",
        "auto",
    ).strip().lower()
    spacing_a = None if spacing_text in {"", "auto", "a"} else float(spacing_text)
    elevation = float(_ask("Shared 3D camera elevation", "30"))
    azimuth = float(_ask("Shared 3D camera azimuth", "-60"))
    roll = float(_ask("Shared 3D camera roll", "0"))
    requested_output = Path(_ask("Grand-aligned output directory", "grand-aligned")).expanduser()
    if not requested_output.is_absolute():
        requested_output = scan_root / requested_output
    output = _available_grand_output(requested_output.resolve())
    if output != requested_output.resolve():
        _note(f"The requested output already exists; using {output.name} instead.")

    print()
    print(_style("Grand-alignment summary", "bold", "white"))
    print(f"  Analysis roots: {len(selected)}")
    print(f"  Saved maps available: {sum(len(item.maps) for item in selected)}")
    print(f"  Fixed substrate(s): {', '.join(substrates)}")
    print(f"  Reference: {reference.root}")
    print(f"  Output: {output}")
    print("  Execution: local saved-NPZ post-processing; no trajectory scan or sbatch.")
    if not _yes_no("Run grand alignment now", default=True):
        _note("Grand alignment was cancelled; no output was written.")
        return
    result = grand_align_analysis_roots(
        selected,
        output,
        fixed_substrates=substrates,
        reference_root=reference,
        spacing_a=spacing_a,
        elevation=elevation,
        azimuth=azimuth,
        roll=roll,
    )
    print()
    print(_style("Grand alignment complete", "bold", "green"))
    print(f"  Aligned maps: {len(result.aligned_maps)}")
    print(f"  Rendered plots: {len(result.plots)}")
    print(f"  Skipped incompatible maps: {len(result.skipped_maps)}")
    print(f"  Output: {result.output_root}")
    print(f"  Manifest: {result.manifest}")
    for warning in result.warnings:
        _note(f"WARNING: {warning}")


def _next_output_root(path: Path) -> tuple[Path, int]:
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            return candidate, index
        index += 1


def _unique_sibling(path: Path, label: str) -> Path:
    candidate = path.with_name(f"{path.stem}_{label}{path.suffix}")
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{label}_{index}{path.suffix}")
        index += 1
    return candidate


def _replace_table_value(lines: list[str], section: str, key: str, rendered: str) -> None:
    header = f"[{section}]"
    try:
        start = next(index for index, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        lines.extend(["", header, f"{key} = {rendered}"])
        return
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].strip().startswith("[")),
        len(lines),
    )
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(start + 1, end):
        if pattern.match(lines[index]):
            lines[index] = f"{key} = {rendered}"
            return
    lines.insert(end, f"{key} = {rendered}")


def _write_repaired_config(
    source: Path,
    target: Path,
    *,
    anchor_updates: dict[str, str],
    output_root: Path | None = None,
) -> Path:
    lines = source.read_text(encoding="utf-8").splitlines()
    case_starts = [index for index, line in enumerate(lines) if line.strip() == "[[case]]"]
    case_blocks: list[tuple[int, int, str]] = []
    for position, start in enumerate(case_starts):
        next_case = case_starts[position + 1] if position + 1 < len(case_starts) else len(lines)
        end = next_case
        if position + 1 == len(case_starts):
            end = next(
                (
                    index
                    for index in range(start + 1, len(lines))
                    if lines[index].strip().startswith("[") and lines[index].strip() != "[[case]]"
                ),
                len(lines),
            )
        run_id = ""
        for line in lines[start + 1 : end]:
            match = re.match(r'^\s*(?:id|run_id)\s*=\s*"(.*)"\s*$', line)
            if match:
                run_id = match.group(1)
                break
        case_blocks.append((start, end, run_id))

    if case_blocks:
        for start, end, run_id in reversed(case_blocks):
            if run_id not in anchor_updates:
                continue
            replacement = f"cavity_anchor = {_q(anchor_updates[run_id])}"
            field = re.compile(r"^\s*cavity_anchor\s*=")
            existing = next((index for index in range(start + 1, end) if field.match(lines[index])), None)
            if existing is None:
                lines.insert(end, replacement)
            else:
                lines[existing] = replacement
    elif anchor_updates:
        # Legacy single-input configs have no [[case]] table.
        _replace_table_value(lines, "cavity", "anchor", _q(next(iter(anchor_updates.values()))))
    if output_root is not None:
        _replace_table_value(lines, "output", "root", _q(portable_path(output_root, target.parent)))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return target


def _repair_anchor_failures(config: AnalysisConfig, failures: tuple[dict[str, str], ...]) -> dict[str, str]:
    failed_ids = {
        str(item.get("run_id", ""))
        for item in failures
        if "anchor" in str(item.get("error", "")).lower()
    }
    updates: dict[str, str] = {}
    suggestions: list[tuple[str, str, str]] = []
    for dataset in config.datasets:
        if dataset.run_id not in failed_ids or dataset.kind != "md" or dataset.topology is None:
            continue
        case_config = config_for_dataset(config, dataset)
        if case_config.cavity.mode == "mask":
            _note(
                f"{dataset.run_id}: this was a mask-mode anchor failure. The repaired analysis uses the mask metadata "
                "reference point directly, so no anchor edit is required."
            )
            continue
        requested = case_config.cavity.anchor
        try:
            import MDAnalysis as mda

            universe = mda.Universe(str(dataset.topology))
            candidates = list(
                dict.fromkeys(
                    f"{int(residue.resid)}{str(residue.resname).upper()}"
                    for residue in mda_anchor_candidates(universe, requested)
                )
            )
            close = getattr(universe.trajectory, "close", None)
            if close is not None:
                close()
        except Exception as exc:
            _note(f"{dataset.run_id}: topology inspection failed ({exc}); enter its anchor manually.")
            updates[dataset.run_id] = _ask(f"Exact anchor residue for {dataset.run_id}", requested)
            continue
        if len(candidates) == 1:
            updates[dataset.run_id] = candidates[0]
            suggestions.append((dataset.run_id, requested, candidates[0]))
            continue
        if len(candidates) > 1:
            _table(
                f"Ambiguous anchor for failed case {dataset.run_id}",
                [(str(index), token, "Matching residue in the MD topology") for index, token in enumerate(candidates, start=1)]
                + [("M", "Manual residue token", "Enter an exact residue number and name.")],
            )
            raw = _ask("Choose the correct anchor number", "1")
            if raw.isdigit() and 1 <= int(raw) <= len(candidates):
                updates[dataset.run_id] = candidates[int(raw) - 1]
            else:
                updates[dataset.run_id] = _ask(f"Exact anchor residue for {dataset.run_id}", requested)
            continue
        alternatives = _substrate_candidates(dataset.topology)
        _table(
            f"No residue matched failed anchor {requested!r} in {dataset.run_id}",
            [(str(index), token, "Available non-protein residue") for index, token in enumerate(alternatives, start=1)]
            + [("M", "Manual residue token", "Enter an exact residue number and name.")],
        )
        raw = _ask("Choose a replacement anchor number", "1" if alternatives else "M")
        if raw.isdigit() and 1 <= int(raw) <= len(alternatives):
            updates[dataset.run_id] = alternatives[int(raw) - 1]
        else:
            updates[dataset.run_id] = _ask(f"Exact anchor residue for {dataset.run_id}", requested)
    if suggestions:
        _table(
            "Safe case-specific anchor repairs detected",
            [(run_id, resolved, f"requested={requested}; unique same-resname match") for run_id, requested, resolved in suggestions],
        )
        if not _yes_no("Apply these case-specific anchor repairs", default=True):
            for run_id, requested, resolved in suggestions:
                updates[run_id] = _ask(f"Exact anchor residue for {run_id}", resolved or requested)
    return updates


def _resume_existing_analysis(existing: ExistingAnalysis) -> tuple[Path, AnalysisConfig] | None:
    print()
    print(_style("Existing partial analysis detected", "bold", "yellow"))
    print(f"  Config: {existing.config_path}")
    print(f"  Output: {existing.output_root}")
    _table(
        "Recorded failures",
        [
            (
                str(index),
                str(item.get("run_id", "unknown")),
                f"{item.get('phase', 'analysis')}: {item.get('error', 'unknown error')}",
            )
            for index, item in enumerate(existing.failures, start=1)
        ],
    )
    CHOICE_DETAILS.update(
        {
            "resume": ("Repair and resume", "Recommended: reuse successful caches and retry failed work."),
            "new-output": ("Use a new output", "Keep the old results and append _1, _2, ... to the result directory."),
            "overwrite": ("Recompute/overwrite", "Use the same output directory and run every selected task with --force."),
            "new-setup": ("Start a new setup", "Ignore this partial run and return to case discovery."),
        }
    )
    action = _choice(
        "How should the existing analysis be handled?",
        ("resume", "new-output", "overwrite", "new-setup"),
        "resume",
    )
    if action == "new-setup":
        return None
    config = load_analysis_config(existing.config_path)
    anchor_updates = _repair_anchor_failures(config, existing.failures)
    if action == "new-output":
        output_root, suffix = _next_output_root(config.output.root)
        target = existing.config_path.with_name(f"{existing.config_path.stem}_{suffix}{existing.config_path.suffix}")
        while target.exists():
            suffix += 1
            output_root = config.output.root.with_name(f"{config.output.root.name}_{suffix}")
            target = existing.config_path.with_name(f"{existing.config_path.stem}_{suffix}{existing.config_path.suffix}")
        stem = f"run_analyses_{suffix}"
        force = False
        resume = False
    elif action == "overwrite":
        output_root = config.output.root
        target = _unique_sibling(existing.config_path, "overwrite")
        stem = "overwrite_analyses"
        force = True
        resume = False
    else:
        output_root = config.output.root
        target = _unique_sibling(existing.config_path, "resume")
        stem = "resume_analyses"
        force = False
        resume = True
    _write_repaired_config(
        existing.config_path,
        target,
        anchor_updates=anchor_updates,
        output_root=output_root,
    )
    repaired = load_analysis_config(target)
    launchers = render_analysis_launchers(
        repaired,
        target.parent,
        name_stem=stem,
        force=force,
        resume=resume,
    )
    print()
    print(_style("Recovery setup complete", "bold", "green"))
    print(f"  Repaired config: {target}")
    print(f"  Output root: {repaired.output.root}")
    for label, path in launchers.items():
        print(f"  {label}: {path}")
    if resume:
        _note("The resume launchers reuse valid trajectory/feature/cluster/hydration caches and retry failed finalize/case stages.")
    elif force:
        _note("The overwrite launchers include --force and recompute cached stages in the existing output directory.")
    return target, repaired


def _optional_float(prompt: str, current: object) -> float | None:
    rendered = "auto" if current is None else str(current)
    raw = _ask(prompt, rendered).strip().lower()
    return None if raw in {"", "auto", "none", "native"} else float(raw)


def _recompute_stale_pose_hydration(existing: ExistingAnalysis) -> None:
    """Run only the invalidated pose-hydration/finalize stages before interactive replotting."""
    config = load_analysis_config(existing.config_path)
    if config.output.root.resolve() != existing.output_root.resolve():
        raise ValueError(
            f"Existing manifest output ({existing.output_root}) does not match config output ({config.output.root})"
        )
    print(_style("Preparing minimal pose-hydration repair now.", "bold", "cyan"))
    print("  Reusing saved pose-training features and pooled cluster models.")
    print("  Reading trajectories only for invalidated pose-hydration maps; --force is not used.")
    prepare_analysis_cavities(config)
    validate_analysis_config(config, check_files=True)
    hydration = run_pose_stage(config, "hydrate", force=False)
    if hydration.failures:
        details = "; ".join(
            f"{item.get('run_id', 'unknown')} [{item.get('phase', 'hydrate')}]: {item.get('error', 'unknown error')}"
            for item in hydration.failures
        )
        raise RuntimeError("Pose-hydration repair failed: " + details)
    if "compare-hydration" in config.analysis.tasks:
        print(_style("Updating aggregate pose-hydration maps.", "cyan"))
        finalized = run_pose_stage(config, "finalize", force=False)
        if finalized.failures:
            details = "; ".join(
                f"{item.get('run_id', 'unknown')} [{item.get('phase', 'finalize')}]: {item.get('error', 'unknown error')}"
                for item in finalized.failures
            )
            raise RuntimeError("Pose-hydration aggregate repair failed: " + details)
    remaining = stale_pose_hydration_runs(existing.output_root)
    if remaining:
        raise RuntimeError(
            "Pose-hydration repair did not refresh every stale case: " + ", ".join(remaining)
        )
    print(_style("Minimal pose-hydration repair complete; continuing to plot settings.", "bold", "green"))


def _recompute_stale_analysis(existing: ExistingAnalysis) -> None:
    """Refresh stale generic MD/PocketMC outputs while retaining independent pose caches."""
    config = load_analysis_config(existing.config_path)
    if config.output.root.resolve() != existing.output_root.resolve():
        raise ValueError(
            f"Existing manifest output ({existing.output_root}) does not match config output ({config.output.root})"
        )
    stale_ids = set(stale_analysis_runs(existing.output_root))
    print(_style("Preparing minimal case-analysis repair now.", "bold", "cyan"))
    print("  Re-reading only stale MD trajectories or PocketMC accepted states for saved-table/density correctness.")
    print("  Valid pose-training and pooled cluster-model caches are reused; --force is not used.")
    from .aggregate import write_aggregate
    from .cache import load_analysis_cache
    from .runner import _copy_plot_script, _fingerprint, run_dataset

    repaired = []
    for dataset in config.datasets:
        if dataset.run_id not in stale_ids:
            continue
        repaired.append(
            run_dataset(
                replace(config, output=replace(config.output, overwrite=True)),
                dataset,
                force=False,
                reset_plot_style=False,
                pose_stage=None,
            )
        )
    # Refresh aggregate occupancy outputs from every available generic cache;
    # Independent pose stages are deliberately not launched here.
    cached_results = {item.dataset.run_id: item for item in repaired}
    for dataset in config.datasets:
        if dataset.run_id in cached_results:
            continue
        current_fingerprint = _fingerprint(config_for_dataset(config, dataset), dataset)
        result = load_analysis_cache(
            existing.output_root / dataset.run_id,
            dataset,
            analysis_version=ANALYSIS_CACHE_VERSION,
            fingerprint=current_fingerprint,
        )
        if result is not None:
            cached_results[dataset.run_id] = result
    if len(cached_results) == len(config.datasets):
        write_aggregate(list(cached_results.values()), existing.output_root)
    _copy_plot_script(existing.output_root, reset=False)
    remaining = stale_analysis_runs(existing.output_root)
    if remaining:
        raise RuntimeError("Case-analysis repair did not refresh every stale case: " + ", ".join(remaining))
    try:
        root_payload = json.loads(existing.manifest.read_text(encoding="utf-8"))
        root_payload["analysis_cache_version"] = ANALYSIS_CACHE_VERSION
        existing.manifest.write_text(json.dumps(root_payload, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    print(_style("Minimal case-analysis repair complete; continuing to plot settings.", "bold", "green"))


def _recompute_stale_md_analysis(existing: ExistingAnalysis) -> None:
    """Backward-compatible alias for the generic case-analysis repair."""
    _recompute_stale_analysis(existing)


def _edit_completed_plots(existing: ExistingAnalysis) -> None:
    targets = discover_plot_targets(existing.output_root)
    if not targets:
        raise ValueError(f"No saved plot data were found under {existing.output_root}")
    print()
    print(_style("Completed analysis: result-plot update", "bold", "green"))
    print(f"  Output: {existing.output_root}")
    print(_style("  Valid caches are reused; stale case-frame or pose-density data can be minimally repaired on request.", "dim"))
    _table(
        "Available plot groups (multiple selections allowed; A or 0 selects all)",
        [(str(index), target.label, target.description) for index, target in enumerate(targets, start=1)],
    )
    selected_numbers = _select_numbers("Choose plot group numbers", len(targets), default="A", allow_all=True)
    selected = [targets[index - 1] for index in selected_numbers]
    if any(target.run_directory is not None and ":pose-" not in target.key for target in selected):
        stale_cases = stale_analysis_runs(existing.output_root)
        if stale_cases:
            _note(
                "These saved case results predate the current coordinate/membership semantics. Redrawing alone "
                "cannot correct the saved tables or density grids: " + ", ".join(stale_cases)
            )
            if not _yes_no("Run the minimal case-analysis repair now and then continue replotting", default=True):
                print(f'  Manual command: python analyses.py run -c "{existing.config_path}"')
                print("  Do not add --force; independent pose caches will be reused.")
                return
            selected_keys = [target.key for target in selected]
            _recompute_stale_analysis(existing)
            refreshed = {target.key: target for target in discover_plot_targets(existing.output_root)}
            selected = [refreshed[key] for key in selected_keys if key in refreshed]
            missing = [key for key in selected_keys if key not in refreshed]
            if missing:
                raise RuntimeError("Recomputed plot group(s) were not rediscovered: " + ", ".join(missing))
    if any(":pose-density-" in target.key for target in selected):
        stale_runs = stale_pose_hydration_runs(existing.output_root)
        if stale_runs:
            _note(
                "These saved pose-hydration grids predate the cavity-frame/PBC repair and cannot be "
                "corrected by redrawing PNG files alone: " + ", ".join(stale_runs)
            )
            if not _yes_no("Run the minimal pose-hydration repair now and then continue replotting", default=True):
                print(f'  Manual command: python analyses.py run -c "{existing.config_path}"')
                print("  Valid pose-training and pooled cluster caches will be reused; --force is not required.")
                return
            selected_keys = [target.key for target in selected]
            _recompute_stale_pose_hydration(existing)
            refreshed = {target.key: target for target in discover_plot_targets(existing.output_root)}
            selected = [refreshed[key] for key in selected_keys if key in refreshed]
            missing = [key for key in selected_keys if key not in refreshed]
            if missing:
                raise RuntimeError("Recomputed plot group(s) were not rediscovered: " + ", ".join(missing))
    style = load_plot_style(existing.output_root)
    _table(
        "Current plot settings",
        [
            ("1", "Axes", f"limits={style.get('axis_limits', {})}"),
            ("2", "2D heatmap display grid", f"spacing_A={style.get('density_display_grid_A')} (native=None)"),
            ("3", "Grid lines", f"visible={style.get('grid_visible', True)}, spacing_A={style.get('grid_spacing_A')}, alpha={style.get('grid_alpha', 0.22)}"),
            ("4", "Fonts", f"base/title/axis/tick={style.get('font_size')}/{style.get('title_font_size')}/{style.get('axis_label_font_size')}/{style.get('tick_font_size')}"),
            ("5", "Density colors", f"cmap={style.get('density_cmap')}, vmin={style.get('density_vmin')}, vmax={style.get('density_vmax')}"),
            ("6", "3D isosurfaces", f"levels%={style.get('density_3d_isosurface_levels_percent', [8, 25, 50])}, opacity={style.get('density_3d_opacity', 0.22)}"),
            (
                "7", "Substrate overlay",
                f"enabled={style.get('substrate_overlay', True)}, labels={style.get('substrate_labels', True)}, "
                f"3D color={style.get('substrate_atom_color_3d') or 'element'}, "
                f"brightness={style.get('substrate_atom_brightness_3d', 1.8)}, "
                f"atom opacity={style.get('substrate_atom_opacity_3d', 1.0)}",
            ),
            ("8", "Figure/export", f"figure={style.get('figure_size')}, density={style.get('density_figure_size')}, dpi={style.get('dpi')}"),
            ("9", "Redraw only", "Keep every current setting and regenerate the selected plot groups."),
        ],
    )
    categories = _select_numbers(
        "Choose setting groups to modify (9 redraws without changes; A edits all)",
        9,
        default="9",
        allow_all=True,
    )
    if 1 in categories:
        raw = _ask(
            "Axis limits; use key=xmin,xmax,ymin,ymax (semicolon-separated, blank keeps current)",
            "",
        )
        if raw:
            limits = dict(style.get("axis_limits", {}))
            for assignment in raw.split(";"):
                key, separator, values = assignment.partition("=")
                parsed = [float(value.strip()) for value in values.split(",") if value.strip()]
                if not separator or len(parsed) != 4:
                    raise ValueError(f"Invalid axis limit assignment: {assignment}")
                limits[key.strip()] = parsed
            style["axis_limits"] = limits
    if 2 in categories:
        style["density_display_grid_A"] = _optional_float(
            "2D display-grid spacing in A (native keeps the analysis grid)", style.get("density_display_grid_A")
        )
    if 3 in categories:
        style["grid_visible"] = _yes_no("Show plot grid lines", default=bool(style.get("grid_visible", True)))
        style["grid_spacing_A"] = _optional_float("Major coordinate grid-line spacing in A", style.get("grid_spacing_A"))
        style["grid_alpha"] = float(_ask("Grid-line opacity", str(style.get("grid_alpha", 0.22))))
    if 4 in categories:
        for key, label in (
            ("font_size", "Base font size"), ("title_font_size", "Title font size"),
            ("axis_label_font_size", "Axis-label font size"), ("tick_font_size", "Tick font size"),
            ("legend_font_size", "Legend font size"),
        ):
            style[key] = float(_ask(label, str(style.get(key, 11))))
    if 5 in categories:
        style["density_cmap"] = _ask("Density colormap", str(style.get("density_cmap", "viridis")))
        style["density_vmin"] = _optional_float("Density color minimum", style.get("density_vmin"))
        style["density_vmax"] = _optional_float("Density color maximum", style.get("density_vmax"))
    if 6 in categories:
        current = ",".join(str(value) for value in style.get("density_3d_isosurface_levels_percent", [8, 25, 50]))
        style["density_3d_isosurface_levels_percent"] = [float(value) for value in _csv(_ask("Isosurface levels (% of maximum)", current))]
        style["density_3d_opacity"] = float(
            _ask("Isosurface opacity (0=transparent, 1=opaque; 0.12-0.25 exposes substrate well)", str(style.get("density_3d_opacity", 0.22)))
        )
        style["density_3d_surface_step_size"] = int(_ask("Surface mesh step (1=smoothest)", str(style.get("density_3d_surface_step_size", 1))))
    if 7 in categories:
        style["substrate_overlay"] = _yes_no("Draw OPP/ATC/selected substrate", default=bool(style.get("substrate_overlay", True)))
        style["substrate_labels"] = _yes_no("Label substrate residue names", default=bool(style.get("substrate_labels", True)))
        style["substrate_atom_size"] = float(_ask("2D substrate atom size", str(style.get("substrate_atom_size", 28.0))))
        style["substrate_atom_size_3d"] = float(_ask("3D substrate atom size", str(style.get("substrate_atom_size_3d", 48.0))))
        current_color = style.get("substrate_atom_color_3d")
        color = _ask(
            "3D substrate atom color (element, color name, or #RRGGBB)",
            "element" if current_color in {None, ""} else str(current_color),
        ).strip()
        style["substrate_atom_color_3d"] = None if color.lower() in {"", "auto", "element", "elements", "none"} else color
        style["substrate_atom_brightness_3d"] = float(
            _ask("3D substrate brightness multiplier (1=original; try 1.8-2.5)", str(style.get("substrate_atom_brightness_3d", 1.8)))
        )
        style["substrate_atom_opacity_3d"] = float(
            _ask("3D substrate atom opacity (0-1)", str(style.get("substrate_atom_opacity_3d", 1.0)))
        )
        style["substrate_atom_depthshade_3d"] = _yes_no(
            "Use 3D depth shading on substrate atoms (off is brighter)",
            default=bool(style.get("substrate_atom_depthshade_3d", False)),
        )
        style["substrate_draw_on_top_3d"] = _yes_no(
            "Draw substrate over translucent density surfaces for emphasis",
            default=bool(style.get("substrate_draw_on_top_3d", True)),
        )
        style["substrate_bond_color_3d"] = _ask(
            "3D substrate bond color", str(style.get("substrate_bond_color_3d", "#707070"))
        )
        style["substrate_bond_width_3d"] = float(
            _ask("3D substrate bond width", str(style.get("substrate_bond_width_3d", 2.4)))
        )
        style["substrate_label_color_3d"] = _ask(
            "3D substrate label color", str(style.get("substrate_label_color_3d", "#111111"))
        )
    if 8 in categories:
        figure = _csv(_ask("General figure width,height", ",".join(str(value) for value in style.get("figure_size", (9, 5.5)))))
        density = _csv(_ask("Density figure width,height", ",".join(str(value) for value in style.get("density_figure_size", (8, 6.4)))))
        if len(figure) != 2 or len(density) != 2:
            raise ValueError("Figure sizes require width,height")
        style["figure_size"] = [float(value) for value in figure]
        style["density_figure_size"] = [float(value) for value in density]
        style["dpi"] = int(_ask("Output DPI", str(style.get("dpi", 300))))
    style_path = save_plot_style(existing.output_root, style)
    print(_style(f"Regenerating selected plots now; settings saved to {style_path}", "cyan"))
    outputs = render_plot_targets(existing.output_root, style, selected)
    print(_style(f"Plot-only update complete: {len(outputs)} file(s) regenerated.", "bold", "green"))
    print(f"  Standalone rerun: python analyses.py replot {existing.output_root}")


def _write_and_finish(
    destination_path: Path,
    sections: list[list[str]],
    *,
    configure_pose: bool,
    md_selected: list[DiscoveredCase],
) -> tuple[Path, AnalysisConfig]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sections = _portable_config_sections(sections, destination_path.parent)
    destination_path.write_text("\n\n".join("\n".join(section) for section in sections) + "\n", encoding="utf-8")
    config = load_analysis_config(destination_path)
    launchers = render_analysis_launchers(config, destination_path.parent)
    print()
    print(_style("Analysis setup complete", "bold", "green"))
    print(f"  {_style('Config:', 'bold', 'white')} {destination_path}")
    for label, path in launchers.items():
        print(f"  {_style(label + ':', 'bold', 'white')} {path}")
    print("  Expanded tasks:", ", ".join(config.analysis.tasks))
    if configure_pose:
        counts = (_frame_count(case.topology, case.trajectory) for case in md_selected)
        total = sum(
            min(count, config.pose.max_frames_per_trajectory)
            if config.pose.max_frames_per_trajectory and count
            else (count or 0)
            for count in counts
        )
        print(
            f"  Pose training cap: {config.pose.max_frames_per_trajectory or 'all'} frame(s) per trajectory; "
            f"expected pooled maximum: {total or 'unknown'}"
        )
    print("  Output root:", config.output.root)
    print(_style("  No analysis or GUI was launched. Run one of the generated shell/Slurm files when ready.", "yellow"))
    return destination_path, config


def _run_discovery_wizard(destination: str | Path | None = None) -> tuple[Path, AnalysisConfig] | None:
    _banner()
    skip_completed_prompt = False
    if destination is None:
        partial = _failed_analyses(Path.cwd().resolve())
        if partial:
            recovered = _resume_existing_analysis(_choose_failed_analysis(partial))
            if recovered is not None:
                return recovered
        existing_grand = discover_grand_alignment_outputs(Path.cwd().resolve(), max_depth=3)
        grand_action = ""
        if existing_grand:
            grand_action = _manage_existing_grand_alignment(
                _choose_grand_alignment(Path.cwd().resolve(), existing_grand)
            )
            if grand_action in {"done", "exit"}:
                return None
            if grand_action == "new-setup":
                skip_completed_prompt = True
        grand_candidates = (
            []
            if grand_action == "new-setup"
            else discover_completed_analysis_roots(Path.cwd().resolve(), max_depth=5)
        )
        if grand_action == "new-grand":
            if len(grand_candidates) < 2:
                raise ValueError("Fewer than two completed analyses are available for a new grand alignment")
            _run_grand_alignment_wizard(Path.cwd().resolve(), grand_candidates)
            return None
        if len(grand_candidates) >= 2:
            action = _choice(
                "Multiple completed analysis batches were found",
                ("grand-align", "new-setup", "exit"),
                "grand-align",
            )
            if action == "grand-align":
                _run_grand_alignment_wizard(Path.cwd().resolve(), grand_candidates)
                return None
            if action == "exit":
                return None
            skip_completed_prompt = True
        completed = [] if skip_completed_prompt else _completed_analyses(Path.cwd().resolve())
        if completed:
            existing = _choose_failed_analysis(completed)
            CHOICE_DETAILS.update(
                {
                    "modify-plots": (
                        "Modify result plots",
                        "Reuse completed data, minimally repair stale pose-density maps when needed, and redraw selected plots.",
                    ),
                    "new-setup": ("Start a new setup", "Ignore completed outputs and continue to case discovery."),
                }
            )
            action = _choice("A completed analysis was found", ("modify-plots", "new-setup"), "modify-plots")
            if action == "modify-plots":
                _edit_completed_plots(existing)
                return None
    _step(1, "Discover Cases", "Recursively locate physical MD pairs and PocketMC accepted-state outputs.")
    scan_root = Path(_ask("Scan root", str(Path.cwd()))).expanduser().resolve()
    depth = int(_ask("Maximum subdirectory depth", "4"))
    cases = discover_cases(scan_root, max_depth=depth, deep=False)
    print()
    print(format_cases(cases))
    if not cases:
        _note("No analyzable cases were detected; switching to manual setup.")
        return run_wizard(destination=destination, discover_first=False)
    indices = _select_numbers("Select case numbers (comma-separated; A/0 = all)", len(cases), default="1")
    selected = [cases[index - 1] for index in indices]

    _step(2, "Choose Analysis Inputs", "Physical MD is always preferred. MC analysis is offered only when MD results are absent.")
    analysis_cases: list[tuple[DiscoveredCase, str]] = []
    md_selected: list[DiscoveredCase] = []
    for case in selected:
        md_available = case.md_status == "confirmed" and case.topology is not None and case.trajectory is not None
        mc_available = case.pocketmc_status in {"confirmed", "probable"} and case.mc_trajectory is not None
        if md_available:
            analysis_cases.append((case, "md"))
            md_selected.append(case)
            print(_style(f"  {case.case_id}: physical MD selected ({case.topology.name} + {case.trajectory.name})", "green"))
            continue
        if mc_available:
            _note(
                f"{case.case_id}: no physical MD topology/trajectory pair was found. "
                "PocketMC accepted states can be analyzed, but they do not represent physical time."
            )
            if _yes_no(f"Analyze the MC results for {case.case_id}", default=True):
                analysis_cases.append((case, "pocketmc"))
            continue
        print(_style(f"  {case.case_id}: skipped; no confirmed physical MD or complete PocketMC result was found.", "red"))
    if not analysis_cases:
        raise ValueError("No analyzable input remained after MD/MC confirmation")

    _step(3, "Tracked Molecule", "Choose the molecule whose cavity occupancy, paths, and density will be analyzed.")
    preset = _choice("Tracked molecule", ("water", "co", "custom"), "water")
    molecule_lines = ["[molecule]", f"preset = {_q(preset)}"]
    if preset == "custom":
        molecule_lines.extend(
            [
                f"label = {_q(_ask('Display label', 'Tracked molecule'))}",
                f"resnames = {_array(_csv(_ask('Residue names (comma-separated)')))}",
                f"point_mode = {_q(_choice('Representative point', ('atom', 'cog', 'com'), 'cog'))}",
                f"atom_names = {_array(_csv(_ask('Atom names (blank means all)', '')))}",
            ]
        )

    _step(4, "Cavity Definition", "Reuse each MC run's detected sphere/mask mode by default and review every mask bundle together.")
    chosen_cases = [case for case, _kind in analysis_cases]
    modes = _choose_cavity_modes(chosen_cases)
    initial_anchor_candidates = _substrate_candidates(md_selected[0].topology) if md_selected else []
    anchor = _choose_anchor_request(initial_anchor_candidates)
    case_anchors = _resolve_case_anchors(chosen_cases, anchor)
    anchor_atoms = _csv(_ask("Anchor atoms", "C2,C4,C7"))
    radius_nm = 0.60
    if any(mode == "sphere" for mode in modes.values()):
        radius_nm = float(_ask("Sphere radius in nm", "0.60"))

    bundles: dict[str, CavityBundle] = {}
    existing_mask_cases = [
        case
        for case in chosen_cases
        if modes[case.case_id] == "mask" and case.pocketmc_status in {"confirmed", "probable"}
    ]
    if existing_mask_cases:
        bundles.update(_review_existing_bundles(existing_mask_cases, scan_root, modes))
    for case in chosen_cases:
        if modes[case.case_id] != "mask" or case.case_id in bundles:
            continue
        _note(
            f"{case.case_id}: no prior MC cavity exists. A seeded GCMC voxel cavity will be built later by the generated launcher; "
            "the wizard is only recording the required inputs now."
        )
        bundles[case.case_id] = _deferred_bundle(case)

    _step(5, "Analysis Tasks", "Select one or more outputs. Use A or 0 as the all-applicable shortcut.")
    has_md = bool(md_selected)
    has_mc = any(kind == "pocketmc" for _case, kind in analysis_cases)
    requested = _task_selection(has_md, has_mc)
    normalized = {item.lower().replace("_", "-") for item in requested}
    explicit_pose = bool(normalized & {"pose", "cluster", "clusters", "pose-clusters", "pose-hydration", "compare-hydration"})
    if explicit_pose:
        configure_pose = True
        _note("Substrate configuration is required by the selected pose task(s).")
    else:
        configure_pose = has_md and _yes_no("Configure substrate pose/hydration analysis", default=True)
    substrate_lines: list[str] = []
    pose_lines: list[str] = []
    if configure_pose:
        candidates = _substrate_candidates(md_selected[0].topology)
        default_tokens = ",".join(candidates[:1]) if candidates else anchor
        substrate_selection = _select_substrate(candidates, default_tokens)
        fit_selection = _ask("Substrate fit selection (blank uses all heavy substrate atoms)", "")
        substrate_lines = ["[substrate]", "enabled = true", f"selection = {_q(substrate_selection)}"]
        if fit_selection:
            substrate_lines.append(f"fit_selection = {_q(fit_selection)}")
        for case in md_selected:
            count = _frame_count(case.topology, case.trajectory)
            print(f"  {case.case_id}: trajectory frames = {count if count is not None else 'unknown'}")
        maximum_text = _ask("Maximum clustering-training frames per trajectory", "5000").strip().lower()
        maximum = 0 if maximum_text in {"all", "a", "0"} else int(maximum_text)
        clusters = int(_ask("Number of common pose clusters", "3"))
        reference_default = str(md_selected[0].directory / "previous.gro")
        if not Path(reference_default).exists() and md_selected[0].topology is not None:
            reference_default = str(md_selected[0].topology)
        reference = _ask("Canonical reference structure", reference_default)
        pocket_selection = _ask("Conserved local-pocket fit selection", "protein and backbone")
        reference_selection = _ask("Reference pocket selection (blank uses same selection)", "")
        pose_lines = [
            "[pose]", f"clusters = {clusters}", "seed = 2026", "restarts = 20",
            f"reference = {_q(reference)}", f"pocket_selection = {_q(pocket_selection)}",
        ]
        if reference_selection:
            pose_lines.append(f"reference_pocket_selection = {_q(reference_selection)}")
        pose_lines.extend(
            ["", "[pose.sampling]", f"max_frames_per_trajectory = {maximum}", 'strategy = "uniform"', "write_trajectory = true"]
        )
        if "all" not in normalized:
            for task in ("pose-clusters", "pose-hydration", "compare-hydration"):
                if task not in requested:
                    requested.append(task)

    _step(6, "Outputs", "Choose the result directory and save a reproducible TOML plus three run templates.")
    output_root = _ask("Results directory", "analysis-results")
    destination_path = Path(destination or _ask("Save configuration as", "analyses.toml")).expanduser().resolve()
    default_mode = next(iter(set(modes.values()))) if len(set(modes.values())) == 1 else "sphere"
    global_anchor = case_anchors[chosen_cases[0].case_id]
    cavity_lines = [
        "[cavity]", f"mode = {_q(default_mode)}", f"anchor = {_q(global_anchor)}",
        f"anchor_atoms = {_array(anchor_atoms)}", f"radius_nm = {radius_nm}", "membership_padding_nm = 0.02",
    ]
    case_sections = [
        _case_lines(
            case, kind, modes[case.case_id], bundles.get(case.case_id),
            anchor=case_anchors[case.case_id], anchor_atoms=anchor_atoms, radius_nm=radius_nm,
        )
        for case, kind in analysis_cases
    ]
    analysis_lines = [
        "[analysis]", f"tasks = {_array(requested)}", "stride = 1", "gap_ps = 1000.0",
        "path_sample_ps = 1000.0", "contact_cutoff_a = 4.0", "density_bin_a = 1.0",
        "density_sigma_a = 2.0", 'density_quantity = "occupancy"',
    ]
    output_lines = ["[output]", f"root = {_q(output_root)}", "cache = true", "overwrite = true"]
    sections = [*case_sections, molecule_lines, cavity_lines]
    if substrate_lines:
        sections.append(substrate_lines)
    if pose_lines:
        sections.append(pose_lines)
    sections.extend([analysis_lines, output_lines])
    return _write_and_finish(
        destination_path, sections, configure_pose=configure_pose, md_selected=md_selected
    )


def run_wizard(
    *,
    destination: str | Path | None = None,
    discover_first: bool = False,
) -> tuple[Path, AnalysisConfig] | None:
    """Collect portable analysis options and write TOML plus three launchers."""
    if discover_first:
        return _run_discovery_wizard(destination)
    _banner()
    _step(1, "Input", "Configure one run or expand run directories from a GCMC TOML.")
    kind = _choice("Input type", ("md", "pocketmc"), "md")
    source = _choice("Input scope", ("single", "gcmc"), "single")
    input_lines = ["[input]", f"kind = {_q(kind)}"]
    if source == "gcmc":
        input_lines.append(f"gcmc_config = {_q(_ask('PocketMC/GCMC config TOML', 'config.toml'))}")
        if kind == "md":
            input_lines.extend(
                [
                    f"topology = {_q(_ask('Run-relative topology pattern', 'md.tpr'))}",
                    f"trajectory = {_q(_ask('Run-relative trajectory pattern', 'md.xtc'))}",
                    f"reference = {_q(_ask('Run-relative reference pattern', 'previous.gro'))}",
                ]
            )
        else:
            input_lines.extend(
                [
                    f"trajectory = {_q(_ask('Run-relative accepted-state trajectory', 'trajectory.gro'))}",
                    f"mc_log = {_q(_ask('Run-relative MC log', 'mc.log'))}",
                    f"trajectory_meta = {_q(_ask('Run-relative state metadata', 'trajectory.meta.jsonl'))}",
                ]
            )
    else:
        input_lines.append(f"run_id = {_q(_ask('Run ID', 'run'))}")
        if kind == "md":
            input_lines.extend(
                [
                    f"topology = {_q(_ask('Topology (TPR/GRO/PDB)', 'md.tpr'))}",
                    f"trajectory = {_q(_ask('Trajectory (XTC/TRR/GRO)', 'md.xtc'))}",
                    f"reference = {_q(_ask('Alignment reference (optional)', ''))}",
                ]
            )
        else:
            input_lines.extend(
                [
                    f"trajectory = {_q(_ask('PocketMC accepted-state trajectory', 'trajectory.gro'))}",
                    f"mc_log = {_q(_ask('MC log', 'mc.log'))}",
                    f"trajectory_meta = {_q(_ask('State metadata sidecar', 'trajectory.meta.jsonl'))}",
                ]
            )

    _step(2, "Tracked Molecule", "Choose Water, CO, or a manual residue definition by number.")
    preset = _choice("Tracked molecule", ("water", "co", "custom"), "water")
    molecule_lines = ["[molecule]", f"preset = {_q(preset)}"]
    if preset == "custom":
        resnames = _csv(_ask("Residue names (comma-separated)"))
        point_mode = _choice("Representative point", ("atom", "cog", "com"), "cog")
        atom_names = _csv(_ask("Atom names (comma-separated; blank means all atoms)", ""))
        molecule_lines.extend(
            [
                f"label = {_q(_ask('Display label', 'Tracked molecule'))}",
                f"resnames = {_array(resnames)}", f"point_mode = {_q(point_mode)}", f"atom_names = {_array(atom_names)}",
            ]
        )

    _step(3, "Cavity", "Sphere asks for a radius; mask uses an existing GCMC voxel bundle.")
    cavity_mode = _choice("Cavity definition", ("sphere", "mask"), "sphere")
    cavity_lines = ["[cavity]", f"mode = {_q(cavity_mode)}"]
    if cavity_mode == "mask":
        while True:
            try:
                bundle = _bundle_from_user_path(_ask("Cavity directory or *_mask.dat path", "cavity_mask.dat"))
                if bundle.mask is None:
                    raise ValueError(f"No cavity mask was found in {bundle.directory}")
                break
            except (FileNotFoundError, ValueError) as exc:
                print(_style(str(exc), "red"))
        cavity_lines.append(f"mask = {_q(bundle.mask)}")
        if bundle.meta:
            cavity_lines.append(f"meta = {_q(bundle.meta)}")
        if bundle.points:
            cavity_lines.append(f"points = {_q(bundle.points)}")
        if bundle.nearby:
            cavity_lines.append(f"nearby_residues = {_q(bundle.nearby)}")
    anchor = _ask("Anchor residue (for example 800ATC)", "800ATC")
    anchor_atoms = _csv(_ask("Anchor atoms", "C2,C4,C7"))
    cavity_lines.extend([f"anchor = {_q(anchor)}", f"anchor_atoms = {_array(anchor_atoms)}"])
    if cavity_mode == "sphere":
        cavity_lines.append(f"radius_nm = {_ask('Sphere radius in nm', '0.60')}")

    _step(4, "Tasks", "Select task numbers; use A or 0 for all applicable tasks.")
    requested = _task_selection(kind == "md", kind == "pocketmc")
    requested_normalized = {item.lower().replace("_", "-") for item in requested}
    manual_pose = kind == "md" and bool(
        requested_normalized & {"pose", "cluster", "clusters", "pose-clusters", "pose-hydration", "compare-hydration"}
    )
    substrate_lines: list[str] = []
    pose_lines: list[str] = []
    if manual_pose:
        substrate_selection = _selection_from_tokens(
            _ask("Substrate residues, or selection=<MDAnalysis selection>", "800ATC")
        )
        substrate_lines = ["[substrate]", "enabled = true", f"selection = {_q(substrate_selection)}"]
        fit_selection = _ask("Substrate fit selection (blank uses all heavy substrate atoms)", "")
        if fit_selection:
            substrate_lines.append(f"fit_selection = {_q(fit_selection)}")
        maximum_text = _ask("Maximum clustering-training frames per trajectory", "5000").strip().lower()
        maximum = 0 if maximum_text in {"all", "a", "0"} else int(maximum_text)
        pose_lines = [
            "[pose]", f"clusters = {int(_ask('Number of common pose clusters', '3'))}",
            f"reference = {_q(_ask('Canonical reference structure', 'previous.gro'))}",
            f"pocket_selection = {_q(_ask('Conserved local-pocket fit selection', 'protein and backbone'))}",
            "", "[pose.sampling]", f"max_frames_per_trajectory = {maximum}", 'strategy = "uniform"', "write_trajectory = true",
        ]
    expanded = expand_tasks(kind, requested, pose_enabled=manual_pose)
    _step(5, "Outputs", "Save the TOML and generate direct-shell, generic Slurm, and Tahoma-only Slurm launchers.")
    output_root = _ask("Results directory", "analysis-results")
    destination_path = Path(destination or _ask("Save configuration as", "analyses.toml")).expanduser().resolve()
    analysis_lines = [
        "[analysis]", f"tasks = {_array(requested)}", "stride = 1", "gap_ps = 1000.0",
        "path_sample_ps = 1000.0", "contact_cutoff_a = 4.0", "density_bin_a = 1.0",
        "density_sigma_a = 2.0", 'density_quantity = "occupancy"',
    ]
    output_lines = ["[output]", f"root = {_q(output_root)}", "cache = true", "overwrite = true"]
    sections = [input_lines, molecule_lines, cavity_lines]
    if substrate_lines:
        sections.append(substrate_lines)
    if pose_lines:
        sections.append(pose_lines)
    sections.extend([analysis_lines, output_lines])
    path, config = _write_and_finish(destination_path, sections, configure_pose=manual_pose, md_selected=[])
    print("  Expanded tasks (manual selection):", ", ".join(expanded))
    return path, config
