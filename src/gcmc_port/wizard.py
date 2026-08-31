from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

from .assets import default_asset_path
from .config import CavityBuildConfig, CavityConfig, ExecutionConfig, SimulationConfig, SlurmConfig, load_config
from .reference import SKIP_RESIDUE_NAMES, StructureResidue, collect_structure_residues
from .slurm import render_local_script, render_sbatch, render_tahoma_only_sbatch


@dataclass(frozen=True, slots=True)
class WizardArtifacts:
    config_path: Path
    local_script_path: Path
    generic_sbatch_path: Path
    tahoma_only_sbatch_path: Path


@dataclass(frozen=True, slots=True)
class WizardChoice:
    key: str
    label: str
    description: str
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class InsertedMolecule:
    key: str
    label: str
    description: str
    water_itp: str
    gas_gro: str
    mu0: float


@dataclass(frozen=True, slots=True)
class ResidueOption:
    residue: StructureResidue
    category: str


ANSI_RESET = "\033[0m"
ANSI = {
    "cyan": "\033[96m",
    "blue": "\033[94m",
    "white": "\033[97m",
    "yellow": "\033[93m",
    "green": "\033[92m",
    "dim": "\033[2m",
    "bold": "\033[1m",
}

INSERTED_MOLECULES = (
    InsertedMolecule(
        key="water",
        label="Water",
        description="TIP3P water using the bundled WAT.itp and water template.",
        water_itp="WAT.itp",
        gas_gro="COM.gro",
        mu0=-25.48056,
    ),
    InsertedMolecule(
        key="co",
        label="CO",
        description="Three-site carbon monoxide model with a massless dummy site.",
        water_itp="co/COM.itp",
        gas_gro="co/COM.gro",
        mu0=-58.9323290,
    ),
)


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _style(text: str, *styles: str) -> str:
    if not _supports_color():
        return text
    prefix = "".join(ANSI[style] for style in styles)
    return f"{prefix}{text}{ANSI_RESET}"


def _print_banner() -> None:
    print()
    print(_style("╔════════════════════════════════════════════════════════════════════╗", "cyan"))
    print(_style("║", "cyan") + _style(" PocketMC Interactive Wizard".ljust(66), "bold", "white") + _style("║", "cyan"))
    print(_style("║", "cyan") + _style(" Generate a TOML plus direct-shell and Slurm launchers.".ljust(66), "dim", "cyan") + _style("║", "cyan"))
    print(_style("╚════════════════════════════════════════════════════════════════════╝", "cyan"))


def _print_step_header(step_number: int, title: str, description: str) -> None:
    print()
    print(f"{_style(f' Step {step_number} ', 'bold', 'blue')}{_style(f' {title}', 'bold', 'white')}")
    print(_style(description, "cyan"))


def _display_choice_table(title: str, rows: list[tuple[str, str, str]]) -> None:
    print()
    print(_style(title, "bold", "white"))
    print(_style("─" * max(24, len(title)), "cyan"))
    for number, label, description in rows:
        print(f"  {_style(number.rjust(2), 'bold', 'cyan')}  {_style(label, 'bold', 'white')}  {_style(description, 'dim')}")


def _prompt_message(text: str) -> str:
    return _style(text, "green")


def _print_prompt_hint(text: str) -> None:
    print(_style(f"  {text}", "dim"))


def _print_banner() -> None:
    print()
    print(_style("=" * 72, "cyan"))
    print(_style(" PocketMC Interactive Wizard ", "bold", "white"))
    print(_style(" Generate a TOML plus direct-shell and Slurm launchers.", "dim", "cyan"))
    print(_style("=" * 72, "cyan"))


def _print_step_header(step_number: int, title: str, description: str) -> None:
    print()
    print(_style("-" * 72, "blue"))
    print(f"{_style(f'[Step {step_number}]', 'bold', 'blue')} {_style(title, 'bold', 'white')}")
    print(_style(description, "cyan"))


def _display_choice_table(title: str, rows: list[tuple[str, str, str]]) -> None:
    print()
    print(_style(title, "bold", "white"))
    print(_style("-" * max(24, len(title)), "cyan"))
    for number, label, description in rows:
        print(f"  {_style(number.rjust(2), 'bold', 'cyan')}  {_style(label, 'bold', 'white')}  {_style(description, 'dim')}")


def build_wizard_case(write_config: str | None = None) -> WizardArtifacts:
    _print_banner()
    _print_step_header(1, "Input Files", "Choose the initial GRO structure and topology for this case.")
    gro_path = _prompt_file_selection("initial GRO structure", "*.gro", preferred_names=("init.gro",))
    top_path = _prompt_file_selection("topology file", "*.top", preferred_names=("topol.top",))

    _print_step_header(2, "Built-in Defaults", "Review the currently supported force field and water model.")
    _print_forcefield_notice()

    _print_step_header(3, "Inserted Molecule", "Choose what the MC moves insert into the selected region.")
    molecule = _prompt_inserted_molecule()

    _print_step_header(4, "Cavity Mode", "Choose whether the MC region is sphere-based or voxel-mask-based.")
    mode_choices = [
        WizardChoice("sphere", "Sphere mode", "Use a sphere around the reference point."),
        WizardChoice("mask", "Mask mode", "Build and use a seeded voxel cavity automatically."),
    ]
    mode = _prompt_choice(
        "Choose the cavity mode",
        mode_choices,
        default_key="sphere",
    )

    residue_types = _load_residue_types(default_asset_path("residuetypes.dat"))
    _print_step_header(5, "Reference Residues", "Pick substrate or catalytic residues that define the reference point.")
    selected_residues = _prompt_anchor_residues(gro_path, residue_types)

    reference_mode = "com"
    center_atoms: list[str] = []
    if len(selected_residues) == 1:
        single_residue = selected_residues[0]
        available_atoms = _unique_atom_names(single_residue)
        default_reference = "atoms" if any(atom in {"C2", "C4", "C7"} for atom in available_atoms) else "com"
        reference_mode = _prompt_choice(
            "Choose how to define the reference point",
            [
                WizardChoice("com", "Residue COM", "Use the full residue center of mass."),
                WizardChoice("atoms", "Selected atoms", "Choose one or more atom names from the residue."),
            ],
            default_key=default_reference,
        )
        if reference_mode == "atoms":
            center_atoms = _prompt_atom_selection(available_atoms)
    else:
        print(_style("Multiple residues were selected, so the wizard will use their combined COM as the reference point.", "yellow"))

    sim_defaults = SimulationConfig()
    cavity_build_defaults = CavityBuildConfig()
    cavity_defaults = CavityConfig()
    wizard_target_nmol_default = 0

    _print_step_header(6, "Simulation Basics", "Set the main MC conditions. Fine-grained tuning can stay at defaults.")
    _print_prompt_hint("Simulation temperature used in the MC acceptance and chemical-potential correction.")
    temperature = _prompt_float("Temperature (K)", sim_defaults.temperature)
    _print_prompt_hint("Dimensionless activity/fugacity ratio p/p0 used in mu = mu_ex + R*T*ln(p/p0); it must be positive.")
    pressure = _prompt_float("Activity/fugacity ratio p/p0", sim_defaults.pressure)
    _print_prompt_hint("Maximum number of MC trial moves attempted in this run.")
    max_trials = _prompt_int("Maximum MC trials", sim_defaults.max_trials)
    _print_prompt_hint("Target occupancy stop condition. Use 0 to disable it; the wizard defaults to 0 because this is often not known ahead of time.")
    target_nmol = _prompt_int("Target number of inserted molecules in the region (0 disables this stop condition)", wizard_target_nmol_default)
    _print_prompt_hint(
        f"Model-calibrated excess chemical potential for {molecule.label}. "
        f"The bundled default for this selection is {molecule.mu0} kJ/mol."
    )
    mu0 = _prompt_float("mu0 (kJ/mol)", molecule.mu0)
    rmax = sim_defaults.rmax
    if mode == "sphere":
        _print_prompt_hint("Sphere radius around the selected reference point that defines the insertion/proposal region.")
        rmax = _prompt_float("Sphere radius rmax (nm)", sim_defaults.rmax)

    advanced_values = {
        "gas_constant": sim_defaults.gas_constant,
        "rvdw": sim_defaults.rvdw,
        "rfree": sim_defaults.rfree,
        "v0": sim_defaults.v0,
        "kres": sim_defaults.kres,
        "max_e0_tries": sim_defaults.max_e0_tries,
        "mask_dx": sim_defaults.mask_dx,
        "dx": cavity_build_defaults.dx,
        "probe_radius": cavity_build_defaults.probe_radius,
        "search_radius": cavity_build_defaults.search_radius,
        "nearby_cutoff": cavity_build_defaults.nearby_cutoff,
        "min_points": cavity_build_defaults.min_points,
    }
    _print_step_header(7, "Advanced Options", "Review optional low-level controls. The default answer is No.")
    _print_advanced_summary(mode, advanced_values)
    if _prompt_yes_no("Would you like to edit the advanced options?", default=False):
        advanced_values["gas_constant"] = _prompt_float("gas_constant (kJ/mol/K)", advanced_values["gas_constant"])
        advanced_values["rvdw"] = _prompt_float("rvdw (nm)", advanced_values["rvdw"])
        advanced_values["rfree"] = _prompt_float("rfree", advanced_values["rfree"])
        advanced_values["v0"] = _prompt_float("v0 (nm^3)", advanced_values["v0"])
        advanced_values["kres"] = _prompt_float("kres", advanced_values["kres"])
        advanced_values["max_e0_tries"] = _prompt_int("max_e0_tries", int(advanced_values["max_e0_tries"]))
        advanced_values["mask_dx"] = _prompt_float("mask_dx (nm)", advanced_values["mask_dx"])
        if mode == "mask":
            advanced_values["dx"] = _prompt_float("Cavity voxel spacing dx (nm)", advanced_values["dx"])
            advanced_values["probe_radius"] = _prompt_float("Cavity probe radius (nm)", advanced_values["probe_radius"])
            advanced_values["search_radius"] = _prompt_float("Cavity search radius (nm)", advanced_values["search_radius"])
            advanced_values["nearby_cutoff"] = _prompt_float("Nearby-residue cutoff (nm)", advanced_values["nearby_cutoff"])
            advanced_values["min_points"] = _prompt_int("Minimum cavity points", int(advanced_values["min_points"]))

    _print_step_header(8, "GROMACS Command", "Choose the GROMACS executable style used by the generated config.")
    gmx_cmd = _prompt_choice(
        "Choose the GROMACS command",
        [
            WizardChoice("gmx_mpi", "gmx_mpi", "MPI-enabled GROMACS command."),
            WizardChoice("gmx", "gmx", "Threaded GROMACS command."),
        ],
        default_key="gmx_mpi",
    )

    config_path = _resolve_config_path(gro_path, top_path, write_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    job_name = f"{config_path.parent.name or 'gcmc'}-gcmc"

    config_text = _render_wizard_toml(
        config_dir=config_path.parent,
        gro_path=gro_path,
        top_path=top_path,
        job_name=job_name,
        molecule=molecule,
        mode=mode,
        selected_residues=selected_residues,
        reference_mode=reference_mode,
        center_atoms=center_atoms,
        temperature=temperature,
        pressure=pressure,
        max_trials=max_trials,
        target_nmol=target_nmol,
        mu0=mu0,
        rmax=rmax,
        gmx_cmd=gmx_cmd,
        advanced_values=advanced_values,
    )
    config_path.write_text(config_text, encoding="utf-8")

    config = load_config(config_path)
    local_script_path = render_local_script(config, config_path.parent / "run_gcmc.sh")
    generic_sbatch_path = render_sbatch(config, config_path.parent / "run_gcmc.sbatch")
    tahoma_only_sbatch_path = render_tahoma_only_sbatch(
        config,
        config_path.parent / "run_gcmc_tahoma_only.sbatch",
    )

    print("\nWizard complete.")
    print(f"  {_style('Config:', 'bold', 'white')} {config_path}")
    print(f"  {_style('Direct shell launcher:', 'bold', 'white')} {local_script_path}")
    print(f"  {_style('Generic Slurm launcher:', 'bold', 'white')} {generic_sbatch_path}")
    print(f"  {_style('Tahoma-only Slurm launcher:', 'bold', 'white')} {tahoma_only_sbatch_path}")
    return WizardArtifacts(
        config_path=config_path,
        local_script_path=local_script_path,
        generic_sbatch_path=generic_sbatch_path,
        tahoma_only_sbatch_path=tahoma_only_sbatch_path,
    )


def _prompt_file_selection(label: str, pattern: str, *, preferred_names: tuple[str, ...] = ()) -> Path:
    candidates = sorted(
        Path.cwd().resolve().glob(pattern),
        key=lambda path: (0 if path.name in preferred_names else 1, path.name.lower()),
    )
    if candidates:
        rows = [(str(index), path.name, str(path.resolve())) for index, path in enumerate(candidates, start=1)]
        rows.append(("0", "Enter manually", f"Type the path to the {label} yourself."))
        _display_choice_table(f"Detected {label} candidates in {Path.cwd().resolve()}", rows)
        while True:
            raw = input(_prompt_message(f"Choose the {label} number [1]: ")).strip() or "1"
            if raw == "0":
                return _prompt_existing_path(f"Path to the {label}")
            if raw.isdigit():
                index = int(raw)
                if 1 <= index <= len(candidates):
                    return candidates[index - 1].resolve()
            print("Please enter a valid number from the list.")
    return _prompt_existing_path(f"Path to the {label}")


def _prompt_existing_path(message: str) -> Path:
    while True:
        raw = input(_prompt_message(f"{message}: ")).strip()
        if not raw:
            print("A path is required.")
            continue
        path = Path(raw).expanduser().resolve()
        if path.exists():
            return path
        print(f"Path not found: {path}")


def _print_forcefield_notice() -> None:
    print(_style("Force-field and water-model notice", "bold", "white"))
    print(_style("  This wizard supports the bundled AMBER14SB setup with TIP3P water and a three-site CO insert.", "yellow"))
    print(_style("  For other inserted molecules or force fields, provide those inputs manually in the TOML.", "dim"))


def _prompt_inserted_molecule() -> InsertedMolecule:
    choices = [
        WizardChoice(molecule.key, molecule.label, molecule.description)
        for molecule in INSERTED_MOLECULES
    ]
    selected_key = _prompt_choice("Choose the inserted molecule", choices, default_key="water")
    for molecule in INSERTED_MOLECULES:
        if molecule.key == selected_key:
            return molecule
    raise ValueError(f"Unsupported inserted molecule selection: {selected_key}")


def _load_residue_types(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";") or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) >= 2:
            mapping[fields[0].upper()] = fields[1].strip().lower()
    return mapping


def _prompt_anchor_residues(gro_path: Path, residue_types: dict[str, str]) -> list[StructureResidue]:
    residues = collect_structure_residues(gro_path)
    substrate_candidates = [
        ResidueOption(residue=residue, category=_residue_category(residue, residue_types))
        for residue in residues
        if residue.resname.upper() not in SKIP_RESIDUE_NAMES and _residue_category(residue, residue_types) != "protein"
    ]
    if substrate_candidates:
        print()
        print(_style("Possible non-protein substrate residues detected", "bold", "white"))
        _print_residue_options(substrate_candidates)
        print(_style("These residues are not protein, water, Na, or Cl, so they may be substrates or cofactors.", "dim"))
        print(_style("If you prefer a catalytic residue or another anchor instead, choose 0 to browse all non-water residues.", "dim"))
        selected = _prompt_residue_option_list(
            substrate_candidates,
            prompt="Choose one or more residue numbers (comma-separated, 0 = browse all non-water residues)",
            allow_browse_all=True,
        )
        if selected:
            return selected
    print()
    print(_style("The wizard will now show all non-water, non-Na/Cl residues so you can choose an alternative anchor.", "yellow"))
    all_candidates = [
        ResidueOption(residue=residue, category=_residue_category(residue, residue_types))
        for residue in residues
        if residue.resname.upper() not in SKIP_RESIDUE_NAMES
    ]
    if not all_candidates:
        raise ValueError(f"No anchorable residues were found in {gro_path}")
    _print_residue_options(all_candidates)
    selected = _prompt_residue_option_list(
        all_candidates,
        prompt="Choose one or more residue numbers (comma-separated)",
        allow_browse_all=False,
    )
    if not selected:
        raise ValueError("At least one anchor residue must be selected")
    return selected


def _residue_category(residue: StructureResidue, residue_types: dict[str, str]) -> str:
    return residue_types.get(residue.resname.upper(), "unknown")


def _print_residue_options(options: list[ResidueOption]) -> None:
    rows = []
    for index, option in enumerate(options, start=1):
        atom_preview = ", ".join(option.residue.atom_names[:6])
        rows.append(
            (
                str(index),
                option.residue.token,
                f"type={option.category}; atoms={len(option.residue.atom_names)}; sample={atom_preview}",
            )
        )
    _display_choice_table("Residue choices", rows)


def _prompt_residue_option_list(
    options: list[ResidueOption],
    *,
    prompt: str,
    allow_browse_all: bool,
) -> list[StructureResidue]:
    default = "1"
    while True:
        raw = input(_prompt_message(f"{prompt} [{default}]: ")).strip() or default
        if allow_browse_all and raw == "0":
            return []
        selected_numbers: list[int] = []
        valid = True
        for token in raw.split(","):
            value = token.strip()
            if not value.isdigit():
                valid = False
                break
            number = int(value)
            if number < 1 or number > len(options):
                valid = False
                break
            if number not in selected_numbers:
                selected_numbers.append(number)
        if valid and selected_numbers:
            return [options[number - 1].residue for number in selected_numbers]
        print("Please enter one or more valid residue numbers.")


def _unique_atom_names(residue: StructureResidue) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for atom_name in residue.atom_names:
        if atom_name in seen:
            continue
        seen.add(atom_name)
        names.append(atom_name)
    return names


def _prompt_atom_selection(atom_names: list[str]) -> list[str]:
    _display_choice_table(
        "Available atom names for this residue",
        [(str(index), atom_name, "Selectable reference atom") for index, atom_name in enumerate(atom_names, start=1)],
    )
    default_atoms = [name for name in ("C2", "C4", "C7") if name in atom_names]
    default = ",".join(str(atom_names.index(name) + 1) for name in default_atoms) if default_atoms else "1"
    while True:
        raw = input(_prompt_message(f"Choose atom numbers for the reference point (comma-separated) [{default}]: ")).strip() or default
        selected_numbers: list[int] = []
        valid = True
        for token in raw.split(","):
            value = token.strip()
            if not value.isdigit():
                valid = False
                break
            number = int(value)
            if number < 1 or number > len(atom_names):
                valid = False
                break
            if number not in selected_numbers:
                selected_numbers.append(number)
        if valid and selected_numbers:
            return [atom_names[number - 1] for number in selected_numbers]
        print("Please enter one or more valid atom numbers.")


def _print_advanced_summary(mode: str, values: dict[str, float | int]) -> None:
    print(_style("Advanced options available", "bold", "white"))
    lines = [
        f"  gas_constant = {values['gas_constant']} ; gas constant in kJ/mol/K",
        f"  rvdw = {values['rvdw']} ; hard clash cutoff for proposal generation radius in nm",
        f"  rfree = {values['rfree']} ; restraint-shell multiplier around the reference region",
        f"  v0 = {values['v0']} ; model-specific reference molecular volume in nm^3",
        f"  kres = {values['kres']} ; position-restraint force constant",
        f"  max_e0_tries = {values['max_e0_tries']} ; retries for the initial reference energy",
        f"  mask_dx = {values['mask_dx']} ; proposal jitter in mask mode, 0 uses the voxel spacing",
    ]
    if mode == "mask":
        lines.extend(
            [
                f"  dx = {values['dx']} ; cavity voxel spacing in nm",
                f"  probe_radius = {values['probe_radius']} ; minimum cavity clearance in nm",
                f"  search_radius = {values['search_radius']} ; local cavity search radius in nm",
                f"  nearby_cutoff = {values['nearby_cutoff']} ; nearby-residue reporting cutoff in nm",
                f"  min_points = {values['min_points']} ; minimum accepted cavity voxel count",
            ]
        )
    for line in lines:
        print(_style(line, "dim"))


def _prompt_choice(message: str, choices: list[WizardChoice], *, default_key: str) -> str:
    index_by_key = {choice.key: index for index, choice in enumerate(choices, start=1)}
    default_number = index_by_key[default_key]
    _display_choice_table(message, [(str(index), choice.label, choice.description) for index, choice in enumerate(choices, start=1)])
    while True:
        raw = input(_prompt_message(f"Choose a number [{default_number}]: ")).strip() or str(default_number)
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1].key
        print("Please enter a valid choice number.")


def _prompt_yes_no(message: str, *, default: bool) -> bool:
    default_text = "N" if not default else "Y"
    while True:
        raw = input(_prompt_message(f"{message} [Y/N, default {default_text}]: ")).strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer Y or N.")


def _prompt_float(message: str, default: float) -> float:
    while True:
        raw = input(_prompt_message(f"{message} [{default}]: ")).strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number.")


def _prompt_int(message: str, default: int) -> int:
    while True:
        raw = input(_prompt_message(f"{message} [{default}]: ")).strip()
        if not raw:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")


def _resolve_config_path(gro_path: Path, top_path: Path, write_config: str | None) -> Path:
    if write_config:
        return Path(write_config).expanduser().resolve()
    common_parent = Path(os.path.commonpath([str(gro_path.parent), str(top_path.parent)])).resolve()
    return common_parent / "gcmc_wizard.toml"


def _render_wizard_toml(
    *,
    config_dir: Path,
    gro_path: Path,
    top_path: Path,
    job_name: str,
    molecule: InsertedMolecule,
    mode: str,
    selected_residues: list[StructureResidue],
    reference_mode: str,
    center_atoms: list[str],
    temperature: float,
    pressure: float,
    max_trials: int,
    target_nmol: int,
    mu0: float,
    rmax: float,
    gmx_cmd: str,
    advanced_values: dict[str, float | int],
) -> str:
    execution_defaults = _execution_defaults(gmx_cmd)
    slurm_defaults = SlurmConfig(
        shebang="#!/usr/bin/env bash",
        job_name=job_name,
    )
    cavity_defaults = CavityConfig()
    cavity_build_defaults = CavityBuildConfig()
    mask_prefix = cavity_build_defaults.output_prefix
    section_map: list[tuple[str, dict[str, object]]] = [
        (
            "paths",
            {
                "project_root": ".",
                "work_root": ".",
                "forcefield_dir": "amber14sb_parmbsc1.ff",
                "residue_types": "residuetypes.dat",
                "topology": _relative_path(top_path, config_dir),
                "water_itp": molecule.water_itp,
                "chk_mdp": "chk.mdp",
                "steep_mdp": "steep.mdp",
                "em_mdp": "em.mdp",
                "init_gro": _relative_path(gro_path, config_dir),
                "gas_gro": molecule.gas_gro,
            },
        ),
        (
            "execution",
            {
                "gmx_cmd": execution_defaults.gmx_cmd,
                "launcher_single": execution_defaults.launcher_single,
                "launcher_multi": execution_defaults.launcher_multi,
                "mdrun_multi_args": execution_defaults.mdrun_multi_args,
                "nodes": execution_defaults.nodes,
                "cores_per_node": execution_defaults.cores_per_node,
                "module_setup": execution_defaults.module_setup,
            },
        ),
        (
            "execution.env",
            {
                "GMX_MAXBACKUP": "-1",
            },
        ),
        (
            "anchor",
            {
                "anchor": selected_residues[0].token,
                "resid": selected_residues[0].resid,
                "resname": selected_residues[0].resname,
                "residues": [residue.token for residue in selected_residues],
                "reference_mode": reference_mode,
                "center_atoms": center_atoms,
            },
        ),
        (
            "simulation",
            {
                "test_insertion": False,
                "temperature": temperature,
                "pressure": pressure,
                "max_trials": max_trials,
                "max_consecutive_insertion_failures": SimulationConfig().max_consecutive_insertion_failures,
                "target_nmol": target_nmol,
                "mu0": mu0,
                "gas_constant": advanced_values["gas_constant"],
                "rmax": rmax,
                "rvdw": advanced_values["rvdw"],
                "rfree": advanced_values["rfree"],
                "v0": advanced_values["v0"],
                "kres": advanced_values["kres"],
                "max_e0_tries": advanced_values["max_e0_tries"],
                "mask_dx": advanced_values["mask_dx"],
            },
        ),
        (
            "cavity",
            {
                "mode": mode,
                "mask_file": f"{mask_prefix.name}_mask.dat" if mode == "mask" else None,
                "mask_meta": f"{mask_prefix.name}.meta.json" if mode == "mask" else None,
                "restraint_radius": cavity_defaults.restraint_radius,
                "membership_padding": cavity_defaults.membership_padding,
                "initial_delete_padding": cavity_defaults.initial_delete_padding,
            },
        ),
        (
            "cavity_build",
            {
                "enabled": mode == "mask",
                "mode": "seeded",
                "output_prefix": mask_prefix.as_posix(),
                "exclude_residues": [residue.token for residue in selected_residues],
                "dx": advanced_values["dx"],
                "probe_radius": advanced_values["probe_radius"],
                "search_radius": advanced_values["search_radius"],
                "nearby_cutoff": advanced_values["nearby_cutoff"],
                "min_points": advanced_values["min_points"],
            },
        ),
        (
            "loop",
            {
                "sweep_values": [],
                "replica_dirs": ["run"],
                "replica_count": 1,
                "sweep_dir_format": "{value}",
                "replica_dir_format": "{replica:02d}",
            },
        ),
        (
            "slurm",
            {
                "enabled": True,
                "shebang": slurm_defaults.shebang,
                "time_limit": slurm_defaults.time_limit,
                "partition": slurm_defaults.partition,
                "account": "YOUR_ACCOUNT",
                "job_name": slurm_defaults.job_name,
                "output": slurm_defaults.output,
                "extra_directives": [],
            },
        ),
    ]
    lines: list[str] = []
    for section_name, values in section_map:
        lines.append(f"[{section_name}]")
        for key, value in values.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _execution_defaults(gmx_cmd: str) -> ExecutionConfig:
    if gmx_cmd == "gmx":
        return ExecutionConfig(
            shell_executable="/bin/bash",
            module_setup=[],
            gmx_cmd="gmx",
            launcher_single="",
            launcher_multi="",
            mdrun_multi_args=["-ntmpi", "1", "-ntomp", "{cores}"],
        )
    return ExecutionConfig(
        shell_executable="/bin/bash",
        module_setup=[],
        gmx_cmd="gmx_mpi",
        launcher_single="mpirun -np 1",
        launcher_multi="mpirun -np {cores}",
        mdrun_multi_args=[],
    )


def _relative_path(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")
