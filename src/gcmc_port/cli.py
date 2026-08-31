from __future__ import annotations

import argparse
from pathlib import Path

from .assets import export_default_assets, export_example_case
from .cavity import build_cavity_from_structure
from .config import load_config
from .helpers import (
    alcove_remove_initial_waters,
    build_cavity_mask,
    downsize_gro_legacy,
    downsize_top_legacy,
    get_alcove_center,
    get_alcove_residues,
    remove_waters_near_centroid,
)
from .moves import write_position_restraints, write_position_restraints_gas, write_trajectory
from .slurm import render_sbatch, submit_sbatch
from .wizard import build_wizard_case
from .workflow import GCMCWorkflow


class RichHelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = action.help or ""
        if "%(default)" in help_text:
            return help_text
        if action.default in (argparse.SUPPRESS, None):
            return help_text
        if action.required:
            return help_text
        return f"{help_text} (default: %(default)s)"


class RichArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("formatter_class", RichHelpFormatter)
        super().__init__(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = RichArgumentParser(description="Standalone Python port of the shell-based GCMC workflow")
    parser.add_argument("--interactive", action="store_true", help="Launch the interactive setup wizard and write a TOML config")
    parser.add_argument("--write-config", help="When using --interactive, save the generated TOML to this path")
    subparsers = parser.add_subparsers(dest="command", parser_class=RichArgumentParser)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the Python GCMC workflow",
        description=(
            "Run the MC workflow described by the TOML config.\n"
            "Plain mask configs still require a prebuilt cavity, but wizard-generated mask configs can auto-build it."
        ),
        epilog=(
            "For voxel-mask runs:\n"
            "  1. Build the cavity first with 'python build_cavity.py ...' or 'python gcmc.py build-cavity ...'\n"
            "  2. Inspect the generated *_points.pdb / *.meta.json outputs\n"
            "  3. Set [cavity].mode = \"mask\" and point mask_file / mask_meta to those outputs,\n"
            "     or use the interactive wizard to generate a mask config that auto-builds them.\n"
            "  4. Then run 'python gcmc.py run -c config.toml'\n"
        ),
    )
    run_parser.add_argument("-c", "--config", required=True, help="Path to TOML config")

    emit_parser = subparsers.add_parser("emit-sbatch", help="Render an sbatch script from config")
    emit_parser.add_argument("-c", "--config", required=True, help="Path to TOML config")
    emit_parser.add_argument("-o", "--output", default="run_gcmc.sbatch", help="Output sbatch script path")

    defaults_parser = subparsers.add_parser("init-defaults", help="Copy bundled default inputs into a target directory")
    defaults_parser.add_argument("-o", "--output", default=".", help="Target directory for bundled default inputs")
    defaults_parser.add_argument("--force", action="store_true", help="Overwrite existing files in the target directory")

    example_parser = subparsers.add_parser("init-example", help="Copy a runnable bundled example into a target directory")
    example_parser.add_argument("-o", "--output", default="example-case", help="Target directory for the bundled example")
    example_parser.add_argument("--force", action="store_true", help="Overwrite existing files in the target directory")

    cavity_parser = subparsers.add_parser(
        "build-cavity",
        help="Build a voxel cavity mask from a structure",
        description=(
            "Build a fixed apo-style voxel cavity mask from a GRO structure.\n"
            "Use seeded mode when you already know the site; use auto mode to rank candidate pockets.\n"
            "This is a preprocessing step; it does not launch the MC workflow."
        ),
        epilog=(
            "Examples:\n"
            "  python gcmc.py build-cavity -f init.gro -m seeded -S 800ATC -E 800ATC\n"
            "  python gcmc.py build-cavity -f init.gro -m auto -E 800ATC -C 3\n"
            "\n"
            "Typical workflow:\n"
            "  1. Build the cavity and inspect *_points.pdb / *.meta.json\n"
            "  2. Copy mask_file / mask_meta into config.toml under [cavity]\n"
            "  3. Run 'python gcmc.py run -c config.toml'\n"
        ),
    )
    cavity_parser.add_argument("-f", "--gro", required=True, help="Input GRO structure")
    cavity_parser.add_argument("-o", "--output-prefix", default="cavity", help="Prefix used for cavity output files")
    cavity_parser.add_argument(
        "-m",
        "--mode",
        choices=("seeded", "auto"),
        default="seeded",
        help=(
            "Pocket search mode. 'seeded' is deterministic around a known site; "
            "'auto' searches for enclosed pockets and ranks candidates."
        ),
    )
    cavity_parser.add_argument(
        "-S",
        "--seed-residue",
        help="Seed residue near the target cavity, e.g. 800ATC. Best when the site is already known.",
    )
    cavity_parser.add_argument(
        "-p",
        "--seed-point",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Explicit seed point in nm. Use this when you want full control over the cavity center.",
    )
    cavity_parser.add_argument(
        "-a",
        "--seed-atoms",
        nargs="*",
        default=["C2", "C4", "C7"],
        help=(
            "Atom names used when computing the center of --seed-residue. "
            "Add or remove atoms to bias the seed toward one end of a ligand or cofactor."
        ),
    )
    cavity_parser.add_argument(
        "-E",
        "--exclude-residue",
        action="append",
        default=[],
        help=(
            "Residue to exclude during discovery, e.g. 800ATC. "
            "Excluded residues are also omitted from the final cavity envelope, which is useful when a bound ligand blocks the site; "
            "auto mode also uses excluded residues as a local guide. Repeat as needed."
        ),
    )
    cavity_parser.add_argument(
        "-x",
        "--dx",
        type=float,
        default=0.075,
        help=(
            "Voxel spacing in nm. Lower values resolve narrower voids but increase runtime and memory; "
            "higher values run faster but can miss thin channels or create blockier masks."
        ),
    )
    cavity_parser.add_argument(
        "-r",
        "--probe-radius",
        type=float,
        default=0.10,
        help=(
            "Minimum clearance required for a voxel center in nm. Lower values admit tighter crevices; "
            "higher values make the cavity more conservative and can leave holes through narrow necks."
        ),
    )
    cavity_parser.add_argument(
        "-R",
        "--search-radius",
        type=float,
        default=0.9,
        help=(
            "Radius of the local search sphere in nm. Higher values can recover larger connected pockets but cost more "
            "and may merge nearby voids; lower values focus locally and may clip extended cavities."
        ),
    )
    cavity_parser.add_argument(
        "-n",
        "--nearby-cutoff",
        type=float,
        default=0.45,
        help=(
            "Residue-reporting cutoff in nm. Higher values list more surrounding residues; "
            "lower values keep the report tighter to the cavity wall."
        ),
    )
    cavity_parser.add_argument(
        "-k",
        "--min-peak-clearance",
        type=float,
        default=0.16,
        help=(
            "Auto mode only: minimum peak clearance in nm used to keep a seed candidate. "
            "Higher values suppress shallow pockets and bulk noise; lower values explore weaker or tighter sites."
        ),
    )
    cavity_parser.add_argument(
        "-C",
        "--candidate-limit",
        type=int,
        default=5,
        help=(
            "Auto mode only: maximum number of candidate pockets to write. "
            "Higher values examine more alternatives and generate more files; lower values are faster but may omit backups."
        ),
    )
    cavity_parser.add_argument(
        "-N",
        "--min-points",
        type=int,
        default=20,
        help=(
            "Reject pockets smaller than this many voxels. Higher values remove tiny fragments and speckles; "
            "lower values keep small pockets but may admit noisy slivers."
        ),
    )

    submit_parser = subparsers.add_parser("submit", help="Render and submit an sbatch script")
    submit_parser.add_argument("-c", "--config", required=True, help="Path to TOML config")
    submit_parser.add_argument("-o", "--output", default="run_gcmc.sbatch", help="Output sbatch script path")

    helper_parser = subparsers.add_parser("helper", help="Run a helper tool equivalent")
    helper_sub = helper_parser.add_subparsers(dest="helper_command", required=True)

    center = helper_sub.add_parser("get-alcove-center")
    center.add_argument("gro")
    center.add_argument("resid", type=int)
    center.add_argument("resname")
    center.add_argument("center_atoms", nargs="*", default=["C2", "C4", "C7"])

    residues = helper_sub.add_parser("get-alcove-residues")
    residues.add_argument("gro")
    residues.add_argument("rmax", type=float)
    residues.add_argument("x", type=float)
    residues.add_argument("y", type=float)
    residues.add_argument("z", type=float)
    residues.add_argument("rfree", type=float)
    residues.add_argument("--out-dir", default=".")

    mask = helper_sub.add_parser("build-cavity-mask")
    mask.add_argument("xgro")
    mask.add_argument("x", type=float)
    mask.add_argument("y", type=float)
    mask.add_argument("z", type=float)
    mask.add_argument("rseed", type=float)
    mask.add_argument("dx", type=float)
    mask.add_argument("rexcl", type=float)
    mask.add_argument("outprefix")

    downsize_gro = helper_sub.add_parser("downsize-gro")
    downsize_gro.add_argument("gro")
    downsize_gro.add_argument("x", type=float)
    downsize_gro.add_argument("y", type=float)
    downsize_gro.add_argument("z", type=float)
    downsize_gro.add_argument("acs", type=int)
    downsize_gro.add_argument("output")

    downsize_top = helper_sub.add_parser("downsize-top")
    downsize_top.add_argument("top")
    downsize_top.add_argument("gro")
    downsize_top.add_argument("acs", type=int)
    downsize_top.add_argument("output")

    remove = helper_sub.add_parser("remove-water-near-centroid")
    remove.add_argument("input")
    remove.add_argument("output")
    remove.add_argument("x", type=float)
    remove.add_argument("y", type=float)
    remove.add_argument("z", type=float)
    remove.add_argument("radius", type=float)

    initial = helper_sub.add_parser("alcove-remove-initial-wat")
    initial.add_argument("input")
    initial.add_argument("output")
    initial.add_argument("radius", type=float)
    initial.add_argument("x", type=float)
    initial.add_argument("y", type=float)
    initial.add_argument("z", type=float)

    posre = helper_sub.add_parser("position-restraints")
    posre.add_argument("gro")
    posre.add_argument("kres", type=float)
    posre.add_argument("--out-dir", default=".")

    posre_gas = helper_sub.add_parser("position-restraints-gas")
    posre_gas.add_argument("gro")
    posre_gas.add_argument("kres", type=float)
    posre_gas.add_argument("rvdw", type=float)
    posre_gas.add_argument("rfree", type=float)
    posre_gas.add_argument("--out-dir", default=".")

    traj = helper_sub.add_parser("write-trajectory")
    traj.add_argument("gro")
    traj.add_argument("energy", type=float)
    traj.add_argument("naccepted", type=int)
    traj.add_argument("nmol", type=int)
    traj.add_argument("target_nmol", type=int)
    traj.add_argument("--trajectory", default="trajectory.gro")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_config and not args.interactive:
        parser.error("--write-config can only be used together with --interactive.")

    if args.interactive:
        artifacts = build_wizard_case(args.write_config)
        print(artifacts.config_path)
        return 0

    if not args.command:
        parser.error("a command is required unless --interactive is used")

    if args.command == "run":
        config = load_config(args.config)
        GCMCWorkflow(config).run()
        return 0

    if args.command == "emit-sbatch":
        config = load_config(args.config)
        path = render_sbatch(config, args.output)
        print(path)
        return 0

    if args.command == "init-defaults":
        try:
            path = export_default_assets(args.output, force=args.force)
        except FileExistsError as exc:
            parser.exit(2, f"{exc}\n")
        print(path)
        return 0

    if args.command == "init-example":
        try:
            path = export_example_case(args.output, force=args.force)
        except FileExistsError as exc:
            parser.exit(2, f"{exc}\n")
        print(path)
        return 0

    if args.command == "build-cavity":
        outputs = build_cavity_from_structure(
            args.gro,
            outprefix=args.output_prefix,
            mode=args.mode,
            dx=args.dx,
            probe_radius=args.probe_radius,
            search_radius=args.search_radius,
            seed_point=tuple(args.seed_point) if args.seed_point else None,
            seed_residue=args.seed_residue,
            seed_atoms=args.seed_atoms,
            exclude_residues=args.exclude_residue,
            nearby_cutoff=args.nearby_cutoff,
            min_peak_clearance=args.min_peak_clearance,
            candidate_limit=args.candidate_limit,
            min_points=args.min_points,
        )
        for path in outputs:
            print(path)
        return 0

    if args.command == "submit":
        config = load_config(args.config)
        path = render_sbatch(config, args.output)
        result = submit_sbatch(path)
        print(result.stdout.strip() or result.stderr.strip())
        return result.returncode

    if args.command == "helper":
        if args.helper_command == "get-alcove-center":
            x, y, z = get_alcove_center(args.gro, args.resid, args.resname, args.center_atoms)
            print(f"{x:.8f} {y:.8f} {z:.8f}")
            return 0
        if args.helper_command == "get-alcove-residues":
            get_alcove_residues(args.gro, args.rmax, (args.x, args.y, args.z), args.rfree, out_dir=args.out_dir)
            return 0
        if args.helper_command == "build-cavity-mask":
            build_cavity_mask(args.xgro, (args.x, args.y, args.z), args.rseed, args.dx, args.rexcl, args.outprefix)
            return 0
        if args.helper_command == "downsize-gro":
            downsize_gro_legacy(args.gro, (args.x, args.y, args.z), args.acs, args.output)
            return 0
        if args.helper_command == "downsize-top":
            downsize_top_legacy(args.top, args.gro, args.acs, args.output)
            return 0
        if args.helper_command == "remove-water-near-centroid":
            remove_waters_near_centroid(args.input, args.output, (args.x, args.y, args.z), args.radius)
            return 0
        if args.helper_command == "alcove-remove-initial-wat":
            alcove_remove_initial_waters(args.input, args.output, args.radius, (args.x, args.y, args.z))
            return 0
        if args.helper_command == "position-restraints":
            write_position_restraints(args.gro, args.kres, out_dir=args.out_dir)
            return 0
        if args.helper_command == "position-restraints-gas":
            write_position_restraints_gas(args.gro, args.kres, rvdw=args.rvdw, rfree=args.rfree, out_dir=args.out_dir)
            return 0
        if args.helper_command == "write-trajectory":
            write_trajectory(args.gro, args.energy, args.naccepted, args.nmol, args.target_nmol, trajectory_path=args.trajectory)
            return 0

    parser.error("Unsupported command")
    return 2
