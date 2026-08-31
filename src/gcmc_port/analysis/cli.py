from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

from .cavity_setup import prepare_analysis_cavities
from .config import load_analysis_config, validate_analysis_config, with_runtime_overrides
from .discovery import cases_json, discover_cases, format_cases
from .grand_alignment import (
    default_fixed_substrates,
    grand_align_analysis_roots,
    load_completed_analysis_root,
    load_grand_plot_style,
    repair_grand_alignment_output,
    replot_grand_alignment,
)
from .pose import run_pose_stage
from .plot_editor import apply_style_assignments, discover_plot_targets, load_plot_style, render_plot_targets, save_plot_style
from .runner import run_analysis
from .slurm import render_analysis_launchers, render_analysis_sbatch, submit_analysis_sbatch
from .viewer import view_density
from .vmd import launch_vmd, render_vmd
from .wizard import run_wizard


def _csv(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    return [item.strip() for value in values for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pocketmc-analyses",
        description="Physical-MD, PocketMC accepted-state, substrate-pose, and hydration analysis",
    )
    parser.add_argument("--interactive", action="store_true", help="open the configuration wizard")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="run analyses from a TOML configuration")
    run.add_argument("-c", "--config", required=True, type=Path)
    run.add_argument("--tasks", action="append", help="comma-separated task override")
    run.add_argument("--runs", action="append", help="comma-separated run ID or replica filter")
    run.add_argument("--force", action="store_true", help="ignore versioned analysis caches")
    run.add_argument("--fail-fast", action="store_true", help="stop a batch at the first failure")
    run.add_argument("--jobs", type=int, default=1, help="number of independently processed runs")
    run.add_argument("--reset-plot-style", action="store_true", help="replace generated plot_results.py before plotting")

    validate = subparsers.add_parser("validate", help="validate files, selections, and task dependencies")
    validate.add_argument("-c", "--config", required=True, type=Path)
    validate.add_argument("--tasks", action="append", help="comma-separated task override")
    validate.add_argument("--runs", action="append", help="comma-separated run ID or replica filter")

    prepare = subparsers.add_parser(
        "prepare-cavities",
        help="build any deferred GCMC voxel cavities recorded by the analysis wizard",
    )
    prepare.add_argument("-c", "--config", required=True, type=Path)
    prepare.add_argument("--force", action="store_true", help="rebuild cavity bundles even when all outputs exist")

    density = subparsers.add_parser("view-density", help="open an interactive Matplotlib 3D density viewer")
    density.add_argument("npz", type=Path)
    density.add_argument("--cutoff-percent", type=float, default=8.0)
    density.add_argument("--opacity", type=float, default=0.30)
    density.add_argument("--cmap", default="viridis")

    replot = subparsers.add_parser("replot", help="regenerate selected plots from saved tables/NPZ data only")
    replot.add_argument("root", type=Path, help="analysis result root")
    replot.add_argument("--targets", action="append", help="comma-separated target keys; default is all")
    replot.add_argument("--set", dest="settings", action="append", default=[], help="plot setting as KEY=JSON_VALUE")
    replot.add_argument("--list", action="store_true", help="list plot target keys without rendering")

    grand = subparsers.add_parser(
        "grand-align",
        help="align completed density results onto one fixed-substrate coordinate frame",
    )
    grand.add_argument("roots", nargs="+", type=Path, help="two or more completed analysis result roots")
    grand.add_argument("-o", "--output", type=Path, default=Path("grand-aligned"))
    grand.add_argument("--reference", type=Path, help="one selected analysis root to use as the target frame")
    grand.add_argument("--substrate", action="append", help="fixed substrate resname(s), comma-separated; auto-prefers OPP")
    grand.add_argument("--spacing-a", type=float, help="common Cartesian grid spacing in Angstrom")
    grand.add_argument("--elevation", type=float, default=30.0, help="shared 3D camera elevation")
    grand.add_argument("--azimuth", type=float, default=-60.0, help="shared 3D camera azimuth")
    grand.add_argument("--roll", type=float, default=0.0, help="shared 3D camera roll")
    grand.add_argument("--no-plots", action="store_true", help="write aligned NPZ data without rendering PNG plots")

    grand_replot = subparsers.add_parser(
        "grand-replot",
        help="redraw an existing grand alignment from saved NPZ files only",
    )
    grand_replot.add_argument("root", type=Path, help="grand-aligned output root")
    grand_replot.add_argument(
        "--repair",
        action="store_true",
        help="upgrade an old grand output to content-aware axes and add the editable plot script",
    )

    vmd = subparsers.add_parser("launch-vmd", help="explicitly launch a generated VMD Tcl session")
    vmd.add_argument("tcl", type=Path)
    vmd.add_argument("--vmd", default="auto", help="VMD executable path (default: discover vmd)")
    vmd.add_argument("--wait", action="store_true")

    render = subparsers.add_parser("render-vmd", help="headlessly render a generated canonical-view VMD Tcl script")
    render.add_argument("tcl", type=Path)
    render.add_argument("--vmd", default="auto", help="VMD executable path (default: discover vmd)")

    discover = subparsers.add_parser("discover", help="scan subdirectories and report analyzable MD/PocketMC cases")
    discover.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    discover.add_argument("--max-depth", type=int, default=4)
    discover.add_argument("--deep", action="store_true", help="open trajectory candidates to report frame/time metadata")
    discover.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    emit = subparsers.add_parser("emit-sbatch", help="render the staged pose-analysis Slurm pipeline")
    emit.add_argument("-c", "--config", required=True, type=Path)
    emit.add_argument("-o", "--output", type=Path, default=Path("analysis-jobs"))

    launchers = subparsers.add_parser("emit-launchers", help="render direct-shell, generic Slurm, and Tahoma-only Slurm launchers")
    launchers.add_argument("-c", "--config", required=True, type=Path)
    launchers.add_argument("-o", "--output", type=Path, default=Path("."))

    submit = subparsers.add_parser("submit-sbatch", help="render and explicitly submit the staged Slurm pipeline")
    submit.add_argument("-c", "--config", required=True, type=Path)
    submit.add_argument("-o", "--output", type=Path, default=Path("analysis-jobs"))

    stage = subparsers.add_parser("_pose-stage", help="internal stage command used by generated Slurm scripts")
    stage.add_argument("-c", "--config", required=True, type=Path)
    stage.add_argument("--phase", required=True, choices=("features", "cluster", "hydrate", "finalize"))
    stage.add_argument("--case-index", type=int)
    stage.add_argument("--force", action="store_true")
    return parser


def _print_plan(config: object) -> None:
    # Kept separate so validate output remains stable and easy to capture in tests.
    cfg = config
    print(f"Input kind: {cfg.kind}")
    print("Cases:")
    for dataset in cfg.datasets:
        derived = dataset.kind == "md" and dataset.pocketmc_status in {"confirmed", "probable"}
        cavity = dataset.cavity or cfg.cavity
        cavity_input = cavity.mask if cavity.mode == "mask" else f"r={cavity.radius_nm:g} nm"
        print(
            f"  {dataset.run_id}: kind={dataset.kind}, system={dataset.system_id or dataset.run_id}, "
            f"PocketMC={dataset.pocketmc_status}, PocketMC-derived-MD={'yes' if derived else 'no'}, "
            f"cavity={cavity.mode} ({cavity_input})"
        )
    print("Expanded tasks:", ", ".join(cfg.analysis.tasks))
    print("Output root:", cfg.output.root)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.interactive or args.command is None:
            wizard_result = run_wizard(discover_first=True)
            if wizard_result is None:
                return 0
            _path, config = wizard_result
            validate_analysis_config(config, check_files=True)
            _print_plan(config)
            print("Setup complete. No analysis was launched; run one of the generated .sh/.sbatch files when ready.")
            return 0
        if args.command in {"run", "validate"}:
            config = load_analysis_config(args.config)
            config = with_runtime_overrides(config, tasks=_csv(args.tasks), runs=_csv(args.runs))
            _print_plan(config)
            if args.command == "validate":
                validate_analysis_config(config, check_files=True)
                print("Validation successful.")
                return 0
            if args.jobs < 1:
                raise ValueError("--jobs must be at least 1")
            results, failures = run_analysis(
                config,
                force=args.force,
                fail_fast=args.fail_fast,
                jobs=args.jobs,
                reset_plot_style=args.reset_plot_style,
            )
            print(f"Completed {len(results)} run(s); failures: {len(failures)}")
            for failure in failures:
                phase = f" [{failure['phase']}]" if failure.get("phase") else ""
                print(f"FAILED {failure['run_id']}{phase}: {failure['error']}", file=sys.stderr)
            return 2 if failures else 0
        if args.command == "prepare-cavities":
            config = load_analysis_config(args.config)
            outputs = prepare_analysis_cavities(config, force=args.force)
            for path in outputs:
                print(path)
            return 0
        if args.command == "view-density":
            view_density(args.npz, cutoff_percent=args.cutoff_percent, opacity=args.opacity, cmap=args.cmap)
            return 0
        if args.command == "replot":
            targets = discover_plot_targets(args.root)
            if args.list:
                for target in targets:
                    print(f"{target.key}\t{target.label}\t{target.description}")
                return 0
            requested = set(_csv(args.targets) or [target.key for target in targets])
            unknown = sorted(requested - {target.key for target in targets})
            if unknown:
                raise ValueError("Unknown plot target(s): " + ", ".join(unknown))
            chosen = [target for target in targets if target.key in requested]
            if not chosen:
                raise ValueError(f"No plot targets were found under {args.root}")
            style = apply_style_assignments(load_plot_style(args.root), args.settings)
            style_path = save_plot_style(args.root, style)
            print(f"Saved plot settings: {style_path}")
            outputs = render_plot_targets(args.root, style, chosen)
            print(f"Regenerated {len(outputs)} plot(s) from saved analysis data.")
            return 0
        if args.command == "grand-align":
            analyses = [load_completed_analysis_root(path) for path in args.roots]
            substrates = _csv(args.substrate) or list(default_fixed_substrates(analyses))
            result = grand_align_analysis_roots(
                analyses,
                args.output,
                fixed_substrates=substrates,
                reference_root=args.reference,
                spacing_a=args.spacing_a,
                elevation=args.elevation,
                azimuth=args.azimuth,
                roll=args.roll,
                render_plots=not args.no_plots,
            )
            print(f"Grand-aligned {len(result.aligned_maps)} saved density map(s).")
            print(f"Output: {result.output_root}")
            print(f"Manifest: {result.manifest}")
            if result.skipped_maps:
                print(f"Skipped incompatible map(s): {len(result.skipped_maps)}")
            for warning in result.warnings:
                print(f"WARNING: {warning}", file=sys.stderr)
            return 0
        if args.command == "grand-replot":
            outputs = (
                repair_grand_alignment_output(args.root)
                if args.repair
                else replot_grand_alignment(args.root, load_grand_plot_style(args.root))
            )
            print(f"Regenerated {len(outputs)} grand-aligned plot(s) from saved NPZ data.")
            print(f"Editable script: {args.root.expanduser().resolve() / 'plot_grand_aligned.py'}")
            return 0
        if args.command == "launch-vmd":
            return launch_vmd(args.tcl, executable=args.vmd, wait=args.wait)
        if args.command == "render-vmd":
            return render_vmd(args.tcl, executable=args.vmd)
        if args.command == "discover":
            cases = discover_cases(args.root, max_depth=args.max_depth, deep=args.deep)
            print(cases_json(cases) if args.json else format_cases(cases))
            return 0 if cases else 1
        if args.command in {"emit-sbatch", "submit-sbatch"}:
            config = load_analysis_config(args.config)
            validate_analysis_config(config, check_files=True)
            paths = render_analysis_sbatch(config, args.output)
            print("Rendered Slurm pipeline:")
            for path in paths:
                print(" ", path)
            if args.command == "submit-sbatch":
                result = submit_analysis_sbatch(args.output)
                if result.stdout:
                    print(result.stdout.rstrip())
                if result.stderr:
                    print(result.stderr.rstrip(), file=sys.stderr)
                return result.returncode
            return 0
        if args.command == "emit-launchers":
            config = load_analysis_config(args.config)
            validate_analysis_config(config, check_files=True)
            paths = render_analysis_launchers(config, args.output)
            for label, path in paths.items():
                print(f"{label}: {path}")
            return 0
        if args.command == "_pose-stage":
            config = load_analysis_config(args.config)
            prepare_analysis_cavities(config)
            validate_analysis_config(config, check_files=True)
            result = run_pose_stage(config, args.phase, case_index=args.case_index, force=args.force)
            for failure in result.failures:
                print(f"FAILED {failure['run_id']} [{failure['phase']}]: {failure['error']}", file=sys.stderr)
            return 2 if result.failures else 0
        raise RuntimeError(f"Unhandled command: {args.command}")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if os.environ.get("POCKETMC_ANALYSIS_TRACEBACK"):
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
