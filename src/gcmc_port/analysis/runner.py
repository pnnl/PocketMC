from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
from time import strftime
from typing import Any

from ..pathing import portable_path
from .aggregate import write_aggregate
from .cache import load_analysis_cache, write_analysis_cache
from .config import validate_analysis_config
from .cavity_setup import prepare_analysis_cavities
from .density import build_density
from .events import build_paths, build_visits
from .mc_reader import read_mc_dataset
from .md_reader import read_md_dataset
from .masking import mask_dependency_paths
from .models import ANALYSIS_CACHE_VERSION, AnalysisConfig, DatasetSpec, RunResult, config_for_dataset
from .models import POSE_TASKS
from .pose import PoseStageResult, run_pose_stage
from . import plot_template
from .plotting import render_aggregate_pose_plots, render_result_plots
from .tables import write_run_tables
from .vmd import write_vmd_session


CACHE_VERSION = ANALYSIS_CACHE_VERSION
_PLOT_LOCK = threading.Lock()


def _progress(message: str) -> None:
    print(f"[{strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def _jsonable(value: Any, *, base_directory: Path | None = None) -> Any:
    if isinstance(value, Path):
        return portable_path(value, base_directory) if base_directory is not None else str(value)
    if isinstance(value, tuple):
        return [_jsonable(item, base_directory=base_directory) for item in value]
    if isinstance(value, list):
        return [_jsonable(item, base_directory=base_directory) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item, base_directory=base_directory) for key, item in value.items()}
    return value


def _resolved_settings(config: AnalysisConfig, *, base_directory: Path | None = None) -> dict[str, Any]:
    return _jsonable(
        {
            "molecule": asdict(config.molecule),
            "cavity": asdict(config.cavity),
            "analysis": asdict(config.analysis),
            "output": asdict(config.output),
            "substrate": asdict(config.substrate),
            "pose": asdict(config.pose),
            "gmx_cmd": config.gmx_cmd,
        },
        base_directory=base_directory,
    )


def _write_failure_manifest(config: AnalysisConfig, dataset: DatasetSpec, exc: Exception) -> None:
    run_dir = config.output.root / dataset.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "analysis_cache_version": CACHE_VERSION,
        "status": "failed",
        "run_id": dataset.run_id,
        "kind": dataset.kind,
        "tasks": list(config.analysis.tasks),
        "fingerprint": _fingerprint(config, dataset),
        "settings": _resolved_settings(config, base_directory=run_dir),
        "warnings": [],
        "error": f"{type(exc).__name__}: {exc}",
        "outputs": [],
    }
    (run_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _fingerprint(config: AnalysisConfig, dataset: DatasetSpec) -> str:
    digest = hashlib.sha256()
    digest.update(str(CACHE_VERSION).encode())
    digest.update(repr((config.kind, config.molecule, config.cavity, config.analysis, config.substrate, config.pose, dataset)).encode())
    paths = [
        dataset.topology,
        dataset.trajectory,
        dataset.reference,
        dataset.mc_log,
        dataset.trajectory_meta,
        config.cavity.mask,
        config.cavity.meta,
        config.cavity.mask_trajectory,
        config.analysis.canonical_source,
        config.analysis.homolog_source,
        config.pose.reference,
    ]
    paths.extend(
        mask_dependency_paths(
            mask_path=config.cavity.mask,
            meta_path=config.cavity.meta,
            build_source=config.cavity.build_source,
            run_dir=dataset.run_dir,
            config_dir=config.config_path.parent,
            membership_padding=config.cavity.membership_padding_nm,
        )
    )
    for path in paths:
        if path is None:
            continue
        digest.update(str(path).encode())
        if path.exists():
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _copy_plot_script(run_dir: Path, *, reset: bool) -> Path:
    target = run_dir / "plot_results.py"
    with _PLOT_LOCK:
        if reset or not target.exists():
            source = Path(plot_template.__file__).read_text(encoding="utf-8")
            target.write_text(source, encoding="utf-8")
    return target


def _run_plot_script(path: Path, run_dir: Path) -> None:
    """Render through trusted package code; ``path`` remains a manual-edit artifact only."""
    del path
    with _PLOT_LOCK:
        style = dict(plot_template.STYLE)
        if (run_dir / "tables").exists():
            render_result_plots(run_dir, style)
            return
        for case_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and (path / "tables").exists()):
            render_result_plots(case_dir, style)
        render_aggregate_pose_plots(run_dir, style)


def _load_or_analyze(config: AnalysisConfig, dataset: DatasetSpec, run_dir: Path, force: bool) -> RunResult:
    fingerprint = _fingerprint(config, dataset)
    if config.output.cache and not force:
        cached = load_analysis_cache(
            run_dir,
            dataset,
            analysis_version=CACHE_VERSION,
            fingerprint=fingerprint,
        )
        if cached is not None:
            _progress(f"[cache] {dataset.run_id}: reusing completed trajectory analysis.")
            return cached
    result = read_md_dataset(config, dataset) if dataset.kind == "md" else read_mc_dataset(config, dataset)
    if dataset.kind == "md":
        result.visits = build_visits(result, config.analysis.gap_ps)
        if "paths" in config.analysis.tasks:
            result.path_samples = build_paths(result, config.analysis.path_sample_ps, config.analysis.contact_cutoff_a)
    final_fingerprint = _fingerprint(config, dataset)
    if final_fingerprint != fingerprint:
        raise RuntimeError(
            f"{dataset.run_id}: an analysis input changed while its trajectory was being read; "
            "the incomplete result was not cached. Rerun this case after inputs stop changing."
        )
    result.metadata["analysis_fingerprint"] = fingerprint
    if config.output.cache:
        write_analysis_cache(
            run_dir,
            result,
            analysis_version=CACHE_VERSION,
            fingerprint=fingerprint,
        )
    return result


def run_dataset(
    config: AnalysisConfig,
    dataset: DatasetSpec,
    *,
    force: bool = False,
    reset_plot_style: bool = False,
    pose_stage: PoseStageResult | None = None,
) -> RunResult:
    config = config_for_dataset(config, dataset)
    run_dir = config.output.root / dataset.run_id
    if run_dir.exists() and not config.output.overwrite and any(run_dir.iterdir()):
        raise FileExistsError(f"Output exists and output.overwrite=false: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    result = _load_or_analyze(config, dataset, run_dir, force)
    if pose_stage is not None:
        result.outputs.extend(pose_stage.outputs_by_run.get(dataset.run_id, []))
        result.warnings.extend(pose_stage.warnings_by_run.get(dataset.run_id, []))
        if pose_stage.metadata_by_run.get(dataset.run_id):
            result.metadata["pose"] = pose_stage.metadata_by_run[dataset.run_id]
    result.outputs.extend(write_run_tables(result, run_dir / "tables"))
    if "density" in config.analysis.tasks:
        result.outputs.extend(build_density(config, result, run_dir / "density"))
    if "vmd" in config.analysis.tasks:
        result.outputs.extend(write_vmd_session(config, result, run_dir))
    plot_script = _copy_plot_script(config.output.root, reset=reset_plot_style)
    result.outputs.append(plot_script)
    if "plots" in config.analysis.tasks:
        _run_plot_script(plot_script, run_dir)
        result.outputs.extend(sorted((run_dir / "plots").glob("*.png")))
        result.outputs.extend(sorted((run_dir / "poses").glob("**/*.png")))
    manifest = {
        "schema_version": 1,
        "analysis_cache_version": CACHE_VERSION,
        "status": "complete",
        "run_id": dataset.run_id,
        "kind": dataset.kind,
        "system_id": dataset.system_id or dataset.run_id,
        "comparison_group": dataset.comparison_group,
        "pocketmc_detection": {
            "status": dataset.pocketmc_status,
            "evidence": list(dataset.pocketmc_evidence),
            "pocketmc_derived_md": dataset.kind == "md" and dataset.pocketmc_status in {"confirmed", "probable"},
        },
        "tasks": list(config.analysis.tasks),
        "settings": _resolved_settings(config, base_directory=run_dir),
        "inputs": {
            "topology": None if dataset.topology is None else portable_path(dataset.topology, run_dir),
            "trajectory": portable_path(dataset.trajectory, run_dir),
            "reference": None if dataset.reference is None else portable_path(dataset.reference, run_dir),
            "mc_log": None if dataset.mc_log is None else portable_path(dataset.mc_log, run_dir),
            "trajectory_meta": None if dataset.trajectory_meta is None else portable_path(dataset.trajectory_meta, run_dir),
        },
        "fingerprint": str(result.metadata.get("analysis_fingerprint") or _fingerprint(config, dataset)),
        "warnings": result.warnings,
        "outputs": [portable_path(path, run_dir) for path in result.outputs],
        "scientific_scope": (
            "PocketMC accepted states are a heuristic search sequence, not equilibrium or physical time."
            if dataset.kind == "pocketmc"
            else "MD visit durations are sampled lower bounds with explicit boundary censoring."
        ),
    }
    manifest_path = run_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result.outputs.append(manifest_path)
    return result


def run_analysis(
    config: AnalysisConfig,
    *,
    force: bool = False,
    fail_fast: bool = False,
    jobs: int = 1,
    reset_plot_style: bool = False,
) -> tuple[list[RunResult], list[dict[str, str]]]:
    _progress("[step 1/5] Preparing configured cavity inputs.")
    prepare_analysis_cavities(config)
    _progress("[step 2/5] Validating trajectories, topologies, cavity files, and task dependencies.")
    validate_analysis_config(config, check_files=True)
    if not config.output.overwrite:
        conflicts = [
            config.output.root / dataset.run_id
            for dataset in config.datasets
            if (config.output.root / dataset.run_id).exists() and any((config.output.root / dataset.run_id).iterdir())
        ]
        if conflicts:
            raise FileExistsError(
                "Output exists and output.overwrite=false: " + ", ".join(str(path) for path in conflicts)
            )
    config.output.root.mkdir(parents=True, exist_ok=True)
    root_plot_script = _copy_plot_script(config.output.root, reset=reset_plot_style)
    results: list[RunResult] = []
    failures: list[dict[str, str]] = []
    pose_stage: PoseStageResult | None = None
    pose_aggregate_outputs: list[Path] = []
    if set(config.analysis.tasks) & POSE_TASKS:
        _progress(
            "[step 3/5] Building pooled pose features/clusters and pose-conditioned hydration. "
            "This can take minutes to hours for long trajectories."
        )
        pose_stage = run_pose_stage(config, "all", force=force)
        failures.extend(pose_stage.failures)
        pose_aggregate_outputs.extend(pose_stage.aggregate_outputs)
        if fail_fast and pose_stage.failures:
            raise RuntimeError(pose_stage.failures[0]["error"])

    case_positions = {dataset.run_id: index for index, dataset in enumerate(config.datasets, start=1)}

    def execute(dataset: DatasetSpec) -> RunResult:
        _progress(
            f"[step 4/5] Case {case_positions[dataset.run_id]}/{len(config.datasets)} "
            f"{dataset.run_id}: {dataset.kind} trajectory analysis started."
        )
        _progress(f"[case tasks] {dataset.run_id}: {', '.join(config.analysis.tasks)}")
        _progress("[case timing] Long trajectories, density, and plotting can take minutes to hours.")
        return run_dataset(
            config,
            dataset,
            force=force,
            reset_plot_style=reset_plot_style,
            pose_stage=pose_stage,
        )

    if jobs > 1 and len(config.datasets) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(execute, dataset): dataset for dataset in config.datasets}
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    _progress(f"[case complete] {dataset.run_id}: wrote {len(result.outputs)} output artifact(s).")
                except Exception as exc:
                    if config.output.overwrite or not isinstance(exc, FileExistsError):
                        _write_failure_manifest(config_for_dataset(config, dataset), dataset, exc)
                    failures.append({"run_id": dataset.run_id, "error": str(exc)})
                    if fail_fast:
                        raise
    else:
        for dataset in config.datasets:
            try:
                result = execute(dataset)
                results.append(result)
                _progress(f"[case complete] {dataset.run_id}: wrote {len(result.outputs)} output artifact(s).")
            except Exception as exc:
                if config.output.overwrite or not isinstance(exc, FileExistsError):
                    _write_failure_manifest(config_for_dataset(config, dataset), dataset, exc)
                failures.append({"run_id": dataset.run_id, "error": str(exc)})
                if fail_fast or len(config.datasets) == 1:
                    raise
    results.sort(key=lambda item: item.dataset.run_id)
    _progress("[step 5/5] Writing aggregate tables, plots, and the run manifest.")
    try:
        aggregate_outputs = write_aggregate(results, config.output.root)
    except Exception as exc:
        aggregate_outputs = []
        failures.append({"run_id": "aggregate", "phase": "tables", "error": f"{type(exc).__name__}: {exc}"})
    aggregate_outputs.extend(pose_aggregate_outputs)
    if "plots" in config.analysis.tasks:
        try:
            _run_plot_script(root_plot_script, config.output.root)
            aggregate_outputs.extend(sorted((config.output.root / "aggregate" / "pose-groups").glob("**/*.png")))
        except Exception as exc:
            failures.append({"run_id": "aggregate", "phase": "plots", "error": f"{type(exc).__name__}: {exc}"})
    root_manifest = {
        "schema_version": 1,
        "analysis_cache_version": CACHE_VERSION,
        "status": "partial" if failures else "complete",
        "config": portable_path(config.config_path, config.output.root),
        "kind": config.kind,
        "tasks": list(config.analysis.tasks),
        "settings": _resolved_settings(config, base_directory=config.output.root),
        "completed_runs": [result.dataset.run_id for result in results],
        "failures": failures,
        "aggregate_outputs": [portable_path(path, config.output.root) for path in aggregate_outputs],
        "plot_script": portable_path(root_plot_script, config.output.root),
    }
    (config.output.root / "analysis_manifest.json").write_text(json.dumps(root_manifest, indent=2) + "\n", encoding="utf-8")
    _progress(f"[complete] Analysis finished: {len(results)} case(s), {len(failures)} failure(s).")
    return results, failures
