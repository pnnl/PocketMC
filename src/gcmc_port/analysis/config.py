from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import tomllib
from typing import Any

from gcmc_port.config import load_config as load_gcmc_config
from gcmc_port.gro import parse_gro

from .anchors import select_mda_anchor
from .models import (
    AnalysisConfig,
    AnalysisOptions,
    CavitySpec,
    DatasetSpec,
    MoleculeSpec,
    OutputOptions,
    POSE_TASKS,
    PoseFeatureSpec,
    PoseOptions,
    SubstrateSpec,
    cavity_for_dataset,
    config_for_dataset,
    expand_tasks,
)


def _path(base: Path, value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _tuple_strings(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def _render_pattern(pattern: str, *, run_dir: Path, replica: str, sweep: str) -> Path:
    text = pattern.format(run_dir=str(run_dir), replica=replica, sweep=sweep)
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (run_dir / path).resolve()


def _run_id(run_dir: Path, replica: str, sweep: str) -> str:
    raw = "-".join(item for item in (sweep, replica) if item) or run_dir.name or "run"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _infer_molecule_from_gcmc(gcmc: Any) -> MoleculeSpec:
    gas = parse_gro(gcmc.paths.gas_gro)
    resname = gas.atoms[0].resname.upper() if gas.atoms else ""
    if resname in {"SOL", "WAT", "HOH"}:
        return MoleculeSpec.from_values(preset="water")
    if resname == "COM":
        return MoleculeSpec.from_values(preset="co")
    return MoleculeSpec.from_values(
        preset="custom",
        label=resname or "Tracked molecule",
        resnames=[resname],
        point_mode="cog",
        atom_names=[atom.atomname for atom in gas.atoms],
    )


def _datasets_from_gcmc(
    *,
    gcmc: Any,
    kind: str,
    input_raw: dict[str, Any],
) -> tuple[DatasetSpec, ...]:
    if kind == "md":
        topology_pattern = str(input_raw.get("topology", "md.tpr"))
        trajectory_pattern = str(input_raw.get("trajectory", "md.xtc"))
        reference_pattern = str(input_raw.get("reference", "previous.gro"))
    else:
        topology_pattern = str(input_raw.get("topology", "previous.gro"))
        trajectory_pattern = str(input_raw.get("trajectory", "trajectory.gro"))
        reference_pattern = str(input_raw.get("reference", "previous.gro"))
    mc_log_pattern = str(input_raw.get("mc_log", "mc.log"))
    meta_pattern = str(input_raw.get("trajectory_meta", "trajectory.meta.jsonl"))
    datasets: list[DatasetSpec] = []
    for spec in gcmc.run_specs():
        sweep = "" if spec.sweep_value is None else str(spec.sweep_value)
        run_dir = spec.run_dir.resolve()
        datasets.append(
            DatasetSpec(
                run_id=_run_id(run_dir, spec.replica_label, sweep),
                kind=kind,  # type: ignore[arg-type]
                run_dir=run_dir,
                topology=_render_pattern(topology_pattern, run_dir=run_dir, replica=spec.replica_label, sweep=sweep),
                trajectory=_render_pattern(trajectory_pattern, run_dir=run_dir, replica=spec.replica_label, sweep=sweep),
                reference=_render_pattern(reference_pattern, run_dir=run_dir, replica=spec.replica_label, sweep=sweep),
                mc_log=_render_pattern(mc_log_pattern, run_dir=run_dir, replica=spec.replica_label, sweep=sweep),
                trajectory_meta=_render_pattern(meta_pattern, run_dir=run_dir, replica=spec.replica_label, sweep=sweep),
                replica=spec.replica_label,
                sweep=sweep,
                system_id=str(input_raw.get("system_id", gcmc.config_path.parent.name or "system")),
                comparison_group=str(input_raw.get("comparison_group", "")),
                pocketmc_status="confirmed",
                pocketmc_evidence=("PocketMC/GCMC config",),
            )
        )
    return tuple(datasets)


def _safe_id(value: Any, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or fallback)).strip(".") or fallback


def _case_cavity(base: Path, raw: dict[str, Any]) -> CavitySpec | None:
    """Load optional per-case cavity fields from a ``[[case]]`` table."""
    keys = {
        "cavity_mode", "cavity_mask", "cavity_meta", "cavity_mask_trajectory",
        "cavity_points", "cavity_nearby_residues", "cavity_anchor", "cavity_anchor_atoms",
        "cavity_radius_nm", "cavity_membership_padding_nm", "cavity_protein_selection",
        "cavity_align_selection", "cavity_build_enabled", "cavity_build_source",
        "cavity_build_output_prefix", "cavity_build_mode", "cavity_build_exclude_residues",
        "cavity_build_dx", "cavity_build_probe_radius", "cavity_build_search_radius",
        "cavity_build_nearby_cutoff", "cavity_build_min_points",
    }
    if not keys.intersection(raw):
        return None

    def path_value(key: str) -> Path | None:
        return _path(base, raw.get(key))

    return CavitySpec(
        mode=str(raw.get("cavity_mode", "sphere")).strip().lower(),
        mask=path_value("cavity_mask"),
        meta=path_value("cavity_meta"),
        mask_trajectory=path_value("cavity_mask_trajectory"),
        anchor=str(raw.get("cavity_anchor", "800ATC")),
        anchor_atoms=_tuple_strings(raw.get("cavity_anchor_atoms"), ("C2", "C4", "C7")),
        radius_nm=float(raw.get("cavity_radius_nm", 0.60)),
        membership_padding_nm=float(raw.get("cavity_membership_padding_nm", 0.02)),
        protein_selection=str(raw.get("cavity_protein_selection", "protein and not name H*")),
        align_selection=(str(raw["cavity_align_selection"]).strip() or None) if "cavity_align_selection" in raw else None,
        points=path_value("cavity_points"),
        nearby_residues=path_value("cavity_nearby_residues"),
        build_enabled=bool(raw.get("cavity_build_enabled", False)),
        build_source=path_value("cavity_build_source"),
        build_output_prefix=path_value("cavity_build_output_prefix"),
        build_mode=str(raw.get("cavity_build_mode", "seeded")).strip().lower(),
        build_exclude_residues=_tuple_strings(raw.get("cavity_build_exclude_residues"), ()),
        build_dx=float(raw.get("cavity_build_dx", 0.075)),
        build_probe_radius=float(raw.get("cavity_build_probe_radius", 0.10)),
        build_search_radius=float(raw.get("cavity_build_search_radius", 0.90)),
        build_nearby_cutoff=float(raw.get("cavity_build_nearby_cutoff", 0.45)),
        build_min_points=int(raw.get("cavity_build_min_points", 20)),
    )


def _dataset_from_case(base: Path, raw: dict[str, Any], index: int) -> DatasetSpec:
    kind = str(raw.get("kind", "md")).strip().lower()
    if kind not in {"md", "pocketmc"}:
        raise ValueError(f"case {index}: kind must be md or pocketmc")
    run_dir = _path(base, raw.get("run_dir", ".")) or base

    def case_path(key: str, default: str | None = None) -> Path | None:
        value = raw.get(key)
        if value in (None, ""):
            return (run_dir / default).resolve() if default else None
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    trajectory = case_path("trajectory")
    if trajectory is None:
        raise ValueError(f"case {index}: trajectory is required")
    fallback = run_dir.name or f"case-{index}"
    run_id = _safe_id(raw.get("id", raw.get("run_id")), fallback)
    evidence = _tuple_strings(raw.get("pocketmc_evidence"), ())
    return DatasetSpec(
        run_id=run_id,
        kind=kind,  # type: ignore[arg-type]
        run_dir=run_dir,
        topology=case_path("topology"),
        trajectory=trajectory,
        reference=case_path("reference"),
        mc_log=case_path("mc_log", "mc.log") if kind == "pocketmc" else case_path("mc_log"),
        trajectory_meta=case_path("trajectory_meta", "trajectory.meta.jsonl") if kind == "pocketmc" else case_path("trajectory_meta"),
        replica=str(raw.get("replica", "")),
        sweep=str(raw.get("sweep", "")),
        system_id=_safe_id(raw.get("system_id"), run_id),
        comparison_group=_safe_id(raw.get("comparison_group"), "") if raw.get("comparison_group") else "",
        pocketmc_status=str(raw.get("pocketmc_status", "not_detected")),
        pocketmc_evidence=evidence,
        substrate_selection=str(raw.get("substrate_selection", "")).strip(),
        pocket_selection=str(raw.get("pocket_selection", "")).strip(),
        cavity=_case_cavity(base, raw),
    )


def load_analysis_config(config_path: str | Path) -> AnalysisConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    base = path.parent
    input_raw = dict(raw.get("input", {}))
    molecule_raw = dict(raw.get("molecule", {}))
    cavity_raw = dict(raw.get("cavity", {}))
    analysis_raw = dict(raw.get("analysis", {}))
    output_raw = dict(raw.get("output", {}))
    substrate_raw = dict(raw.get("substrate", {}))
    pose_raw = dict(raw.get("pose", {}))

    case_rows = raw.get("case", [])
    if case_rows:
        if not isinstance(case_rows, list):
            raise ValueError("[[case]] must be an array of tables")
        datasets = tuple(_dataset_from_case(base, dict(item), index) for index, item in enumerate(case_rows, start=1))
        kinds = {item.kind for item in datasets}
        kind = next(iter(kinds)) if len(kinds) == 1 else "mixed"
        gcmc = None
        inherited_molecule = MoleculeSpec.from_values(preset="water")
    else:
        kind = str(input_raw.get("kind", "md")).strip().lower()
        if kind not in {"md", "pocketmc"}:
            raise ValueError("input.kind must be md or pocketmc")

        gcmc_path = _path(base, input_raw.get("gcmc_config"))
        gcmc = load_gcmc_config(gcmc_path) if gcmc_path is not None else None
        if gcmc is not None:
            datasets = _datasets_from_gcmc(gcmc=gcmc, kind=kind, input_raw=input_raw)
            inherited_molecule = _infer_molecule_from_gcmc(gcmc)
        else:
            run_dir = _path(base, input_raw.get("run_dir", ".")) or base
            topology = _path(base, input_raw.get("topology"))
            trajectory = _path(base, input_raw.get("trajectory"))
            if trajectory is None:
                raise ValueError("input.trajectory is required when input.gcmc_config is not set")
            raw_run_id = str(input_raw.get("run_id", run_dir.name or "run"))
            safe_run_id = _safe_id(raw_run_id, "run")
            datasets = (
                DatasetSpec(
                    run_id=safe_run_id,
                    kind=kind,  # type: ignore[arg-type]
                    run_dir=run_dir,
                    topology=topology,
                    trajectory=trajectory,
                    reference=_path(base, input_raw.get("reference")),
                    mc_log=_path(base, input_raw.get("mc_log")),
                    trajectory_meta=_path(base, input_raw.get("trajectory_meta", "trajectory.meta.jsonl")),
                    system_id=_safe_id(input_raw.get("system_id"), safe_run_id),
                    comparison_group=str(input_raw.get("comparison_group", "")),
                    pocketmc_status="confirmed" if kind == "pocketmc" else "not_detected",
                    pocketmc_evidence=("configured as PocketMC accepted states",) if kind == "pocketmc" else (),
                ),
            )
            inherited_molecule = MoleculeSpec.from_values(preset="water")

    preset = str(molecule_raw.get("preset", inherited_molecule.preset))
    molecule = MoleculeSpec.from_values(
        preset=preset,
        label=molecule_raw.get("label", inherited_molecule.label),
        resnames=molecule_raw.get("resnames", inherited_molecule.resnames),
        point_mode=molecule_raw.get("point_mode", inherited_molecule.point_mode),
        atom_names=molecule_raw.get("atom_names", inherited_molecule.atom_names),
    )

    inherited_anchor = gcmc.anchor.anchor if gcmc is not None else "800ATC"
    inherited_anchor_atoms = tuple(gcmc.anchor.center_atoms) if gcmc is not None else ("C2", "C4", "C7")
    inherited_mode = gcmc.cavity.mode if gcmc is not None else "sphere"
    inherited_mask = gcmc.cavity.mask_file if gcmc is not None else None
    inherited_meta = gcmc.cavity.mask_meta if gcmc is not None else None
    cavity = CavitySpec(
        mode=str(cavity_raw.get("mode", inherited_mode)).strip().lower(),
        mask=_path(base, cavity_raw.get("mask")) or inherited_mask,
        meta=_path(base, cavity_raw.get("meta")) or inherited_meta,
        mask_trajectory=_path(base, cavity_raw.get("mask_trajectory")),
        anchor=str(cavity_raw.get("anchor", inherited_anchor)),
        anchor_atoms=_tuple_strings(cavity_raw.get("anchor_atoms"), inherited_anchor_atoms),
        radius_nm=float(cavity_raw.get("radius_nm", gcmc.simulation.rmax if gcmc is not None else 0.60)),
        membership_padding_nm=float(cavity_raw.get("membership_padding_nm", 0.02)),
        protein_selection=str(cavity_raw.get("protein_selection", "protein and not name H*")),
        align_selection=(str(cavity_raw["align_selection"]).strip() or None) if "align_selection" in cavity_raw else None,
        points=_path(base, cavity_raw.get("points")),
        nearby_residues=_path(base, cavity_raw.get("nearby_residues")),
        build_enabled=bool(cavity_raw.get("build_enabled", False)),
        build_source=_path(base, cavity_raw.get("build_source")),
        build_output_prefix=_path(base, cavity_raw.get("build_output_prefix")),
        build_mode=str(cavity_raw.get("build_mode", "seeded")).strip().lower(),
        build_exclude_residues=_tuple_strings(cavity_raw.get("build_exclude_residues"), ()),
        build_dx=float(cavity_raw.get("build_dx", 0.075)),
        build_probe_radius=float(cavity_raw.get("build_probe_radius", 0.10)),
        build_search_radius=float(cavity_raw.get("build_search_radius", 0.90)),
        build_nearby_cutoff=float(cavity_raw.get("build_nearby_cutoff", 0.45)),
        build_min_points=int(cavity_raw.get("build_min_points", 20)),
    )

    substrate = SubstrateSpec(
        enabled=bool(substrate_raw.get("enabled", bool(substrate_raw.get("selection")))),
        selection=str(substrate_raw.get("selection", "")).strip(),
        fit_selection=str(substrate_raw.get("fit_selection", "")).strip(),
        heavy_selection=str(substrate_raw.get("heavy_selection", "not name H*")).strip() or "not name H*",
    )
    sampling_raw = dict(pose_raw.get("sampling", {}))
    features: list[PoseFeatureSpec] = []
    for index, item in enumerate(pose_raw.get("feature", []), start=1):
        feature = dict(item)
        features.append(
            PoseFeatureSpec(
                name=str(feature.get("name", f"feature_{index}")),
                kind=str(feature.get("kind", "distance")).strip().lower(),
                selections=_tuple_strings(feature.get("selections"), ()),
                point_modes=_tuple_strings(feature.get("point_modes"), ()),
                weight=float(feature.get("weight", 1.0)),
            )
        )
    pose = PoseOptions(
        clusters=int(pose_raw.get("clusters", pose_raw.get("k", 3))),
        max_frames_per_trajectory=int(sampling_raw.get("max_frames_per_trajectory", 5000)),
        sampling_strategy=str(sampling_raw.get("strategy", "uniform")).strip().lower(),
        write_trajectory=bool(sampling_raw.get("write_trajectory", True)),
        seed=int(pose_raw.get("seed", 2026)),
        restarts=int(pose_raw.get("restarts", 20)),
        reference=_path(base, pose_raw.get("reference")),
        reference_pocket_selection=str(pose_raw.get("reference_pocket_selection", "")).strip(),
        pocket_selection=str(pose_raw.get("pocket_selection", "protein and backbone")).strip(),
        site_hpd=float(pose_raw.get("site_hpd", 0.70)),
        site_min_separation_a=float(pose_raw.get("site_min_separation_a", 2.4)),
        site_max_count=int(pose_raw.get("site_max_count", 20)),
        site_integral_radius_a=float(pose_raw.get("site_integral_radius_a", 1.4)),
        features=tuple(features),
    )

    tasks = _tuple_strings(analysis_raw.get("tasks"), ("all",))
    options = AnalysisOptions(
        tasks=expand_tasks(kind, tasks, pose_enabled=substrate.enabled),  # type: ignore[arg-type]
        start_ps=float(analysis_raw["start_ps"]) if analysis_raw.get("start_ps") is not None else None,
        stop_ps=float(analysis_raw["stop_ps"]) if analysis_raw.get("stop_ps") is not None else None,
        stride=int(analysis_raw.get("stride", 1)),
        gap_ps=float(analysis_raw.get("gap_ps", 1000.0)),
        path_sample_ps=float(analysis_raw.get("path_sample_ps", 1000.0)),
        contact_cutoff_a=float(analysis_raw.get("contact_cutoff_a", 4.0)),
        density_bin_a=float(analysis_raw.get("density_bin_a", 1.0)),
        density_sigma_a=float(analysis_raw.get("density_sigma_a", 2.0)),
        density_cutoff_sigma=float(analysis_raw.get("density_cutoff_sigma", 3.0)),
        density_quantity=str(analysis_raw.get("density_quantity", "occupancy")),
        rolling_window=int(analysis_raw.get("rolling_window", 10)),
        canonical_source=_path(base, analysis_raw.get("canonical_source")),
        canonical_chain=str(analysis_raw.get("canonical_chain", "")).strip(),
        homolog_source=_path(base, analysis_raw.get("homolog_source")),
        homolog_chain=str(analysis_raw.get("homolog_chain", "")).strip(),
        verbose=bool(analysis_raw.get("verbose", True)),
    )
    output = OutputOptions(
        root=_path(base, output_raw.get("root", "analysis-results")) or (base / "analysis-results"),
        cache=bool(output_raw.get("cache", True)),
        overwrite=bool(output_raw.get("overwrite", True)),
    )
    config = AnalysisConfig(
        config_path=path,
        kind=kind,  # type: ignore[arg-type]
        datasets=datasets,
        molecule=molecule,
        cavity=cavity,
        analysis=options,
        output=output,
        substrate=substrate,
        pose=pose,
        gmx_cmd=str(cavity_raw.get("gmx_cmd", gcmc.execution.gmx_cmd if gcmc is not None else "gmx_mpi")),
    )
    validate_analysis_config(config, check_files=False)
    return config


def validate_analysis_config(config: AnalysisConfig, *, check_files: bool = True) -> None:
    run_ids = [dataset.run_id for dataset in config.datasets]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Analysis run IDs must be unique after replica/sweep expansion")
    for dataset in config.datasets:
        if dataset.pocketmc_status not in {"confirmed", "probable", "not_detected"}:
            raise ValueError(f"{dataset.run_id}: pocketmc_status must be confirmed, probable, or not_detected")
    for dataset in config.datasets:
        cavity = cavity_for_dataset(config, dataset)
        prefix = f"{dataset.run_id}: " if dataset.cavity is not None else ""
        if cavity.mode not in {"mask", "sphere"}:
            raise ValueError(f"{prefix}cavity.mode must be mask or sphere")
        if cavity.mode == "mask" and cavity.mask is None:
            raise ValueError(f"{prefix}cavity.mask is required for mask mode")
        if cavity.radius_nm <= 0 or cavity.membership_padding_nm < 0:
            raise ValueError(f"{prefix}cavity radius must be positive and padding must be non-negative")
        if cavity.build_enabled:
            if cavity.mode != "mask":
                raise ValueError(f"{prefix}cavity auto-build is valid only in mask mode")
            if cavity.build_source is None or cavity.build_output_prefix is None:
                raise ValueError(f"{prefix}cavity auto-build requires source and output prefix")
            if cavity.build_mode != "seeded":
                raise ValueError(f"{prefix}cavity build mode currently supports only seeded")
            if min(
                cavity.build_dx,
                cavity.build_probe_radius,
                cavity.build_search_radius,
                cavity.build_nearby_cutoff,
                float(cavity.build_min_points),
            ) <= 0:
                raise ValueError(f"{prefix}cavity build dimensions and minimum points must be positive")
    opts = config.analysis
    if opts.stride < 1 or opts.gap_ps < 0 or opts.path_sample_ps <= 0:
        raise ValueError("analysis stride/path interval must be positive and gap must be non-negative")
    if opts.density_bin_a <= 0 or opts.density_sigma_a <= 0 or opts.density_cutoff_sigma <= 0:
        raise ValueError("density bin, sigma, and cutoff must be positive")
    if opts.density_quantity not in {"occupancy", "probability"}:
        raise ValueError("analysis.density_quantity must be occupancy or probability")
    if opts.start_ps is not None and opts.stop_ps is not None and opts.stop_ps < opts.start_ps:
        raise ValueError("analysis.stop_ps must be >= start_ps")
    if config.pose.clusters < 1 or config.pose.restarts < 1 or config.pose.max_frames_per_trajectory < 0:
        raise ValueError("pose clusters/restarts must be positive and max_frames_per_trajectory must be non-negative")
    if config.pose.sampling_strategy != "uniform":
        raise ValueError("pose.sampling.strategy currently supports only 'uniform'")
    if not 0 < config.pose.site_hpd <= 1 or config.pose.site_min_separation_a <= 0:
        raise ValueError("pose hydration-site HPD and separation values are invalid")
    if config.pose.site_max_count < 1 or config.pose.site_integral_radius_a <= 0:
        raise ValueError("pose hydration-site count and integration radius must be positive")
    feature_kinds = {"distance", "end_to_end", "angle", "dihedral", "orientation", "pocket_axis_orientation", "ring_pucker"}
    for feature in config.pose.features:
        if feature.kind not in feature_kinds:
            raise ValueError(f"unsupported pose feature kind: {feature.kind}")
        if feature.weight < 0:
            raise ValueError(f"pose feature weight must be non-negative: {feature.name}")
    if set(opts.tasks) & POSE_TASKS:
        if not config.substrate.enabled:
            raise ValueError("pose tasks require [substrate] enabled=true")
        if not config.substrate.selection and not any(item.substrate_selection for item in config.datasets if item.kind == "md"):
            raise ValueError("pose tasks require substrate.selection or case.substrate_selection")
        if not any(item.kind == "md" for item in config.datasets):
            raise ValueError("pose tasks require at least one physical MD case")
    if not check_files:
        return
    errors: list[str] = []
    for dataset in config.datasets:
        if not dataset.trajectory.exists():
            errors.append(f"{dataset.run_id}: trajectory not found: {dataset.trajectory}")
        if dataset.kind == "md" and (dataset.topology is None or not dataset.topology.exists()):
            errors.append(f"{dataset.run_id}: MD topology not found: {dataset.topology}")
        elif (
            dataset.kind == "md"
            and dataset.topology is not None
            and config_for_dataset(config, dataset).cavity.mode == "sphere"
        ):
            case_config = config_for_dataset(config, dataset)
            try:
                import MDAnalysis as mda

                universe = mda.Universe(str(dataset.topology))
                _group, resolution = select_mda_anchor(
                    universe,
                    case_config.cavity.anchor,
                    case_config.cavity.anchor_atoms,
                    context=f"{dataset.run_id} cavity",
                )
                if resolution.warning:
                    print(f"[anchor validation] {dataset.run_id}: {resolution.warning}", flush=True)
                close = getattr(universe.trajectory, "close", None)
                if close is not None:
                    close()
            except Exception as exc:
                errors.append(f"{dataset.run_id}: anchor preflight failed: {type(exc).__name__}: {exc}")
        if dataset.kind == "pocketmc" and dataset.mc_log is not None and not dataset.mc_log.exists():
            errors.append(f"{dataset.run_id}: mc.log not found: {dataset.mc_log}")
    for dataset in config.datasets:
        cavity = cavity_for_dataset(config, dataset)
        if cavity.mode != "mask":
            continue
        if cavity.mask is not None and not cavity.mask.exists() and not cavity.build_enabled:
            errors.append(
                f"{dataset.run_id}: cavity mask not found: {cavity.mask} "
                "(review this case's cavity_mask path or enable deferred cavity building)"
            )
        if cavity.meta is not None and not cavity.meta.exists() and not cavity.build_enabled:
            errors.append(f"{dataset.run_id}: cavity metadata not found: {cavity.meta}")
        if cavity.mask_trajectory is not None and not cavity.mask_trajectory.exists():
            errors.append(f"{dataset.run_id}: cavity mask trajectory not found: {cavity.mask_trajectory}")
        if cavity.build_enabled and (cavity.build_source is None or not cavity.build_source.exists()):
            errors.append(f"{dataset.run_id}: cavity build source not found: {cavity.build_source}")
    for label, source in (("canonical", opts.canonical_source), ("homolog", opts.homolog_source)):
        if source is not None and not source.exists():
            errors.append(f"{label} mapping source not found: {source}")
    if config.pose.reference is not None and not config.pose.reference.exists():
        errors.append(f"pose reference not found: {config.pose.reference}")
    if errors:
        raise ValueError("Analysis input validation failed:\n  " + "\n  ".join(errors))


def with_runtime_overrides(
    config: AnalysisConfig,
    *,
    tasks: list[str] | None = None,
    runs: list[str] | None = None,
) -> AnalysisConfig:
    analysis = config.analysis
    if tasks is not None:
        analysis = replace(analysis, tasks=expand_tasks(config.kind, tasks, pose_enabled=config.substrate.enabled))
    datasets = config.datasets
    if runs:
        wanted = {item.strip() for item in runs}
        datasets = tuple(item for item in datasets if item.run_id in wanted or item.replica in wanted)
        if not datasets:
            raise ValueError(f"No datasets matched --runs: {', '.join(runs)}")
    return replace(config, analysis=analysis, datasets=datasets)
