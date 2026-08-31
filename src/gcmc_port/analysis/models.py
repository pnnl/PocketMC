from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal


InputKind = Literal["md", "pocketmc"]
ConfigKind = Literal["md", "pocketmc", "mixed"]
PointMode = Literal["atom", "cog", "com"]

POSE_TASKS = {"pose-clusters", "pose-hydration", "compare-hydration"}
BASE_MD_TASKS = {"lifetime", "paths", "density", "plots", "vmd"}
MD_TASKS = BASE_MD_TASKS | POSE_TASKS
MC_TASKS = {"mc-states", "density", "plots", "vmd"}
ALL_TASKS = MD_TASKS | MC_TASKS

# Increment whenever cached FrameRecord coordinates/membership semantics change.
ANALYSIS_CACHE_VERSION = 6


@dataclass(frozen=True, slots=True)
class MoleculeSpec:
    preset: str = "water"
    label: str = "Water"
    resnames: tuple[str, ...] = ("SOL", "WAT", "HOH")
    point_mode: PointMode = "atom"
    atom_names: tuple[str, ...] = ("OW", "O", "OH2")

    @classmethod
    def from_values(
        cls,
        *,
        preset: str,
        label: str | None = None,
        resnames: list[str] | tuple[str, ...] | None = None,
        point_mode: str | None = None,
        atom_names: list[str] | tuple[str, ...] | None = None,
    ) -> "MoleculeSpec":
        key = preset.strip().lower()
        if key == "water":
            defaults = cls()
        elif key == "co":
            defaults = cls(
                preset="co",
                label="CO",
                resnames=("COM",),
                point_mode="cog",
                atom_names=("CJ", "OJ1"),
            )
        elif key == "custom":
            defaults = cls(preset="custom", label="Tracked molecule", resnames=(), point_mode="cog", atom_names=())
        else:
            raise ValueError("molecule.preset must be water, co, or custom")

        resolved_mode = str(point_mode or defaults.point_mode).strip().lower()
        if resolved_mode not in {"atom", "cog", "com"}:
            raise ValueError("molecule.point_mode must be atom, cog, or com")
        resolved_resnames = tuple(str(item).strip().upper() for item in (resnames or defaults.resnames) if str(item).strip())
        resolved_atoms = tuple(str(item).strip().upper() for item in (atom_names or defaults.atom_names) if str(item).strip())
        if not resolved_resnames:
            raise ValueError("molecule.resnames must contain at least one residue name")
        if resolved_mode == "atom" and not resolved_atoms:
            raise ValueError("molecule.atom_names is required when point_mode='atom'")
        return cls(
            preset=key,
            label=str(label or defaults.label).strip() or defaults.label,
            resnames=resolved_resnames,
            point_mode=resolved_mode,  # type: ignore[arg-type]
            atom_names=resolved_atoms,
        )


@dataclass(frozen=True, slots=True)
class CavitySpec:
    mode: str = "sphere"
    mask: Path | None = None
    meta: Path | None = None
    mask_trajectory: Path | None = None
    anchor: str = "800ATC"
    anchor_atoms: tuple[str, ...] = ("C2", "C4", "C7")
    radius_nm: float = 0.60
    membership_padding_nm: float = 0.02
    protein_selection: str = "protein and not name H*"
    align_selection: str | None = None
    points: Path | None = None
    nearby_residues: Path | None = None
    build_enabled: bool = False
    build_source: Path | None = None
    build_output_prefix: Path | None = None
    build_mode: str = "seeded"
    build_exclude_residues: tuple[str, ...] = ()
    build_dx: float = 0.075
    build_probe_radius: float = 0.10
    build_search_radius: float = 0.90
    build_nearby_cutoff: float = 0.45
    build_min_points: int = 20


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    run_id: str
    kind: InputKind
    run_dir: Path
    topology: Path | None
    trajectory: Path
    reference: Path | None = None
    mc_log: Path | None = None
    trajectory_meta: Path | None = None
    replica: str = ""
    sweep: str = ""
    system_id: str = ""
    comparison_group: str = ""
    pocketmc_status: str = "not_detected"
    pocketmc_evidence: tuple[str, ...] = ()
    substrate_selection: str = ""
    pocket_selection: str = ""
    cavity: CavitySpec | None = None


@dataclass(frozen=True, slots=True)
class PoseFeatureSpec:
    name: str
    kind: str
    selections: tuple[str, ...] = ()
    point_modes: tuple[str, ...] = ()
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class SubstrateSpec:
    enabled: bool = False
    selection: str = ""
    fit_selection: str = ""
    heavy_selection: str = "not name H*"


@dataclass(frozen=True, slots=True)
class PoseOptions:
    clusters: int = 3
    max_frames_per_trajectory: int = 5000
    sampling_strategy: str = "uniform"
    write_trajectory: bool = True
    seed: int = 2026
    restarts: int = 20
    reference: Path | None = None
    reference_pocket_selection: str = ""
    pocket_selection: str = "protein and backbone"
    site_hpd: float = 0.70
    site_min_separation_a: float = 2.4
    site_max_count: int = 20
    site_integral_radius_a: float = 1.4
    features: tuple[PoseFeatureSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisOptions:
    tasks: tuple[str, ...] = ("all",)
    start_ps: float | None = None
    stop_ps: float | None = None
    stride: int = 1
    gap_ps: float = 1000.0
    path_sample_ps: float = 1000.0
    contact_cutoff_a: float = 4.0
    density_bin_a: float = 1.0
    density_sigma_a: float = 2.0
    density_cutoff_sigma: float = 3.0
    density_quantity: str = "occupancy"
    rolling_window: int = 10
    canonical_source: Path | None = None
    canonical_chain: str = ""
    homolog_source: Path | None = None
    homolog_chain: str = ""
    verbose: bool = True


@dataclass(frozen=True, slots=True)
class OutputOptions:
    root: Path
    cache: bool = True
    overwrite: bool = True


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    config_path: Path
    kind: ConfigKind
    datasets: tuple[DatasetSpec, ...]
    molecule: MoleculeSpec
    cavity: CavitySpec
    analysis: AnalysisOptions
    output: OutputOptions
    substrate: SubstrateSpec = SubstrateSpec()
    pose: PoseOptions = PoseOptions()
    gmx_cmd: str = "gmx_mpi"


def cavity_for_dataset(config: AnalysisConfig, dataset: DatasetSpec) -> CavitySpec:
    """Return a case-specific cavity when present, otherwise the global default."""
    return dataset.cavity or config.cavity


def config_for_dataset(config: AnalysisConfig, dataset: DatasetSpec) -> AnalysisConfig:
    """Bind a case-specific cavity to a temporary config used by existing readers."""
    cavity = cavity_for_dataset(config, dataset)
    return config if cavity is config.cavity else replace(config, cavity=cavity)


@dataclass(frozen=True, slots=True)
class MoleculeFrame:
    uid: str
    resid: int
    resname: str
    point_nm: tuple[float, float, float]
    inside: bool
    nearest_residue: str = ""
    nearest_distance_nm: float = float("nan")
    nearest_residue_sim: str = ""
    nearest_residue_homolog: str = ""


@dataclass(frozen=True, slots=True)
class FrameRecord:
    frame: int
    time_ps: float
    molecules: tuple[MoleculeFrame, ...]
    occupancy: int
    energy_kj_mol: float | None = None
    trial: int | None = None
    move: str = ""


@dataclass(frozen=True, slots=True)
class VisitRecord:
    run_id: str
    molecule_uid: str
    resid: int
    resname: str
    visit_index: int
    event_type: str
    start_frame: int
    end_frame: int
    start_ps: float
    end_ps: float
    lifetime_ps: float
    sample_count: int
    left_censored: bool
    right_censored: bool
    dominant_residue: str


@dataclass(frozen=True, slots=True)
class PathSample:
    run_id: str
    molecule_uid: str
    sample_index: int
    frame: int
    time_ps: float
    label: str
    nearest_residue: str
    distance_nm: float
    inside: bool
    point_nm: tuple[float, float, float]
    nearest_residue_sim: str = ""
    nearest_residue_homolog: str = ""


@dataclass(frozen=True, slots=True)
class MCMove:
    run_id: str
    trial: int
    accepted_before: int
    move: str
    accepted: bool
    energy_kj_mol: float | None
    delta_energy_kj_mol: float | None
    n_inside_before: int | None


@dataclass(slots=True)
class RunResult:
    dataset: DatasetSpec
    frames: list[FrameRecord]
    visits: list[VisitRecord] = field(default_factory=list)
    path_samples: list[PathSample] = field(default_factory=list)
    mc_moves: list[MCMove] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def expand_tasks(
    kind: ConfigKind,
    requested: tuple[str, ...] | list[str],
    *,
    pose_enabled: bool = False,
) -> tuple[str, ...]:
    normalized = {str(item).strip().lower().replace("_", "-") for item in requested if str(item).strip()}
    allowed = MD_TASKS if kind == "md" else (MC_TASKS if kind == "pocketmc" else ALL_TASKS)
    if not normalized or "all" in normalized:
        if kind == "md":
            selected = set(BASE_MD_TASKS)
            if pose_enabled:
                selected.update(POSE_TASKS)
            return tuple(sorted(selected))
        if kind == "pocketmc":
            return tuple(sorted(MC_TASKS))
        selected = set(BASE_MD_TASKS | MC_TASKS)
        if pose_enabled:
            selected.update(POSE_TASKS)
        return tuple(sorted(selected))
    aliases = {
        "occupancy": "lifetime" if kind == "md" else "mc-states",
        "mc": "mc-states",
        "states": "mc-states",
        "path": "paths",
        "plot": "plots",
        "trace": "vmd",
        "water-trace": "vmd",
        "2dmap": "density",
        "2d-map": "density",
        "3dmap": "density",
        "3d-map": "density",
        "cluster": "pose-clusters",
        "clusters": "pose-clusters",
        "pose": "pose-clusters",
        "pose-density": "pose-hydration",
        "cluster-density": "pose-hydration",
        "compare": "compare-hydration",
    }
    normalized = {aliases.get(item, item) for item in normalized}
    unsupported = normalized - allowed
    if unsupported:
        if kind == "pocketmc" and unsupported & ({"lifetime", "paths"} | POSE_TASKS):
            raise ValueError("PocketMC accepted-state trajectories do not support physical lifetime or path tasks")
        raise ValueError(f"Unsupported {kind} analysis task(s): {', '.join(sorted(unsupported))}")
    if normalized & {"paths", "density", "plots", "vmd"} and kind in {"md", "mixed"}:
        normalized.add("lifetime")
    if normalized & {"density", "plots", "vmd"} and kind in {"pocketmc", "mixed"}:
        normalized.add("mc-states")
    if normalized & {"pose-hydration", "compare-hydration"}:
        normalized.add("pose-clusters")
    if "compare-hydration" in normalized:
        normalized.add("pose-hydration")
    return tuple(sorted(normalized))
