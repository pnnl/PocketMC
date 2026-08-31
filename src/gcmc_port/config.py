from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from .assets import resolve_project_input
from .reference import normalize_residue_tokens
from .safe_config import (
    safe_single_line,
    validated_environment,
    validated_module_setup,
    validated_shebang,
    validated_slurm_directives,
)


@dataclass(slots=True)
class PathsConfig:
    project_root: Path
    work_root: Path
    forcefield_dir: Path
    residue_types: Path
    topology: Path
    water_itp: Path
    chk_mdp: Path
    steep_mdp: Path
    em_mdp: Path
    init_gro: Path
    gas_gro: Path


@dataclass(slots=True)
class ExecutionConfig:
    shell_executable: str = "/bin/bash"
    module_setup: list[str] = field(default_factory=list)
    gmx_cmd: str = "gmx_mpi"
    launcher_single: str = "mpirun -np 1"
    launcher_multi: str = "mpirun -np {cores}"
    mdrun_multi_args: list[str] = field(default_factory=list)
    nodes: int = 1
    cores_per_node: int = 32
    env: dict[str, str] = field(default_factory=dict)

    @property
    def total_cores(self) -> int:
        return self.nodes * self.cores_per_node


@dataclass(slots=True)
class AnchorConfig:
    anchor: str = "800ATC"
    resid: int = 800
    resname: str = "ATC"
    residues: list[str] = field(default_factory=lambda: ["800ATC"])
    reference_mode: str = "atoms"
    center_atoms: list[str] = field(default_factory=lambda: ["C2", "C4", "C7"])

    def normalized(self) -> "AnchorConfig":
        self.reference_mode = str(self.reference_mode).strip().lower() or "atoms"
        if self.reference_mode not in {"atoms", "com"}:
            raise ValueError(f"Unsupported anchor.reference_mode={self.reference_mode!r}; expected 'atoms' or 'com'")

        if self.residues:
            self.residues = normalize_residue_tokens(self.residues)
        elif self.anchor:
            self.residues = normalize_residue_tokens([self.anchor])
        elif self.resid and self.resname:
            self.residues = [f"{self.resid}{self.resname.upper()}"]
        else:
            self.residues = []

        if self.residues:
            self.anchor = self.residues[0]
            digits = []
            letters = []
            for char in self.anchor:
                if char.isdigit() and not letters:
                    digits.append(char)
                else:
                    letters.append(char)
            if digits and letters:
                self.resid = int("".join(digits))
                self.resname = "".join(letters)

        if len(self.residues) > 1 and self.reference_mode == "atoms":
            self.reference_mode = "com"
        return self


@dataclass(slots=True)
class SimulationConfig:
    test_insertion: bool = False
    temperature: float = 298.15
    pressure: float = 1.0
    max_trials: int = 500
    max_consecutive_insertion_failures: int = 5
    target_nmol: int = 31
    mu0: float = -25.48056
    gas_constant: float = 0.008314
    rmax: float = 0.6
    rvdw: float = 0.2
    rfree: float = 2.0
    v0: float = 0.030345
    kres: float = 10000000.0
    max_e0_tries: int = 10
    mask_file: str | None = None
    mask_dx: float = 0.0


@dataclass(slots=True)
class CavityConfig:
    mode: str = "sphere"
    mask_file: Path | None = None
    mask_meta: Path | None = None
    restraint_radius: float = 1.2
    membership_padding: float = 0.02
    initial_delete_padding: float = 0.0


@dataclass(slots=True)
class CavityBuildConfig:
    enabled: bool = False
    mode: str = "seeded"
    output_prefix: Path = Path("cavity")
    exclude_residues: list[str] = field(default_factory=list)
    dx: float = 0.075
    probe_radius: float = 0.10
    search_radius: float = 0.9
    nearby_cutoff: float = 0.45
    min_points: int = 20


@dataclass(slots=True)
class LoopConfig:
    sweep_values: list[int] = field(default_factory=list)
    replica_count: int = 5
    replica_dirs: list[str] = field(default_factory=list)
    sweep_dir_format: str = "{value}"
    replica_dir_format: str = "{replica:02d}"

    def expanded_replica_dirs(self) -> list[str]:
        if self.replica_dirs:
            return self.replica_dirs
        return [self.replica_dir_format.format(replica=idx) for idx in range(self.replica_count)]


@dataclass(slots=True)
class SlurmConfig:
    enabled: bool = True
    shebang: str = "#!/usr/bin/env bash"
    time_limit: str = "24:00:00"
    partition: str = "compute"
    account: str = ""
    job_name: str = "gcmc-port"
    output: str = "slurm-%j.out"
    extra_directives: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RunSpec:
    run_dir: Path
    replica_label: str
    sweep_value: int | None = None


@dataclass(slots=True)
class Config:
    config_path: Path
    paths: PathsConfig
    execution: ExecutionConfig
    anchor: AnchorConfig
    simulation: SimulationConfig
    cavity: CavityConfig
    loop: LoopConfig
    slurm: SlurmConfig
    cavity_build: CavityBuildConfig = field(default_factory=CavityBuildConfig)

    def run_specs(self) -> list[RunSpec]:
        run_specs: list[RunSpec] = []
        replica_dirs = self.loop.expanded_replica_dirs()
        if self.loop.sweep_values:
            for value in self.loop.sweep_values:
                sweep_dir = self.paths.work_root / self.loop.sweep_dir_format.format(value=value)
                for replica in replica_dirs:
                    run_specs.append(RunSpec(run_dir=sweep_dir / replica, replica_label=replica, sweep_value=value))
        else:
            for replica in replica_dirs:
                run_specs.append(RunSpec(run_dir=self.paths.work_root / replica, replica_label=replica))
        return run_specs


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_config(config_path: str | Path) -> Config:
    path = Path(config_path).resolve()
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    paths_raw = raw.get("paths", {})
    base_root = _resolve(path.parent, paths_raw.get("project_root", ".."))
    work_root = _resolve(base_root, paths_raw.get("work_root", "."))
    paths = PathsConfig(
        project_root=base_root,
        work_root=work_root,
        forcefield_dir=resolve_project_input(
            base_root,
            paths_raw.get("forcefield_dir", "amber14sb_parmbsc1.ff"),
            default_name="amber14sb_parmbsc1.ff",
        ),
        residue_types=resolve_project_input(
            base_root,
            paths_raw.get("residue_types", "residuetypes.dat"),
            default_name="residuetypes.dat",
        ),
        topology=_resolve(base_root, paths_raw.get("topology", "topol.top")),
        water_itp=resolve_project_input(base_root, paths_raw.get("water_itp", "WAT.itp"), default_name="WAT.itp"),
        chk_mdp=resolve_project_input(base_root, paths_raw.get("chk_mdp", "chk.mdp"), default_name="chk.mdp"),
        steep_mdp=resolve_project_input(base_root, paths_raw.get("steep_mdp", "steep.mdp"), default_name="steep.mdp"),
        em_mdp=resolve_project_input(base_root, paths_raw.get("em_mdp", "em.mdp"), default_name="em.mdp"),
        init_gro=_resolve(base_root, paths_raw.get("init_gro", "init.gro")),
        gas_gro=resolve_project_input(base_root, paths_raw.get("gas_gro", "COM.gro"), default_name="COM.gro"),
    )

    exec_raw = raw.get("execution", {})
    env = validated_environment(dict(exec_raw.get("env", {})))
    env.setdefault("GMX_MAXBACKUP", "-1")
    execution = ExecutionConfig(
        shell_executable=safe_single_line(exec_raw.get("shell_executable", "/bin/bash"), "execution.shell_executable"),
        module_setup=validated_module_setup(exec_raw.get("module_setup", []), "execution.module_setup"),
        gmx_cmd=safe_single_line(exec_raw.get("gmx_cmd", "gmx_mpi"), "execution.gmx_cmd", allow_empty=False),
        launcher_single=safe_single_line(exec_raw.get("launcher_single", "mpirun -np 1"), "execution.launcher_single"),
        launcher_multi=safe_single_line(exec_raw.get("launcher_multi", "mpirun -np {cores}"), "execution.launcher_multi"),
        mdrun_multi_args=[
            safe_single_line(value, f"execution.mdrun_multi_args[{index}]", allow_empty=False)
            for index, value in enumerate(exec_raw.get("mdrun_multi_args", []))
        ],
        nodes=int(exec_raw.get("nodes", 1)),
        cores_per_node=int(exec_raw.get("cores_per_node", 32)),
        env=env,
    )

    anchor_raw = raw.get("anchor", {})
    anchor_residues = [str(value) for value in anchor_raw.get("residues", [])]
    anchor = AnchorConfig(
        anchor=anchor_raw.get("anchor", "800ATC"),
        resid=int(anchor_raw.get("resid", 800)),
        resname=anchor_raw.get("resname", "ATC"),
        residues=anchor_residues,
        reference_mode=anchor_raw.get("reference_mode", "com" if len(anchor_residues) > 1 else "atoms"),
        center_atoms=list(anchor_raw.get("center_atoms", ["C2", "C4", "C7"])),
    ).normalized()

    sim_raw = raw.get("simulation", {})
    simulation = SimulationConfig(
        test_insertion=bool(sim_raw.get("test_insertion", False)),
        temperature=float(sim_raw.get("temperature", 298.15)),
        pressure=float(sim_raw.get("pressure", 1.0)),
        max_trials=int(sim_raw.get("max_trials", 500)),
        max_consecutive_insertion_failures=int(sim_raw.get("max_consecutive_insertion_failures", 5)),
        target_nmol=int(sim_raw.get("target_nmol", 31)),
        mu0=float(sim_raw.get("mu0", -25.48056)),
        gas_constant=float(sim_raw.get("gas_constant", 0.008314)),
        rmax=float(sim_raw.get("rmax", 0.6)),
        rvdw=float(sim_raw.get("rvdw", 0.2)),
        rfree=float(sim_raw.get("rfree", 2.0)),
        v0=float(sim_raw.get("v0", 0.030345)),
        kres=float(sim_raw.get("kres", 10000000.0)),
        max_e0_tries=int(sim_raw.get("max_e0_tries", 10)),
        mask_file=sim_raw.get("mask_file"),
        mask_dx=float(sim_raw.get("mask_dx", 0.0)),
    )

    cavity_raw = raw.get("cavity", {})
    cavity_mode = str(cavity_raw.get("mode", "sphere")).strip().lower() or "sphere"
    if cavity_mode not in {"sphere", "mask"}:
        raise ValueError(f"Unsupported cavity.mode={cavity_mode!r}; expected 'sphere' or 'mask'")
    mask_file_value = cavity_raw.get("mask_file", simulation.mask_file)
    mask_meta_value = cavity_raw.get("mask_meta")
    cavity = CavityConfig(
        mode=cavity_mode,
        mask_file=_resolve(base_root, mask_file_value) if mask_file_value else None,
        mask_meta=_resolve(base_root, mask_meta_value) if mask_meta_value else None,
        restraint_radius=float(cavity_raw.get("restraint_radius", simulation.rfree * simulation.rmax)),
        membership_padding=float(cavity_raw.get("membership_padding", 0.02)),
        initial_delete_padding=float(cavity_raw.get("initial_delete_padding", 0.0)),
    )

    cavity_build_raw = raw.get("cavity_build", {})
    cavity_build = CavityBuildConfig(
        enabled=bool(cavity_build_raw.get("enabled", False)),
        mode=str(cavity_build_raw.get("mode", "seeded")).strip().lower() or "seeded",
        output_prefix=_resolve(base_root, str(cavity_build_raw.get("output_prefix", "cavity"))),
        exclude_residues=normalize_residue_tokens([str(value) for value in cavity_build_raw.get("exclude_residues", [])]),
        dx=float(cavity_build_raw.get("dx", 0.075)),
        probe_radius=float(cavity_build_raw.get("probe_radius", 0.10)),
        search_radius=float(cavity_build_raw.get("search_radius", 0.9)),
        nearby_cutoff=float(cavity_build_raw.get("nearby_cutoff", 0.45)),
        min_points=int(cavity_build_raw.get("min_points", 20)),
    )
    if cavity_build.mode != "seeded":
        raise ValueError(f"Unsupported cavity_build.mode={cavity_build.mode!r}; expected 'seeded'")

    loop_raw = raw.get("loop", {})
    loop = LoopConfig(
        sweep_values=[int(value) for value in loop_raw.get("sweep_values", [])],
        replica_count=int(loop_raw.get("replica_count", 5)),
        replica_dirs=[str(value) for value in loop_raw.get("replica_dirs", [])],
        sweep_dir_format=str(loop_raw.get("sweep_dir_format", "{value}")),
        replica_dir_format=str(loop_raw.get("replica_dir_format", "{replica:02d}")),
    )

    slurm_raw = raw.get("slurm", {})
    slurm = SlurmConfig(
        enabled=bool(slurm_raw.get("enabled", True)),
        shebang=validated_shebang(slurm_raw.get("shebang", "#!/usr/bin/env bash")),
        time_limit=safe_single_line(slurm_raw.get("time_limit", "24:00:00"), "slurm.time_limit", allow_empty=False),
        partition=safe_single_line(slurm_raw.get("partition", "compute"), "slurm.partition"),
        account=safe_single_line(slurm_raw.get("account", ""), "slurm.account"),
        job_name=safe_single_line(slurm_raw.get("job_name", "gcmc-port"), "slurm.job_name", allow_empty=False),
        output=safe_single_line(slurm_raw.get("output", "slurm-%j.out"), "slurm.output", allow_empty=False),
        extra_directives=validated_slurm_directives(slurm_raw.get("extra_directives", [])),
    )

    return Config(
        config_path=path,
        paths=paths,
        execution=execution,
        anchor=anchor,
        simulation=simulation,
        cavity=cavity,
        loop=loop,
        slurm=slurm,
        cavity_build=cavity_build,
    )
