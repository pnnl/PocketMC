from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import log, pi, sqrt
import os
from pathlib import Path
import random
import shutil
from time import perf_counter

from .cavity import (
    VoxelMask,
    align_voxel_mask_to_structure,
    build_cavity_from_structure,
    load_voxel_mask,
    molecule_residue_ids_in_mask,
    remove_waters_in_mask,
    water_residue_ids_in_mask,
)
from .clash import ClashDetail, describe_clash, residue_clash_in_structure
from .config import Config, RunSpec
from .gromacs import ShellRunner, parse_potential_energy
from .gro import (
    WATER_NAMES,
    contiguous_residue_groups,
    coordinates_center,
    count_water_molecules,
    last_residue_atoms,
    parse_atom_line,
    parse_gro,
)
from .helpers import get_alcove_residues, remove_waters_near_centroid
from .moves import (
    acceptance_probability,
    propose_deletion,
    propose_insertion,
    propose_rotation,
    propose_translation,
    write_mask_trajectory,
    write_position_restraints,
    write_trajectory,
)
from .reference import resolve_reference_center
from .topology import INCLUDE_RE, ensure_molecule_include, insert_posre_include, merge_wat_into_sol, set_molecule_count

NO_VALID_INSERTION_POINT_ERROR = "Could not find valid insertion point"
MAX_MASK_REFERENCE_CLEANUP_CYCLES = 5
CENTERING_QC_TARGET_FRACTION = 0.15
CENTERING_QC_HARD_STOP_FRACTION = 0.25
CENTERING_GROUP_NAME = "CENTER_SOLUTE"
SOLVENT_CATEGORIES = {"water"}
CENTERING_FALLBACK_SKIP_CATEGORIES = {"water", "ion"}
SCIENTIFIC_SCOPE_NOTICE = (
    "Scientific scope: this minimization-assisted workflow is a heuristic structure/occupancy search, "
    "not a rigorous equilibrium muVT sampler; per-proposal energy minimization does not preserve "
    "Metropolis detailed balance. Do not use its occupancy histogram for equilibrium free energies."
)


@dataclass(slots=True)
class WorkflowContext:
    run_dir: Path
    log_file: Path
    progress_file: Path
    positions_file: Path
    mask_trajectory_file: Path
    previous_top: Path
    current_top: Path
    previous_gro: Path
    current_gro: Path
    gas_name: str
    na_gas: int
    orig_atom_count: int


class GCMCWorkflow:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.runner = ShellRunner(config.execution)
        self.echo_progress = "SLURM_JOB_ID" not in os.environ
        self._mask_cache: VoxelMask | None = None
        self._residue_categories: dict[str, str] | None = None

    def run(self) -> None:
        self._prepare_runtime_inputs()
        self._validate_inputs()
        run_specs = self.config.run_specs()
        continue_on_case_failure = len(run_specs) > 1
        failures: list[tuple[RunSpec, Exception]] = []
        for run_spec in run_specs:
            try:
                self._run_single(run_spec)
            except Exception as exc:  # noqa: BLE001
                if not continue_on_case_failure:
                    raise
                self._record_run_failure(run_spec, exc)
                failures.append((run_spec, exc))
        if failures:
            labels = ", ".join(str(run_spec.run_dir) for run_spec, _ in failures)
            raise RuntimeError(f"{len(failures)} GCMC run(s) failed: {labels}")

    def _run_single(self, run_spec: RunSpec) -> None:
        if self.config.cavity.mode == "mask":
            self._run_single_mask(run_spec)
            return

        run_dir = run_spec.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._copy_inputs(run_dir)

        topol = run_dir / "topol.top"
        merge_wat_into_sol(topol)
        ensure_molecule_include(topol, self.config.paths.water_itp.name, self._molecule_atomtypes_include_name())

        ctx = WorkflowContext(
            run_dir=run_dir,
            log_file=run_dir / "mc.log",
            progress_file=run_dir / "workflow.log",
            positions_file=run_dir / "trials.xyz",
            mask_trajectory_file=run_dir / "cavity_trajectory.gro",
            previous_top=run_dir / "previous.top",
            current_top=run_dir / "current.top",
            previous_gro=run_dir / "previous.gro",
            current_gro=run_dir / "current.gro",
            gas_name=parse_gro(self.config.paths.gas_gro).atoms[0].resname,
            na_gas=len(parse_gro(self.config.paths.gas_gro).atoms),
            orig_atom_count=0,
        )

        rng = random.Random()
        if run_spec.sweep_value is not None:
            seed_hint = run_spec.sweep_value * 100 + int(run_spec.replica_label)
            rng.seed(seed_hint)

        self._prepare_logs(ctx)
        self._progress(
            ctx,
            f"Run start: run_dir={ctx.run_dir} replica={run_spec.replica_label}"
            + (f" sweep={run_spec.sweep_value}" if run_spec.sweep_value is not None else ""),
        )
        self._progress(ctx, SCIENTIFIC_SCOPE_NOTICE)
        self._progress(
            ctx,
            "GCMC region: "
            f"rins={self._insertion_radius():.8f} nm  "
            f"r2={self._insertion_radius_squared():.8f} nm^2  "
            f"Veff={self._effective_volume():.10f} nm^3",
        )
        self._progress(ctx, f"{ctx.gas_name} chemical potential = {self._chemical_potential():.2f}")

        self._initialize_system(ctx)
        ctx.orig_atom_count = len(parse_gro(run_dir / "start.gro").atoms)
        self._progress(ctx, f"Initial setup complete: start_atoms={ctx.orig_atom_count}")

        center = self._reference_center(ctx.previous_gro)
        get_alcove_residues(
            ctx.previous_gro,
            self.config.simulation.rmax,
            center,
            self.config.simulation.rfree,
            out_dir=ctx.run_dir,
        )
        write_position_restraints(ctx.previous_gro, self.config.simulation.kres, out_dir=ctx.run_dir)

        e0 = self._initial_reference_energy(ctx)
        self._progress(ctx, f"Reference energy established: E0={e0:.3f}")
        naccepted = 0
        nmol = 1
        provenance: dict[int, str] = {}
        mu_tp = self._chemical_potential()
        veff = self._effective_volume()
        consecutive_insertion_point_failures = 0

        self._write_log_header(ctx)
        for trial in range(1, self.config.simulation.max_trials + 1):
            trial_started_at = perf_counter()
            if self.config.simulation.target_nmol > 0 and nmol >= self.config.simulation.target_nmol:
                self._append_log(ctx, f"Reached TARGET_NMOL={self.config.simulation.target_nmol} (nmol={nmol}) -> stop this run")
                break

            center = self._reference_center(ctx.previous_gro)
            get_alcove_residues(
                ctx.previous_gro,
                self.config.simulation.rmax,
                center,
                self.config.simulation.rfree,
                out_dir=ctx.run_dir,
            )

            nins = nmol - 1
            move = self._select_move(nins, rng)
            move_name = type_move_name(move)
            self._progress(
                ctx,
                f"MC {trial}/{self.config.simulation.max_trials}: move={move_name} nmol={nmol} accepted={naccepted}",
                echo=False,
            )
            moved_resid: int | None = None
            try:
                if move == 1:
                    if self.config.simulation.target_nmol > 0 and nmol >= self.config.simulation.target_nmol:
                        self._append_log(
                            ctx,
                            f"Reached TARGET_NMOL={self.config.simulation.target_nmol} (nmol={nmol}) -> stop this run",
                        )
                        break
                    propose_insertion(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        rvdw=self.config.simulation.rvdw,
                        gas_name=ctx.gas_name,
                        gas_gro=self.config.paths.gas_gro,
                        rmax=self.config.simulation.rmax,
                        xyz_path=ctx.positions_file,
                        out_dir=ctx.run_dir,
                        mask_file=self._resolved_mask_file(),
                        mask_dx=self.config.simulation.mask_dx,
                        seed_point=center,
                        rng=rng,
                    )
                    consecutive_insertion_point_failures = 0
                    moved_resid = parse_gro(ctx.current_gro).atoms[-1].resid
                    type_move = "I"
                    mu = mu_tp
                elif move == 2:
                    moved_resid = propose_rotation(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        nmol=nmol,
                        gas_name=ctx.gas_name,
                        orig_atom_count=ctx.orig_atom_count,
                        rvdw=self.config.simulation.rvdw,
                        rng=rng,
                    )
                    type_move = "R"
                    mu = 0.0
                elif move == 3:
                    moved_resid = propose_translation(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        nmol=nmol,
                        gas_name=ctx.gas_name,
                        orig_atom_count=ctx.orig_atom_count,
                        rvdw=self.config.simulation.rvdw,
                        rng=rng,
                    )
                    type_move = "T"
                    mu = 0.0
                else:
                    moved_resid = propose_deletion(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        nmol=nmol,
                        gas_name=ctx.gas_name,
                        orig_atom_count=ctx.orig_atom_count,
                        out_dir=ctx.run_dir,
                        rng=rng,
                    )
                    type_move = "D"
                    mu = -mu_tp
            except Exception as exc:  # noqa: BLE001
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: {move_name} proposal failed ({exc})",
                    echo=False,
                )
                self._append_log(ctx, f"  {type_move_name(move)} proposal failed on trial {trial}: {exc}")
                if self._is_insertion_point_failure(move, exc):
                    consecutive_insertion_point_failures += 1
                    if self._should_stop_after_insertion_point_failures(
                        ctx,
                        trial=trial,
                        consecutive_failures=consecutive_insertion_point_failures,
                    ):
                        break
                elif move == 1:
                    consecutive_insertion_point_failures = 0
                continue

            self._progress(
                ctx,
                f"MC {trial}/{self.config.simulation.max_trials}: {move_name} proposed, running EM",
                echo=False,
            )
            write_position_restraints(ctx.current_gro, self.config.simulation.kres, out_dir=ctx.run_dir)
            e1 = self._relax_and_center(ctx, current=True)
            if e1 is None:
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: {move_name} EM failed",
                    echo=False,
                )
                self._append_log(ctx, f"  simulation failed on trial {trial} (move={type_move})")
                continue
            hard_clash = self._post_em_clash(ctx.run_dir / "confout.gro", moved_resid, type_move=type_move)
            if hard_clash is not None:
                details = describe_clash(hard_clash)
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: hard-clash reject after EM ({details})",
                    echo=False,
                )
                self._append_log(ctx, f"  hard-clash reject on trial {trial} (move={type_move}, {details})")
                continue

            sphere_resids = self._inserted_residue_ids_in_sphere(ctx.run_dir / "confout.gro", ctx)
            expected_n = nins + {1: 1, 2: 0, 3: 0, 4: -1}[move]
            if len(sphere_resids) != expected_n:
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: rejected by sphere membership "
                    f"before={nins} after={len(sphere_resids)} expected={expected_n}",
                    echo=False,
                )
                self._append_log(
                    ctx,
                    f"  sphere-membership mismatch on trial {trial} "
                    f"(move={type_move}, before={nins}, after={len(sphere_resids)}, expected={expected_n})",
                )
                continue

            de = e1 - (e0 + mu)
            acceptance = acceptance_probability(
                de=de,
                temperature=self.config.simulation.temperature,
                move=move,
                veff=veff,
                v0=self.config.simulation.v0,
                nins=nins,
                rng=rng,
                gas_constant=self.config.simulation.gas_constant,
            )
            resname, resid, info = self._move_log_context(
                type_move=type_move,
                moved_resid=moved_resid,
                confout_gro=ctx.run_dir / "confout.gro",
                x_gro=ctx.run_dir / "x.gro",
                previous_gro=ctx.previous_gro,
                gas_name=ctx.gas_name,
                provenance=provenance,
            )
            elapsed_seconds = int(perf_counter() - trial_started_at)
            line = (
                f"{trial:5d} {naccepted:4d} {type_move} | "
                f"{self._fmt_energy(e1, de):>8s} {e0:8.3f} {self._fmt_delta(de):>6s} | "
                f"{resname:>3s} {resid:>5s} {info:>5s} | "
                f"{elapsed_seconds:4d} {self._acceptance_label(acceptance):>4s} | {nins:4d}"
            )
            self._append_log(ctx, line)

            if acceptance > 0:
                shutil.copyfile(ctx.run_dir / "confout.gro", ctx.previous_gro)
                shutil.copyfile(ctx.current_top, ctx.previous_top)
                e0 = e1
                naccepted += 1
                if move == 1:
                    nmol += 1
                    if moved_resid is not None:
                        provenance[moved_resid] = f"t{trial:03d}"
                elif move == 4:
                    nmol = max(1, nmol - 1)
                    if moved_resid is not None:
                        provenance.pop(moved_resid, None)
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: ACCEPTED move={type_move} "
                    f"E1={e1:.3f} DE={de:.3f} nmol={nmol} accepted={naccepted}",
                    echo=False,
                )
                shutil.copyfile(ctx.run_dir / "posre_cavity.itp", ctx.run_dir / f"posre_cavity-{nmol}.itp")
                shutil.copyfile(ctx.run_dir / "confout.gro", ctx.run_dir / f"confout-{nmol}.gro")
                write_trajectory(
                    ctx.run_dir / "confout.gro",
                    e0,
                    naccepted,
                    nmol,
                    self.config.simulation.target_nmol,
                    trajectory_path=ctx.run_dir / "trajectory.gro",
                    gas_gro=self.config.paths.gas_gro,
                    trial=trial,
                    move=type_move,
                    active_resids=sphere_resids,
                    provenance=provenance,
                    trajectory_meta_path=ctx.run_dir / "trajectory.meta.jsonl",
                )
                if self.config.simulation.target_nmol > 0 and nmol >= self.config.simulation.target_nmol:
                    self._append_log(
                        ctx,
                        f"Reached TARGET_NMOL={self.config.simulation.target_nmol} (nmol={nmol}) after acceptance -> stop this run",
                    )
                    break
            else:
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: rejected move={type_move} "
                    f"E1={e1:.3f} DE={de:.3f} nmol={nmol}",
                    echo=False,
                )

        self._progress(ctx, f"Run complete: accepted={naccepted} final_nmol={nmol}", echo=False)

    def _copy_inputs(self, run_dir: Path) -> None:
        forcefield_dirs = [self.config.paths.forcefield_dir]
        topology_forcefield_dir = self._topology_forcefield_dir()
        if topology_forcefield_dir is not None:
            forcefield_dirs.append(topology_forcefield_dir)

        copied_forcefields: set[Path] = set()
        for forcefield_dir in forcefield_dirs:
            resolved_forcefield = forcefield_dir.resolve()
            if resolved_forcefield in copied_forcefields or not forcefield_dir.exists():
                continue
            copied_forcefields.add(resolved_forcefield)
            shutil.copytree(forcefield_dir, run_dir / forcefield_dir.name, dirs_exist_ok=True)

        sources = [
            self.config.paths.residue_types,
            self.config.paths.topology,
            self.config.paths.water_itp,
            self.config.paths.chk_mdp,
            self.config.paths.steep_mdp,
            self.config.paths.em_mdp,
            self.config.paths.init_gro,
            self.config.paths.gas_gro,
        ]
        molecule_atomtypes = self._molecule_atomtypes_itp()
        if molecule_atomtypes is not None:
            sources.insert(3, molecule_atomtypes)

        copied: set[Path] = set()
        for src in sources:
            self._copy_file_once(src, run_dir / src.name, copied)

        for src, relative_destination in self._topology_local_include_files(copied_forcefields):
            self._copy_file_once(src, run_dir / relative_destination, copied)

    @staticmethod
    def _copy_file_once(src: Path, destination: Path, copied: set[Path]) -> None:
        resolved_destination = destination.resolve()
        if resolved_destination in copied:
            return
        copied.add(resolved_destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, destination)

    def _topology_local_include_files(self, forcefield_dirs: set[Path]) -> list[tuple[Path, Path]]:
        topology_path = self.config.paths.topology
        if not topology_path.exists():
            return []

        copied_forcefield_dirs = {path.resolve() for path in forcefield_dirs if path.exists()}
        found: list[tuple[Path, Path]] = []
        visited: set[Path] = set()

        def visit(source_file: Path, destination_dir: Path) -> None:
            resolved_source = source_file.resolve()
            if resolved_source in visited or not source_file.exists():
                return
            visited.add(resolved_source)

            for raw_line in source_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                match = INCLUDE_RE.match(raw_line)
                if not match:
                    continue
                include_path = Path(match.group("path"))
                if include_path.is_absolute():
                    continue

                resolved_include = self._resolve_include_from_file(source_file, include_path)
                if resolved_include is None or not resolved_include.exists():
                    continue
                if self._is_inside_any(resolved_include, copied_forcefield_dirs):
                    continue

                destination = self._safe_relative_destination(destination_dir / include_path)
                if destination is None:
                    continue
                found.append((resolved_include, destination))
                visit(resolved_include, destination.parent)

        visit(topology_path, Path("."))
        return found

    def _resolve_include_from_file(self, source_file: Path, include_path: Path) -> Path | None:
        candidates = [
            (source_file.parent / include_path).resolve(),
            (self.config.paths.topology.parent / include_path).resolve(),
            (self.config.paths.project_root / include_path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _safe_relative_destination(path: Path) -> Path | None:
        parts: list[str] = []
        for part in path.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                if not parts:
                    return None
                parts.pop()
                continue
            parts.append(part)
        if not parts:
            return None
        return Path(*parts)

    @staticmethod
    def _is_inside_any(path: Path, roots: set[Path]) -> bool:
        resolved = path.resolve()
        for root in roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def _molecule_atomtypes_itp(self) -> Path | None:
        molecule_itp = self.config.paths.water_itp
        candidate = molecule_itp.with_name(f"{molecule_itp.stem}_atomtypes{molecule_itp.suffix}")
        if not candidate.exists():
            return None
        required_atomtypes = self._atomtypes_defined_in_file(candidate)
        if required_atomtypes and self._topology_forcefield_has_atomtypes(required_atomtypes):
            return None
        return candidate

    def _molecule_atomtypes_include_name(self) -> str | None:
        molecule_atomtypes = self._molecule_atomtypes_itp()
        return molecule_atomtypes.name if molecule_atomtypes is not None else None

    def _topology_forcefield_has_atomtypes(self, required_atomtypes: set[str]) -> bool:
        forcefield_dir = self._topology_forcefield_dir()
        if forcefield_dir is None:
            return False
        return required_atomtypes.issubset(self._atomtypes_defined_in_file(forcefield_dir / "ffnonbonded.itp"))

    def _topology_forcefield_dir(self) -> Path | None:
        topology_path = self.config.paths.topology
        if not topology_path.exists():
            return None
        for line in topology_path.read_text(encoding="utf-8").splitlines():
            match = INCLUDE_RE.match(line)
            if not match:
                continue
            include_path = Path(match.group("path"))
            if include_path.name.lower() != "forcefield.itp":
                continue
            resolved = self._resolve_topology_include(include_path)
            if resolved is not None:
                return resolved.parent
        return None

    def _resolve_topology_include(self, include_path: Path) -> Path | None:
        candidates: list[Path] = []
        if include_path.is_absolute():
            candidates.append(include_path)
        else:
            candidates.extend(
                [
                    (self.config.paths.topology.parent / include_path).resolve(),
                    (self.config.paths.project_root / include_path).resolve(),
                ]
            )
            if include_path.parent.name == self.config.paths.forcefield_dir.name:
                candidates.append(self.config.paths.forcefield_dir / include_path.name)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        if not include_path.is_absolute() and include_path.parent.name == self.config.paths.forcefield_dir.name:
            return self.config.paths.forcefield_dir / include_path.name
        return None

    @staticmethod
    def _atomtypes_defined_in_file(path: Path) -> set[str]:
        if not path.exists():
            return set()
        atomtypes: set[str] = set()
        in_atomtypes = False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                in_atomtypes = "".join(line.lower().split()) == "[atomtypes]"
                continue
            if in_atomtypes and not line.startswith("#"):
                fields = line.split()
                if fields:
                    atomtypes.add(fields[0].upper())
        return atomtypes

    def _prepare_runtime_inputs(self) -> None:
        if self.config.cavity.mode != "mask" or not self.config.cavity_build.enabled:
            return
        build = self.config.cavity_build
        seed_point = self._reference_center(self.config.paths.init_gro)
        build_cavity_from_structure(
            self.config.paths.init_gro,
            outprefix=build.output_prefix,
            mode=build.mode,
            dx=build.dx,
            probe_radius=build.probe_radius,
            search_radius=build.search_radius,
            seed_point=seed_point,
            exclude_residues=build.exclude_residues,
            nearby_cutoff=build.nearby_cutoff,
            min_points=build.min_points,
        )
        self.config.cavity.mask_file = build.output_prefix.with_name(f"{build.output_prefix.name}_mask.dat")
        self.config.cavity.mask_meta = build.output_prefix.with_suffix(".meta.json")
        self._mask_cache = None

    def _reference_center(self, gro_or_structure: str | Path | object) -> tuple[float, float, float]:
        return resolve_reference_center(
            gro_or_structure,
            residue_tokens=self.config.anchor.residues,
            reference_mode=self.config.anchor.reference_mode,
            center_atoms=self.config.anchor.center_atoms,
            fallback_resid=self.config.anchor.resid,
            fallback_resname=self.config.anchor.resname,
        )

    def _validate_inputs(self) -> None:
        sim = self.config.simulation
        positive_parameters = {
            "simulation.temperature": sim.temperature,
            "simulation.pressure (activity/fugacity ratio p/p0)": sim.pressure,
            "simulation.gas_constant": sim.gas_constant,
            "simulation.rmax": sim.rmax,
            "simulation.rvdw": sim.rvdw,
            "simulation.v0": sim.v0,
        }
        invalid = [f"{name}={value}" for name, value in positive_parameters.items() if value <= 0]
        if invalid:
            raise ValueError("The following GCMC parameters must be positive: " + ", ".join(invalid))
        if sim.max_trials < 0 or sim.max_e0_tries <= 0 or sim.target_nmol < 0 or sim.mask_dx < 0:
            raise ValueError(
                "simulation.max_trials and target_nmol must be non-negative; "
                "max_e0_tries must be positive; mask_dx must be non-negative"
            )

        required_paths = {
            "forcefield_dir": self.config.paths.forcefield_dir,
            "residue_types": self.config.paths.residue_types,
            "topology": self.config.paths.topology,
            "water_itp": self.config.paths.water_itp,
            "chk_mdp": self.config.paths.chk_mdp,
            "steep_mdp": self.config.paths.steep_mdp,
            "em_mdp": self.config.paths.em_mdp,
            "init_gro": self.config.paths.init_gro,
            "gas_gro": self.config.paths.gas_gro,
        }
        missing = [f"{name}={path}" for name, path in required_paths.items() if not path.exists()]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(
                "Missing required GCMC inputs. Provide at least topol.top and init.gro, or override paths in the config: "
                f"{joined}"
            )
        if self.config.cavity.mode == "mask":
            mask_file = self._resolved_mask_file()
            if mask_file is None or not mask_file.exists():
                raise FileNotFoundError(
                    "Mask mode requires an existing [cavity].mask_file.\n"
                    "Build the cavity first with 'python build_cavity.py ...' or 'python gcmc.py build-cavity ...', "
                    "inspect the outputs, then set [cavity].mode='mask' and point mask_file/mask_meta to the generated files."
                )

    def _initialize_system(self, ctx: WorkflowContext) -> None:
        self._progress(ctx, "Centering the initial image and Removing initial water in the alcove")
        result = self.runner.run_gmx(
            ["editconf", "-f", self.config.paths.init_gro.name, "-o", "starti.gro"],
            cwd=ctx.run_dir,
            log_path="inichk.log",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "starti.gro").exists(), "Initial setup: editconf", ctx)

        result = self.runner.run_gmx(
            ["make_ndx", "-f", "starti.gro", "-o", "index.ndx"],
            cwd=ctx.run_dir,
            log_path="ichec.log",
            stdin_text="q\n",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "index.ndx").exists(), "Initial setup: make_ndx", ctx)

        result = self.runner.run_gmx(
            ["grompp", "-p", "topol.top", "-f", self.config.paths.chk_mdp.name, "-r", "starti.gro", "-c", "starti.gro"],
            cwd=ctx.run_dir,
            log_path="igrompp.log",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "topol.tpr").exists(), "Initial setup: grompp", ctx)

        self._center_initial_image(ctx)
        center = self._reference_center(ctx.run_dir / "tempA.gro")
        remove_waters_near_centroid(
            ctx.run_dir / "tempA.gro",
            ctx.run_dir / "iistart.gro",
            center,
            self.config.simulation.rmax,
        )
        self._require_step((ctx.run_dir / "iistart.gro").exists(), "Initial setup: delete cavity waters", ctx)

        result = self.runner.run_gmx(["editconf", "-f", "iistart.gro", "-o", "startii.gro"], cwd=ctx.run_dir, log_path="inichk.log")
        self._require_step(result.returncode == 0 and (ctx.run_dir / "startii.gro").exists(), "Initial setup: final editconf", ctx)

        previous_top = ctx.previous_top
        shutil.copy2(ctx.run_dir / "topol.top", previous_top)
        initial_waters = count_water_molecules(parse_gro(ctx.run_dir / "tempA.gro"))
        ewat = count_water_molecules(parse_gro(ctx.run_dir / "startii.gro"))
        removed_waters = max(0, initial_waters - ewat)
        self._progress(
            ctx,
            f"Initial setup: cavity waters removed: SOL {initial_waters} -> {ewat} (deleted {removed_waters} water molecules)",
        )
        set_molecule_count(previous_top, "SOL", ewat)
        insert_posre_include(previous_top)

        self._progress(ctx, f"Initial setup: steep minimization with {ewat} waters")
        result = self.runner.run_gmx(
            [
                "grompp",
                "-p",
                previous_top.name,
                "-f",
                self.config.paths.steep_mdp.name,
                "-r",
                "startii.gro",
                "-c",
                "startii.gro",
                "-o",
                "start.tpr",
            ],
            cwd=ctx.run_dir,
            log_path="steep.grompp.log",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "start.tpr").exists(), "Initial setup: steep grompp", ctx)

        result = self.runner.run_gmx(["mdrun", "-deffnm", "start"], cwd=ctx.run_dir, log_path="steep.mdrun.log", multi=True)
        self._require_step(result.returncode == 0 and (ctx.run_dir / "start.gro").exists(), "Initial setup: steep mdrun", ctx)
        shutil.copy2(ctx.run_dir / "start.gro", ctx.previous_gro)
        shutil.copy2(ctx.previous_gro, ctx.current_gro)
        shutil.copy2(ctx.previous_top, ctx.current_top)

    def _initialize_system_mask(self, ctx: WorkflowContext) -> VoxelMask:
        self._progress(ctx, "Centering the initial image and removing initial waters inside the voxel cavity")
        result = self.runner.run_gmx(
            ["editconf", "-f", self.config.paths.init_gro.name, "-o", "starti.gro"],
            cwd=ctx.run_dir,
            log_path="inichk.log",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "starti.gro").exists(), "Initial setup: editconf", ctx)

        result = self.runner.run_gmx(
            ["make_ndx", "-f", "starti.gro", "-o", "index.ndx"],
            cwd=ctx.run_dir,
            log_path="ichec.log",
            stdin_text="q\n",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "index.ndx").exists(), "Initial setup: make_ndx", ctx)

        result = self.runner.run_gmx(
            ["grompp", "-p", "topol.top", "-f", self.config.paths.chk_mdp.name, "-r", "starti.gro", "-c", "starti.gro"],
            cwd=ctx.run_dir,
            log_path="igrompp.log",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "topol.tpr").exists(), "Initial setup: grompp", ctx)

        self._center_initial_image(ctx)
        initial_mask, shift = align_voxel_mask_to_structure(self._mask_model(), ctx.run_dir / "tempA.gro")
        shift_norm = sqrt(sum(component * component for component in shift))
        if shift_norm > 1.0e-6:
            self._progress(
                ctx,
                f"Aligned cavity mask to the centered structure: shift=({shift[0]:.4f}, {shift[1]:.4f}, {shift[2]:.4f}) nm",
            )
        delete_mask = self._initial_delete_mask(initial_mask)
        removed_waters = remove_waters_in_mask(ctx.run_dir / "tempA.gro", ctx.run_dir / "iistart.gro", delete_mask)
        self._require_step((ctx.run_dir / "iistart.gro").exists(), "Initial setup: delete cavity waters", ctx)

        result = self.runner.run_gmx(["editconf", "-f", "iistart.gro", "-o", "startii.gro"], cwd=ctx.run_dir, log_path="inichk.log")
        self._require_step(result.returncode == 0 and (ctx.run_dir / "startii.gro").exists(), "Initial setup: final editconf", ctx)

        previous_top = ctx.previous_top
        shutil.copy2(ctx.run_dir / "topol.top", previous_top)
        initial_waters = count_water_molecules(parse_gro(ctx.run_dir / "tempA.gro"))
        ewat = count_water_molecules(parse_gro(ctx.run_dir / "startii.gro"))
        self._progress(
            ctx,
            f"Initial setup: cavity waters removed: SOL {initial_waters} -> {ewat} (deleted {removed_waters} water molecules)",
        )
        if self.config.cavity.initial_delete_padding > 0:
            self._progress(
                ctx,
                f"Initial setup: voxel cleanup used membership_padding={delete_mask.membership_padding:.4f} nm",
            )
        set_molecule_count(previous_top, "SOL", ewat)
        insert_posre_include(previous_top)

        self._progress(ctx, f"Initial setup: steep minimization with {ewat} waters")
        result = self.runner.run_gmx(
            [
                "grompp",
                "-p",
                previous_top.name,
                "-f",
                self.config.paths.steep_mdp.name,
                "-r",
                "startii.gro",
                "-c",
                "startii.gro",
                "-o",
                "start.tpr",
            ],
            cwd=ctx.run_dir,
            log_path="steep.grompp.log",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "start.tpr").exists(), "Initial setup: steep grompp", ctx)

        result = self.runner.run_gmx(["mdrun", "-deffnm", "start"], cwd=ctx.run_dir, log_path="steep.mdrun.log", multi=True)
        self._require_step(result.returncode == 0 and (ctx.run_dir / "start.gro").exists(), "Initial setup: steep mdrun", ctx)
        shutil.copy2(ctx.run_dir / "start.gro", ctx.previous_gro)
        shutil.copy2(ctx.previous_gro, ctx.current_gro)
        shutil.copy2(ctx.previous_top, ctx.current_top)
        return self._runtime_mask_for_structure(ctx.previous_gro)

    def _run_single_mask(self, run_spec: RunSpec) -> None:
        run_dir = run_spec.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        self._copy_inputs(run_dir)

        topol = run_dir / "topol.top"
        merge_wat_into_sol(topol)
        ensure_molecule_include(topol, self.config.paths.water_itp.name, self._molecule_atomtypes_include_name())

        ctx = WorkflowContext(
            run_dir=run_dir,
            log_file=run_dir / "mc.log",
            progress_file=run_dir / "workflow.log",
            positions_file=run_dir / "trials.xyz",
            mask_trajectory_file=run_dir / "cavity_trajectory.gro",
            previous_top=run_dir / "previous.top",
            current_top=run_dir / "current.top",
            previous_gro=run_dir / "previous.gro",
            current_gro=run_dir / "current.gro",
            gas_name=parse_gro(self.config.paths.gas_gro).atoms[0].resname,
            na_gas=len(parse_gro(self.config.paths.gas_gro).atoms),
            orig_atom_count=0,
        )

        rng = random.Random()
        if run_spec.sweep_value is not None:
            seed_hint = run_spec.sweep_value * 100 + int(run_spec.replica_label)
            rng.seed(seed_hint)

        envelope_mask = self._mask_model()
        self._prepare_logs(ctx)
        self._progress(
            ctx,
            f"Run start: run_dir={ctx.run_dir} replica={run_spec.replica_label}"
            + (f" sweep={run_spec.sweep_value}" if run_spec.sweep_value is not None else ""),
        )
        self._progress(ctx, SCIENTIFIC_SCOPE_NOTICE)
        self._progress(ctx, self._mask_summary(envelope_mask, prefix="GCMC envelope"))
        self._progress(ctx, f"{ctx.gas_name} chemical potential = {self._chemical_potential():.2f}")

        mask = self._initialize_system_mask(ctx)
        self._require_accessible_mask(mask, phase="Initial setup", ctx=ctx)
        ctx.orig_atom_count = len(parse_gro(run_dir / "start.gro").atoms)
        self._progress(ctx, f"Initial setup complete: start_atoms={ctx.orig_atom_count}")

        self._populate_restraint_shell(
            ctx.previous_gro,
            out_dir=ctx.run_dir,
            center=mask.reference_point,
            radius=self.config.cavity.restraint_radius,
        )
        write_position_restraints(ctx.previous_gro, self.config.simulation.kres, out_dir=ctx.run_dir)

        e0 = self._initial_reference_energy(ctx)
        mask = self._runtime_mask_for_structure(ctx.previous_gro)
        self._require_accessible_mask(mask, phase="Reference EM", ctx=ctx)
        self._progress(ctx, f"Reference energy established: E0={e0:.3f}")
        self._progress(ctx, self._mask_summary(mask, prefix="Active cavity"))
        naccepted = 0
        cavity_resids_initial = molecule_residue_ids_in_mask(ctx.previous_gro, mask, ctx.gas_name)
        provenance: dict[int, str] = {resid: f"initial:{resid}" for resid in cavity_resids_initial}
        nmol = len(cavity_resids_initial)
        mu_tp = self._chemical_potential()
        consecutive_insertion_point_failures = 0
        self._write_mask_frame(ctx, mask, accepted=0, nmol=nmol, label="start")

        self._write_log_header(ctx)
        for trial in range(1, self.config.simulation.max_trials + 1):
            trial_started_at = perf_counter()
            if self.config.simulation.target_nmol > 0 and nmol >= self.config.simulation.target_nmol:
                self._append_log(ctx, f"Reached TARGET_NMOL={self.config.simulation.target_nmol} (nmol={nmol}) -> stop this run")
                break

            self._populate_restraint_shell(
                ctx.previous_gro,
                out_dir=ctx.run_dir,
                center=mask.reference_point,
                radius=self.config.cavity.restraint_radius,
            )
            cavity_resids = molecule_residue_ids_in_mask(ctx.previous_gro, mask, ctx.gas_name)
            nins = len(cavity_resids)
            veff = mask.effective_volume
            move = self._select_move(nins, rng)
            move_name = type_move_name(move)
            self._progress(
                ctx,
                f"MC {trial}/{self.config.simulation.max_trials}: move={move_name} cavity_n={nins} accepted={naccepted}",
                echo=False,
            )
            moved_resid: int | None = None
            try:
                if move == 1:
                    propose_insertion(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        rvdw=self.config.simulation.rvdw,
                        gas_name=ctx.gas_name,
                        gas_gro=self.config.paths.gas_gro,
                        rmax=self.config.simulation.rmax,
                        xyz_path=ctx.positions_file,
                        out_dir=ctx.run_dir,
                        mask_model=mask,
                        mask_dx=self._mask_proposal_jitter(),
                        seed_point=mask.reference_point,
                        rng=rng,
                    )
                    consecutive_insertion_point_failures = 0
                    moved_resid = parse_gro(ctx.current_gro).atoms[-1].resid
                    type_move = "I"
                    mu = mu_tp
                elif move == 2:
                    moved_resid = propose_rotation(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        nmol=max(1, nins),
                        gas_name=ctx.gas_name,
                        orig_atom_count=ctx.orig_atom_count,
                        rvdw=self.config.simulation.rvdw,
                        candidate_resids=cavity_resids,
                        rng=rng,
                    )
                    type_move = "R"
                    mu = 0.0
                elif move == 3:
                    moved_resid = propose_translation(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        nmol=max(1, nins),
                        gas_name=ctx.gas_name,
                        orig_atom_count=ctx.orig_atom_count,
                        rvdw=self.config.simulation.rvdw,
                        candidate_resids=cavity_resids,
                        rng=rng,
                    )
                    type_move = "T"
                    mu = 0.0
                else:
                    moved_resid = propose_deletion(
                        ctx.previous_gro,
                        ctx.current_gro,
                        ctx.previous_top,
                        ctx.current_top,
                        nmol=max(1, nins + 1),
                        gas_name=ctx.gas_name,
                        orig_atom_count=ctx.orig_atom_count,
                        out_dir=ctx.run_dir,
                        candidate_resids=cavity_resids,
                        rng=rng,
                    )
                    type_move = "D"
                    mu = -mu_tp
            except Exception as exc:  # noqa: BLE001
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: {move_name} proposal failed ({exc})",
                    echo=False,
                )
                self._append_log(ctx, f"  {move_name} proposal failed on trial {trial}: {exc}")
                if self._is_insertion_point_failure(move, exc):
                    consecutive_insertion_point_failures += 1
                    if self._should_stop_after_insertion_point_failures(
                        ctx,
                        trial=trial,
                        consecutive_failures=consecutive_insertion_point_failures,
                    ):
                        break
                elif move == 1:
                    consecutive_insertion_point_failures = 0
                continue

            self._progress(
                ctx,
                f"MC {trial}/{self.config.simulation.max_trials}: {move_name} proposed, running EM",
                echo=False,
            )
            write_position_restraints(ctx.current_gro, self.config.simulation.kres, out_dir=ctx.run_dir)
            e1 = self._relax_and_center(ctx, current=True)
            if e1 is None:
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: {move_name} EM failed",
                    echo=False,
                )
                self._append_log(ctx, f"  simulation failed on trial {trial} (move={type_move})")
                continue
            hard_clash = self._post_em_clash(ctx.run_dir / "confout.gro", moved_resid, type_move=type_move)
            if hard_clash is not None:
                details = describe_clash(hard_clash)
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: hard-clash reject after EM ({details})",
                    echo=False,
                )
                self._append_log(ctx, f"  hard-clash reject on trial {trial} (move={type_move}, {details})")
                continue

            after_mask = self._runtime_mask_for_structure(ctx.run_dir / "confout.gro")
            n_after = len(molecule_residue_ids_in_mask(ctx.run_dir / "confout.gro", after_mask, ctx.gas_name))
            expected_n = nins + {1: 1, 2: 0, 3: 0, 4: -1}[move]
            if n_after != expected_n:
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: rejected by cavity membership before={nins} after={n_after} expected={expected_n}",
                    echo=False,
                )
                self._append_log(
                    ctx,
                    f"  cavity-membership mismatch on trial {trial} (move={type_move}, before={nins}, after={n_after}, expected={expected_n})",
                )
                continue

            de = e1 - (e0 + mu)
            acceptance = acceptance_probability(
                de=de,
                temperature=self.config.simulation.temperature,
                move=move,
                veff=veff,
                v0=self.config.simulation.v0,
                nins=nins,
                rng=rng,
                gas_constant=self.config.simulation.gas_constant,
            )
            resname, resid, info = self._move_log_context(
                type_move=type_move,
                moved_resid=moved_resid,
                confout_gro=ctx.run_dir / "confout.gro",
                x_gro=ctx.run_dir / "x.gro",
                previous_gro=ctx.previous_gro,
                gas_name=ctx.gas_name,
                provenance=provenance,
            )
            elapsed_seconds = int(perf_counter() - trial_started_at)
            line = (
                f"{trial:5d} {naccepted:4d} {type_move} | "
                f"{self._fmt_energy(e1, de):>8s} {e0:8.3f} {self._fmt_delta(de):>6s} | "
                f"{resname:>3s} {resid:>5s} {info:>5s} | "
                f"{elapsed_seconds:4d} {self._acceptance_label(acceptance):>4s} | {nins:4d}"
            )
            self._append_log(ctx, line)

            if acceptance > 0:
                shutil.copyfile(ctx.run_dir / "confout.gro", ctx.previous_gro)
                shutil.copyfile(ctx.current_top, ctx.previous_top)
                e0 = e1
                naccepted += 1
                if move == 1 and moved_resid is not None:
                    provenance[moved_resid] = f"t{trial:03d}"
                elif move == 4 and moved_resid is not None:
                    provenance.pop(moved_resid, None)
                mask = self._runtime_mask_for_structure(ctx.previous_gro)
                self._require_accessible_mask(mask, phase=f"Accepted move trial {trial}", ctx=ctx)
                active_resids = molecule_residue_ids_in_mask(ctx.previous_gro, mask, ctx.gas_name)
                nmol = len(active_resids)
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: ACCEPTED move={type_move} "
                    f"E1={e1:.3f} DE={de:.3f} nmol={nmol} accepted={naccepted}",
                    echo=False,
                )
                self._write_mask_frame(ctx, mask, accepted=naccepted, nmol=nmol, label=f"trial={trial}")
                shutil.copyfile(ctx.run_dir / "posre_cavity.itp", ctx.run_dir / f"posre_cavity-{nmol}.itp")
                shutil.copyfile(ctx.run_dir / "confout.gro", ctx.run_dir / f"confout-{nmol}.gro")
                write_trajectory(
                    ctx.run_dir / "confout.gro",
                    e0,
                    naccepted,
                    nmol,
                    self.config.simulation.target_nmol,
                    trajectory_path=ctx.run_dir / "trajectory.gro",
                    gas_gro=self.config.paths.gas_gro,
                    trial=trial,
                    move=type_move,
                    active_resids=active_resids,
                    provenance=provenance,
                    trajectory_meta_path=ctx.run_dir / "trajectory.meta.jsonl",
                )
                if self.config.simulation.target_nmol > 0 and nmol >= self.config.simulation.target_nmol:
                    self._append_log(
                        ctx,
                        f"Reached TARGET_NMOL={self.config.simulation.target_nmol} (nmol={nmol}) after acceptance -> stop this run",
                    )
                    break
            else:
                self._progress(
                    ctx,
                    f"MC {trial}/{self.config.simulation.max_trials}: rejected move={type_move} "
                    f"E1={e1:.3f} DE={de:.3f} nmol={nmol}",
                    echo=False,
                )

        self._progress(ctx, f"Run complete: accepted={naccepted} final_nmol={nmol}", echo=False)

    def _prepare_logs(self, ctx: WorkflowContext) -> None:
        self._cleanup_gromacs_artifacts(ctx.run_dir)
        for path in (
            ctx.log_file,
            ctx.progress_file,
            ctx.positions_file,
            ctx.mask_trajectory_file,
            ctx.run_dir / "trajectory.gro",
            ctx.run_dir / "trajectory.meta.jsonl",
        ):
            if path.exists():
                path.unlink()

    def _record_run_failure(self, run_spec: RunSpec, exc: Exception) -> None:
        run_spec.run_dir.mkdir(parents=True, exist_ok=True)
        progress_path = run_spec.run_dir / "workflow.log"
        stamped = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Run failed and will be skipped: run_dir={run_spec.run_dir} "
            f"replica={run_spec.replica_label}"
            + (f" sweep={run_spec.sweep_value}" if run_spec.sweep_value is not None else "")
            + f" error={exc}"
        )
        if self.echo_progress:
            print(stamped, flush=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    def _write_log_header(self, ctx: WorkflowContext) -> None:
        header = "\n Step  Acc   |       E1       E0     DE | Res    id  Info | Time Stat | nins"
        ctx.log_file.write_text(header + "\n", encoding="utf-8")
        if self.echo_progress:
            print(header, flush=True)

    def _append_log(self, ctx: WorkflowContext, line: str) -> None:
        if self.echo_progress:
            print(line, flush=True)
        with ctx.log_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _progress(self, ctx: WorkflowContext, line: str, *, echo: bool | None = None) -> None:
        stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}"
        should_echo = self.echo_progress if echo is None else echo
        if should_echo:
            print(stamped, flush=True)
        with ctx.progress_file.open("a", encoding="utf-8") as handle:
            handle.write(stamped + "\n")

    def _require_step(self, condition: bool, label: str, ctx: WorkflowContext) -> None:
        if condition:
            return
        self._progress(ctx, f"{label}: FAILED")
        raise RuntimeError(f"{label} failed; see logs in {ctx.run_dir}")

    def _is_insertion_point_failure(self, move: int, exc: Exception) -> bool:
        return move == 1 and str(exc).strip() == NO_VALID_INSERTION_POINT_ERROR

    def _should_stop_after_insertion_point_failures(
        self,
        ctx: WorkflowContext,
        *,
        trial: int,
        consecutive_failures: int,
    ) -> bool:
        limit = int(self.config.simulation.max_consecutive_insertion_failures)
        if limit <= 0:
            return False
        self._progress(
            ctx,
            (
                "Insertion-point failure streak: "
                f"{consecutive_failures}/{limit} consecutive insertion proposals returned "
                f"'{NO_VALID_INSERTION_POINT_ERROR}'"
            ),
            echo=False,
        )
        if consecutive_failures < limit:
            return False
        message = (
            "Stopping run early after "
            f"{consecutive_failures} consecutive insertion proposal failures with "
            f"'{NO_VALID_INSERTION_POINT_ERROR}' at trial {trial}. "
            "This usually means the current cavity is effectively saturated for the present proposal settings."
        )
        self._append_log(ctx, message)
        self._progress(ctx, message, echo=False)
        return True

    def _chemical_potential(self) -> float:
        sim = self.config.simulation
        if sim.temperature <= 0:
            raise ValueError(f"simulation.temperature must be positive, got {sim.temperature}")
        if sim.gas_constant <= 0:
            raise ValueError(f"simulation.gas_constant must be positive, got {sim.gas_constant}")
        if sim.pressure <= 0:
            raise ValueError(
                "simulation.pressure is used as the dimensionless activity/fugacity ratio p/p0 "
                f"and must be positive, got {sim.pressure}"
            )
        return sim.mu0 + sim.gas_constant * sim.temperature * log(sim.pressure)

    def _effective_volume(self) -> float:
        if self.config.cavity.mode == "mask":
            return self._mask_model().effective_volume
        radius = self._insertion_radius()
        return (4.0 / 3.0) * pi * radius ** 3

    def _inserted_residue_ids_in_sphere(self, gro_path: str | Path, ctx: WorkflowContext) -> list[int]:
        structure = parse_gro(gro_path)
        center = self._reference_center(structure)
        radius2 = self._insertion_radius_squared()
        gas_name = ctx.gas_name.upper()
        water_like = gas_name in WATER_NAMES
        residue_ids: list[int] = []
        for atoms in contiguous_residue_groups(structure):
            resname = atoms[0].resname.upper()
            target = resname in WATER_NAMES if water_like else resname == gas_name
            if not target or not any(atom.atomnr > ctx.orig_atom_count for atom in atoms):
                continue
            if water_like:
                marker = next((atom for atom in atoms if atom.atomname.upper() in {"OW", "O"}), atoms[0])
                point = (marker.x, marker.y, marker.z)
            else:
                point = coordinates_center(atoms)
            distance2 = sum((point[index] - center[index]) ** 2 for index in range(3))
            if distance2 <= radius2:
                residue_ids.append(atoms[0].resid)
        return residue_ids

    def _insertion_radius(self) -> float:
        return self.config.simulation.rmax

    def _insertion_radius_squared(self) -> float:
        radius = self._insertion_radius()
        return radius * radius

    def _resolved_mask_meta(self) -> Path | None:
        if self.config.cavity.mask_meta:
            return self.config.cavity.mask_meta
        mask_file = self._resolved_mask_file()
        if mask_file is None:
            return None
        candidate = mask_file.with_name(mask_file.name[:-9] + ".meta.json") if mask_file.name.endswith("_mask.dat") else None
        if candidate and candidate.exists():
            return candidate
        return None

    def _mask_model(self) -> VoxelMask:
        if self._mask_cache is None:
            mask_file = self._resolved_mask_file()
            if mask_file is None:
                raise RuntimeError("Mask mode requested without a cavity mask file")
            self._mask_cache = load_voxel_mask(
                mask_file,
                self._resolved_mask_meta(),
                membership_padding=self.config.cavity.membership_padding,
            )
        return self._mask_cache

    def _initial_delete_mask(self, mask: VoxelMask) -> VoxelMask:
        extra_padding = self.config.cavity.initial_delete_padding
        if extra_padding <= 0:
            return mask
        return mask.with_membership_padding(mask.membership_padding + extra_padding)

    def _runtime_mask_for_structure(self, gro_or_structure: str | Path | object) -> VoxelMask:
        base_mask, _ = align_voxel_mask_to_structure(self._mask_model(), gro_or_structure)
        return base_mask

    def _require_accessible_mask(self, mask: VoxelMask, *, phase: str, ctx: WorkflowContext) -> None:
        if mask.point_count > 0:
            return
        details = (
            f"{phase}: active cavity is empty. "
            "The loaded voxel mask has no accessible points after alignment to the current structure."
        )
        self._progress(ctx, details)
        guidance = (
            "Rebuild the cavity with "
            "'python build_cavity.py ... --exclude-residue ...', "
            "inspect *_points.pdb / *.meta.json, and make sure the selected cavity actually encloses the intended pocket."
        )
        raise RuntimeError(f"{details} {guidance}")

    def _mask_summary(self, mask: VoxelMask, *, prefix: str) -> str:
        return (
            f"{prefix}: dx={mask.dx:.5f} nm  "
            f"points={mask.point_count}  "
            f"Veff={mask.effective_volume:.10f} nm^3  "
            f"ref=({mask.reference_point[0]:.4f}, {mask.reference_point[1]:.4f}, {mask.reference_point[2]:.4f})"
        )

    def _write_mask_frame(self, ctx: WorkflowContext, mask: VoxelMask, *, accepted: int, nmol: int, label: str) -> None:
        write_mask_trajectory(
            mask,
            ctx.previous_gro,
            trajectory_path=ctx.mask_trajectory_file,
            accepted=accepted,
            nmol=nmol,
            label=label,
        )

    def _mask_reference_point(self) -> tuple[float, float, float]:
        return self._mask_model().reference_point

    def _mask_proposal_jitter(self) -> float:
        if self.config.simulation.mask_dx > 0:
            return self.config.simulation.mask_dx
        return self._mask_model().dx

    def _populate_restraint_shell(
        self,
        gro_path: str | Path,
        *,
        out_dir: str | Path,
        center: tuple[float, float, float],
        radius: float,
    ) -> None:
        get_alcove_residues(gro_path, radius, center, 1.0, out_dir=out_dir)

    def _initial_reference_energy(self, ctx: WorkflowContext) -> float:
        if self.config.cavity.mode == "mask":
            return self._initial_reference_energy_mask(ctx)

        e0: float | None = None
        for attempt in range(1, self.config.simulation.max_e0_tries + 1):
            self._progress(ctx, f"Reference EM {attempt}/{self.config.simulation.max_e0_tries}: running em.mdp")
            e0 = self._relax_and_center(ctx, current=False)
            if e0 is not None:
                shutil.copy2(ctx.run_dir / "confout.gro", ctx.run_dir / "confout-0.gro")
                shutil.copy2(ctx.run_dir / "confout.gro", ctx.previous_gro)
                shutil.copy2(ctx.run_dir / "posre_cavity.itp", ctx.run_dir / "posre_cavity-0.itp")
                self._progress(
                    ctx,
                    f"Reference EM {attempt}/{self.config.simulation.max_e0_tries}: success E0={e0:.3f}",
                )
                return e0
            self._progress(ctx, f"Reference EM {attempt}/{self.config.simulation.max_e0_tries}: failed")
        raise RuntimeError("Could not obtain valid E0")

    def _initial_reference_energy_mask(self, ctx: WorkflowContext) -> float:
        e0: float | None = None
        for cleanup_cycle in range(1, MAX_MASK_REFERENCE_CLEANUP_CYCLES + 1):
            e0 = None
            for attempt in range(1, self.config.simulation.max_e0_tries + 1):
                self._progress(
                    ctx,
                    (
                        f"Reference EM cleanup cycle {cleanup_cycle}/{MAX_MASK_REFERENCE_CLEANUP_CYCLES}, "
                        f"attempt {attempt}/{self.config.simulation.max_e0_tries}: running em.mdp"
                    ),
                )
                e0 = self._relax_and_center(ctx, current=False)
                if e0 is not None:
                    break
                self._progress(
                    ctx,
                    (
                        f"Reference EM cleanup cycle {cleanup_cycle}/{MAX_MASK_REFERENCE_CLEANUP_CYCLES}, "
                        f"attempt {attempt}/{self.config.simulation.max_e0_tries}: failed"
                    ),
                )
            if e0 is None:
                raise RuntimeError("Could not obtain valid E0")

            trial_gro = ctx.run_dir / "confout.gro"
            mask = self._runtime_mask_for_structure(trial_gro)
            cavity_resids = water_residue_ids_in_mask(trial_gro, mask)
            if not cavity_resids:
                shutil.copy2(trial_gro, ctx.run_dir / "confout-0.gro")
                shutil.copy2(trial_gro, ctx.previous_gro)
                shutil.copy2(ctx.run_dir / "posre_cavity.itp", ctx.run_dir / "posre_cavity-0.itp")
                self._progress(
                    ctx,
                    (
                        f"Reference EM cleanup cycle {cleanup_cycle}/{MAX_MASK_REFERENCE_CLEANUP_CYCLES}: "
                        f"success E0={e0:.3f} with empty cavity"
                    ),
                )
                return e0

            if cleanup_cycle >= MAX_MASK_REFERENCE_CLEANUP_CYCLES:
                raise RuntimeError(
                    "Could not obtain an empty reference cavity after "
                    f"{MAX_MASK_REFERENCE_CLEANUP_CYCLES} cleanup cycles; "
                    f"{len(cavity_resids)} water(s) remain in the mask"
                )

            self._progress(
                ctx,
                (
                    f"Reference EM cleanup cycle {cleanup_cycle}/{MAX_MASK_REFERENCE_CLEANUP_CYCLES}: "
                    f"{len(cavity_resids)} water(s) entered the voxel cavity after EM; deleting and retrying"
                ),
            )
            self._reset_reference_after_mask_cleanup(ctx, trial_gro)

        raise RuntimeError("Could not obtain valid E0")

    def _reset_reference_after_mask_cleanup(self, ctx: WorkflowContext, source_gro: Path) -> None:
        cleanup_mask = self._initial_delete_mask(self._runtime_mask_for_structure(source_gro))
        removed_waters = remove_waters_in_mask(source_gro, ctx.previous_gro, cleanup_mask)
        if removed_waters <= 0:
            raise RuntimeError("Reference cleanup requested, but no cavity waters were removed")

        remaining_waters = count_water_molecules(parse_gro(ctx.previous_gro))
        set_molecule_count(ctx.previous_top, "SOL", remaining_waters)
        insert_posre_include(ctx.previous_top)
        shutil.copy2(ctx.previous_gro, ctx.current_gro)
        shutil.copy2(ctx.previous_top, ctx.current_top)

        refreshed_mask = self._runtime_mask_for_structure(ctx.previous_gro)
        self._require_accessible_mask(refreshed_mask, phase="Reference cleanup", ctx=ctx)
        self._populate_restraint_shell(
            ctx.previous_gro,
            out_dir=ctx.run_dir,
            center=refreshed_mask.reference_point,
            radius=self.config.cavity.restraint_radius,
        )
        write_position_restraints(ctx.previous_gro, self.config.simulation.kres, out_dir=ctx.run_dir)
        self._progress(
            ctx,
            (
                "Reference cleanup: "
                f"removed {removed_waters} cavity water(s), remaining SOL={remaining_waters}"
            ),
        )

    def _relax_and_center(self, ctx: WorkflowContext, *, current: bool) -> float | None:
        gro_name = ctx.current_gro.name if current else ctx.previous_gro.name
        top_name = ctx.current_top.name if current else ctx.previous_top.name
        # E0 and E1 must be generated with the same MDP/Hamiltonian.  chk.mdp
        # remains an initialization/centering input; all scored states use
        # em.mdp so their reported potential energies are at least comparable.
        mdp_name = self.config.paths.em_mdp.name
        self._cleanup_gromacs_artifacts(ctx.run_dir)

        result = self.runner.run_gmx(
            ["grompp", "-p", top_name, "-f", mdp_name, "-r", gro_name, "-c", gro_name],
            cwd=ctx.run_dir,
            log_path="grompp.log",
        )
        if result.returncode != 0 or not (ctx.run_dir / "topol.tpr").exists():
            return None

        result = self.runner.run_gmx(["mdrun", "-g", "md.log"], cwd=ctx.run_dir, log_path="mdrun.stdout", multi=True)
        if (
            result.returncode != 0
            or not (ctx.run_dir / "md.log").exists()
            or not (ctx.run_dir / "confout.gro").exists()
        ):
            return None
        energy = parse_potential_energy(ctx.run_dir / "md.log")
        if energy is None:
            return None

        center_group = self._prepare_centering_index(ctx, "confout.gro", log_path="chec.log")
        if center_group is None:
            return None
        result1 = self.runner.run_gmx(
            [
                "trjconv",
                "-f",
                "confout.gro",
                "-s",
                "topol.tpr",
                "-o",
                "test.gro",
                "-n",
                "index.ndx",
                "-pbc",
                "cluster",
                "-ur",
                "compact",
                "-boxcenter",
                "tric",
                "-trans",
                "0",
                "0",
                "0",
            ],
            cwd=ctx.run_dir,
            log_path="trjconf.log",
            stdin_text=f"{center_group}\n0\n",
        )
        result2 = self.runner.run_gmx(
            [
                "trjconv",
                "-f",
                "test.gro",
                "-s",
                "topol.tpr",
                "-o",
                "confout.gro",
                "-n",
                "index.ndx",
                "-pbc",
                "mol",
                "-ur",
                "compact",
                "-boxcenter",
                "tric",
                "-trans",
                "0",
                "0",
                "0",
            ],
            cwd=ctx.run_dir,
            log_path="trjconf.log",
            stdin_text="0\n",
        )
        if result1.returncode != 0 or result2.returncode != 0 or not (ctx.run_dir / "confout.gro").exists():
            return None
        return energy

    def _cleanup_gromacs_artifacts(self, run_dir: Path) -> None:
        transient_names = (
            "topol.tpr",
            "md.log",
            "confout.gro",
            "test.gro",
            "tempA.gro",
            "index.ndx",
            "traj.trr",
            "ener.edr",
            "state.cpt",
            "state_prev.cpt",
            "mdout.mdp",
        )
        for name in transient_names:
            path = run_dir / name
            if path.exists():
                path.unlink()
        for backup in run_dir.glob("#*"):
            if backup.is_file():
                backup.unlink()

    def _prepare_centering_index(self, ctx: WorkflowContext, gro_name: str, *, log_path: str) -> int | None:
        result = self.runner.run_gmx(
            ["make_ndx", "-f", gro_name, "-o", "index.ndx"],
            cwd=ctx.run_dir,
            log_path=log_path,
            stdin_text="q\n",
        )
        index_path = ctx.run_dir / "index.ndx"
        if result.returncode != 0 or not index_path.exists():
            self._progress(ctx, f"Could not create centering index for {gro_name}")
            return None

        atom_numbers = self._centering_atom_numbers(ctx.run_dir / gro_name, ctx.gas_name)
        if not atom_numbers:
            self._progress(ctx, f"Could not find solute atoms for centering in {gro_name}")
            return None
        group_number = self._append_index_group(index_path, CENTERING_GROUP_NAME, atom_numbers)
        self._progress(
            ctx,
            f"Centering group {CENTERING_GROUP_NAME}={group_number} contains {len(atom_numbers)} atoms from {gro_name}",
            echo=False,
        )
        return group_number

    def _centering_atom_numbers(self, gro_path: str | Path, gas_name: str) -> list[int]:
        structure = parse_gro(gro_path)
        categories = self._load_residue_categories()
        gas_resname = gas_name.upper()

        first_water_index = next(
            (
                index
                for index, atom in enumerate(structure.atoms)
                if self._residue_category(atom.resname, categories) in SOLVENT_CATEGORIES
                or atom.resname.upper() in WATER_NAMES
            ),
            None,
        )
        if first_water_index is not None:
            selected = [
                atom.atomnr
                for atom in structure.atoms[:first_water_index]
                if atom.resname.upper() != gas_resname
            ]
            if selected:
                return selected

        selected = [
            atom.atomnr
            for atom in structure.atoms
            if atom.resname.upper() != gas_resname
            and self._residue_category(atom.resname, categories) not in CENTERING_FALLBACK_SKIP_CATEGORIES
            and atom.resname.upper() not in WATER_NAMES
        ]
        if selected:
            return selected

        return [
            atom.atomnr
            for atom in structure.atoms
            if atom.resname.upper() != gas_resname and atom.resname.upper() not in WATER_NAMES
        ]

    def _load_residue_categories(self) -> dict[str, str]:
        if self._residue_categories is not None:
            return self._residue_categories

        categories: dict[str, str] = {}
        path = self.config.paths.residue_types
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(";") or stripped.startswith("#"):
                    continue
                fields = stripped.split()
                if len(fields) >= 2:
                    categories[fields[0].upper()] = fields[1].strip().lower()
        self._residue_categories = categories
        return categories

    @staticmethod
    def _residue_category(resname: str, categories: dict[str, str]) -> str:
        return categories.get(resname.upper(), "")

    @staticmethod
    def _append_index_group(index_path: Path, group_name: str, atom_numbers: list[int]) -> int:
        lines = index_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        group_number = sum(1 for line in lines if line.strip().startswith("["))
        lines.append(f"[ {group_name} ]")
        for start in range(0, len(atom_numbers), 15):
            lines.append(" ".join(str(atomnr) for atomnr in atom_numbers[start : start + 15]))
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return group_number

    def _center_initial_image(self, ctx: WorkflowContext) -> None:
        primary_output = ctx.run_dir / "tempA.gro"
        fallback_output = ctx.run_dir / "tempB.gro"
        self._progress(ctx, ">> Centering attempt 1 (cluster/mol)")
        center_group = self._prepare_centering_index(ctx, "starti.gro", log_path="center_make_ndx.log")
        self._require_step(center_group is not None, "Initial setup: centering index", ctx)
        result = self.runner.run_gmx(
            [
                "trjconv",
                "-f",
                "starti.gro",
                "-s",
                "topol.tpr",
                "-o",
                "test.gro",
                "-n",
                "index.ndx",
                "-pbc",
                "cluster",
                "-ur",
                "compact",
                "-boxcenter",
                "tric",
                "-trans",
                "0",
                "0",
                "0",
            ],
            cwd=ctx.run_dir,
            log_path="itrjconf.log",
            stdin_text=f"{center_group}\n0\n",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "test.gro").exists(), "Initial setup: cluster trjconv", ctx)

        result = self.runner.run_gmx(
            [
                "trjconv",
                "-f",
                "test.gro",
                "-s",
                "topol.tpr",
                "-o",
                "tempA.gro",
                "-n",
                "index.ndx",
                "-pbc",
                "mol",
                "-ur",
                "compact",
                "-boxcenter",
                "tric",
                "-trans",
                "0",
                "0",
                "0",
            ],
            cwd=ctx.run_dir,
            log_path="itrjconf.log",
            stdin_text="0\n",
        )
        self._require_step(result.returncode == 0 and primary_output.exists(), "Initial setup: compact trjconv", ctx)

        primary_fraction = self._check_centering_ok(ctx, primary_output.name, qc_log="qc_trjconv.log")
        self._progress(ctx, f">> Centering attempt 1 boundary fraction = {primary_fraction:.7g}")
        if primary_fraction <= CENTERING_QC_TARGET_FRACTION:
            self._progress(ctx, ">> Centering OK on attempt 1. Continue.")
            return

        self._progress(ctx, ">> Attempt 1 failed: fallback to nojump/center")
        result = self.runner.run_gmx(
            [
                "trjconv",
                "-s",
                "topol.tpr",
                "-f",
                "starti.gro",
                "-o",
                "nojump.gro",
                "-n",
                "index.ndx",
                "-pbc",
                "nojump",
            ],
            cwd=ctx.run_dir,
            log_path="itrjconf_fallback.log",
            stdin_text="0\n",
        )
        self._require_step(result.returncode == 0 and (ctx.run_dir / "nojump.gro").exists(), "Initial setup: nojump trjconv", ctx)

        fallback_group = self._prepare_centering_index(ctx, "nojump.gro", log_path="center_make_ndx_fallback.log")
        self._require_step(fallback_group is not None, "Initial setup: fallback centering index", ctx)
        result = self.runner.run_gmx(
            [
                "trjconv",
                "-s",
                "topol.tpr",
                "-f",
                "nojump.gro",
                "-o",
                fallback_output.name,
                "-n",
                "index.ndx",
                "-pbc",
                "mol",
                "-center",
                "-ur",
                "compact",
            ],
            cwd=ctx.run_dir,
            log_path="itrjconf_fallback.log",
            stdin_text=f"{fallback_group}\n0\n",
        )
        self._require_step(result.returncode == 0 and fallback_output.exists(), "Initial setup: fallback centering", ctx)

        fallback_fraction = self._check_centering_ok(ctx, fallback_output.name, qc_log="qc_trjconv_fallback.log")
        self._progress(ctx, f">> Fallback centering boundary fraction = {fallback_fraction:.7g}")

        chosen_name, chosen_fraction = self._select_centering_candidate(primary_fraction, fallback_fraction)
        if chosen_name == "fallback":
            shutil.copy2(fallback_output, primary_output)

        if chosen_fraction > CENTERING_QC_HARD_STOP_FRACTION:
            self._progress(
                ctx,
                (
                    "FATAL: Centering quality remained too poor after both methods. "
                    f"attempt1={primary_fraction:.7g}, fallback={fallback_fraction:.7g}, "
                    f"chosen={chosen_name}:{chosen_fraction:.7g}"
                ),
            )
            raise RuntimeError(
                "Centering failed quality control after both methods; "
                f"best boundary fraction={chosen_fraction:.7g} exceeds the hard-stop threshold "
                f"of {CENTERING_QC_HARD_STOP_FRACTION:.2f}. See logs in {ctx.run_dir}"
            )

        if chosen_fraction > CENTERING_QC_TARGET_FRACTION:
            self._progress(
                ctx,
                (
                    "WARNING: Both centering methods exceeded the preferred boundary threshold, "
                    f"but continuing with the better result from {chosen_name} "
                    f"(attempt1={primary_fraction:.7g}, fallback={fallback_fraction:.7g}, "
                    f"chosen={chosen_fraction:.7g})"
                ),
            )
            return

        self._progress(ctx, f">> Centering OK after selecting the {chosen_name} result. Continue.")

    def _check_centering_ok(self, ctx: WorkflowContext, gro_name: str, *, qc_log: str) -> float:
        gro_path = ctx.run_dir / gro_name
        structure = parse_gro(gro_path)
        box_fields = structure.box_line.split()
        if len(box_fields) < 3:
            self._progress(ctx, "QC boundary fraction = 1")
            return 1.0
        try:
            boxx, boxy, boxz = (float(box_fields[0]), float(box_fields[1]), float(box_fields[2]))
        except ValueError:
            self._progress(ctx, "QC boundary fraction = 1")
            return 1.0

        qc_group = self._prepare_centering_index(ctx, gro_name, log_path=f"{Path(qc_log).stem}_make_ndx.log")
        if qc_group is None:
            self._progress(ctx, "QC boundary fraction = 1")
            return 1.0

        qc_path = ctx.run_dir / "__prot.gro"
        result = self.runner.run_gmx(
            [
                "trjconv",
                "-f",
                gro_name,
                "-s",
                gro_name,
                "-o",
                qc_path.name,
                "-n",
                "index.ndx",
            ],
            cwd=ctx.run_dir,
            log_path=qc_log,
            stdin_text=f"{qc_group}\n",
        )
        if result.returncode != 0 or not qc_path.exists():
            self._progress(ctx, "QC boundary fraction = 1")
            return 1.0

        protein = parse_gro(qc_path)
        qc_path.unlink(missing_ok=True)
        if not protein.atoms:
            self._progress(ctx, "QC boundary fraction = 1")
            return 1.0

        boundary_hits = 0
        for atom in protein.atoms:
            if (
                atom.x < 0.15
                or atom.x > boxx - 0.15
                or atom.y < 0.15
                or atom.y > boxy - 0.15
                or atom.z < 0.15
                or atom.z > boxz - 0.15
            ):
                boundary_hits += 1
        fraction = boundary_hits / len(protein.atoms)
        self._progress(ctx, f"QC boundary fraction = {fraction:.7g}")
        return fraction

    @staticmethod
    def _select_centering_candidate(primary_fraction: float, fallback_fraction: float) -> tuple[str, float]:
        if fallback_fraction < primary_fraction:
            return "fallback", fallback_fraction
        return "attempt1", primary_fraction

    def _resolved_mask_file(self) -> Path | None:
        if self.config.cavity.mode != "mask":
            return None
        return self.config.cavity.mask_file

    def _select_move(self, nins: int, rng: random.Random) -> int:
        if nins <= 0:
            return 1
        x = rng.random()
        if x < 0.25:
            return 1
        if x < 0.50:
            return 4
        if x < 0.75:
            return 2
        return 3

    def _nearest_context(self, confout_gro: Path, x_gro: Path, target_resid: int | None = None) -> tuple[str, str, float]:
        conf = parse_gro(confout_gro)
        if target_resid is None:
            trial_atoms = last_residue_atoms(conf)
        else:
            trial_atoms = [atom for atom in conf.atoms if atom.resid == target_resid]
        if not trial_atoms or not x_gro.exists():
            return "---", "---", 0.0
        trial_atom = trial_atoms[0]
        best_name = "---"
        best_resid = "---"
        best_dist = float("inf")
        for atom in parse_raw_context(x_gro):
            if atom.resid == trial_atom.resid:
                continue
            dist = sqrt((atom.x - trial_atom.x) ** 2 + (atom.y - trial_atom.y) ** 2 + (atom.z - trial_atom.z) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_name = atom.resname
                best_resid = str(atom.resid)
        if best_dist == float("inf"):
            return "---", "---", 0.0
        return best_name, best_resid, best_dist

    def _move_log_context(
        self,
        *,
        type_move: str,
        moved_resid: int | None,
        confout_gro: Path,
        x_gro: Path,
        previous_gro: Path,
        gas_name: str,
        provenance: dict[int, str],
    ) -> tuple[str, str, str]:
        if type_move == "D" and moved_resid is not None:
            resname = self._residue_resname(previous_gro, moved_resid, gas_name)
            return resname, str(moved_resid), provenance.get(moved_resid, "unk")
        resname, resid, dmin = self._nearest_context(confout_gro, x_gro, moved_resid)
        return resname, resid, f"{dmin:5.3f}"

    @staticmethod
    def _residue_resname(gro_path: Path, resid: int, gas_name: str) -> str:
        structure = parse_gro(gro_path)
        for atom in structure.atoms:
            if atom.resid == resid:
                return atom.resname
        return gas_name

    @staticmethod
    def _fmt_delta(de: float) -> str:
        return "---" if de > 1.0e4 else f"{de:.3f}"

    @staticmethod
    def _fmt_energy(e1: float, de: float) -> str:
        return "---" if de > 1.0e4 else f"{e1:.3f}"

    @staticmethod
    def _acceptance_label(acceptance: int) -> str:
        return "ACC." if acceptance > 0 else "REJ."

    def _post_em_clash(self, confout_gro: Path, moved_resid: int | None, *, type_move: str) -> ClashDetail | None:
        if type_move not in {"I", "R", "T"} or moved_resid is None:
            return None
        return residue_clash_in_structure(confout_gro, moved_resid, self.config.simulation.rvdw)


def parse_raw_context(path: str | Path):
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            yield parse_atom_line(line)
        except ValueError:
            continue


def type_move_name(move: int) -> str:
    return {1: "insertion", 2: "rotation", 3: "translation", 4: "deletion"}.get(move, "move")
