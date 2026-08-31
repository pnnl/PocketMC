from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import tomllib
from typing import Any

from ..pathing import portable_path
from ..safe_config import (
    safe_single_line,
    validated_module_setup,
    validated_shebang,
    validated_slurm_directives,
)
from .models import AnalysisConfig


def _settings(config: AnalysisConfig) -> dict[str, Any]:
    try:
        with config.config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except Exception:
        raw = {}
    slurm = dict(raw.get("slurm", {}))
    return {
        "shebang": validated_shebang(slurm.get("shebang", "#!/usr/bin/env bash")),
        "partition": safe_single_line(slurm.get("partition", "compute"), "slurm.partition"),
        "account": safe_single_line(slurm.get("account", ""), "slurm.account").strip() or "YOUR_ACCOUNT",
        "time": safe_single_line(
            slurm.get("time", slurm.get("time_limit", "24:00:00")),
            "slurm.time",
            allow_empty=False,
        ),
        "nodes": int(slurm.get("nodes", 0)),
        "tasks_per_node": int(slurm.get("ntasks_per_node", slurm.get("tasks_per_node", 0))),
        "cpus": int(slurm.get("cpus_per_task", 1)),
        "memory": safe_single_line(slurm.get("memory", "8G"), "slurm.memory"),
        "job_name": safe_single_line(slurm.get("job_name", "pocketmc-analysis"), "slurm.job_name", allow_empty=False),
        "module_setup": validated_module_setup(slurm.get("module_setup", []), "slurm.module_setup"),
        "extra_directives": validated_slurm_directives(slurm.get("extra_directives", [])),
    }


def _header(settings: dict[str, Any], name: str, *, array: str | None = None) -> list[str]:
    lines = [
        settings["shebang"],
        f"#SBATCH --job-name={name}",
        f"#SBATCH --time={settings['time']}",
    ]
    if settings.get("partition"):
        lines.append(f"#SBATCH --partition={settings['partition']}")
    if int(settings.get("nodes", 0)) > 0:
        lines.append(f"#SBATCH --nodes={settings['nodes']}")
    if int(settings.get("tasks_per_node", 0)) > 0:
        lines.append(f"#SBATCH --ntasks-per-node={settings['tasks_per_node']}")
    if int(settings.get("cpus", 0)) > 0:
        lines.append(f"#SBATCH --cpus-per-task={settings['cpus']}")
    if settings.get("memory"):
        lines.append(f"#SBATCH --mem={settings['memory']}")
    lines.extend(
        [
            f"#SBATCH --output={name}-%A_%a.out" if array else f"#SBATCH --output={name}-%j.out",
            f"#SBATCH --error={name}-%A_%a.err" if array else f"#SBATCH --error={name}-%j.err",
        ]
    )
    if settings["account"]:
        lines.append(f"#SBATCH --account={settings['account']}")
    if array:
        lines.append(f"#SBATCH --array={array}")
    lines.extend(settings["extra_directives"])
    lines.extend(
        [
            "", "set -Eeuo pipefail",
            "CURRENT_STEP=initialization",
            "trap 'status=$?; printf \"[%s] ERROR step=%s line=%s status=%s command=%s\\n\" \"$(date \"+%F %T\")\" \"$CURRENT_STEP\" \"$LINENO\" \"$status\" \"$BASH_COMMAND\" >&2; exit $status' ERR",
            "export POCKETMC_ANALYSIS_TRACEBACK=1",
            *settings["module_setup"], "",
        ]
    )
    return lines


def _launcher_body(
    config: AnalysisConfig,
    *,
    launcher_directory: Path,
    tee_log: bool,
    tahoma: bool,
    force: bool = False,
    resume: bool = False,
    slurm_cpu_workers: bool = False,
) -> list[str]:
    config_path = portable_path(config.config_path, launcher_directory)
    case_dir = portable_path(config.config_path.parent, launcher_directory)
    lines = [
        "set -Eeuo pipefail",
        'CURRENT_STEP="initialization"',
        'on_error() { status=$?; printf "[%s] ERROR step=%s line=%s status=%s command=%s\\n" "$(date "+%F %T")" "$CURRENT_STEP" "${BASH_LINENO[0]:-$LINENO}" "$status" "$BASH_COMMAND" >&2; exit "$status"; }',
        "trap on_error ERR",
        'log() { printf "[%s] %s\\n" "$(date "+%F %T")" "$*"; }',
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        f'CASE_DIR="$SCRIPT_DIR"/{shlex.quote(case_dir)}',
        'cd "$CASE_DIR"',
    ]
    if tee_log:
        log_name = "analysis-resume.log" if resume else "analysis-run.log"
        lines.extend(
            [
                f'LOG_FILE="$CASE_DIR"/{shlex.quote(log_name)}',
                'exec > >(tee -a "$LOG_FILE") 2>&1',
                'log "Combined screen log: $LOG_FILE"',
            ]
        )
    if tahoma:
        lines.extend(
            [
                'CURRENT_STEP="Tahoma environment setup"',
                'log "Loading Tahoma modules (override POCKETMC_ANALYSES_BIN if needed)."',
                "source /etc/profile.d/modules.sh",
                "module purge",
                "module load gromacs",
            ]
        )
    if slurm_cpu_workers:
        default_jobs = max(1, min(len(config.datasets), 32))
        lines.extend(
            [
                f'DEFAULT_ANALYSIS_JOBS={default_jobs}',
                'ANALYSIS_JOBS="${ANALYSIS_JOBS:-$DEFAULT_ANALYSIS_JOBS}"',
                'if (( ANALYSIS_JOBS < 1 )); then echo "ANALYSIS_JOBS must be at least 1" >&2; exit 2; fi',
                'if (( ANALYSIS_JOBS > ${SLURM_CPUS_PER_TASK:-1} )); then ANALYSIS_JOBS=${SLURM_CPUS_PER_TASK:-1}; fi',
                'THREADS_PER_WORKER=$(( ${SLURM_CPUS_PER_TASK:-1} / ANALYSIS_JOBS ))',
                'if (( THREADS_PER_WORKER < 1 )); then THREADS_PER_WORKER=1; fi',
                'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS_PER_WORKER}"',
                'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS_PER_WORKER}"',
                'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS_PER_WORKER}"',
                'export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$THREADS_PER_WORKER}"',
                'log "CPU execution: one Python process; jobs=$ANALYSIS_JOBS independent cases; threads/worker=$THREADS_PER_WORKER (no MPI ranks)."',
            ]
        )
    lines.extend(
        [
            'POCKETMC_ANALYSES_BIN="${POCKETMC_ANALYSES_BIN:-pocketmc-analyses}"',
            'export POCKETMC_ANALYSIS_TRACEBACK=1',
            f'CONFIG="$SCRIPT_DIR"/{shlex.quote(config_path)}',
            'CURRENT_STEP="1/3 cavity preparation"',
            'log "STEP 1/3: Preparing cavity inputs. Voxel construction may take several minutes for a large structure."',
            '"$POCKETMC_ANALYSES_BIN" prepare-cavities -c "$CONFIG"',
            'CURRENT_STEP="2/3 input validation"',
            'log "STEP 2/3: Validating every case, topology, trajectory, and cavity path."',
            '"$POCKETMC_ANALYSES_BIN" validate -c "$CONFIG"',
            'CURRENT_STEP="3/3 analysis"',
            (
                'log "STEP 3/3: Resuming from versioned caches; completed feature/cluster stages are reused and failed work is retried."'
                if resume and not force
                else 'log "STEP 3/3: Running analyses. Long trajectories and pose/density tasks may take minutes to hours; progress messages will continue below."'
            ),
            '"$POCKETMC_ANALYSES_BIN" run -c "$CONFIG"'
            + (' --jobs "$ANALYSIS_JOBS"' if slurm_cpu_workers else "")
            + (" --force" if force else ""),
            'CURRENT_STEP="complete"',
            'log "Analysis completed successfully."',
        ]
    )
    return lines


def _write_executable(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(path, path.stat().st_mode | 0o111)
    except OSError:
        pass
    return path


def render_analysis_launchers(
    config: AnalysisConfig,
    output_directory: str | Path,
    *,
    name_stem: str = "run_analyses",
    force: bool = False,
    resume: bool = False,
) -> dict[str, Path]:
    """Write direct-shell, generic Slurm, and Tahoma-only Slurm launchers."""
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = _settings(config)
    settings["shebang"] = "#!/usr/bin/env bash"
    settings["account"] = settings["account"] or "YOUR_ACCOUNT"
    generic_sh = _write_executable(
        output / f"{name_stem}.sh",
        [
            "#!/bin/bash",
            "",
            *_launcher_body(
                config,
                launcher_directory=output,
                tee_log=True,
                tahoma=False,
                force=force,
                resume=resume,
            ),
        ],
    )
    generic_sbatch = _write_executable(
        output / f"{name_stem}.sbatch",
        _header(settings, settings["job_name"] + ("-resume" if resume else ""))
        + _launcher_body(
            config,
            launcher_directory=output,
            tee_log=False,
            tahoma=False,
            force=force,
            resume=resume,
        ),
    )
    tahoma_settings = dict(settings)
    tahoma_settings.update(
        {
            "account": "YOUR_ACCOUNT",
            "time": "48:00:00",
            "partition": "",
            "nodes": 1,
            # The analysis is not MPI: request one task with 32 CPUs and use
            # those CPUs for case-level workers / numerical-library threads.
            "tasks_per_node": 1,
            "cpus": 32,
            "memory": "",
            "job_name": "GCMC",
        }
    )
    tahoma_name = tahoma_settings["job_name"] + ("-resume" if resume else "")
    tahoma_sbatch = _write_executable(
        output / f"{name_stem}_tahoma_only.sbatch",
        _header(tahoma_settings, tahoma_name)
        + _launcher_body(
            config,
            launcher_directory=output,
            tee_log=False,
            tahoma=True,
            force=force,
            resume=resume,
            slurm_cpu_workers=True,
        ),
    )
    return {
        "Direct shell": generic_sh,
        "Generic Slurm": generic_sbatch,
        "Tahoma-only Slurm": tahoma_sbatch,
    }


def render_analysis_sbatch(config: AnalysisConfig, output_directory: str | Path) -> list[Path]:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = _settings(config)
    settings["shebang"] = "#!/usr/bin/env bash"
    config_path = portable_path(config.config_path, output)
    case_dir = portable_path(config.config_path.parent, output)
    md_cases = [item for item in config.datasets if item.kind == "md"]
    if not md_cases:
        raise ValueError("Slurm pose pipeline requires at least one physical MD case")
    mapping = output / "md_cases.tsv"
    mapping.write_text(
        "array_index\trun_id\tsystem_id\treplica\n"
        + "\n".join(
            f"{index}\t{item.run_id}\t{item.system_id or item.run_id}\t{item.replica}"
            for index, item in enumerate(md_cases)
        )
        + "\n",
        encoding="utf-8",
    )

    def write(name: str, lines: list[str]) -> Path:
        path = output / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    runtime = [
        'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
        f'CASE_DIR="$SCRIPT_DIR"/{shlex.quote(case_dir)}',
        f'CONFIG="$SCRIPT_DIR"/{shlex.quote(config_path)}',
        'POCKETMC_ANALYSES_BIN="${POCKETMC_ANALYSES_BIN:-pocketmc-analyses}"',
        'cd "$CASE_DIR"',
    ]

    array = f"0-{len(md_cases) - 1}"
    features = write(
        "01_features.sbatch",
        _header(settings, settings["job_name"] + "-features", array=array)
        + [*runtime, 'CURRENT_STEP="pose features"', 'echo "[$(date \"+%F %T\")] Building pose features; this can take minutes to hours."', '"$POCKETMC_ANALYSES_BIN" _pose-stage -c "$CONFIG" --phase features --case-index "$SLURM_ARRAY_TASK_ID"'],
    )
    cluster = write(
        "02_cluster.sbatch",
        _header(settings, settings["job_name"] + "-cluster")
        + [*runtime, 'CURRENT_STEP="pooled pose clustering"', 'echo "[$(date \"+%F %T\")] Fitting pooled pose clusters."', '"$POCKETMC_ANALYSES_BIN" _pose-stage -c "$CONFIG" --phase cluster'],
    )
    hydrate = write(
        "03_hydrate.sbatch",
        _header(settings, settings["job_name"] + "-hydrate", array=array)
        + [*runtime, 'CURRENT_STEP="pose hydration"', 'echo "[$(date \"+%F %T\")] Computing pose-conditioned hydration; this can take minutes to hours."', '"$POCKETMC_ANALYSES_BIN" _pose-stage -c "$CONFIG" --phase hydrate --case-index "$SLURM_ARRAY_TASK_ID"'],
    )
    finalize = write(
        "04_finalize.sbatch",
        _header(settings, settings["job_name"] + "-finalize")
        + [*runtime, 'CURRENT_STEP="analysis finalize"', 'echo "[$(date \"+%F %T\")] Finalizing aggregate comparisons."', '"$POCKETMC_ANALYSES_BIN" _pose-stage -c "$CONFIG" --phase finalize'],
    )
    helper = write(
        "submit_pipeline.sh",
        [
            "#!/bin/bash", "set -euo pipefail",
            'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
            'cd "$SCRIPT_DIR"',
            'feature_job=$(sbatch --parsable 01_features.sbatch)',
            'cluster_job=$(sbatch --parsable --dependency=afterany:${feature_job} 02_cluster.sbatch)',
            'hydrate_job=$(sbatch --parsable --dependency=afterany:${cluster_job} 03_hydrate.sbatch)',
            'finalize_job=$(sbatch --parsable --dependency=afterany:${hydrate_job} 04_finalize.sbatch)',
            'printf "features=%s\\ncluster=%s\\nhydrate=%s\\nfinalize=%s\\n" "$feature_job" "$cluster_job" "$hydrate_job" "$finalize_job"',
        ],
    )
    manifest = output / "slurm_pipeline.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": portable_path(config.config_path, output),
                "md_cases": [item.run_id for item in md_cases],
                "dependency": "features array -> pooled cluster -> hydration array -> finalize",
                "scripts": [portable_path(path, output) for path in (features, cluster, hydrate, finalize, helper)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [mapping, features, cluster, hydrate, finalize, helper, manifest]


def submit_analysis_sbatch(directory: str | Path) -> subprocess.CompletedProcess[str]:
    helper = Path(directory).expanduser().resolve() / "submit_pipeline.sh"
    if not helper.exists():
        raise FileNotFoundError(helper)
    return subprocess.run(["bash", str(helper)], cwd=helper.parent, capture_output=True, text=True, check=False)
