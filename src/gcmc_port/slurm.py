from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess

from .config import Config


def _script_relative_path(path: Path, script_directory: Path) -> str:
    return Path(os.path.relpath(path.resolve(), script_directory.resolve())).as_posix()


def _runtime_path_lines(config: Config, output: Path, *, slurm: bool) -> list[str]:
    config_path = config.config_path.resolve()
    case_dir = config_path.parent.resolve()
    script_directory = output.parent.resolve()
    config_relative = shlex.quote(_script_relative_path(config_path, script_directory))
    case_relative = shlex.quote(_script_relative_path(case_dir, script_directory))
    if slurm:
        # Slurm executes a copy of the submitted script from its spool directory,
        # so BASH_SOURCE[0] does not point at the user's case directory in a job.
        runtime_directory = 'SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"'
    else:
        runtime_directory = 'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"'
    return [
        runtime_directory,
        f'CONFIG="$SCRIPT_DIR"/{config_relative}',
        f'CASE_DIR="$SCRIPT_DIR"/{case_relative}',
        'POCKETMC_BIN="${POCKETMC_BIN:-gcmc-port}"',
    ]


def _execution_lines(
    config: Config,
    output: Path,
    *,
    module_setup: list[str],
    slurm: bool = False,
) -> list[str]:
    lines = ["set -euo pipefail", "", *module_setup]
    for key, value in config.execution.env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.extend(
        [
            "",
            *_runtime_path_lines(config, output, slurm=slurm),
            'cd "$CASE_DIR"',
            'if ! command -v "$POCKETMC_BIN" >/dev/null 2>&1; then',
            '  echo "PocketMC command not found: $POCKETMC_BIN" >&2',
            '  echo "Activate the installed environment or set POCKETMC_BIN." >&2',
            "  exit 127",
            "fi",
            'exec "$POCKETMC_BIN" run -c "$CONFIG"',
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


def _resource_directives(config: Config) -> list[str]:
    threaded = not config.execution.launcher_multi.strip()
    return [
        f"#SBATCH --nodes={config.execution.nodes}",
        f"#SBATCH --ntasks-per-node={1 if threaded else config.execution.cores_per_node}",
        f"#SBATCH --cpus-per-task={config.execution.cores_per_node if threaded else 1}",
    ]


def render_local_script(config: Config, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "",
        "# Local/direct launcher: no scheduler account or resource directives are required.",
        *_execution_lines(config, output, module_setup=config.execution.module_setup),
    ]
    return _write_executable(output, lines)


def render_sbatch(config: Config, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    account = config.slurm.account.strip() or "YOUR_ACCOUNT"
    directives = [
        "#!/usr/bin/env bash",
        "# Generic Slurm launcher: replace placeholders and review every resource request.",
        f"#SBATCH --account={account}",
        f"#SBATCH --job-name={config.slurm.job_name}",
        f"#SBATCH --time={config.slurm.time_limit}",
        f"#SBATCH --partition={config.slurm.partition}",
        *_resource_directives(config),
        f"#SBATCH --output={config.slurm.output}",
    ]
    directives.extend(config.slurm.extra_directives)
    directives.append("")
    directives.extend(
        _execution_lines(config, output, module_setup=config.execution.module_setup, slurm=True)
    )
    return _write_executable(output, directives)


def render_tahoma_only_sbatch(config: Config, output_path: str | Path) -> Path:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directives = [
        "#!/usr/bin/env bash",
        "# Tahoma-only Slurm launcher: replace YOUR_ACCOUNT before submission.",
        "#SBATCH --account=YOUR_ACCOUNT",
        f"#SBATCH --time={config.slurm.time_limit}",
        *_resource_directives(config),
        f"#SBATCH --job-name={config.slurm.job_name}",
        f"#SBATCH --error={config.slurm.job_name}-%j.err",
        f"#SBATCH --output={config.slurm.job_name}-%j.out",
        "",
    ]
    directives.extend(
        _execution_lines(
            config,
            output,
            slurm=True,
            module_setup=[
                "# Tahoma environment preset",
                "source /etc/profile.d/modules.sh",
                "module purge",
                "module load openmpi",
                "module load gromacs",
            ],
        )
    )
    return _write_executable(output, directives)


def render_tahoma_sbatch(config: Config, output_path: str | Path) -> Path:
    """Backward-compatible alias for the explicitly named Tahoma-only renderer."""
    return render_tahoma_only_sbatch(config, output_path)


def submit_sbatch(script_path: str | Path) -> subprocess.CompletedProcess[str]:
    script = Path(script_path).resolve()
    return subprocess.run(
        ["sbatch", script.name],
        cwd=script.parent,
        capture_output=True,
        text=True,
        check=False,
    )
