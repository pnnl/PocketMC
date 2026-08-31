from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import os
from pathlib import Path
import re
import shlex
import subprocess

from .config import ExecutionConfig


@dataclass(slots=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class ShellRunner:
    def __init__(self, execution: ExecutionConfig) -> None:
        self.execution = execution

    def _expand_cores(self, value: str, label: str) -> str:
        expanded = value.replace("{cores}", str(self.execution.total_cores))
        if "{" in expanded or "}" in expanded:
            raise ValueError(f"{label} supports only the {{cores}} placeholder")
        return expanded

    def _tokens(self, value: str, label: str) -> list[str]:
        expanded = self._expand_cores(value, label).strip()
        if not expanded:
            return []
        try:
            tokens = shlex.split(expanded, posix=True)
        except ValueError as exc:
            raise ValueError(f"Invalid {label}: {exc}") from exc
        if not tokens or any("\x00" in token for token in tokens):
            raise ValueError(f"Invalid {label}")
        return tokens

    def _launcher(self, multi: bool) -> list[str]:
        value = self.execution.launcher_multi if multi else self.execution.launcher_single
        label = "execution.launcher_multi" if multi else "execution.launcher_single"
        tokens = self._tokens(value, label)
        if not tokens:
            return []
        executable = Path(tokens[0]).name.lower()
        if executable in {"mpirun", "mpiexec"}:
            if len(tokens) != 3 or tokens[1] not in {"-n", "-np"} or not tokens[2].isdigit():
                raise ValueError(f"{label} permits only 'mpirun/mpiexec -n/-np COUNT'")
            return tokens
        if executable == "srun":
            if len(tokens) == 1:
                return tokens
            if len(tokens) == 3 and tokens[1] in {"-n", "--ntasks", "-c", "--cpus-per-task"} and tokens[2].isdigit():
                return tokens
            if len(tokens) == 2 and re.fullmatch(r"--(?:ntasks|cpus-per-task|mpi)=[A-Za-z0-9_.+-]+", tokens[1]):
                return tokens
            raise ValueError(f"{label} contains unsupported srun options")
        raise ValueError(f"{label} executable must be mpirun, mpiexec, or srun")

    def _gmx_command(self) -> list[str]:
        tokens = self._tokens(self.execution.gmx_cmd, "execution.gmx_cmd")
        if len(tokens) != 1 or re.fullmatch(r"gmx(?:_mpi)?(?:_d)?(?:\.exe)?", Path(tokens[0]).name, re.IGNORECASE) is None:
            raise ValueError("execution.gmx_cmd must be a path or command name for gmx/gmx_mpi")
        return tokens

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        for key, value in self.execution.env.items():
            if not key or not (key[0].isalpha() or key[0] == "_") or not all(
                character.isalnum() or character == "_" for character in key
            ):
                raise ValueError(f"Invalid environment variable name: {key!r}")
            if "\x00" in value:
                raise ValueError(f"Environment variable {key!r} contains a NUL character")
            environment[key] = value
        return environment

    def _expanded_mdrun_multi_args(self) -> list[str]:
        return [self._expand_cores(arg, "execution.mdrun_multi_args") for arg in self.execution.mdrun_multi_args]

    def run_gmx(
        self,
        args: list[str],
        *,
        cwd: str | Path,
        log_path: str | Path | None = None,
        stdin_text: str | None = None,
        multi: bool = False,
    ) -> CommandResult:
        launcher = self._launcher(multi)
        command_args = list(args)
        if multi and command_args and command_args[0] == "mdrun" and self.execution.mdrun_multi_args:
            command_args.extend(self._expanded_mdrun_multi_args())
        command = [
            *launcher,
            *self._gmx_command(),
            *command_args,
        ]
        result = subprocess.run(
            command,
            cwd=str(cwd),
            input=stdin_text,
            text=True,
            capture_output=True,
            env=self._environment(),
            shell=False,
            check=False,
        )
        if log_path is not None:
            log = Path(cwd) / log_path
            log.write_text(result.stdout + result.stderr, encoding="utf-8")
        return CommandResult(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def parse_potential_energy(md_log: str | Path) -> float | None:
    path = Path(md_log)
    if not path.exists():
        return None
    energy: float | None = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("Potential Energy"):
            try:
                candidate = float(line.split()[-1])
            except ValueError:
                continue
            if isfinite(candidate):
                energy = candidate
    return energy
