from __future__ import annotations

import re
import shlex
from typing import Iterable, Mapping


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_MODULE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+/@:-]*")
_ALLOWED_SHEBANGS = {"#!/bin/bash", "#!/usr/bin/bash", "#!/usr/bin/env bash"}
_DANGEROUS_ENV_NAMES = {
    "PATH", "PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "BASH_ENV", "ENV", "SHELLOPTS",
}


def safe_single_line(value: object, label: str, *, allow_empty: bool = True) -> str:
    text = str(value)
    if not allow_empty and not text.strip():
        raise ValueError(f"{label} must not be empty")
    if any(character in text for character in ("\r", "\n", "\x00")):
        raise ValueError(f"{label} must be a single line without NUL characters")
    return text


def validated_environment(values: Mapping[object, object]) -> dict[str, str]:
    environment: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        value = safe_single_line(raw_value, f"execution.env.{key}")
        if _ENV_NAME.fullmatch(key) is None:
            raise ValueError(f"Invalid environment variable name in execution.env: {key!r}")
        if key.upper() in _DANGEROUS_ENV_NAMES:
            raise ValueError(f"execution.env may not override security-sensitive variable {key!r}")
        environment[key] = value
    return environment


def validated_module_setup(lines: Iterable[object], label: str = "module_setup") -> list[str]:
    """Allow only declarative environment-module operations, never general shell code."""
    validated: list[str] = []
    for index, raw in enumerate(lines):
        line = safe_single_line(raw, f"{label}[{index}]", allow_empty=False).strip()
        if line in {"source /etc/profile.d/modules.sh", ". /etc/profile.d/modules.sh"}:
            validated.append("source /etc/profile.d/modules.sh")
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError as exc:
            raise ValueError(f"Invalid {label}[{index}]: {exc}") from exc
        if tokens == ["module", "purge"]:
            validated.append("module purge")
            continue
        if len(tokens) >= 3 and tokens[:2] in (["module", "load"], ["module", "unload"]):
            if all(_MODULE_NAME.fullmatch(token) for token in tokens[2:]):
                validated.append(" ".join(tokens))
                continue
        if len(tokens) == 4 and tokens[:2] == ["module", "swap"]:
            if all(_MODULE_NAME.fullmatch(token) for token in tokens[2:]):
                validated.append(" ".join(tokens))
                continue
        raise ValueError(
            f"Unsafe {label}[{index}]={line!r}. Only the fixed modules.sh initialization and "
            "module purge/load/unload/swap operations are allowed. Preload any other setup "
            "before running PocketMC."
        )
    return validated


def validated_shebang(value: object) -> str:
    shebang = safe_single_line(value, "slurm.shebang", allow_empty=False).strip()
    if shebang not in _ALLOWED_SHEBANGS:
        raise ValueError(
            "slurm.shebang must select Bash using #!/bin/bash, #!/usr/bin/bash, or #!/usr/bin/env bash"
        )
    return shebang


def validated_slurm_directives(values: Iterable[object]) -> list[str]:
    directives: list[str] = []
    for index, raw in enumerate(values):
        value = safe_single_line(raw, f"slurm.extra_directives[{index}]", allow_empty=False).strip()
        if value.startswith("#SBATCH "):
            directives.append(value)
        elif value.startswith("--"):
            directives.append(f"#SBATCH {value}")
        else:
            raise ValueError(
                f"slurm.extra_directives[{index}] must start with '#SBATCH ' or '--', got {value!r}"
            )
    return directives
