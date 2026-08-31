from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from gcmc_port.config import ExecutionConfig, load_config
from gcmc_port.gromacs import ShellRunner, parse_potential_energy


class PotentialEnergyParsingTests(unittest.TestCase):
    def test_reads_last_finite_potential_with_leading_whitespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "md.log"
            log.write_text(
                " Potential Energy  = -123.5\n"
                "Potential Energy  = nan\n"
                " Potential Energy  = -120.25\n",
                encoding="utf-8",
            )
            energy = parse_potential_energy(log)

        self.assertEqual(energy, -120.25)

    def test_rejects_non_finite_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log = Path(tmpdir) / "md.log"
            log.write_text("Potential Energy  = inf\n", encoding="utf-8")
            energy = parse_potential_energy(log)

        self.assertIsNone(energy)


class SafeCommandExecutionTests(unittest.TestCase):
    def test_gromacs_uses_argv_environment_and_shell_false(self) -> None:
        execution = ExecutionConfig(
            gmx_cmd="gmx_mpi",
            launcher_single="mpirun -np 1",
            launcher_multi="mpirun -np {cores}",
            nodes=1,
            cores_per_node=4,
            env={"GMX_MAXBACKUP": "-1"},
        )
        completed = subprocess.CompletedProcess([], 0, "stdout", "stderr")
        with tempfile.TemporaryDirectory() as raw, patch(
            "gcmc_port.gromacs.subprocess.run", return_value=completed
        ) as run:
            result = ShellRunner(execution).run_gmx(
                ["grompp", "-f", "input file.mdp"], cwd=raw
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            run.call_args.args[0],
            ["mpirun", "-np", "1", "gmx_mpi", "grompp", "-f", "input file.mdp"],
        )
        self.assertIs(run.call_args.kwargs["shell"], False)
        self.assertEqual(run.call_args.kwargs["env"]["GMX_MAXBACKUP"], "-1")

    def test_toml_rejects_arbitrary_module_shell_command(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = Path(raw) / "config.toml"
            config.write_text(
                '[execution]\nmodule_setup = ["touch should-not-run"]\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unsafe execution.module_setup"):
                load_config(config)

    def test_command_field_cannot_select_a_shell(self) -> None:
        execution = ExecutionConfig(
            gmx_cmd='bash -c "echo unsafe"',
            launcher_single="",
        )
        with tempfile.TemporaryDirectory() as raw, patch("gcmc_port.gromacs.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "execution.gmx_cmd"):
                ShellRunner(execution).run_gmx(["--version"], cwd=raw)
            run.assert_not_called()

    def test_toml_rejects_loader_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            config = Path(raw) / "config.toml"
            config.write_text(
                '[execution.env]\nLD_PRELOAD = "/tmp/untrusted.so"\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "security-sensitive"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
