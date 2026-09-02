from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.config import load_config
from gcmc_port.slurm import (
    render_local_script,
    render_sbatch,
    render_tahoma_only_sbatch,
    submit_sbatch,
)


class RenderSbatchTests(unittest.TestCase):
    def test_submit_sbatch_submits_from_the_launcher_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "jobs" / "run_gcmc.sbatch"
            script.parent.mkdir()
            script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

            with patch("gcmc_port.slurm.subprocess.run") as run:
                submit_sbatch(script)

            run.assert_called_once_with(
                ["sbatch", script.name],
                cwd=script.parent.resolve(),
                capture_output=True,
                text=True,
                check=False,
            )

    def test_render_local_and_sbatch_use_script_relative_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "case"
            jobs_dir = case_dir / "jobs"
            case_dir.mkdir(parents=True)
            config_path = case_dir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'project_root = "."',
                        'work_root = "."',
                        "",
                        "[execution]",
                        "module_setup = []",
                        "",
                        "[slurm]",
                        'job_name = "test-job"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            local = render_local_script(config, jobs_dir / "run_gcmc.sh")
            sbatch = render_sbatch(config, jobs_dir / "run_gcmc.sbatch")
            local_text = local.read_text(encoding="utf-8")
            self.assertIn(
                'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
                local_text,
            )
            sbatch_text = sbatch.read_text(encoding="utf-8")
            self.assertIn(
                'SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"',
                sbatch_text,
            )
            self.assertNotIn("BASH_SOURCE", sbatch_text)
            for text in (local_text, sbatch_text):
                self.assertIn('CONFIG="$SCRIPT_DIR"/../config.toml', text)
                self.assertIn('CASE_DIR="$SCRIPT_DIR"/..', text)
                self.assertIn('exec "$POCKETMC_BIN" run -c "$CONFIG"', text)
                self.assertNotIn(case_dir.resolve().as_posix(), text)
            self.assertNotIn("#SBATCH", local_text)
            self.assertIn("#SBATCH --account=YOUR_ACCOUNT", sbatch_text)

    def test_render_tahoma_only_sbatch_uses_placeholder_and_single_run_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir) / "case"
            jobs_dir = case_dir / "jobs"
            case_dir.mkdir(parents=True)
            config_path = case_dir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'project_root = "."',
                        'work_root = "."',
                        "",
                        "[loop]",
                        'replica_dirs = ["run"]',
                        "replica_count = 1",
                        "",
                        "[slurm]",
                        'job_name = "test-job"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)
            output = render_tahoma_only_sbatch(config, jobs_dir / "run_gcmc_tahoma_only.sbatch")
            text = output.read_text(encoding="utf-8")

            self.assertIn("#SBATCH --account=YOUR_ACCOUNT", text)
            self.assertIn("module load openmpi", text)
            self.assertIn("module load gromacs", text)
            self.assertNotIn("for SWEEP", text)
            self.assertIn(
                'SCRIPT_DIR="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"',
                text,
            )
            self.assertNotIn("BASH_SOURCE", text)
            self.assertIn('CONFIG="$SCRIPT_DIR"/../config.toml', text)
            self.assertIn('exec "$POCKETMC_BIN" run -c "$CONFIG"', text)
            self.assertNotIn(case_dir.resolve().as_posix(), text)


if __name__ == "__main__":
    unittest.main()
