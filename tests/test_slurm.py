from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.config import load_config
from gcmc_port.slurm import render_local_script, render_sbatch, render_tahoma_only_sbatch


class RenderSbatchTests(unittest.TestCase):
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
            for output in (local, sbatch):
                text = output.read_text(encoding="utf-8")
                self.assertIn('SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"', text)
                self.assertIn('CONFIG="$SCRIPT_DIR"/../config.toml', text)
                self.assertIn('CASE_DIR="$SCRIPT_DIR"/..', text)
                self.assertIn('exec "$POCKETMC_BIN" run -c "$CONFIG"', text)
                self.assertNotIn(case_dir.resolve().as_posix(), text)
            self.assertNotIn("#SBATCH", local.read_text(encoding="utf-8"))
            self.assertIn("#SBATCH --account=YOUR_ACCOUNT", sbatch.read_text(encoding="utf-8"))

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
            self.assertIn('CONFIG="$SCRIPT_DIR"/../config.toml', text)
            self.assertIn('exec "$POCKETMC_BIN" run -c "$CONFIG"', text)
            self.assertNotIn(case_dir.resolve().as_posix(), text)


if __name__ == "__main__":
    unittest.main()
