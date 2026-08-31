from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.analysis.cache import load_analysis_cache, write_analysis_cache
from gcmc_port.analysis.models import (
    DatasetSpec,
    FrameRecord,
    MCMove,
    MoleculeFrame,
    PathSample,
    RunResult,
    VisitRecord,
)
from gcmc_port.analysis.runner import _run_plot_script


class SafeAnalysisCacheTests(unittest.TestCase):
    def test_json_npz_cache_round_trip_preserves_records_without_absolute_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "results" / "case-one"
            run_dir.mkdir(parents=True)
            source = root / "input" / "trajectory.xtc"
            source.parent.mkdir()
            source.write_text("trajectory", encoding="utf-8")
            dataset = DatasetSpec("case-one", "md", source.parent, None, source)
            molecule = MoleculeFrame(
                "SOL:1", 1, "SOL", (0.1, 0.2, 0.3), True, "42ARG", 0.25, "42ARG", "40ARG"
            )
            result = RunResult(
                dataset=dataset,
                frames=[FrameRecord(4, 2.0, (molecule,), 1, -12.5, 7, "insert")],
                visits=[VisitRecord("case-one", "SOL:1", 1, "SOL", 1, "entry", 4, 4, 2.0, 2.0, 0.0, 1, False, True, "42ARG")],
                path_samples=[PathSample("case-one", "SOL:1", 0, 4, 2.0, "inside", "42ARG", 0.25, True, (0.1, 0.2, 0.3), "42ARG", "40ARG")],
                mc_moves=[MCMove("case-one", 7, 2, "insert", True, -12.5, -1.5, 1)],
                warnings=["test warning"],
                metadata={"source": str(source), "nonfinite": float("nan")},
            )
            write_analysis_cache(run_dir, result, analysis_version=9, fingerprint="abc")

            metadata_text = (run_dir / ".analysis_cache" / "metadata.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root), metadata_text)
            self.assertTrue((run_dir / ".analysis_cache" / "records.npz").is_file())
            loaded = load_analysis_cache(
                run_dir, dataset, analysis_version=9, fingerprint="abc"
            )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.frames, result.frames)
            self.assertEqual(loaded.visits, result.visits)
            self.assertEqual(loaded.path_samples, result.path_samples)
            self.assertEqual(loaded.mc_moves, result.mc_moves)
            self.assertEqual(loaded.metadata["source"], str(source.resolve()))

    def test_legacy_pickle_file_is_never_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            sentinel = run_dir / "executed.txt"
            (run_dir / ".analysis_cache.pkl").write_text(
                f"legacy pickle placeholder; must not create {sentinel}", encoding="utf-8"
            )
            dataset = DatasetSpec("case", "md", run_dir, None, run_dir / "trajectory.xtc")
            loaded = load_analysis_cache(
                run_dir, dataset, analysis_version=1, fingerprint="unused"
            )
            self.assertIsNone(loaded)
            self.assertFalse(sentinel.exists())


class TrustedPlotExecutionTests(unittest.TestCase):
    def test_result_directory_python_is_not_imported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "tables").mkdir()
            sentinel = root / "executed.txt"
            generated = root / "plot_results.py"
            generated.write_text(
                "from pathlib import Path\nPath(__file__).with_name('executed.txt').write_text('bad')\n",
                encoding="utf-8",
            )
            with patch("gcmc_port.analysis.runner.render_result_plots") as render:
                _run_plot_script(generated, root)
            render.assert_called_once()
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
