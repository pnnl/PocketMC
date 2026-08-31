from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.cli import build_parser


def _subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices[name]
    raise AssertionError(f"Could not locate subparser {name}")


class BuildCavityHelpTests(unittest.TestCase):
    def test_top_level_parser_accepts_interactive_without_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--interactive"])

        self.assertTrue(args.interactive)
        self.assertIsNone(args.command)

    def test_run_help_explains_mask_build_is_separate(self) -> None:
        parser = build_parser()
        run = _subparser(parser, "run")
        help_text = run.format_help()

        self.assertIn("wizard-generated mask configs can auto-build it", help_text)
        self.assertIn("python build_cavity.py", help_text)
        self.assertIn("mask_file / mask_meta", help_text)

    def test_build_cavity_help_lists_short_flags_defaults_and_tradeoffs(self) -> None:
        parser = build_parser()
        cavity = _subparser(parser, "build-cavity")
        help_text = cavity.format_help()

        self.assertIn("-E, --exclude-residue", help_text)
        self.assertIn("This is a preprocessing step", help_text)
        self.assertIn("-x, --dx", help_text)
        self.assertIn("(default: 0.075)", help_text)
        self.assertIn("Lower values resolve narrower voids", help_text)
        self.assertIn("higher values make the cavity more conservative", help_text)

    def test_build_cavity_short_aliases_parse(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "build-cavity",
                "-f",
                "init.gro",
                "-m",
                "auto",
                "-E",
                "800ATC",
                "-x",
                "0.05",
                "-r",
                "0.08",
                "-R",
                "1.1",
                "-C",
                "3",
            ]
        )

        self.assertEqual(args.mode, "auto")
        self.assertEqual(args.exclude_residue, ["800ATC"])
        self.assertAlmostEqual(args.dx, 0.05)
        self.assertAlmostEqual(args.probe_radius, 0.08)
        self.assertAlmostEqual(args.search_radius, 1.1)
        self.assertEqual(args.candidate_limit, 3)


if __name__ == "__main__":
    unittest.main()
