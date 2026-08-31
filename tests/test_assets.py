from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.assets import default_asset_path, export_example_case
from gcmc_port.config import load_config


class AssetExportTests(unittest.TestCase):
    def test_export_example_case_keeps_tip3p_assets_in_bundled_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = export_example_case(Path(tmpdir) / "example-case")

            self.assertTrue((target / "init.gro").exists())
            self.assertTrue((target / "topol.top").exists())
            self.assertTrue((target / "config.example.toml").exists())
            self.assertFalse((target / "WAT.itp").exists())
            self.assertFalse((target / "COM.gro").exists())

            config = load_config(target / "config.example.toml")

        self.assertEqual(config.paths.water_itp, default_asset_path("WAT.itp"))
        self.assertEqual(config.paths.gas_gro, default_asset_path("COM.gro"))

    def test_config_can_resolve_bundled_co_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "topol.top").write_text("[ system ]\ndemo\n", encoding="utf-8")
            (root / "init.gro").write_text("demo\n0\n1.0 1.0 1.0\n", encoding="utf-8")
            config_path = root / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'project_root = "."',
                        'topology = "topol.top"',
                        'init_gro = "init.gro"',
                        'water_itp = "co/COM.itp"',
                        'gas_gro = "co/COM.gro"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.paths.water_itp, default_asset_path("co/COM.itp"))
        self.assertEqual(config.paths.gas_gro, default_asset_path("co/COM.gro"))
        self.assertTrue(default_asset_path("co/COM_atomtypes.itp").exists())


if __name__ == "__main__":
    unittest.main()
