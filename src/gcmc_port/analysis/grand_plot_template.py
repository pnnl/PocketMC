from __future__ import annotations

# ================== USER-EDITABLE GRAND-ALIGNMENT PLOTS ==================
# Edit the values below, save this file, then run:
#
#     python plot_grand_aligned.py
#
# Only saved grand-aligned NPZ files are read. Trajectories, lifetimes,
# occupancies, cavity membership, and alignment transforms are not recalculated.
STYLE = {
    # Export and fonts
    "figure_size": (9.0, 5.5),
    "density_figure_size": (9.2, 7.4),
    "dpi": 300,
    "font_size": 11,
    "title_font_size": 14,
    "axis_label_font_size": 12,
    "tick_font_size": 10,
    "legend_font_size": 9,

    # 2D/3D density colors.  By default every saved map gets its own automatic
    # 0..max scale with rounded 0.01/0.02/0.05-style colorbar increments.
    "density_cmap": "viridis",
    "difference_cmap": "coolwarm",
    "density_vmin": 0.0,
    "density_vmax": None,
    "grand_shared_color_scale": False,  # True uses one scale per comparable map group.

    # Shared axes. auto-content fits the lowest requested isosurface and the
    # visible substrate, not the much larger underlying interpolation grid.
    # per-map zooms each PNG independently around its own density + substrate;
    # shared-group keeps identical limits across comparable maps.
    # Set grand_axis_mode="manual" and supply six limits to override it.
    "grand_axis_mode": "auto-content",  # auto-content, manual, or full-grid
    "grand_axis_scope": "per-map",  # per-map or shared-group
    "grand_axis_padding_fraction": 0.10,
    "grand_axis_padding_A": 1.0,
    "density_3d_axis_limits": None,  # [xmin, xmax, ymin, ymax, zmin, zmax]
    "axis_limits": {},  # Optional 2D overrides: {"xy": [xmin,xmax,ymin,ymax], ...}
    "grid_visible": True,
    "grid_spacing_A": None,
    "grid_alpha": 0.22,
    "grand_square_2d_panels": True,
    "grand_2d_panel_spacing": 0.36,

    # 3D isosurfaces and camera
    "density_3d_isosurface_levels_percent": [8.0, 25.0, 50.0],
    "density_3d_opacity": 0.22,
    "density_3d_max_faces": 120000,
    "density_3d_surface_step_size": 1,
    "density_3d_legend_location": "upper right",
    "density_3d_elev": 30.0,
    "density_3d_azim": -60.0,
    "density_3d_roll": 0.0,

    # Aligned cavity boundary.  mask mode draws its projected/3D voxel shell;
    # sphere mode draws circles and a wireframe sphere.
    "cavity_boundary": True,
    "cavity_boundary_color": "#00E5FF",
    "cavity_boundary_line_width": 2.2,
    "cavity_boundary_line_style": "--",
    "cavity_boundary_alpha": 0.95,
    "cavity_boundary_line_width_3d": 0.75,
    "cavity_boundary_alpha_3d": 0.68,
    "cavity_boundary_point_size_3d": 5.0,
    "cavity_boundary_3d_max_points": 20000,

    # OPP/ATC substrate drawing
    "substrate_overlay": True,
    "substrate_show_hydrogens": False,
    "substrate_labels": True,
    "substrate_atom_size": 28.0,
    "substrate_atom_size_3d": 48.0,
    "substrate_atom_color_3d": None,  # None/"element", color name, or #RRGGBB
    "substrate_atom_brightness_3d": 1.8,
    "substrate_atom_opacity_3d": 1.0,
    "substrate_atom_depthshade_3d": False,
    "substrate_draw_on_top_3d": True,
    "substrate_atom_edge_color_3d": "white",
    "substrate_atom_edge_width_3d": 0.70,
    "substrate_bond_width": 1.8,
    "substrate_bond_color": "white",
    "substrate_bond_width_3d": 2.4,
    "substrate_bond_color_3d": "#707070",
    "substrate_bond_opacity_3d": 1.0,
    "substrate_label_color": "white",
    "substrate_label_color_3d": "#111111",
    "substrate_label_background_3d": "white",
    "substrate_label_background_opacity_3d": 0.72,
    "substrate_label_font_size": 9,
}
# ========================================================================

from pathlib import Path
import sys

# This compatibility block intentionally contains no checkout path. Installed
# environments resolve gcmc_port normally, and source-tree results use the
# relative fallback below.
# __GCMC_PORT_GENERATED_SOURCE_BEGIN__
_GENERATED_PACKAGE_SOURCE = None
if _GENERATED_PACKAGE_SOURCE:
    _generated_source = Path(_GENERATED_PACKAGE_SOURCE).expanduser()
    if (_generated_source / "gcmc_port" / "analysis" / "grand_alignment.py").exists():
        if str(_generated_source) not in sys.path:
            sys.path.insert(0, str(_generated_source))
# __GCMC_PORT_GENERATED_SOURCE_END__

# Source-checkout fallback; installed users resolve the package normally.
for _parent in Path(__file__).resolve().parents:
    for _source in (_parent / "src", _parent / "Python" / "src"):
        if (_source / "gcmc_port" / "analysis" / "grand_alignment.py").exists():
            if str(_source) not in sys.path:
                sys.path.insert(0, str(_source))
            break

try:
    from gcmc_port.analysis.grand_alignment import replot_grand_alignment
except ModuleNotFoundError as exc:
    if exc.name != "gcmc_port":
        raise
    raise ModuleNotFoundError(
        "Could not import gcmc_port. Activate the PocketMC environment, then rerun this script."
    ) from exc


def render(result_directory: str | Path | None = None) -> None:
    directory = (
        Path(result_directory).expanduser().resolve()
        if result_directory is not None
        else Path(__file__).resolve().parent
    )
    replot_grand_alignment(directory, dict(STYLE))


if __name__ == "__main__":
    render()
