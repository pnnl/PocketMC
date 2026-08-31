from __future__ import annotations

# ========================= USER-EDITABLE PLOT STYLE =========================
# Edit these values after analysis, then run:  python plot_results.py
# density_display_grid_A changes only rendered heatmap sampling; it never rewrites
# the scientific density_maps.npz grid. Use None to display the native analysis grid.
STYLE = {
    "figure_size": (9.0, 5.5),
    "density_figure_size": (8.0, 6.4),
    "dpi": 300,
    "font_size": 11,
    "title_font_size": 14,
    "axis_label_font_size": 12,
    "tick_font_size": 10,
    "legend_font_size": 9,
    "line_width": 1.8,
    "marker_size": 18.0,
    "occupancy_color": "#1f77b4",
    "rolling_color": "#d62728",
    "entry_color": "#d62728",
    "reentry_color": "#ff7f0e",
    "resident_color": "#555555",
    "cluster_colors": ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"],
    "pose_marker_size": 14.0,
    "difference_cmap": "coolwarm",
    "density_cmap": "viridis",
    "density_vmin": None,
    "density_vmax": None,
    "axis_limits": {},  # Example: {"occupancy": [0, 100, 0, 8], "xy": [-5, 20, -5, 20]}
    "density_display_grid_A": None,  # Example: 0.5; plotting only, safe after analysis.
    "grid_visible": True,
    "grid_spacing_A": None,  # Major grid-line spacing on coordinate plots.
    "grid_alpha": 0.22,
    "histogram_bins": 25,
    "rolling_window": 10,
    "density_3d_cutoff_percent": 8.0,
    "density_3d_isosurface_levels_percent": [8.0, 25.0, 50.0],
    # Plot-only 3D visibility controls.  Lower density opacity exposes atoms
    # inside the isosurface; 0 is invisible and 1 is opaque.
    "density_3d_opacity": 0.22,
    "density_3d_max_points": 35000,
    "density_3d_max_faces": 120000,
    "density_3d_surface_step_size": 1,
    "density_3d_legend_location": "upper right",
    "density_3d_axis_limits": None,  # [xmin, xmax, ymin, ymax, zmin, zmax]
    "density_3d_elev": 30.0,
    "density_3d_azim": -60.0,
    "density_3d_roll": 0.0,
    "substrate_overlay": True,
    "substrate_show_hydrogens": False,
    "substrate_labels": True,
    "substrate_atom_size": 28.0,
    "substrate_atom_size_3d": 48.0,
    # None/"element" preserves C/N/O/P element colors.  Set a Matplotlib color
    # such as "cyan" or "#FFD54F" to highlight every selected substrate atom.
    "substrate_atom_color_3d": None,
    "substrate_atom_brightness_3d": 1.8,  # 1=original element color; >1 is brighter.
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
# ============================================================================

from pathlib import Path
import sys

# Source-checkout fallback: installed users resolve the package normally.
for _parent in Path(__file__).resolve().parents:
    _candidate = _parent / "src"
    if (_candidate / "gcmc_port" / "analysis" / "plotting.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from gcmc_port.analysis.plotting import render_aggregate_pose_plots, render_result_plots


def render(result_directory: str | Path | None = None) -> None:
    directory = Path(result_directory) if result_directory is not None else Path(__file__).resolve().parent
    # Direct edits in this generated Python file must win over plot_style.json.
    # Interactive/CLI replotting passes no marker and continues to use the JSON.
    style = dict(STYLE)
    style["_ignore_saved_plot_style"] = True
    if (directory / "tables").exists():
        render_result_plots(directory, style)
        return
    for run_directory in sorted(path for path in directory.iterdir() if path.is_dir() and (path / "tables").exists()):
        render_result_plots(run_directory, style)
    render_aggregate_pose_plots(directory, style)


if __name__ == "__main__":
    render()
