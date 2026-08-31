from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _plt():
    import matplotlib
    # Every plot produced by this module is written to disk.  An interactive
    # terminal (including an SSH session on a headless compute node) must not
    # make Matplotlib select Qt/X11 and abort while trying to load xcb.
    if not matplotlib.get_backend().lower().startswith("agg"):
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _style_with_saved_overrides(result_root: Path, style: dict[str, Any]) -> dict[str, Any]:
    merged = dict(style)
    if bool(merged.pop("_ignore_saved_plot_style", False)):
        return merged
    path = result_root / "plot_style.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                selected = merged.get("_selected_plot_paths")
                merged.update(payload)
                if selected is not None:
                    merged["_selected_plot_paths"] = selected
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return merged


def _apply_global_style(style: dict[str, Any]) -> None:
    plt = _plt()
    plt.rcParams.update(
        {
            "font.size": float(style.get("font_size", 11)),
            "axes.titlesize": float(style.get("title_font_size", 14)),
            "axes.labelsize": float(style.get("axis_label_font_size", 12)),
            "xtick.labelsize": float(style.get("tick_font_size", 10)),
            "ytick.labelsize": float(style.get("tick_font_size", 10)),
            "legend.fontsize": float(style.get("legend_font_size", 9)),
        }
    )


def _save(fig: Any, path: Path, style: dict[str, Any]) -> None:
    selected = style.get("_selected_plot_paths")
    if selected and str(path.resolve()) not in set(selected):
        _plt().close(fig)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Axes that are not compatible with tight_layout.*")
        fig.tight_layout()
    fig.savefig(path, dpi=int(style["dpi"]), bbox_inches="tight")
    _plt().close(fig)


def _wants_plot(path: Path, style: dict[str, Any]) -> bool:
    selected = style.get("_selected_plot_paths")
    return not selected or str(path.resolve()) in set(selected)


def _format_axis(ax: Any, style: dict[str, Any], key: str) -> None:
    ax.tick_params(labelsize=int(style["tick_font_size"]))
    ax.title.set_fontsize(float(style.get("title_font_size", 14)))
    ax.xaxis.label.set_fontsize(float(style.get("axis_label_font_size", 12)))
    ax.yaxis.label.set_fontsize(float(style.get("axis_label_font_size", 12)))
    limits = style.get("axis_limits", {}).get(key)
    if limits and len(limits) == 4:
        ax.set_xlim(float(limits[0]), float(limits[1]))
        ax.set_ylim(float(limits[2]), float(limits[3]))
    spacing = style.get("grid_spacing_A")
    if spacing and key in {"xy", "xz", "yz", "pose_xy", "pose_xz", "pose_yz"}:
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(float(spacing)))
        ax.yaxis.set_major_locator(MultipleLocator(float(spacing)))
    ax.grid(bool(style.get("grid_visible", True)), alpha=float(style.get("grid_alpha", 0.22)))


def _resample_projection(
    image: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    style: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Change only the displayed heatmap grid; the saved scientific density stays untouched."""
    spacing = style.get("density_display_grid_A")
    if spacing in (None, "", 0, 0.0) or len(horizontal) < 2 or len(vertical) < 2:
        return image, horizontal, vertical
    spacing = float(spacing)
    if spacing <= 0:
        return image, horizontal, vertical
    target_horizontal = np.arange(horizontal[0], horizontal[-1] + spacing * 0.25, spacing)
    target_vertical = np.arange(vertical[0], vertical[-1] + spacing * 0.25, spacing)
    if target_horizontal.size < 2 or target_vertical.size < 2:
        return image, horizontal, vertical
    interpolator = RegularGridInterpolator(
        (vertical, horizontal), np.asarray(image, dtype=float), bounds_error=False, fill_value=np.nan
    )
    xx, yy = np.meshgrid(target_horizontal, target_vertical, indexing="xy")
    rendered = interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(yy.shape)
    return rendered, target_horizontal, target_vertical


_COVALENT_RADII = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05,
    "F": 0.57, "Cl": 1.02, "Br": 1.20, "I": 1.39, "Mg": 1.41, "Na": 1.66,
}
_ELEMENT_COLORS = {
    "H": "#dddddd", "C": "#333333", "N": "#3050f8", "O": "#e02020", "P": "#ff8000",
    "S": "#e6c400", "F": "#50d050", "Cl": "#1fbd1f", "Br": "#a62929", "I": "#940094",
    "Mg": "#8aff00", "Na": "#ab5cf2",
}


def _opacity(value: Any, label: str) -> float:
    opacity = float(value)
    if not 0.0 <= opacity <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1, got {opacity}")
    return opacity


def _substrate_colors_3d(elements: np.ndarray, style: dict[str, Any]) -> list[tuple[float, float, float]]:
    from matplotlib.colors import to_rgb

    override = style.get("substrate_atom_color_3d")
    use_elements = override is None or str(override).strip().lower() in {"", "auto", "element", "elements", "none"}
    base = (
        [_ELEMENT_COLORS.get(str(element).title(), "#ff69b4") for element in elements]
        if use_elements
        else [str(override)] * len(elements)
    )
    brightness = float(style.get("substrate_atom_brightness_3d", 1.8))
    if brightness <= 0.0:
        raise ValueError(f"substrate_atom_brightness_3d must be positive, got {brightness}")
    return [tuple(np.clip(np.asarray(to_rgb(color), dtype=float) * brightness, 0.0, 1.0)) for color in base]


def _overlay(path: Path) -> dict[str, np.ndarray] | None:
    overlay_path = path.parent / "substrate_overlay.npz"
    if not overlay_path.exists():
        return None
    with np.load(overlay_path) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _visible_overlay(overlay: dict[str, np.ndarray], style: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(overlay["positions_A"], dtype=float)
    elements = np.asarray(overlay.get("elements", np.asarray(["C"] * len(positions))), dtype=str)
    resnames = np.asarray(overlay.get("resnames", np.asarray(["SUB"] * len(positions))), dtype=str)
    resids = np.asarray(overlay.get("resids", np.zeros(len(positions), dtype=int)), dtype=int)
    keep = np.ones(len(positions), dtype=bool)
    if not bool(style.get("substrate_show_hydrogens", False)):
        keep &= np.char.upper(elements) != "H"
    return positions[keep], elements[keep], resnames[keep], resids[keep]


def _bond_pairs(positions: np.ndarray, elements: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for first in range(len(positions)):
        first_element = str(elements[first]).title()
        first_radius = _COVALENT_RADII.get(first_element, 0.77)
        for second in range(first + 1, len(positions)):
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            second_radius = _COVALENT_RADII.get(str(elements[second]).title(), 0.77)
            if 0.35 <= distance <= 1.25 * (first_radius + second_radius):
                pairs.append((first, second))
    return pairs


def _draw_substrate_2d(
    ax: Any,
    overlay: dict[str, np.ndarray] | None,
    horizontal: str,
    vertical: str,
    style: dict[str, Any],
) -> None:
    if overlay is None or not bool(style.get("substrate_overlay", True)):
        return
    positions, elements, resnames, resids = _visible_overlay(overlay, style)
    if positions.size == 0:
        return
    dimensions = {"x": 0, "y": 1, "z": 2}
    for first, second in _bond_pairs(positions, elements):
        ax.plot(
            positions[[first, second], dimensions[horizontal]],
            positions[[first, second], dimensions[vertical]],
            color=style.get("substrate_bond_color", "white"),
            linewidth=float(style.get("substrate_bond_width", 1.8)),
            alpha=0.95,
            zorder=5,
        )
    colors = [_ELEMENT_COLORS.get(str(element).title(), "#ff69b4") for element in elements]
    ax.scatter(
        positions[:, dimensions[horizontal]], positions[:, dimensions[vertical]],
        c=colors, s=float(style.get("substrate_atom_size", 28.0)), edgecolors="white", linewidths=0.45,
        zorder=6, label="Substrate",
    )
    if bool(style.get("substrate_labels", True)):
        seen: set[tuple[str, int]] = set()
        for resname, resid in zip(resnames, resids):
            key = (str(resname), int(resid))
            if key in seen:
                continue
            seen.add(key)
            selected = (resnames == resname) & (resids == resid)
            center = positions[selected].mean(axis=0)
            label = str(resname) if int(resid) == 0 else f"{resname}{int(resid)}"
            ax.text(
                center[dimensions[horizontal]], center[dimensions[vertical]], label,
                color=style.get("substrate_label_color", "white"),
                fontsize=float(style.get("substrate_label_font_size", 9)), fontweight="bold",
                ha="center", va="center", zorder=7,
                bbox={"facecolor": "black", "alpha": 0.38, "edgecolor": "none", "pad": 1.2},
            )


def _draw_substrate_3d(ax: Any, overlay: dict[str, np.ndarray] | None, style: dict[str, Any]) -> bool:
    if overlay is None or not bool(style.get("substrate_overlay", True)):
        return False
    positions, elements, resnames, resids = _visible_overlay(overlay, style)
    if positions.size == 0:
        return False
    bond_opacity = _opacity(style.get("substrate_bond_opacity_3d", 1.0), "substrate_bond_opacity_3d")
    for first, second in _bond_pairs(positions, elements):
        ax.plot(
            *positions[[first, second]].T,
            color=style.get("substrate_bond_color_3d", "#707070"),
            linewidth=float(style.get("substrate_bond_width_3d", 2.4)),
            alpha=bond_opacity,
            zorder=20,
        )
    colors = _substrate_colors_3d(elements, style)
    ax.scatter(
        *positions.T,
        c=colors,
        s=float(style.get("substrate_atom_size_3d", 48.0)),
        alpha=_opacity(style.get("substrate_atom_opacity_3d", 1.0), "substrate_atom_opacity_3d"),
        edgecolors=style.get("substrate_atom_edge_color_3d", "white"),
        linewidths=float(style.get("substrate_atom_edge_width_3d", 0.70)),
        depthshade=bool(style.get("substrate_atom_depthshade_3d", False)),
        zorder=21,
    )
    if bool(style.get("substrate_labels", True)):
        seen: set[tuple[str, int]] = set()
        for resname, resid in zip(resnames, resids):
            key = (str(resname), int(resid))
            if key in seen:
                continue
            seen.add(key)
            selected = (resnames == resname) & (resids == resid)
            center = positions[selected].mean(axis=0)
            label = str(resname) if int(resid) == 0 else f"{resname}{int(resid)}"
            background = style.get("substrate_label_background_3d", "white")
            bbox = None
            if background not in {None, "", "none", "None"}:
                bbox = {
                    "facecolor": background,
                    "alpha": _opacity(
                        style.get("substrate_label_background_opacity_3d", 0.72),
                        "substrate_label_background_opacity_3d",
                    ),
                    "edgecolor": "none",
                    "pad": 1.2,
                }
            ax.text(
                *center,
                label,
                color=style.get("substrate_label_color_3d", "#111111"),
                fontsize=float(style.get("substrate_label_font_size", 9)),
                fontweight="bold",
                bbox=bbox,
                zorder=22,
            )
    return True


def _voxel_surface(rho: np.ndarray, axes: dict[str, np.ndarray], level: float) -> list[list[tuple[float, float, float]]]:
    mask = np.isfinite(rho) & (rho >= level)
    if not np.any(mask):
        return []
    steps = [float(np.median(np.diff(axes[name]))) if len(axes[name]) > 1 else 1.0 for name in "xyz"]
    polygons: list[list[tuple[float, float, float]]] = []
    directions = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    for index in np.argwhere(mask):
        center = np.asarray([axes[name][index[axis]] for axis, name in enumerate("xyz")], dtype=float)
        low = center - np.asarray(steps) * 0.5
        high = center + np.asarray(steps) * 0.5
        x0, y0, z0 = low
        x1, y1, z1 = high
        faces = (
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
            [(x0, y0, z0), (x0, y0, z1), (x1, y0, z1), (x1, y0, z0)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)],
        )
        for direction, face in zip(directions, faces):
            neighbor = index + np.asarray(direction)
            if np.any(neighbor < 0) or np.any(neighbor >= np.asarray(mask.shape)) or not mask[tuple(neighbor)]:
                polygons.append(face)
    return polygons


def _density_isosurfaces(
    fig: Any,
    ax: Any,
    rho: np.ndarray,
    axes: dict[str, np.ndarray],
    style: dict[str, Any],
    density_label: str,
    overlay: dict[str, np.ndarray] | None = None,
) -> None:
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    maximum = float(np.nanmax(rho)) if rho.size else 0.0
    density_opacity = _opacity(style.get("density_3d_opacity", 0.22), "density_3d_opacity")
    if bool(style.get("substrate_draw_on_top_3d", True)) and hasattr(ax, "computed_zorder"):
        # Keep the substrate legible through translucent surfaces.  This changes
        # only artist compositing; all coordinates and density values are untouched.
        ax.computed_zorder = False
    percentages = [float(value) for value in style.get("density_3d_isosurface_levels_percent", [8.0, 25.0, 50.0])]
    percentages = sorted({value for value in percentages if 0.0 < value < 100.0})
    levels = [maximum * value / 100.0 for value in percentages if maximum > 0.0]
    cmap = colormaps.get_cmap(style.get("density_cmap", "viridis"))
    norm = Normalize(vmin=0.0, vmax=max(maximum, 1e-30))
    handles: list[Any] = []
    for percentage, level in zip(percentages, levels):
        color = cmap(norm(level))
        collection: Any | None = None
        try:
            from skimage.measure import marching_cubes

            spacing = tuple(float(np.median(np.diff(axes[name]))) if len(axes[name]) > 1 else 1.0 for name in "xyz")
            vertices, faces, _normals, _values = marching_cubes(
                np.nan_to_num(rho, nan=0.0), level=level, spacing=spacing,
                step_size=max(1, int(style.get("density_3d_surface_step_size", 1))),
            )
            vertices += np.asarray([axes[name][0] for name in "xyz"])
            maximum_faces = max(1000, int(style.get("density_3d_max_faces", 120000)))
            if len(faces) > maximum_faces:
                faces = faces[:: int(math.ceil(len(faces) / maximum_faces))]
            collection = Poly3DCollection(vertices[faces], facecolor=color, edgecolor="none")
        except (ImportError, RuntimeError, ValueError):
            polygons = _voxel_surface(rho, axes, level)
            maximum_faces = max(1000, int(style.get("density_3d_max_faces", 120000)))
            if len(polygons) > maximum_faces:
                polygons = polygons[:: int(math.ceil(len(polygons) / maximum_faces))]
            if polygons:
                collection = Poly3DCollection(polygons, facecolor=color, edgecolor="none")
        if collection is None:
            continue
        collection.set_alpha(density_opacity)
        collection.set_zorder(1)
        ax.add_collection3d(collection)
        handles.append(Patch(facecolor=color, alpha=density_opacity, label=f"{percentage:g}% max = {level:.3g}"))
    substrate_shown = _draw_substrate_3d(ax, overlay, style)
    limits = style.get("density_3d_axis_limits")
    if limits and len(limits) == 6:
        x_limits = (float(limits[0]), float(limits[1]))
        y_limits = (float(limits[2]), float(limits[3]))
        z_limits = (float(limits[4]), float(limits[5]))
    else:
        x_limits = (float(axes["x"][0]), float(axes["x"][-1]))
        y_limits = (float(axes["y"][0]), float(axes["y"][-1]))
        z_limits = (float(axes["z"][0]), float(axes["z"][-1]))
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_zlim(*z_limits)
    ax.set_box_aspect(
        tuple(max(high - low, 1e-6) for low, high in (x_limits, y_limits, z_limits))
    )
    view = {
        "elev": float(style.get("density_3d_elev", 30.0)),
        "azim": float(style.get("density_3d_azim", -60.0)),
    }
    try:
        ax.view_init(**view, roll=float(style.get("density_3d_roll", 0.0)))
    except TypeError:  # Matplotlib releases before view_init(..., roll=...).
        ax.view_init(**view)
    scalar = ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array(np.asarray([0.0, maximum]))
    fig.colorbar(scalar, ax=ax, shrink=0.68, pad=0.10, label=density_label)
    if substrate_shown:
        legend_color = _substrate_colors_3d(np.asarray(["C"]), style)[0]
        handles.append(
            Line2D(
                [0], [0], marker="o",
                color=style.get("substrate_bond_color_3d", "#707070"),
                markerfacecolor=legend_color,
                markeredgecolor=style.get("substrate_atom_edge_color_3d", "white"),
                markersize=7, label="Substrate", linewidth=1.8,
            )
        )
    if handles:
        ax.legend(handles=handles, loc=style.get("density_3d_legend_location", "upper right"), fontsize=float(style.get("legend_font_size", 9)))


def _kaplan_meier(durations: np.ndarray, censored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if durations.size == 0:
        return np.asarray([0.0]), np.asarray([1.0])
    timeline = np.sort(np.unique(durations))
    survival = 1.0
    x = [0.0]
    y = [1.0]
    for time in timeline:
        at_risk = int(np.count_nonzero(durations >= time))
        observed = int(np.count_nonzero((durations == time) & (~censored)))
        if at_risk and observed:
            survival *= 1.0 - observed / at_risk
        x.append(float(time))
        y.append(survival)
    return np.asarray(x), np.asarray(y)


def _occupancy_plots(directory: Path, style: dict[str, Any], kind: str) -> None:
    plt = _plt()
    rows = _rows(directory / "tables" / "frames.tsv")
    if not rows:
        return
    x = np.asarray([float(row["time_ps"]) for row in rows])
    occupancy = np.asarray([float(row["occupancy"]) for row in rows])
    x_label = "Time (ps)" if kind == "md" else "Accepted state"
    title_suffix = "" if kind == "md" else " (non-equilibrium accepted states)"
    fig, ax = plt.subplots(figsize=style["figure_size"])
    ax.plot(x, occupancy, color=style["occupancy_color"], linewidth=style["line_width"], label="Occupancy")
    window = max(1, int(style["rolling_window"]))
    if occupancy.size >= window and window > 1:
        rolling = np.convolve(occupancy, np.ones(window) / window, mode="valid")
        ax.plot(x[window - 1 :], rolling, color=style["rolling_color"], linewidth=style["line_width"], label=f"Rolling mean ({window})")
    ax.set_title("Cavity occupancy profile" + title_suffix, fontsize=style["title_font_size"])
    ax.set_xlabel(x_label, fontsize=style["axis_label_font_size"])
    ax.set_ylabel("Molecules in cavity", fontsize=style["axis_label_font_size"])
    ax.legend()
    _format_axis(ax, style, "occupancy")
    _save(fig, directory / "plots" / "occupancy_temporal.png", style)

    if kind == "pocketmc":
        trials = np.asarray([float(row["trial"]) if row.get("trial") else np.nan for row in rows])
        finite_trials = np.isfinite(trials)
        if np.any(finite_trials):
            fig, ax = plt.subplots(figsize=style["figure_size"])
            ax.step(
                trials[finite_trials],
                occupancy[finite_trials],
                where="post",
                color=style["occupancy_color"],
                linewidth=style["line_width"],
            )
            ax.set_title("Cavity occupancy by MC trial (accepted states only)", fontsize=style["title_font_size"])
            ax.set_xlabel("MC trial (not physical time)", fontsize=style["axis_label_font_size"])
            ax.set_ylabel("Molecules in cavity", fontsize=style["axis_label_font_size"])
            _format_axis(ax, style, "occupancy_trial")
            _save(fig, directory / "plots" / "occupancy_by_trial.png", style)

    fig, ax = plt.subplots(figsize=style["figure_size"])
    bins = np.arange(int(occupancy.min()), int(occupancy.max()) + 2) - 0.5
    ax.hist(occupancy, bins=bins, color=style["occupancy_color"], edgecolor="white", alpha=0.85)
    ax.set_title("Cavity occupancy distribution" + title_suffix, fontsize=style["title_font_size"])
    ax.set_xlabel("Molecules in cavity", fontsize=style["axis_label_font_size"])
    ax.set_ylabel("Frames/states", fontsize=style["axis_label_font_size"])
    _format_axis(ax, style, "occupancy_distribution")
    _save(fig, directory / "plots" / "occupancy_distribution.png", style)

    if kind == "pocketmc":
        energy = np.asarray([float(row["energy_kj_mol"]) if row.get("energy_kj_mol") else np.nan for row in rows])
        finite = np.isfinite(energy)
        if np.any(finite):
            fig, ax = plt.subplots(figsize=style["figure_size"])
            ax.plot(x[finite], energy[finite], color="#9467bd", linewidth=style["line_width"])
            ax.set_title("Accepted-state energy (not a physical time series)", fontsize=style["title_font_size"])
            ax.set_xlabel(x_label, fontsize=style["axis_label_font_size"])
            ax.set_ylabel("Energy (kJ/mol)", fontsize=style["axis_label_font_size"])
            _format_axis(ax, style, "energy")
            _save(fig, directory / "plots" / "energy_by_accepted_state.png", style)


def _lifetime_plots(directory: Path, style: dict[str, Any]) -> None:
    plt = _plt()
    rows = _rows(directory / "tables" / "events.tsv")
    if not rows:
        for name in ("lifetime_distribution.png", "residence_timeline.png"):
            (directory / "plots" / name).unlink(missing_ok=True)
        return
    durations = np.asarray([float(row["lifetime_ps"]) / 1000.0 for row in rows])
    left = np.asarray([row["left_censored"].lower() == "true" for row in rows])
    right = np.asarray([row["right_censored"].lower() == "true" for row in rows])
    usable = ~left
    fig, axes = plt.subplots(1, 2, figsize=(style["figure_size"][0] * 1.55, style["figure_size"][1]))
    complete = durations[~(left | right)]
    censored = durations[left | right]
    axes[0].hist(complete, bins=int(style["histogram_bins"]), alpha=0.82, color=style["entry_color"], label="Complete")
    if censored.size:
        axes[0].hist(censored, bins=int(style["histogram_bins"]), alpha=0.50, color=style["resident_color"], label="Censored")
    axes[0].set_xlabel("Visit duration (ns)")
    axes[0].set_ylabel("Visits")
    axes[0].set_title("Residence-time distribution")
    axes[0].legend()
    km_x, km_y = _kaplan_meier(durations[usable], right[usable])
    axes[1].step(km_x, km_y, where="post", color=style["occupancy_color"], linewidth=style["line_width"])
    axes[1].set_xlabel("Visit duration (ns)")
    axes[1].set_ylabel("Survival probability")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_title("Kaplan–Meier (left-censored visits excluded)")
    for axis in axes:
        _format_axis(axis, style, "lifetime")
    _save(fig, directory / "plots" / "lifetime_distribution.png", style)

    uids = sorted({row["molecule_uid"] for row in rows})
    if len(uids) > 80:
        totals = {uid: sum(float(row["lifetime_ps"]) for row in rows if row["molecule_uid"] == uid) for uid in uids}
        uids = sorted(uids, key=totals.get, reverse=True)[:80]
    positions = {uid: index for index, uid in enumerate(uids)}
    fig_height = max(style["figure_size"][1], min(18.0, 1.5 + 0.23 * len(uids)))
    fig, ax = plt.subplots(figsize=(style["figure_size"][0], fig_height))
    for row in rows:
        uid = row["molecule_uid"]
        if uid not in positions:
            continue
        color = style["resident_color"] if row["left_censored"].lower() == "true" else (style["reentry_color"] if row["event_type"] == "reentry" else style["entry_color"])
        start = float(row["start_ps"]) / 1000.0
        stop = float(row["end_ps"]) / 1000.0
        ax.plot([start, stop], [positions[uid], positions[uid]], color=color, linewidth=3.2, solid_capstyle="round")
    ax.set_yticks(range(len(uids)), uids)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Molecule UID")
    ax.set_title("Cavity residence timeline")
    _format_axis(ax, style, "timeline")
    _save(fig, directory / "plots" / "residence_timeline.png", style)


def _path_plot(directory: Path, style: dict[str, Any]) -> None:
    plt = _plt()
    rows = _rows(directory / "tables" / "paths.tsv")
    if not rows:
        (directory / "plots" / "path_transition_matrix.png").unlink(missing_ok=True)
        return
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["molecule_uid"], []).append(row)
    transitions: dict[tuple[str, str], int] = {}
    label_counts: dict[str, int] = {}
    for samples in grouped.values():
        samples.sort(key=lambda row: int(row["sample_index"]))
        labels = [row["label"] for row in samples]
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        for first, second in zip(labels, labels[1:]):
            if first != second:
                transitions[(first, second)] = transitions.get((first, second), 0) + 1
    labels = [item[0] for item in sorted(label_counts.items(), key=lambda item: item[1], reverse=True)[:25]]
    if not labels:
        return
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    positions = {label: index for index, label in enumerate(labels)}
    for (first, second), count in transitions.items():
        if first in positions and second in positions:
            matrix[positions[first], positions[second]] += count
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.45), max(6.0, len(labels) * 0.42)))
    image = ax.imshow(matrix, cmap="magma", origin="lower")
    ax.set_xticks(range(len(labels)), labels, rotation=90)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Sampled residue/Bulk path transitions")
    fig.colorbar(image, ax=ax, label="Transition count")
    _save(fig, directory / "plots" / "path_transition_matrix.png", style)


def _mc_acceptance_plot(directory: Path, style: dict[str, Any]) -> None:
    plt = _plt()
    rows = _rows(directory / "tables" / "mc_moves.tsv")
    if not rows:
        return
    names = ["I", "D", "R", "T"]
    totals = [sum(row["move"] == name for row in rows) for name in names]
    accepted = [sum(row["move"] == name and row["accepted"].lower() == "true" for row in rows) for name in names]
    rates = [a / t if t else 0.0 for a, t in zip(accepted, totals)]
    fig, ax = plt.subplots(figsize=style["figure_size"])
    bars = ax.bar(names, rates, color=["#2ca02c", "#d62728", "#1f77b4", "#ff7f0e"])
    for bar, value, total in zip(bars, accepted, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value}/{total}", ha="center", va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Move (insert/delete/rotate/translate)")
    ax.set_ylabel("Acceptance fraction")
    ax.set_title("PocketMC move acceptance (non-equilibrium search)")
    _format_axis(ax, style, "acceptance")
    _save(fig, directory / "plots" / "acceptance_by_move.png", style)


def _density_plots(directory: Path, style: dict[str, Any]) -> None:
    npz_path = directory / "density" / "density_maps.npz"
    if not npz_path.exists():
        return
    plt = _plt()
    substrate_overlay = _overlay(npz_path)
    with np.load(npz_path) as data:
        rho = np.asarray(data["rho"], dtype=float)
        axes = {name: np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz"}
        density_label = "Density"
        meta_path = directory / "density" / "density_maps.meta.json"
        if meta_path.exists():
            try:
                quantity = json.loads(meta_path.read_text(encoding="utf-8")).get("quantity_shown", "density")
                density_label = (
                    "Occupancy density (molecules A$^{-3}$)"
                    if quantity == "occupancy"
                    else "Probability density (A$^{-3}$)"
                )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        for plane, horizontal, vertical in (("xy", "x", "y"), ("xz", "x", "z"), ("yz", "y", "z")):
            output_path = directory / "plots" / f"density_{plane}.png"
            if not _wants_plot(output_path, style):
                continue
            image = np.asarray(data[f"{plane}_projection"], dtype=float)
            image, shown_horizontal, shown_vertical = _resample_projection(
                image, axes[horizontal], axes[vertical], style
            )
            fig, ax = plt.subplots(figsize=style["density_figure_size"])
            shown = ax.imshow(
                image,
                origin="lower",
                extent=(shown_horizontal[0], shown_horizontal[-1], shown_vertical[0], shown_vertical[-1]),
                aspect="equal",
                cmap=style["density_cmap"],
                vmin=style["density_vmin"],
                vmax=style["density_vmax"],
            )
            ax.set_xlabel(f"{horizontal.upper()} (Å)")
            ax.set_ylabel(f"{vertical.upper()} (Å)")
            ax.set_title(f"{plane.upper()} molecule occupancy-density projection")
            fig.colorbar(shown, ax=ax, label="Projected density (molecules A$^{-2}$)")
            _draw_substrate_2d(ax, substrate_overlay, horizontal, vertical, style)
            _format_axis(ax, style, plane)
            _save(fig, output_path, style)

        output_3d = directory / "plots" / "density_3d.png"
        if _wants_plot(output_3d, style):
            fig = plt.figure(figsize=style["density_figure_size"])
            ax = fig.add_subplot(111, projection="3d")
            _density_isosurfaces(fig, ax, rho, axes, style, density_label, substrate_overlay)
            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")
            ax.set_zlabel("Z (Å)")
            ax.set_title("3D density isosurfaces")
            _save(fig, output_3d, style)


def _pose_plots(directory: Path, style: dict[str, Any]) -> None:
    rows = _rows(directory / "poses" / "pose_assignments.tsv")
    summary = _rows(directory / "poses" / "cluster_summary.tsv")
    if not rows or not summary:
        return
    plt = _plt()
    colors = list(style.get("cluster_colors", ["#0072B2", "#D55E00", "#009E73"]))
    clusters = np.asarray([int(row["cluster"]) for row in rows], dtype=int)
    times = np.asarray([float(row["time_ps"]) / 1000.0 for row in rows], dtype=float)
    pc1 = np.asarray([float(row.get("pc1", 0.0)) for row in rows], dtype=float)
    pc2 = np.asarray([float(row.get("pc2", 0.0)) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=style["figure_size"])
    for cluster in sorted(set(clusters)):
        selected = clusters == cluster
        ax.scatter(
            pc1[selected], pc2[selected], s=style.get("pose_marker_size", 14.0),
            color=colors[(cluster - 1) % len(colors)], alpha=0.55, linewidths=0, label=f"Cluster {cluster}",
        )
    ax.set_xlabel("Pose PC1")
    ax.set_ylabel("Pose PC2")
    ax.set_title("Substrate pose space (training PCA; all analyzed frames projected)")
    ax.legend()
    _format_axis(ax, style, "pose_pca")
    _save(fig, directory / "poses" / "plots" / "pose_pca.png", style)

    fig, ax = plt.subplots(figsize=style["figure_size"])
    ax.scatter(times, clusters, c=[colors[(value - 1) % len(colors)] for value in clusters], s=8, linewidths=0)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Common cluster")
    ax.set_yticks(sorted(set(clusters)))
    ax.set_title("Substrate pose state over physical MD time")
    _format_axis(ax, style, "pose_time")
    _save(fig, directory / "poses" / "plots" / "pose_temporal.png", style)

    summary.sort(key=lambda row: int(row["cluster"]))
    fig, ax = plt.subplots(figsize=style["figure_size"])
    labels = [f"C{row['cluster']}" for row in summary]
    values = [float(row["population"]) for row in summary]
    ax.bar(labels, values, color=[colors[index % len(colors)] for index in range(len(values))])
    ax.set_ylim(0, max(1.0, max(values) * 1.12))
    ax.set_xlabel("Common pose cluster")
    ax.set_ylabel("Frame population")
    ax.set_title("Substrate pose populations")
    _format_axis(ax, style, "pose_population")
    _save(fig, directory / "poses" / "plots" / "pose_populations.png", style)

    base_columns = {
        "run_id", "system_id", "comparison_group", "replica", "sweep", "frame", "time_ps",
        "cluster", "distance_to_centroid", "pc1", "pc2",
    }
    descriptor_columns = [name for name in rows[0] if name not in base_columns]
    if descriptor_columns:
        columns = min(3, len(descriptor_columns))
        plot_rows = int(math.ceil(len(descriptor_columns) / columns))
        fig, descriptor_axes = plt.subplots(plot_rows, columns, figsize=(style["figure_size"][0] * columns / 2.0, 4.2 * plot_rows), squeeze=False)
        cluster_ids = sorted(set(clusters))
        for axis, name in zip(descriptor_axes.flat, descriptor_columns):
            values = [
                [float(row[name]) for row in rows if int(row["cluster"]) == cluster and row.get(name, "") != ""]
                for cluster in cluster_ids
            ]
            axis.boxplot(values, tick_labels=[f"C{cluster}" for cluster in cluster_ids], showfliers=False)
            axis.set_title(name)
            axis.set_xlabel("Common cluster")
            axis.set_ylabel("Descriptor value")
            axis.grid(True, axis="y", alpha=0.2)
        for axis in descriptor_axes.flat[len(descriptor_columns):]:
            axis.set_visible(False)
        _save(fig, directory / "poses" / "plots" / "pose_descriptor_distributions.png", style)

    transitions = _rows(directory / "poses" / "cluster_transitions.tsv")
    if transitions:
        count = max(max(int(row["from_cluster"]), int(row["to_cluster"])) for row in transitions)
        matrix = np.zeros((count, count), dtype=float)
        for row in transitions:
            matrix[int(row["from_cluster"]) - 1, int(row["to_cluster"]) - 1] = float(row["count"])
        fig, ax = plt.subplots(figsize=style["density_figure_size"])
        shown = ax.imshow(matrix, cmap="magma", origin="lower")
        ax.set_xticks(range(count), [f"C{value}" for value in range(1, count + 1)])
        ax.set_yticks(range(count), [f"C{value}" for value in range(1, count + 1)])
        ax.set_xlabel("To cluster")
        ax.set_ylabel("From cluster")
        ax.set_title("Pose-state transition counts")
        fig.colorbar(shown, ax=ax)
        _save(fig, directory / "poses" / "plots" / "pose_transition_matrix.png", style)

    for cluster_dir in sorted((directory / "poses").glob("cluster_*")):
        for frame_name in ("pocket-frame", "substrate-frame"):
            npz_path = cluster_dir / frame_name / "density_maps.npz"
            if not npz_path.exists():
                continue
            with np.load(npz_path) as data:
                axes = {name: np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz"}
                rho = np.asarray(data["rho_conditional"], dtype=float)
                substrate_overlay = _overlay(npz_path)
                for plane, horizontal, vertical in (("xy", "x", "y"), ("xz", "x", "z"), ("yz", "y", "z")):
                    output_path = cluster_dir / "plots" / f"{frame_name}_{plane}.png"
                    if not _wants_plot(output_path, style):
                        continue
                    image = np.asarray(data[f"{plane}_projection"], dtype=float)
                    image, shown_horizontal, shown_vertical = _resample_projection(
                        image, axes[horizontal], axes[vertical], style
                    )
                    fig, ax = plt.subplots(figsize=style["density_figure_size"])
                    shown = ax.imshow(
                        image, origin="lower",
                        extent=(shown_horizontal[0], shown_horizontal[-1], shown_vertical[0], shown_vertical[-1]),
                        aspect="equal", cmap=style["density_cmap"], vmin=style["density_vmin"], vmax=style["density_vmax"],
                    )
                    ax.set_xlabel(f"{horizontal.upper()} (A)")
                    ax.set_ylabel(f"{vertical.upper()} (A)")
                    ax.set_title(f"{cluster_dir.name} {frame_name} {plane.upper()} conditional hydration")
                    fig.colorbar(shown, ax=ax, label="Molecules A$^{-2}$ per cluster frame")
                    _draw_substrate_2d(ax, substrate_overlay, horizontal, vertical, style)
                    _format_axis(ax, style, f"pose_{plane}")
                    _save(fig, output_path, style)
                output_3d = cluster_dir / "plots" / f"{frame_name}_3d.png"
                if _wants_plot(output_3d, style):
                    fig = plt.figure(figsize=style["density_figure_size"])
                    ax = fig.add_subplot(111, projection="3d")
                    _density_isosurfaces(
                        fig,
                        ax,
                        rho,
                        axes,
                        style,
                        "Conditional occupancy density (molecules A$^{-3}$)",
                        substrate_overlay,
                    )
                    ax.set_xlabel("X (A)")
                    ax.set_ylabel("Y (A)")
                    ax.set_zlabel("Z (A)")
                    ax.set_title(f"{cluster_dir.name} {frame_name} 3D hydration isosurfaces")
                    _save(fig, output_3d, style)


def render_aggregate_pose_plots(result_root: Path, style: dict[str, Any]) -> None:
    style = _style_with_saved_overrides(result_root, style)
    _apply_global_style(style)
    pose_root = result_root / "aggregate" / "pose-groups"
    if not pose_root.exists():
        return
    plt = _plt()
    for group_dir in (path for path in pose_root.iterdir() if path.is_dir()):
        rows = _rows(group_dir / "system_cluster_summary.tsv")
        if rows:
            systems = sorted({row["system_id"] for row in rows})
            clusters = sorted({int(row["cluster"]) for row in rows})
            width = 0.8 / max(len(systems), 1)
            fig, ax = plt.subplots(figsize=style["figure_size"])
            for index, system in enumerate(systems):
                selected = {int(row["cluster"]): row for row in rows if row["system_id"] == system}
                values = [float(selected[cluster]["mean_population"]) if cluster in selected else 0.0 for cluster in clusters]
                positions = np.arange(len(clusters)) - 0.4 + width / 2 + index * width
                ax.bar(positions, values, width=width, label=system)
            ax.set_xticks(range(len(clusters)), [f"C{cluster}" for cluster in clusters])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Common pose cluster")
            ax.set_ylabel("Replica-mean population")
            ax.set_title(f"{group_dir.name}: homolog/replica pose populations")
            ax.legend()
            _format_axis(ax, style, "pose_aggregate_population")
            _save(fig, group_dir / "system_cluster_populations.png", style)
    for cluster_dir in pose_root.glob("*/cluster_*"):
        means = sorted(cluster_dir.glob("**/*.mean_density.npz"))
        loaded: list[tuple[Path, np.ndarray, dict[str, np.ndarray], float]] = []
        for path in means:
            with np.load(path) as data:
                loaded.append((path, np.asarray(data["rho"], dtype=float), {name: np.asarray(data[f"{name}_A"]) for name in "xyz"}, float(data["bin_A"]) if "bin_A" in data.files else 1.0))
        if loaded:
            projections = [
                projection
                for _path, rho, _axes, bin_a in loaded
                for projection in (rho.sum(axis=2).T * bin_a, rho.sum(axis=1).T * bin_a, rho.sum(axis=0).T * bin_a)
            ]
            vmax = max(float(np.nanmax(item)) for item in projections)
            for path, rho, axes, bin_a in loaded:
                substrate_overlay = _overlay(path)
                output_2d = path.with_suffix(".png")
                if _wants_plot(output_2d, style):
                    fig, plot_axes = plt.subplots(1, 3, figsize=(style["figure_size"][0] * 1.65, style["figure_size"][1]))
                    for axis, (plane, image, horizontal, vertical) in zip(
                        plot_axes,
                        (("XY", rho.sum(axis=2).T * bin_a, "x", "y"), ("XZ", rho.sum(axis=1).T * bin_a, "x", "z"), ("YZ", rho.sum(axis=0).T * bin_a, "y", "z")),
                    ):
                        image, shown_horizontal, shown_vertical = _resample_projection(
                            image, axes[horizontal], axes[vertical], style
                        )
                        shown = axis.imshow(
                            image, origin="lower", aspect="equal", cmap=style["density_cmap"], vmin=0.0,
                            vmax=style["density_vmax"] if style["density_vmax"] is not None else vmax,
                            extent=(shown_horizontal[0], shown_horizontal[-1], shown_vertical[0], shown_vertical[-1]),
                        )
                        axis.set_title(plane)
                        axis.set_xlabel(f"{horizontal.upper()} (A)")
                        axis.set_ylabel(f"{vertical.upper()} (A)")
                        _draw_substrate_2d(axis, substrate_overlay, horizontal, vertical, style)
                        _format_axis(axis, style, f"pose_{plane.lower()}")
                    fig.colorbar(shown, ax=plot_axes.tolist(), label="Shared-scale projected occupancy")
                    fig.suptitle(path.stem)
                    _save(fig, output_2d, style)
                output_3d = path.with_name(path.stem + "_3d.png")
                if _wants_plot(output_3d, style):
                    fig = plt.figure(figsize=style["density_figure_size"])
                    axis_3d = fig.add_subplot(111, projection="3d")
                    _density_isosurfaces(
                        fig,
                        axis_3d,
                        rho,
                        axes,
                        style,
                        "Mean conditional occupancy density (molecules A$^{-3}$)",
                        substrate_overlay,
                    )
                    axis_3d.set_xlabel("X (A)")
                    axis_3d.set_ylabel("Y (A)")
                    axis_3d.set_zlabel("Z (A)")
                    axis_3d.set_title(f"{path.stem}: shared-frame density isosurfaces")
                    _save(fig, output_3d, style)
        difference_loaded: list[tuple[Path, np.ndarray, dict[str, np.ndarray], float]] = []
        for path in cluster_dir.glob("**/difference.*.npz"):
            with np.load(path) as data:
                rho = np.asarray(data["rho_difference"], dtype=float)
                axes = {name: np.asarray(data[f"{name}_A"]) for name in "xyz"}
                bin_a = float(data["bin_A"]) if "bin_A" in data.files else 1.0
            difference_loaded.append((path, rho, axes, bin_a))
        difference_limit = max(
            float(np.nanmax(np.abs(image)))
            for _path, rho, _axes, bin_a in difference_loaded
            for image in (rho.sum(axis=2).T * bin_a, rho.sum(axis=1).T * bin_a, rho.sum(axis=0).T * bin_a)
        ) if difference_loaded else 1.0
        difference_limit = max(difference_limit, 1e-30)
        for path, rho, axes, bin_a in difference_loaded:
            output_2d = path.with_suffix(".png")
            if not _wants_plot(output_2d, style):
                continue
            images = (rho.sum(axis=2).T * bin_a, rho.sum(axis=1).T * bin_a, rho.sum(axis=0).T * bin_a)
            substrate_overlay = _overlay(path)
            fig, plot_axes = plt.subplots(1, 3, figsize=(style["figure_size"][0] * 1.65, style["figure_size"][1]))
            for axis, (plane, image, horizontal, vertical) in zip(
                plot_axes,
                (("XY", images[0], "x", "y"), ("XZ", images[1], "x", "z"), ("YZ", images[2], "y", "z")),
            ):
                image, shown_horizontal, shown_vertical = _resample_projection(
                    image, axes[horizontal], axes[vertical], style
                )
                shown = axis.imshow(
                    image, origin="lower", aspect="equal", cmap=style.get("difference_cmap", "coolwarm"),
                    vmin=-difference_limit, vmax=difference_limit,
                    extent=(shown_horizontal[0], shown_horizontal[-1], shown_vertical[0], shown_vertical[-1]),
                )
                axis.set_title(plane)
                axis.set_xlabel(f"{horizontal.upper()} (A)")
                axis.set_ylabel(f"{vertical.upper()} (A)")
                _draw_substrate_2d(axis, substrate_overlay, horizontal, vertical, style)
                _format_axis(axis, style, f"pose_{plane.lower()}")
            fig.colorbar(shown, ax=plot_axes.tolist(), label="Density difference (symmetric scale)")
            fig.suptitle(path.stem)
            _save(fig, output_2d, style)


def render_result_plots(directory: Path, style: dict[str, Any]) -> None:
    style = _style_with_saved_overrides(directory.parent, style)
    _apply_global_style(style)
    summary_path = directory / "tables" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Analysis summary not found: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    kind = str(summary["kind"])
    _occupancy_plots(directory, style, kind)
    if kind == "md":
        _lifetime_plots(directory, style)
        _path_plot(directory, style)
    else:
        _mc_acceptance_plot(directory, style)
    _density_plots(directory, style)
    if kind == "md":
        _pose_plots(directory, style)
