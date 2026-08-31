from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def view_density(npz_path: str | Path, *, cutoff_percent: float = 8.0, opacity: float = 0.30, cmap: str = "viridis") -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    path = Path(npz_path).expanduser().resolve()
    with np.load(path) as data:
        rho = np.asarray(data["rho"], dtype=float)
        x = np.asarray(data["x_A"], dtype=float)
        y = np.asarray(data["y_A"], dtype=float)
        z = np.asarray(data["z_A"], dtype=float)
    maximum = float(np.nanmax(rho)) if rho.size else 0.0
    fig = plt.figure(figsize=(11.5, 7.8))
    ax = fig.add_axes([0.05, 0.15, 0.72, 0.80], projection="3d")
    cutoff_axis = fig.add_axes([0.82, 0.72, 0.14, 0.03])
    opacity_axis = fig.add_axes([0.82, 0.62, 0.14, 0.03])
    cutoff_slider = Slider(cutoff_axis, "Cutoff %", 0, 100, valinit=cutoff_percent, valstep=0.5)
    opacity_slider = Slider(opacity_axis, "Opacity", 0.02, 1.0, valinit=opacity, valstep=0.01)

    def render(_value: Any = None) -> None:
        elev, azim = ax.elev, ax.azim
        ax.cla()
        threshold = maximum * float(cutoff_slider.val) / 100.0
        indices = np.argwhere(np.isfinite(rho) & (rho >= threshold))
        if indices.shape[0] > 35000:
            values = rho[tuple(indices.T)]
            keep = np.argpartition(values, -35000)[-35000:]
            indices = indices[keep]
        if indices.size:
            values = rho[tuple(indices.T)]
            ax.scatter(x[indices[:, 0]], y[indices[:, 1]], z[indices[:, 2]], c=values, cmap=cmap, s=10, alpha=float(opacity_slider.val), linewidths=0)
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title(f"{path.name}: {indices.shape[0]:,} points")
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw_idle()

    cutoff_slider.on_changed(render)
    opacity_slider.on_changed(render)
    render()
    plt.show()
