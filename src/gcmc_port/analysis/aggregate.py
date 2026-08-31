from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import RunResult
from .tables import write_tsv


def write_aggregate(results: list[RunResult], output_root: Path) -> list[Path]:
    if not results:
        return []
    aggregate = output_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        occupancy = np.asarray([frame.occupancy for frame in result.frames], dtype=float)
        rows.append(
            {
                "run_id": result.dataset.run_id,
                "system_id": result.dataset.system_id or result.dataset.run_id,
                "comparison_group": result.dataset.comparison_group,
                "replica": result.dataset.replica,
                "sweep": result.dataset.sweep,
                "kind": result.dataset.kind,
                "frame_or_state_count": len(result.frames),
                "mean_occupancy": float(occupancy.mean()),
                "std_occupancy": float(occupancy.std()),
                "max_occupancy": int(occupancy.max()),
                "final_occupancy": int(occupancy[-1]),
                "visit_count": len(result.visits),
                "proposal_count": len(result.mc_moves),
                "accepted_move_count": sum(item.accepted for item in result.mc_moves),
            }
        )
    summary_path = write_tsv(aggregate / "run_summary.tsv", rows)
    outputs = [summary_path]
    kinds = {result.dataset.kind for result in results}
    if kinds == {"md"} and len(results) > 1:
        starts = [result.frames[0].time_ps for result in results]
        stops = [result.frames[-1].time_ps for result in results]
        time_start = max(starts)
        time_stop = min(stops)
        steps = [
            float(np.median(np.diff([frame.time_ps for frame in result.frames])))
            for result in results
            if len(result.frames) > 1
        ]
        positive_steps = [step for step in steps if step > 0]
        if time_stop >= time_start and positive_steps:
            common_step = max(positive_steps)
            common_time = np.arange(time_start, time_stop + common_step * 0.25, common_step)
            common_curves = np.vstack(
                [
                    np.interp(
                        common_time,
                        [frame.time_ps for frame in result.frames],
                        [frame.occupancy for frame in result.frames],
                    )
                    for result in results
                ]
            )
            common_mean = common_curves.mean(axis=0)
            common_ci = 1.96 * common_curves.std(axis=0, ddof=1) / np.sqrt(common_curves.shape[0])
            common_rows = [
                {
                    "time_ps": float(time),
                    "mean_occupancy": float(mean),
                    "ci95_low": float(mean - ci),
                    "ci95_high": float(mean + ci),
                    "run_count": len(results),
                }
                for time, mean, ci in zip(common_time, common_mean, common_ci)
            ]
            outputs.append(write_tsv(aggregate / "md_common_time_occupancy.tsv", common_rows))
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        labels = [row["run_id"] for row in rows]
        means = [float(row["mean_occupancy"]) for row in rows]
        finals = [float(row["final_occupancy"]) for row in rows]
        axes[0].bar(labels, means, color="#1f77b4")
        axes[0].set_ylabel("Mean occupancy")
        axes[0].set_title("Run-level mean cavity occupancy")
        axes[1].bar(labels, finals, color="#ff7f0e")
        axes[1].set_ylabel("Final occupancy")
        axes[1].set_title("Run-level final cavity occupancy")
        for axis in axes:
            axis.tick_params(axis="x", rotation=45)
            axis.grid(True, axis="y", alpha=0.2)
        fig.tight_layout()
        plot_path = aggregate / "batch_occupancy_comparison.png"
        fig.savefig(plot_path, dpi=300)
        plt.close(fig)
        outputs.append(plot_path)

        if kinds == {"md"} and len(results) > 1 and 'common_time' in locals():
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.plot(common_time / 1000.0, common_mean, color="black", linewidth=2.2, label="Run mean")
            ax.fill_between(
                common_time / 1000.0,
                common_mean - common_ci,
                common_mean + common_ci,
                color="black",
                alpha=0.18,
                label="Approx. 95% CI",
            )
            ax.set_xlabel("Common physical time (ns)")
            ax.set_ylabel("Cavity occupancy")
            ax.set_title("MD ensemble occupancy on common time bins")
            ax.legend()
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            common_plot = aggregate / "md_common_time_occupancy.png"
            fig.savefig(common_plot, dpi=300)
            plt.close(fig)
            outputs.append(common_plot)

        progress = np.linspace(0.0, 100.0, 101)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        curves = []
        for result in results:
            values = np.asarray([frame.occupancy for frame in result.frames], dtype=float)
            source = np.linspace(0.0, 100.0, values.size)
            curve = np.interp(progress, source, values)
            curves.append(curve)
            ax.plot(progress, curve, alpha=0.28, linewidth=1.0)
        stacked = np.vstack(curves)
        mean = stacked.mean(axis=0)
        if stacked.shape[0] > 1:
            sem95 = 1.96 * stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        else:
            sem95 = np.zeros_like(mean)
        ax.plot(progress, mean, color="black", linewidth=2.2, label="Run mean")
        ax.fill_between(progress, mean - sem95, mean + sem95, color="black", alpha=0.18, label="Approx. 95% CI")
        ax.set_xlabel("Normalized run progress (%)")
        ax.set_ylabel("Cavity occupancy")
        title = "Batch occupancy on normalized progress axis"
        if kinds == {"pocketmc"}:
            title += " (accepted states; not equilibrium or physical time)"
        else:
            title += " (comparison axis, not physical time)"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        progress_path = aggregate / "batch_normalized_progress.png"
        fig.savefig(progress_path, dpi=300)
        plt.close(fig)
        outputs.append(progress_path)
    except Exception as exc:
        warning = aggregate / "plot_warning.txt"
        warning.write_text(f"Could not render aggregate plots: {exc}\n", encoding="utf-8")
        outputs.append(warning)
    return outputs
