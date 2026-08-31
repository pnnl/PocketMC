from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable

from .models import RunResult


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(materialized[0]) if materialized else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(materialized)
    return path


def write_run_tables(result: RunResult, table_dir: Path) -> list[Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    frame_rows = []
    sample_rows = []
    for frame in result.frames:
        frame_rows.append(
            {
                "run_id": result.dataset.run_id,
                "system_id": result.dataset.system_id or result.dataset.run_id,
                "comparison_group": result.dataset.comparison_group,
                "replica": result.dataset.replica,
                "sweep": result.dataset.sweep,
                "frame": frame.frame,
                "time_ps": f"{frame.time_ps:.6f}",
                "accepted_state": frame.frame if result.dataset.kind == "pocketmc" else "",
                "trial": "" if frame.trial is None else frame.trial,
                "move": frame.move,
                "occupancy": frame.occupancy,
                "energy_kj_mol": "" if frame.energy_kj_mol is None else f"{frame.energy_kj_mol:.8f}",
            }
        )
        for molecule in frame.molecules:
            sample_rows.append(
                {
                    "run_id": result.dataset.run_id,
                    "system_id": result.dataset.system_id or result.dataset.run_id,
                    "comparison_group": result.dataset.comparison_group,
                    "replica": result.dataset.replica,
                    "sweep": result.dataset.sweep,
                    "frame": frame.frame,
                    "time_ps": f"{frame.time_ps:.6f}",
                    "molecule_uid": molecule.uid,
                    "resid": molecule.resid,
                    "resname": molecule.resname,
                    "x_nm": f"{molecule.point_nm[0]:.8f}",
                    "y_nm": f"{molecule.point_nm[1]:.8f}",
                    "z_nm": f"{molecule.point_nm[2]:.8f}",
                    "x_A": f"{molecule.point_nm[0] * 10.0:.8f}",
                    "y_A": f"{molecule.point_nm[1] * 10.0:.8f}",
                    "z_A": f"{molecule.point_nm[2] * 10.0:.8f}",
                    "inside_cavity": str(molecule.inside).lower(),
                    "nearest_residue": molecule.nearest_residue,
                    "nearest_residue_sim": molecule.nearest_residue_sim,
                    "nearest_residue_homolog": molecule.nearest_residue_homolog,
                    "nearest_distance_nm": f"{molecule.nearest_distance_nm:.8f}",
                }
            )
    outputs.append(write_tsv(table_dir / "frames.tsv", frame_rows))
    if result.dataset.kind == "pocketmc":
        outputs.append(write_tsv(table_dir / "mc_states.tsv", frame_rows))
    outputs.append(write_tsv(table_dir / "samples.tsv", sample_rows))
    event_rows = []
    for item in result.visits:
        row = asdict(item)
        row.update({"system_id": result.dataset.system_id or result.dataset.run_id, "comparison_group": result.dataset.comparison_group, "replica": result.dataset.replica, "sweep": result.dataset.sweep})
        event_rows.append(row)
    event_fields = [
        "run_id", "molecule_uid", "resid", "resname", "visit_index", "event_type",
        "start_frame", "end_frame", "start_ps", "end_ps", "lifetime_ps",
        "sample_count", "left_censored", "right_censored", "dominant_residue",
        "system_id", "comparison_group", "replica", "sweep",
    ]
    outputs.append(write_tsv(table_dir / "events.tsv", event_rows, event_fields))
    path_rows = []
    for item in result.path_samples:
        row = asdict(item)
        point = row.pop("point_nm")
        row.update(
            {
                "replica": result.dataset.replica,
                "sweep": result.dataset.sweep,
                "system_id": result.dataset.system_id or result.dataset.run_id,
                "comparison_group": result.dataset.comparison_group,
                "x_nm": point[0],
                "y_nm": point[1],
                "z_nm": point[2],
            }
        )
        path_rows.append(row)
    path_fields = [
        "run_id", "molecule_uid", "sample_index", "frame", "time_ps", "label",
        "nearest_residue", "distance_nm", "inside", "x_nm", "y_nm", "z_nm", "nearest_residue_sim",
        "nearest_residue_homolog", "system_id", "comparison_group", "replica", "sweep",
    ]
    outputs.append(write_tsv(table_dir / "paths.tsv", path_rows, path_fields))
    if result.mc_moves:
        move_rows = []
        by_move: dict[str, list[bool]] = {}
        for item in result.mc_moves:
            row = asdict(item)
            row.update({"system_id": result.dataset.system_id or result.dataset.run_id, "comparison_group": result.dataset.comparison_group, "replica": result.dataset.replica, "sweep": result.dataset.sweep})
            move_rows.append(row)
            by_move.setdefault(item.move, []).append(item.accepted)
        outputs.append(write_tsv(table_dir / "mc_moves.tsv", move_rows))
        acceptance_rows = [
            {
                "run_id": result.dataset.run_id,
                "system_id": result.dataset.system_id or result.dataset.run_id,
                "comparison_group": result.dataset.comparison_group,
                "replica": result.dataset.replica,
                "sweep": result.dataset.sweep,
                "move": move,
                "proposed": len(values),
                "accepted": sum(values),
                "acceptance_fraction": sum(values) / len(values),
            }
            for move, values in sorted(by_move.items())
        ]
        outputs.append(write_tsv(table_dir / "mc_acceptance_by_move.tsv", acceptance_rows))
    summary = {
        "run_id": result.dataset.run_id,
        "system_id": result.dataset.system_id or result.dataset.run_id,
        "comparison_group": result.dataset.comparison_group,
        "pocketmc_status": result.dataset.pocketmc_status,
        "pocketmc_evidence": list(result.dataset.pocketmc_evidence),
        "kind": result.dataset.kind,
        "frame_count": len(result.frames),
        "mean_occupancy": sum(frame.occupancy for frame in result.frames) / len(result.frames),
        "max_occupancy": max(frame.occupancy for frame in result.frames),
        "visit_count": len(result.visits),
        "accepted_move_count": sum(move.accepted for move in result.mc_moves),
        "proposal_count": len(result.mc_moves),
        "warnings": result.warnings,
        "scientific_scope": (
            "PocketMC accepted-state statistics are not equilibrium probabilities, free energies, physical lifetimes, or transport paths."
            if result.dataset.kind == "pocketmc"
            else "MD residence times are sampled lower bounds; boundary visits are reported as censored."
        ),
    }
    summary_path = table_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    outputs.append(summary_path)
    return outputs
