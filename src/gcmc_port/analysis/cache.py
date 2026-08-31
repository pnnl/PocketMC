from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .models import (
    DatasetSpec,
    FrameRecord,
    MCMove,
    MoleculeFrame,
    PathSample,
    RunResult,
    VisitRecord,
)


CACHE_DIRECTORY = ".analysis_cache"
CACHE_FORMAT_VERSION = 1
METADATA_NAME = "metadata.json"
RECORDS_NAME = "records.npz"


class CacheFormatError(ValueError):
    """Raised when a safe analysis cache is incomplete or malformed."""


def cache_directory(run_dir: str | Path) -> Path:
    return Path(run_dir) / CACHE_DIRECTORY


def _portable_path(path: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _json_safe(value: Any, *, base: Path) -> Any:
    if isinstance(value, Path):
        return {"__path__": _portable_path(value, base)}
    if isinstance(value, str) and Path(value).expanduser().is_absolute():
        return {"__path_string__": _portable_path(Path(value), base)}
    if isinstance(value, np.ndarray):
        return [_json_safe(item, base=base) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item(), base=base)
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, tuple):
        return [_json_safe(item, base=base) for item in value]
    if isinstance(value, list):
        return [_json_safe(item, base=base) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item, base=base) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported analysis-cache metadata type: {type(value).__name__}")


def _json_restore(value: Any, *, base: Path) -> Any:
    if isinstance(value, list):
        return [_json_restore(item, base=base) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__path__"}:
            path = Path(str(value["__path__"])).expanduser()
            return path if path.is_absolute() else (base / path).resolve()
        if set(value) == {"__path_string__"}:
            path = Path(str(value["__path_string__"])).expanduser()
            return str(path.resolve() if path.is_absolute() else (base / path).resolve())
        if set(value) == {"__float__"}:
            token = str(value["__float__"])
            return {"nan": float("nan"), "inf": float("inf"), "-inf": float("-inf")}[token]
        return {str(key): _json_restore(item, base=base) for key, item in value.items()}
    return value


def _strings(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.str_)


def _optional_floats(values: Iterable[float | None]) -> tuple[np.ndarray, np.ndarray]:
    items = list(values)
    return (
        np.asarray([float("nan") if value is None else float(value) for value in items], dtype=np.float64),
        np.asarray([value is not None for value in items], dtype=np.bool_),
    )


def _optional_ints(values: Iterable[int | None]) -> tuple[np.ndarray, np.ndarray]:
    items = list(values)
    return (
        np.asarray([-1 if value is None else int(value) for value in items], dtype=np.int64),
        np.asarray([value is not None for value in items], dtype=np.bool_),
    )


def _cache_arrays(result: RunResult) -> dict[str, np.ndarray]:
    frames = result.frames
    molecules = [molecule for frame in frames for molecule in frame.molecules]
    offsets = np.zeros(len(frames) + 1, dtype=np.int64)
    for index, frame in enumerate(frames, start=1):
        offsets[index] = offsets[index - 1] + len(frame.molecules)
    frame_energy, frame_energy_valid = _optional_floats(frame.energy_kj_mol for frame in frames)
    frame_trial, frame_trial_valid = _optional_ints(frame.trial for frame in frames)

    visits = result.visits
    paths = result.path_samples
    moves = result.mc_moves
    move_energy, move_energy_valid = _optional_floats(move.energy_kj_mol for move in moves)
    move_delta, move_delta_valid = _optional_floats(move.delta_energy_kj_mol for move in moves)
    move_inside, move_inside_valid = _optional_ints(move.n_inside_before for move in moves)

    return {
        "frame_frame": np.asarray([item.frame for item in frames], dtype=np.int64),
        "frame_time_ps": np.asarray([item.time_ps for item in frames], dtype=np.float64),
        "frame_occupancy": np.asarray([item.occupancy for item in frames], dtype=np.int64),
        "frame_energy": frame_energy,
        "frame_energy_valid": frame_energy_valid,
        "frame_trial": frame_trial,
        "frame_trial_valid": frame_trial_valid,
        "frame_move": _strings(item.move for item in frames),
        "molecule_offsets": offsets,
        "molecule_uid": _strings(item.uid for item in molecules),
        "molecule_resid": np.asarray([item.resid for item in molecules], dtype=np.int64),
        "molecule_resname": _strings(item.resname for item in molecules),
        "molecule_point_nm": np.asarray([item.point_nm for item in molecules], dtype=np.float64).reshape((-1, 3)),
        "molecule_inside": np.asarray([item.inside for item in molecules], dtype=np.bool_),
        "molecule_nearest_residue": _strings(item.nearest_residue for item in molecules),
        "molecule_nearest_distance_nm": np.asarray(
            [item.nearest_distance_nm for item in molecules], dtype=np.float64
        ),
        "molecule_nearest_residue_sim": _strings(item.nearest_residue_sim for item in molecules),
        "molecule_nearest_residue_homolog": _strings(item.nearest_residue_homolog for item in molecules),
        "visit_run_id": _strings(item.run_id for item in visits),
        "visit_molecule_uid": _strings(item.molecule_uid for item in visits),
        "visit_resid": np.asarray([item.resid for item in visits], dtype=np.int64),
        "visit_resname": _strings(item.resname for item in visits),
        "visit_index": np.asarray([item.visit_index for item in visits], dtype=np.int64),
        "visit_event_type": _strings(item.event_type for item in visits),
        "visit_start_frame": np.asarray([item.start_frame for item in visits], dtype=np.int64),
        "visit_end_frame": np.asarray([item.end_frame for item in visits], dtype=np.int64),
        "visit_start_ps": np.asarray([item.start_ps for item in visits], dtype=np.float64),
        "visit_end_ps": np.asarray([item.end_ps for item in visits], dtype=np.float64),
        "visit_lifetime_ps": np.asarray([item.lifetime_ps for item in visits], dtype=np.float64),
        "visit_sample_count": np.asarray([item.sample_count for item in visits], dtype=np.int64),
        "visit_left_censored": np.asarray([item.left_censored for item in visits], dtype=np.bool_),
        "visit_right_censored": np.asarray([item.right_censored for item in visits], dtype=np.bool_),
        "visit_dominant_residue": _strings(item.dominant_residue for item in visits),
        "path_run_id": _strings(item.run_id for item in paths),
        "path_molecule_uid": _strings(item.molecule_uid for item in paths),
        "path_sample_index": np.asarray([item.sample_index for item in paths], dtype=np.int64),
        "path_frame": np.asarray([item.frame for item in paths], dtype=np.int64),
        "path_time_ps": np.asarray([item.time_ps for item in paths], dtype=np.float64),
        "path_label": _strings(item.label for item in paths),
        "path_nearest_residue": _strings(item.nearest_residue for item in paths),
        "path_distance_nm": np.asarray([item.distance_nm for item in paths], dtype=np.float64),
        "path_inside": np.asarray([item.inside for item in paths], dtype=np.bool_),
        "path_point_nm": np.asarray([item.point_nm for item in paths], dtype=np.float64).reshape((-1, 3)),
        "path_nearest_residue_sim": _strings(item.nearest_residue_sim for item in paths),
        "path_nearest_residue_homolog": _strings(item.nearest_residue_homolog for item in paths),
        "mc_run_id": _strings(item.run_id for item in moves),
        "mc_trial": np.asarray([item.trial for item in moves], dtype=np.int64),
        "mc_accepted_before": np.asarray([item.accepted_before for item in moves], dtype=np.int64),
        "mc_move": _strings(item.move for item in moves),
        "mc_accepted": np.asarray([item.accepted for item in moves], dtype=np.bool_),
        "mc_energy": move_energy,
        "mc_energy_valid": move_energy_valid,
        "mc_delta": move_delta,
        "mc_delta_valid": move_delta_valid,
        "mc_inside": move_inside,
        "mc_inside_valid": move_inside_valid,
    }


def write_analysis_cache(
    run_dir: str | Path,
    result: RunResult,
    *,
    analysis_version: int,
    fingerprint: str,
) -> Path:
    directory = cache_directory(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    records_path = directory / RECORDS_NAME
    records_tmp = directory / f"{RECORDS_NAME}.tmp"
    with records_tmp.open("wb") as handle:
        np.savez(handle, **_cache_arrays(result))
    os.replace(records_tmp, records_path)

    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "analysis_version": int(analysis_version),
        "fingerprint": str(fingerprint),
        "run_id": result.dataset.run_id,
        "kind": result.dataset.kind,
        "warnings": _json_safe(result.warnings, base=directory),
        "outputs": _json_safe(result.outputs, base=directory),
        "result_metadata": _json_safe(result.metadata, base=directory),
    }
    metadata_path = directory / METADATA_NAME
    metadata_tmp = directory / f"{METADATA_NAME}.tmp"
    metadata_tmp.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(metadata_tmp, metadata_path)
    return directory


def read_analysis_cache_metadata(run_dir: str | Path) -> dict[str, Any] | None:
    directory = cache_directory(run_dir)
    metadata_path = directory / METADATA_NAME
    records_path = directory / RECORDS_NAME
    if not metadata_path.is_file() or not records_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("format_version", 0)) != CACHE_FORMAT_VERSION:
            return None
        restored = _json_restore(payload, base=directory)
        return restored if isinstance(restored, dict) else None
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _required(data: Any, name: str) -> np.ndarray:
    if name not in data.files:
        raise CacheFormatError(f"Safe analysis cache is missing array {name!r}")
    return np.asarray(data[name])


def load_analysis_cache(
    run_dir: str | Path,
    dataset: DatasetSpec,
    *,
    analysis_version: int,
    fingerprint: str,
) -> RunResult | None:
    payload = read_analysis_cache_metadata(run_dir)
    if payload is None:
        return None
    if (
        int(payload.get("analysis_version", 0)) != int(analysis_version)
        or str(payload.get("fingerprint", "")) != fingerprint
        or str(payload.get("run_id", "")) != dataset.run_id
        or str(payload.get("kind", "")) != dataset.kind
    ):
        return None

    records_path = cache_directory(run_dir) / RECORDS_NAME
    try:
        with np.load(records_path, allow_pickle=False) as data:
            frame_numbers = _required(data, "frame_frame").astype(np.int64, copy=False)
            offsets = _required(data, "molecule_offsets").astype(np.int64, copy=False)
            if offsets.shape != (len(frame_numbers) + 1,) or offsets[0] != 0:
                raise CacheFormatError("Safe analysis cache has invalid molecule offsets")
            molecule_count = int(offsets[-1])
            molecule_points = _required(data, "molecule_point_nm").astype(np.float64, copy=False)
            if molecule_points.shape != (molecule_count, 3):
                raise CacheFormatError("Safe analysis cache has invalid molecule coordinates")
            molecule_columns = {
                name: _required(data, name)
                for name in (
                    "molecule_uid", "molecule_resid", "molecule_resname", "molecule_inside",
                    "molecule_nearest_residue", "molecule_nearest_distance_nm",
                    "molecule_nearest_residue_sim", "molecule_nearest_residue_homolog",
                )
            }
            if any(len(column) != molecule_count for column in molecule_columns.values()):
                raise CacheFormatError("Safe analysis cache has inconsistent molecule columns")

            molecules: list[MoleculeFrame] = []
            for index in range(molecule_count):
                molecules.append(
                    MoleculeFrame(
                        uid=str(molecule_columns["molecule_uid"][index]),
                        resid=int(molecule_columns["molecule_resid"][index]),
                        resname=str(molecule_columns["molecule_resname"][index]),
                        point_nm=tuple(float(value) for value in molecule_points[index]),
                        inside=bool(molecule_columns["molecule_inside"][index]),
                        nearest_residue=str(molecule_columns["molecule_nearest_residue"][index]),
                        nearest_distance_nm=float(molecule_columns["molecule_nearest_distance_nm"][index]),
                        nearest_residue_sim=str(molecule_columns["molecule_nearest_residue_sim"][index]),
                        nearest_residue_homolog=str(molecule_columns["molecule_nearest_residue_homolog"][index]),
                    )
                )

            frame_time = _required(data, "frame_time_ps")
            frame_occupancy = _required(data, "frame_occupancy")
            frame_energy = _required(data, "frame_energy")
            frame_energy_valid = _required(data, "frame_energy_valid")
            frame_trial = _required(data, "frame_trial")
            frame_trial_valid = _required(data, "frame_trial_valid")
            frame_move = _required(data, "frame_move")
            frame_columns = (frame_time, frame_occupancy, frame_energy, frame_energy_valid, frame_trial, frame_trial_valid, frame_move)
            if any(len(column) != len(frame_numbers) for column in frame_columns):
                raise CacheFormatError("Safe analysis cache has inconsistent frame columns")
            frames = [
                FrameRecord(
                    frame=int(frame_numbers[index]),
                    time_ps=float(frame_time[index]),
                    molecules=tuple(molecules[int(offsets[index]):int(offsets[index + 1])]),
                    occupancy=int(frame_occupancy[index]),
                    energy_kj_mol=float(frame_energy[index]) if bool(frame_energy_valid[index]) else None,
                    trial=int(frame_trial[index]) if bool(frame_trial_valid[index]) else None,
                    move=str(frame_move[index]),
                )
                for index in range(len(frame_numbers))
            ]

            visit_count = len(_required(data, "visit_run_id"))
            visits = [
                VisitRecord(
                    run_id=str(data["visit_run_id"][index]), molecule_uid=str(data["visit_molecule_uid"][index]),
                    resid=int(data["visit_resid"][index]), resname=str(data["visit_resname"][index]),
                    visit_index=int(data["visit_index"][index]), event_type=str(data["visit_event_type"][index]),
                    start_frame=int(data["visit_start_frame"][index]), end_frame=int(data["visit_end_frame"][index]),
                    start_ps=float(data["visit_start_ps"][index]), end_ps=float(data["visit_end_ps"][index]),
                    lifetime_ps=float(data["visit_lifetime_ps"][index]), sample_count=int(data["visit_sample_count"][index]),
                    left_censored=bool(data["visit_left_censored"][index]),
                    right_censored=bool(data["visit_right_censored"][index]),
                    dominant_residue=str(data["visit_dominant_residue"][index]),
                )
                for index in range(visit_count)
            ]
            path_count = len(_required(data, "path_run_id"))
            path_points = _required(data, "path_point_nm").reshape((-1, 3))
            if len(path_points) != path_count:
                raise CacheFormatError("Safe analysis cache has inconsistent path coordinates")
            path_samples = [
                PathSample(
                    run_id=str(data["path_run_id"][index]), molecule_uid=str(data["path_molecule_uid"][index]),
                    sample_index=int(data["path_sample_index"][index]), frame=int(data["path_frame"][index]),
                    time_ps=float(data["path_time_ps"][index]), label=str(data["path_label"][index]),
                    nearest_residue=str(data["path_nearest_residue"][index]),
                    distance_nm=float(data["path_distance_nm"][index]), inside=bool(data["path_inside"][index]),
                    point_nm=tuple(float(value) for value in path_points[index]),
                    nearest_residue_sim=str(data["path_nearest_residue_sim"][index]),
                    nearest_residue_homolog=str(data["path_nearest_residue_homolog"][index]),
                )
                for index in range(path_count)
            ]
            move_count = len(_required(data, "mc_run_id"))
            mc_moves = [
                MCMove(
                    run_id=str(data["mc_run_id"][index]), trial=int(data["mc_trial"][index]),
                    accepted_before=int(data["mc_accepted_before"][index]), move=str(data["mc_move"][index]),
                    accepted=bool(data["mc_accepted"][index]),
                    energy_kj_mol=float(data["mc_energy"][index]) if bool(data["mc_energy_valid"][index]) else None,
                    delta_energy_kj_mol=float(data["mc_delta"][index]) if bool(data["mc_delta_valid"][index]) else None,
                    n_inside_before=int(data["mc_inside"][index]) if bool(data["mc_inside_valid"][index]) else None,
                )
                for index in range(move_count)
            ]
    except (OSError, ValueError, TypeError, KeyError, IndexError, CacheFormatError):
        return None

    warnings = payload.get("warnings", [])
    outputs = payload.get("outputs", [])
    metadata = payload.get("result_metadata", {})
    if not isinstance(warnings, list) or not isinstance(outputs, list) or not isinstance(metadata, dict):
        return None
    return RunResult(
        dataset=dataset,
        frames=frames,
        visits=visits,
        path_samples=path_samples,
        mc_moves=mc_moves,
        warnings=[str(item) for item in warnings],
        outputs=[item if isinstance(item, Path) else Path(str(item)) for item in outputs],
        metadata=metadata,
    )
