from __future__ import annotations

from collections import Counter
import math

import numpy as np

from .models import FrameRecord, PathSample, RunResult, VisitRecord


def _sample_dt(frames: list[FrameRecord]) -> float:
    differences = [b.time_ps - a.time_ps for a, b in zip(frames, frames[1:]) if b.time_ps > a.time_ps]
    return float(np.median(np.asarray(differences, dtype=float))) if differences else 0.0


def _heal(values: list[bool], max_gap_frames: int) -> list[bool]:
    output = list(values)
    index = 0
    while index < len(output):
        if output[index]:
            index += 1
            continue
        start = index
        while index < len(output) and not output[index]:
            index += 1
        if start > 0 and index < len(output) and index - start <= max_gap_frames:
            output[start:index] = [True] * (index - start)
    return output


def _segments(values: list[bool]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values):
        if value and start is None:
            start = index
        if not value and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(values) - 1))
    return result


def build_visits(result: RunResult, gap_ps: float) -> list[VisitRecord]:
    frames = result.frames
    dt = _sample_dt(frames)
    gap_frames = max(0, int(math.floor(gap_ps / dt + 1.0e-9))) if dt > 0 else 0
    rows_by_uid = {item.uid: item for frame in frames for item in frame.molecules}
    visits: list[VisitRecord] = []
    for uid, identity in sorted(rows_by_uid.items()):
        occupancy = []
        nearest_by_frame: list[str] = []
        for frame in frames:
            item = next((candidate for candidate in frame.molecules if candidate.uid == uid), None)
            occupancy.append(bool(item and item.inside))
            nearest_by_frame.append(item.nearest_residue if item else "")
        healed = _heal(occupancy, gap_frames)
        for visit_index, (start, end) in enumerate(_segments(healed), start=1):
            counts = Counter(label for label in nearest_by_frame[start : end + 1] if label)
            left = start == 0
            right = end == len(frames) - 1
            event_type = "initial" if visit_index == 1 and left else ("entry" if visit_index == 1 else "reentry")
            visits.append(
                VisitRecord(
                    run_id=result.dataset.run_id,
                    molecule_uid=uid,
                    resid=identity.resid,
                    resname=identity.resname,
                    visit_index=visit_index,
                    event_type=event_type,
                    start_frame=frames[start].frame,
                    end_frame=frames[end].frame,
                    start_ps=frames[start].time_ps,
                    end_ps=frames[end].time_ps,
                    lifetime_ps=max(0.0, frames[end].time_ps - frames[start].time_ps),
                    sample_count=end - start + 1,
                    left_censored=left,
                    right_censored=right,
                    dominant_residue=(counts.most_common(1)[0][0] if counts else ""),
                )
            )
    return visits


def build_paths(result: RunResult, sample_ps: float, cutoff_a: float) -> list[PathSample]:
    frames = result.frames
    if not frames:
        return []
    times = np.asarray([frame.time_ps for frame in frames], dtype=float)
    selected: list[FrameRecord] = []
    used: set[int] = set()
    target = float(times[0])
    while target <= float(times[-1]) + 1.0e-9:
        right = int(np.searchsorted(times, target, side="left"))
        candidates = [index for index in (right - 1, right) if 0 <= index < len(frames)]
        index = min(candidates, key=lambda item: (abs(float(times[item]) - target), item))
        if index not in used:
            used.add(index)
            selected.append(frames[index])
        target += sample_ps
    samples: list[PathSample] = []
    for sample_index, frame in enumerate(selected):
        for item in frame.molecules:
            if not any(visit.molecule_uid == item.uid for visit in result.visits):
                continue
            label = item.nearest_residue if item.nearest_distance_nm * 10.0 <= cutoff_a else "Bulk"
            samples.append(
                PathSample(
                    run_id=result.dataset.run_id,
                    molecule_uid=item.uid,
                    sample_index=sample_index,
                    frame=frame.frame,
                    time_ps=frame.time_ps,
                    label=label,
                    nearest_residue=item.nearest_residue,
                    distance_nm=item.nearest_distance_nm,
                    inside=item.inside,
                    point_nm=item.point_nm,
                    nearest_residue_sim=item.nearest_residue_sim,
                    nearest_residue_homolog=item.nearest_residue_homolog,
                )
            )
    return samples
