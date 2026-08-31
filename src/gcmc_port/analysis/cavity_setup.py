from __future__ import annotations

from pathlib import Path
from time import strftime

from gcmc_port.cavity import build_cavity_from_structure

from .models import AnalysisConfig, CavitySpec, cavity_for_dataset


def _stamp() -> str:
    return strftime("%Y-%m-%d %H:%M:%S")


def _expected_outputs(cavity: CavitySpec) -> tuple[Path, Path, Path, Path]:
    if cavity.build_output_prefix is None:
        raise ValueError("cavity build output prefix is not configured")
    prefix = cavity.build_output_prefix
    return (
        prefix.with_name(f"{prefix.name}_mask.dat"),
        prefix.with_suffix(".meta.json"),
        prefix.with_name(f"{prefix.name}_points.pdb"),
        prefix.with_name(f"{prefix.name}_nearby_residues.tsv"),
    )


def prepare_analysis_cavities(config: AnalysisConfig, *, force: bool = False) -> list[Path]:
    """Build deferred per-case voxel masks before validation and analysis."""
    outputs: list[Path] = []
    completed: set[tuple[Path, Path]] = set()
    planned = [dataset for dataset in config.datasets if cavity_for_dataset(config, dataset).build_enabled]
    if not planned:
        print(f"[{_stamp()}] [cavity] No deferred cavity builds are required.", flush=True)
        return outputs

    for index, dataset in enumerate(planned, start=1):
        cavity = cavity_for_dataset(config, dataset)
        if cavity.build_source is None or cavity.build_output_prefix is None:
            raise ValueError(f"{dataset.run_id}: deferred cavity build is missing source/output-prefix information")
        source = cavity.build_source.resolve()
        prefix = cavity.build_output_prefix.resolve()
        key = (source, prefix)
        expected = _expected_outputs(cavity)
        if key in completed:
            outputs.extend(path for path in expected if path.exists())
            continue
        completed.add(key)
        if cavity.mask is not None and cavity.mask.resolve() != expected[0].resolve():
            raise ValueError(
                f"{dataset.run_id}: cavity_mask={cavity.mask} does not match deferred build output {expected[0]}"
            )
        if cavity.meta is not None and cavity.meta.resolve() != expected[1].resolve():
            raise ValueError(
                f"{dataset.run_id}: cavity_meta={cavity.meta} does not match deferred build output {expected[1]}"
            )
        if not force and all(path.exists() for path in expected):
            print(
                f"[{_stamp()}] [cavity {index}/{len(planned)}] {dataset.run_id}: "
                f"using existing cavity bundle in {prefix.parent}",
                flush=True,
            )
            outputs.extend(expected)
            continue
        if not source.exists():
            raise FileNotFoundError(f"{dataset.run_id}: cavity build source not found: {source}")
        if source.suffix.lower() != ".gro":
            raise ValueError(f"{dataset.run_id}: GCMC voxel cavity construction requires a GRO structure: {source}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[{_stamp()}] [cavity {index}/{len(planned)}] {dataset.run_id}: building voxel mask from {source}",
            flush=True,
        )
        print(
            f"[{_stamp()}] [cavity] This geometry search can take several minutes, and large systems may take longer.",
            flush=True,
        )
        try:
            built = build_cavity_from_structure(
                source,
                outprefix=prefix,
                mode=cavity.build_mode,
                dx=cavity.build_dx,
                probe_radius=cavity.build_probe_radius,
                search_radius=cavity.build_search_radius,
                seed_residue=cavity.anchor,
                seed_atoms=list(cavity.anchor_atoms),
                exclude_residues=list(cavity.build_exclude_residues or (cavity.anchor,)),
                nearby_cutoff=cavity.build_nearby_cutoff,
                min_points=cavity.build_min_points,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{dataset.run_id}: cavity build failed (source={source}, prefix={prefix}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        missing = [path for path in expected if not path.exists()]
        if missing:
            raise RuntimeError(
                f"{dataset.run_id}: cavity builder completed but expected output(s) are missing: "
                + ", ".join(str(path) for path in missing)
            )
        outputs.extend(built)
        print(f"[{_stamp()}] [cavity] {dataset.run_id}: cavity bundle ready at {prefix.parent}", flush=True)
    return outputs
