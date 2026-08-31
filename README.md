# PocketMC

PocketMC is a Monte Carlo workflow for inserting and deleting water and small molecules in protein pockets or other user-defined regions.
It uses GROMACS (`gmx` or `gmx_mpi`) for structure preparation and minimization steps while PocketMC handles cavity definition, move proposals, acceptance logic, and run orchestration.

The central scientific consideration is estimating the chemical potential of small molecules in supramolecular pockets. This value should be calibrated consistently with the molecular model, force field, nonbonded settings, temperature, and reference state used for a given study.

<p align="center">
	<img src="assets/docs/cavity-mask.gif" alt="PocketMC cavity mask mode animation" width="25%" />
</p>

> [!IMPORTANT]
> Current version support is focused on the FF14SB protein force field with TIP3P water.
> If you need other force fields or water models, treat this version as not yet officially supported.

> [!WARNING]
> PocketMC minimizes each proposed structure before applying its acceptance test. The resulting
> sequence is best used as a search over relaxed structures and pocket occupancies. Interpret state
> frequencies and thermodynamic quantities only after validating the complete protocol for the
> chosen molecular model and scientific question.

All commands below assume you are running from the repository root.

## 0. Workflow Diagram

![PocketMC workflow diagram placeholder](assets/docs/pocketmc-workflow.png)

## 1. Get the Repository

Install Git, clone the repository, and move into the checkout:

```bash
git clone https://github.com/pnnl/PocketMC.git
cd PocketMC
```

The directory containing `pyproject.toml`, `gcmc.py`, and this README is the
repository root. Run the remaining installation commands from that directory.
If you already have a checkout, use `git pull` only when you are ready to update
your local copy and then continue below.

## 2. Install PocketMC And Its Runtime

PocketMC requires:

- Python 3.11 or newer; Python 3.12 is a conservative default for a new environment
- `pip`
- GROMACS, exposed as either `gmx` or `gmx_mpi`
- MPI only when you select the `gmx_mpi` execution style

The Python installation and GROMACS installation are separate. `pip install`
installs PocketMC and its Python dependencies, but it does not install GROMACS.
On an HPC system, use the GROMACS/MPI modules provided by that site. On a local
workstation, use a working GROMACS installation supported by your operating
system; the upstream build procedure is documented in the
[GROMACS installation guide](https://manual.gromacs.org/documentation/current/install-guide/index.html).

### Option A: Conda Environment

Conda is useful on clusters or workstations where you need an isolated Python
without changing the system installation. Miniconda, Anaconda, or Mambaforge can
provide the `conda` command.

```bash
conda create -n pocketmc python=3.12 pip -y
conda activate pocketmc
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

`-e .` is an editable install: updates made inside the checkout are used without
reinstalling the package. For a fixed, non-editable installation, use
`python -m pip install .` instead. Keep subsequent `pip` operations inside the
activated Conda environment.

Conda manages the Python environment here; GROMACS may still come from a site
module. For example, after activating the environment on a cluster:

```bash
module avail gromacs
module load gromacs
gmx --version       # use this when the site exposes gmx
# or: gmx_mpi --version
```

Module names vary by site. Record the required commands in
`[execution].module_setup` or use the generated Tahoma-only launcher described
below.

### Option B: Standard Python Virtual Environment

This option uses Python's built-in `venv` and does not require Conda.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If PowerShell blocks activation, either adjust the execution policy according to
your organization's policy or call the environment interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

The Python Packaging User Guide provides additional
[venv and pip instructions](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

### Option C: Install Into An Existing Environment

If you already manage a suitable Python 3.11+ environment, install directly into
that environment:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install .
```

For development, replace the final command with `python -m pip install -e .`.
Avoid installing into the operating system's Python with `sudo pip`; an isolated
Conda or `venv` environment is safer and more reproducible.

### Network-Restricted Cluster

If compute nodes cannot reach package indexes, download the Python packages on a
networked machine with the same operating-system/CPU/Python combination:

```bash
python -m pip download --dest wheelhouse .
```

Copy the checkout and `wheelhouse/` to the cluster, activate the target environment,
and install without network access:

```bash
python -m pip install --no-index --find-links=wheelhouse -e .
```

### Verify The Installation

Confirm the Python entry points:

```bash
gcmc-port --help
pocketmc-analyses --help
python gcmc.py --help
```

Then confirm one GROMACS command works in the same shell:

```bash
gmx --version
# or
gmx_mpi --version
```

If PocketMC is found but GROMACS is not, activate/load GROMACS first or update
`[execution].gmx_cmd` and `[execution].module_setup` in the TOML configuration.
GROMACS commands are executed as argument lists with `shell=False`; TOML does
not accept arbitrary shell commands. `module_setup` permits only validated
environment-module operations in generated shell and Slurm launchers.

## 3. First Run With The Bundled Example

Quick start command flow:

```bash
python gcmc.py init-example -o ./quickrun
python gcmc.py run -c ./quickrun/config.example.toml
```

This creates a runnable example case and, by default, writes replica directories `./quickrun/00` through `./quickrun/04`.
Useful outputs include `workflow.log`, `mc.log`, `trajectory.gro`, and `cavity_trajectory.gro` when mask mode is enabled.

The bundled example is a substrate-bound protein pocket configured to sample water insertion and
deletion around the pocket region.

## 4. Interactive Wizard (Recommended First)

PocketMC can generate a single-run config interactively:

```bash
python gcmc.py --interactive
```

The wizard:

- detects local `*.gro` and `*.top` files and lets you choose by number
- writes a `gcmc_wizard.toml` beside your selected case inputs by default
- writes three launchers beside that TOML
- supports `sphere` mode and a seeded `mask` mode that can auto-build the cavity during `run -c`
- keeps low-level tuning values behind a final `Advanced options` prompt

The three generated launchers have distinct purposes:

| File | Use |
| --- | --- |
| `run_gcmc.sh` | Direct Bash execution with no scheduler account; run `./run_gcmc.sh` |
| `run_gcmc.sbatch` | Generic Slurm template; replace `YOUR_ACCOUNT` and review partition, nodes, tasks, and time |
| `run_gcmc_tahoma_only.sbatch` | Tahoma-only Slurm template with the required module initialization; replace `YOUR_ACCOUNT` before submission |

On a filesystem that does not preserve the executable bit, run
`chmod +x run_gcmc.sh` once. Submit either Slurm file with `sbatch <file>` only
after reviewing its `#SBATCH` directives. The generated files locate the config
relative to their own directory and call the installed `gcmc-port` command rather
than recording the user's absolute checkout path. Set `POCKETMC_BIN` only when
the command has a different name or location.

`init-example` and `run` do not create scheduler files. The three-file set is
created by the interactive wizard. The `emit-sbatch` command remains available
when you only want to regenerate a generic Slurm file from an existing TOML.

## 5. Run Your Own Case

For a normal run on your own system, the only project-specific files you must provide are:

- `topol.top`
- `init.gro`

Suggested setup:

1. Create a case directory such as `./my-case`.
2. Put your own `topol.top` and `init.gro` there.
3. If you want the bundled defaults copied into that directory, run:

```bash
python gcmc.py init-defaults -o ./my-case
```

4. Copy `assets/example/config.example.toml` to `./my-case/config.toml` and edit it.
5. Run:

```bash
python gcmc.py run -c ./my-case/config.toml
```

For a portable case directory, keep the config beside the case data and use:

```toml
[paths]
project_root = "."
work_root = "."
```

PocketMC resolves relative paths from the config file location, not from your current shell directory.
For most users, keep the config beside case files and use `project_root = "."` and `work_root = "."`; see [TUTORIAL.md](TUTORIAL.md) for full path-resolution examples.

## 6. Supported Force Field And Water Model

PocketMC in the current release is configured around:

- FF14SB protein force field
- TIP3P water model

Mask mode currently assumes water-like inserts (`SOL`, `WAT`, `HOH`).

## 7. Common Commands

| Command | What it does |
| --- | --- |
| `python gcmc.py run -c config.toml` | Runs PocketMC using the TOML config |
| `python gcmc.py --interactive` | Launches the interactive wizard and writes a single-run config |
| `python gcmc.py --interactive --write-config ./my-case/gcmc_wizard.toml` | Launches the wizard and writes the config to a specific path |
| `python gcmc.py init-example -o ./quickrun` | Copies a runnable bundled example case |
| `python gcmc.py init-defaults -o ./my-case` | Copies bundled baseline inputs for a new case |
| `python gcmc.py build-cavity ...` | Builds a voxel cavity mask as a preprocessing step |
| `python gcmc.py emit-sbatch -c config.toml -o jobs/run_gcmc.sbatch` | Renders an `sbatch` script without submitting it |
| `python gcmc.py submit -c config.toml -o jobs/run_gcmc.sbatch` | Renders the script and submits it with `sbatch` |
| `python gcmc.py helper ...` | Runs helper utilities |

## 8. More Documentation

See [TUTORIAL.md](TUTORIAL.md) for:

- a detailed own-case workflow
- a side-by-side visual placeholder for cavity mask mode vs sphere mode (GIF-ready)
- the full TOML section reference
- `build-cavity` flag meanings and tuning advice
- the full mask-mode workflow
- Slurm usage details

Tutorial media files should be placed under `assets/docs/`.

## 9. Integrated MD and PocketMC Analysis

`analyses.py` is the analysis entry point. Its implementation lives in
`src/gcmc_port/analysis/`, so an installed checkout also provides the
`pocketmc-analyses` command.

```bash
# Discover cases and write analyses.toml plus three launchers (does not run)
python analyses.py
python analyses.py --interactive

# Reproducible non-interactive use
python analyses.py validate -c analyses.toml
python analyses.py run -c analyses.toml
python analyses.py run -c analyses.toml --tasks density,vmd --runs replica-00 --jobs 4
python analyses.py prepare-cavities -c analyses.toml
python analyses.py emit-launchers -c analyses.toml -o .

# GUI tools are always explicit
python analyses.py view-density analysis-results/run/density/density_maps.npz
python analyses.py launch-vmd analysis-results/run/vmd/session.vmd.tcl

# Discover physical-MD and PocketMC capabilities below a case root
python analyses.py discover /path/to/cases --max-depth 4 --deep

# Generate or explicitly submit staged Slurm pose-analysis jobs
python analyses.py emit-sbatch -c analyses.toml -o analysis-jobs
python analyses.py submit-sbatch -c analyses.toml -o analysis-jobs
```

The interactive wizard begins with the same recursive discovery. Each directory
is shown once with its physical-MD pair, PocketMC evidence, inferred cavity mode,
and full directory. When physical MD exists it is selected automatically; the
wizard asks about accepted-state MC analysis only when that case has no physical
MD result. Known PocketMC minimization trajectories are not silently classified
as physical MD.

Mask cases are reviewed together. For every selected case the wizard searches
its directory and descendants first, then parent directories, for the
`*_mask.dat`, `*.meta.json`, `*_points.pdb`, and `*_nearby_residues.tsv` cavity bundle.
Choose a case number to replace an incorrect/missing path. If a non-MC case is
explicitly switched to mask mode, its TOML records a deferred seeded voxel build;
the generated launchers build it from the chosen GRO file before validation.

Every interactive setup writes `run_analyses.sh`, `run_analyses.sbatch`, and
`run_analyses_tahoma_only.sbatch` beside the TOML. The first runs directly, the
second is a generic Slurm template, and the third contains the Tahoma environment
preset. Replace `YOUR_ACCOUNT` and review all resource directives before
submission. The launchers report the current stage and failures, and the shell
variant also writes `analysis-run.log`. The analysis uses one Python process and
`--jobs` for independent case-level workers rather than MPI ranks.

Before a new scan, the wizard checks direct child result directories for a
partial `analysis_manifest.json`. The default recovery action repairs detected
inputs and writes corresponding resume launchers.
Versioned pose-feature, cluster, and trajectory caches are reused, so only
invalid or previously failed work is repeated. Alternatives create a new
`analysis-results_1`, `_2`, ... directory or recompute the existing directory
with `--force`.

Anchor residues are stored per case. The wizard first checks the exact residue
token and can then resolve a unique matching residue name for that case.
Multiple same-name residues are never combined silently: the wizard/validator
lists them and requires an exact numbered selection before an expensive pose
stage begins.

Start from [analysis.example.toml](analysis.example.toml). All relative paths
are resolved from that TOML. A PocketMC config can be supplied as
`input.gcmc_config`; its run/replica/sweep layout, inserted molecule, cavity,
anchor, and GROMACS command are inherited. MD batches additionally provide the
topology and trajectory patterns present under each run directory.

The output layout is fixed:

```text
analysis-results/
  plot_results.py
  plot_style.json        saved interactive/CLI plot-only overrides
  <run-id>/
    .analysis_cache/  versioned JSON metadata plus uncompressed, non-pickle NPZ records
    tables/       frames, events/paths, or MC state/move statistics
    plots/        temporal, lifetime, path, density, and MC diagnostics
    density/      compressed NPZ, CUBE, projection CSV, and metadata
    vmd/          unified protein/cavity/trace/density Tcl session
    analysis_manifest.json
  aggregate/      batch comparisons and ensemble summaries
```

When substrate pose analysis is enabled, each MD run also contains `poses/`.
The training frame mapping and canonical-aligned `cluster_training.xtc` live at
its top level; common-cluster assignments, representative structures, actual
representative-frame waters, hydration-site PDBs, pocket/substrate-frame maps,
and VMD sessions live below it. Cross-homolog results are collected under
`aggregate/pose-groups/<comparison-group>/`.

Automatic plotting calls the installed PocketMC plotting module directly and
never imports a Python file from the result directory. The generated
`plot_results.py` is an optional, explicitly run editing aid. Edit its `STYLE`
block to change
figure size, DPI, fonts, axes, colors, colormap, line/marker styles, histogram
bins, or rolling window, then run that file again. `density_display_grid_A`
changes only the rendered 2D heatmap spacing; the scientific NPZ grid is never
rewritten. The file is preserved on later analysis runs; use
`--reset-plot-style` only when you want the template restored.

Legacy `.analysis_cache.pkl` files are never loaded. The safe cache stores
small metadata as JSON and large frame/molecule/event columns in an uncompressed
NPZ opened with `allow_pickle=False`; an old result is recalculated once and
then reuses the new cache normally.

The same operation is available without editing Python:

```bash
python analyses.py replot analysis-results --list
python analyses.py replot analysis-results \
  --targets 100-C01:pose-density-2d,100-C01:pose-density-3d \
  --set density_display_grid_A=0.5 \
  --set 'density_3d_isosurface_levels_percent=[8,25,50]'
```

Starting `analyses.py --interactive` after a completed run offers a numbered
plot-only editor. It lists the current axis, display-grid, grid-line, font,
color-scale, 3D-surface, substrate-overlay and export settings, accepts multiple
plot groups, saves changes in `plot_style.json`, and redraws immediately from
tables/NPZ files without rereading trajectories or creating a Slurm job.

MD trajectories have physical time and therefore support occupancy, censored
residence lifetime, entry/re-entry, residue paths, molecule traces, and density.
PocketMC `trajectory.gro` contains accepted optimization/search states. Its
state index records search progression rather than physical time, so PocketMC
output focuses on state, occupancy, energy, and spatial-occurrence diagnostics.
Use physical MD trajectories for lifetime and continuous-transport analysis.

### Substrate pose and hydration

Physical-MD cases may define a substrate from one or more residue components.
Frames are aligned to a conserved local-pocket selection in a chosen reference
structure. Heavy-atom coordinates are clustered together with optional
distance, angle, dihedral, charge-center, orientation, or ring-puckering
descriptors. Homologs and replicas in one comparison group use a shared,
equally weighted cluster model; the default is three clusters.

The wizard asks for `Maximum clustering-training frames per trajectory`
(default 5000). Exactly that many uniformly spaced eligible frames, including
the first and last, are used to fit the cluster/PCA model and are always saved
as `cluster_training.xtc`. If a trajectory has fewer frames, all are used;
enter `all` to disable the cap. Every eligible MD frame is still assigned to a
cluster, so population, temporal profiles, representative selection and water
density retain the full analyzed trajectory. This cap does not reduce
lifetime/path sampling.

Each cluster receives both a canonical pocket-frame water map and a
substrate-frame map. The former preserves protein/substrate pose differences;
the latter isolates hydration relative to the cluster substrate core. Density,
discrete hydration sites, and the actual representative-frame water snapshot
are all retained. 2D maps overlay the selected substrate atoms and inferred
bonds; 3D maps use connected density isosurfaces rather than point clouds. Both
carry density color scales, and substrate-frame plots use the same pooled
cluster representative orientation across cases. Case masks may produce
different native grid extents; aggregate comparison safely interpolates them to
one union grid in the shared aligned coordinate frame before averaging.
Headless molecular snapshots remain explicit:

```bash
python analyses.py render-vmd analysis-results/run/poses/cluster_01/render_headless.vmd.tcl
```

## 10. License and Disclaimer

Copyright Battelle Memorial Institute 2026. Redistribution and use are subject
to the terms and required notices in [LICENSE](LICENSE), which also contains the
applicable U.S. Government, U.S. Department of Energy, Battelle, and PNNL
disclaimer.
