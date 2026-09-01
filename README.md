# PocketMC

<p align="center">
	<img src="assets/docs/logo-concepts/pocketmc-logo-final-aligned.png" alt="PocketMC logo" width="70%" />
</p>

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

`pip install` installs PocketMC and its Python dependencies, but it does not
install GROMACS. If `gmx` and `gmx_mpi` are both missing, Option A below can
install GROMACS from conda-forge in the same environment. On an HPC system,
prefer the GROMACS/MPI modules provided by that site. The upstream build
procedure is documented in the
[GROMACS installation guide](https://manual.gromacs.org/documentation/current/install-guide/index.html).

### Option A: Conda Environment

Create and activate the Python environment first:

```bash
conda create -n pocketmc python=3.12 pip
conda activate pocketmc
```

Before installing another copy, activate any personal GROMACS installation or
load an available HPC module (`module avail gromacs` followed by the site's
`module load ...` command). Then test whichever executable it provides:

```bash
gmx --version
# or: gmx_mpi --version
```

If either version command succeeds, use that installation. If both commands are
unavailable, install the
[compiled conda-forge GROMACS package](https://anaconda.org/conda-forge/gromacs)
in the active environment:

```bash
conda install -c conda-forge gromacs
gmx --version
```

This installs the non-MPI `gmx` build. For the Open MPI `gmx_mpi` build, install
`"gromacs=*=mpi_openmpi_*"` instead. PocketMC supports either build.

Finally, install PocketMC and its Python dependencies:

```bash
python -m pip install -e .
```

`-e .` is an editable install: updates made inside the checkout are used without
reinstalling the package. For a fixed, non-editable installation, use
`python -m pip install .` instead. Keep subsequent `pip` operations inside the
activated Conda environment.

The Conda package supports Linux and macOS, not native Windows; Windows users
can use it inside WSL/Linux. Do not combine a site module and the Conda package
in the same shell. Record any required module commands in
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

`analyses.py` and the installed `pocketmc-analyses` command analyze both physical
MD trajectories and PocketMC accepted states. Supported outputs include:

- 2D density projections (XY, XZ, and YZ) and 3D density isosurfaces, exported
  as PNG/data files and prepared for VMD
- cavity occupancy, entry/re-entry events, residence lifetimes, residue paths,
  and per-molecule traces for physical MD
- PocketMC move acceptance, occupancy, energy, state, and spatial-occurrence
  diagnostics
- shared substrate-pose clustering, representative structures, hydration sites,
  and pocket-frame or substrate-frame density maps across replicas or homologs
- editable plots, reusable VMD sessions, aggregate comparisons, recursive case
  discovery, and reproducible TOML/Slurm workflows

Start the interactive analysis wizard or rerun a saved configuration:

```bash
python analyses.py
python analyses.py validate -c analyses.toml
python analyses.py run -c analyses.toml --jobs 4
```

See [Analysis Wizard and Reproducible Runs](TUTORIAL.md#12-analysis-wizard-and-reproducible-runs)
for the complete workflow, interpretation guidance, output layout, plot editing,
VMD commands, caching, pose analysis, and Slurm execution.

## 10. License and Disclaimer

Copyright Battelle Memorial Institute 2026. Redistribution and use are subject
to the terms and required notices in [LICENSE](LICENSE), which also contains the
applicable U.S. Government, U.S. Department of Energy, Battelle, and PNNL
disclaimer.
