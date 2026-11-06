# QCSC Prefect

This repository provides a modular workspace for portable HPC workflow orchestration with Prefect.
It is designed so the same workflow code can run across multiple HPC systems by switching profile blocks.

The workspace is organized into core packages, optional integrations, and a PyPI meta-package:

```
qcsc-prefect/
├── docs
│   └── concept.md
├── examples
│   ├── fugaku_prefect_hello_demo
│   ├── prefect_bitcount_demo
│   └── miyabi_prefect_hello_demo
├── packages
│   ├── qcsc-prefect
│   ├── qcsc-prefect-core
│   ├── qcsc-prefect-blocks
│   ├── qcsc-prefect-adapters
│   ├── qcsc-prefect-executor
│   ├── qcsc-prefect-qiskit
│   └── qcsc-prefect-dice
├── pyproject.toml
└── .pre-commit-config.yaml
```

## Repository Structure

- `packages/qcsc-prefect/`
  Meta-package for installing the core QCSC Prefect packages from PyPI.
- `packages/qcsc-prefect-core/`
  Common execution model definitions (for example `ExecutionProfile`) shared by all targets.
- `packages/qcsc-prefect-blocks/`
  Prefect Block schemas for command, execution profile, and HPC profile layers.
- `packages/qcsc-prefect-adapters/`
  Local process runtime plus target-specific scheduler script builders and adapters.
- `packages/qcsc-prefect-executor/`
  High-level execution entrypoints that resolve blocks, derive scheduler routing,
  and dispatch to target runtimes.
- `packages/qcsc-prefect-qiskit/`
  Optional native Qiskit Runtime integration utilities.
- `packages/qcsc-prefect-dice/`
  Optional DICE SHCI solver integration.
- `examples/`
  End-to-end runnable examples for Miyabi and Fugaku.
- `docs/`
  Concept and architecture documents for the block-based execution model.

## Installation

Install the core QCSC Prefect packages from PyPI with:

```bash
pip install qcsc-prefect
```

Install the core packages plus the native Qiskit integration with:

```bash
pip install "qcsc-prefect[qiskit]"
```

Install the core packages plus the Python-side DICE integration with:

```bash
pip install "qcsc-prefect[dice]"
```

Install all optional integrations with:

```bash
pip install "qcsc-prefect[all]"
```

The DICE integration can also be installed directly:

```bash
pip install qcsc-prefect-dice
```

For source development, install the workspace with uv from the repository root:

```bash
git clone https://github.com/qiskit-community/qcsc-prefect.git
cd qcsc-prefect
uv sync --all-packages
uv run pytest
```

The root `pyproject.toml` is a workspace coordinator and is not the package published to PyPI.
Each package under `packages/` remains independently buildable.

## DICE Integration

`qcsc-prefect-dice` is a Python-side integration package. Installing it provides Prefect
tasks/blocks, scheduler templates, DICE input/output handling, and documentation. It does
not build, download, vendor, or install the external DICE/SBD executable.

The DICE/SBD executable must already be compiled on the target HPC system. Build instructions
are site-specific because compilers, MPI stacks, BLAS/LAPACK libraries, GPU support, filesystem
layout, and scheduler environments differ across HPC systems.

Configure the DICE executable through the command configuration used by the target HPC profile.
In the current block model, the `CommandBlock` names the executable key and the `HPCProfileBlock`
maps that key to the executable path visible on the target HPC system:

```python
from qcsc_prefect_blocks.common.blocks import CommandBlock, HPCProfileBlock

CommandBlock(
    command_name="dice",
    executable_key="dice_solver",
    description="DICE SHCI solver executable",
    default_args=[],
).save("cmd-dice-solver", overwrite=True)

HPCProfileBlock(
    hpc_target="miyabi",
    queue_cpu="regular-c",
    queue_gpu="regular-g",
    project_cpu="gz00",
    project_gpu="gz00",
    executable_map={
        "dice_solver": "/work/gz00/<user>/dice/bin/Dice",
    },
).save("hpc-miyabi-dice", overwrite=True)
```

Generated Miyabi, Fugaku, and Slurm job scripts include a preflight check that runs on the
HPC node before launching the job. The script fails early with a clear message if the configured
executable path is empty, missing, not executable, or not found on `PATH`.

For execution on the Prefect worker itself, set `HPCProfileBlock.hpc_target="local"`.
Local execution invokes the mapped executable directly without a shell, job script, or scheduler.
`environments` are merged into the process environment; `modules` and `pre_commands` are not
supported and cause an error before the process starts.

See [DICE Integration](./docs/howto/howto_use_dice_prefect.md) for a longer setup example.

## Documentation

- Concept and architecture:
  - [HPC-Prefect Concept](./docs/concept.md)
- Example guides:
  - [BitCount Tutorial for Miyabi](./docs/tutorials/create_qcsc_workflow_for_miyabi.md)
  - [BitCount Tutorial for Fugaku](./docs/tutorials/create_qcsc_workflow_for_fugaku.md)
  - [Native Qiskit on Prefect](./docs/howto/howto_use_native_qiskit_prefect.md)
  - [Miyabi Hello Demo](./examples/miyabi_prefect_hello_demo/README.md)
  - [Fugaku Hello Demo](./examples/fugaku_prefect_hello_demo/README.md)
  - [Run BS-UNO-AVAS on ROQUO](./docs/tutorials/run_bs_uno_avas_roquo.md)

## Code Management

Code quality checks are configured with pre-commit (`.pre-commit-config.yaml`):

- `ruff check --fix`
- `ruff format`
- `bandit` (Python security checks for medium/high-confidence findings)
- `detect-secrets` (secret scanning backed by `.secrets.baseline`)
- basic repository hygiene hooks (`check-yaml`, trailing whitespace, EOF fix, merge conflict checks)

## Versioning Policy

Each sub-package under `packages/` maintains its own version in its own `pyproject.toml`.
The root project is a workspace coordinator (`qcsc-prefect-workspace`) and is not intended for distribution.
For a coordinated PyPI release, keep the `qcsc-prefect-*` package versions and exact internal
dependency pins aligned, for example `0.1.0`.

## Release Workflow

Maintainers publish through GitHub Actions with PyPI Trusted Publishing.
Do not add PyPI API tokens or password secrets to this repository.

Normal release flow:

1. Run the `Prepare release` workflow with `old_version` and `new_version`.
2. Open a pull request manually from the generated `release/v<new_version>` branch.
3. Review and merge the release PR.
4. Push tag `v<new_version>`.
5. Approve the `pypi` environment deployment in GitHub Actions.
6. Confirm PyPI installation.

After release, verify installation with:

```bash
python -m pip install "qcsc-prefect==<version>"
python -m pip install "qcsc-prefect[all]==<version>"
```

Release notes:

- Use the release workflows for normal releases; do not use PyPI API tokens.
- Trusted Publisher is already configured for the PyPI projects.
- The `pypi-publish.yml` workflow validates tag/package version consistency before publishing.
- The DICE/SBD executable is not installed by `pip`.
- DICE/SBD must be built separately for each target HPC environment.
- If a broken release is published, prefer yanking over deleting.

### Release Validation Note

For the `0.1.0` packaging validation:

- GitHub Actions build-only dry run completed.
- All packages built into wheel and sdist distributions.
- `twine check` passed.
- Local install from `dist/` passed with
  `pip install --find-links qcsc-prefect-dist "qcsc-prefect[all]==0.1.0"`.
- The Qiskit integration import path is `qcsc_prefect.integrations.qiskit`.
- The DICE executable is not installed by `pip`; it must be built separately on the target HPC system.
- TestPyPI Trusted Publisher registration for multiple packages is currently blocked by a suspected
  PyPI/TestPyPI pending publisher issue, so no actual upload was performed.

## Contribution Guidelines

1. Install pre-commit hooks:
   - `pre-commit install`
2. Run checks before commit:
   - `pre-commit run --all-files`
   - or let them run automatically on `git commit`
3. Run tests as needed:
   - `uv run pytest`

When adding a new HPC target, include:

- adapter implementation under `packages/qcsc-prefect-adapters/`
- executor integration under `packages/qcsc-prefect-executor/`
- at least one runnable example under `examples/`
