# `bs_uno_avas`: BS-UHF + UNO-AVAS active spaces for Fe–S clusters

A plugin of this repo that stands on its own. It turns a protonated PDB excerpt of an Fe–S
protein site, or a Gaussian broken-symmetry DFT `.fchk`, into an active-space artifact set and
a manifest (schema v1). The BS-UHF SQD flow in `algorithms/sbd` reads that set
(`FlowParameters.active_space_manifest`; see
[Run BS-UHF SQD on ROQUO](../../docs/tutorials/run_bs_uhf_sqd_roquo.md)).

```
PDB  ─► full-system high-spin UHF ─► BS-UHF per spin pattern ─┐
fchk ─► conversion check (Gaussian BS-DFT orbitals) ──────────┴► UNO-AVAS (Fe 3d + μS 3p)
     ─► labels ─► FCIDUMP + RDM ─► verify ─► output/<run>/ + <run>_manifest.json
```

It needs only numpy, scipy and PySCF (gpu4pyscf is optional), and it does not depend on the
`qcsc-prefect` packages. The only link is the manifest schema, which
`qcsc_workflow_utility.active_space` reads.

## Install

```bash
pip install -e plugins/bs_uno_avas          # CPU; add [gpu] for gpu4pyscf
bs-uno-avas --help                          # = python -m bs_uno_avas.cli
```

On ROQUO nothing needs installing: the `roquo/` scripts put this checkout's `src/` on
`PYTHONPATH` inside the gpu4pyscf venv.

## Try it locally (CPU, about 30 s)

```bash
cd plugins/bs_uno_avas
export PYTHONPATH=$PWD/src      # not needed after pip install -e
python -m bs_uno_avas.cli build --config configs/bare_fe2s2.toml \
    --pdb examples/5XTD_FES_O301_SCH3.pdb --state red
ls workspace/output/
```

This writes `workspace/output/bare_Fe2S2_Ered_Rpdb_S01/` (or `_S10`) with its manifest.

## Run it on ROQUO

Follow [Run BS-UNO-AVAS on ROQUO](../../docs/tutorials/run_bs_uno_avas_roquo.md):
its setup section builds the gpu4pyscf environment. Then copy the
inputs into `workspace/data/` and use the scripts:

```bash
source ~/.qcsc_env; cd $REPO/plugins/bs_uno_avas
bash roquo/run_pdb.sh check      # .pdb files (or roquo/run_fchk.sh for .fchk)
bash roquo/run_pdb.sh build
bash roquo/run_pdb.sh report
```

[The tutorial](../../docs/tutorials/run_bs_uno_avas_roquo.md) explains the questions the scripts ask and what to
check.

## Layout

| Path | What |
|---|---|
| `src/bs_uno_avas/` | the package; `data/AOs/` = the dzvp target AOs (`target_aos = "file:AOs"`) |
| `configs/` | `batch_dzvp.toml` (PDB route, default), `fchk_dgdzvp.toml` (fchk route), demos, `batch_dzvp_fast.toml` (fast mode, not the default) |
| `roquo/` | `run_pdb.sh`, `run_fchk.sh` (check / build / report), `build_gpu4pyscf_roquo.sh` |
| `docs/` | [REFERENCE.md](docs/REFERENCE.md) (rules, config knobs, manifest); running on ROQUO: [the tutorial](../../docs/tutorials/run_bs_uno_avas_roquo.md) |
| `examples/` | two small public PDBs (5XTD FES), used by the tests and the demo |
| `tools/` | `make_aos.py` regenerates `data/AOs/` from the read-only `tools/AOs/*_ref.chk` |
| `tests/` | `pytest plugins/bs_uno_avas/tests -q` (about 1 min) |
| `workspace/` | run files, not in git: `data/`, `output/`, `work/`, `logs/` ([README](workspace/README.md)) |

Relative config paths resolve against `workspace/`. Set `BS_UNO_AVAS_WORKSPACE` to use another
folder (REFERENCE §3).

## Command line

| Command | What |
|---|---|
| `prep --config C [--pdb P]` | structure report, charge / 2S per state (no SCF) |
| `build --config C --pdb P --dry-run` | prep + basis size, DF memory, engine, target check |
| `build --config C --pdb P [--state red] [--set k=v]` | full PDB route, one manifest per state |
| `check-fchk --fchk F` | fchk conversion check, cluster, Fe spins, gate |
| `build-fchk --fchk F --state red [--geometry ox]` | fchk route: active space + manifest |
| `verify --manifest M` | re-runs the reference check from the written files |

`python -m bs_uno_avas.make_manifest` wraps older artifact sets that have no manifest
(REFERENCE §5).
