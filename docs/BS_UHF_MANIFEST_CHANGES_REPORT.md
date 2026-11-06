# BS-UHF Reference from an AVAS Manifest: Changes to `algorithms/`

**Branch:** `bk/feature/bs-uhf-sqd-roquo`, on top of `bk/feature/bs-uno-avas` (the manifest
producer) and `bk/fix/sbd-flow-updates` (sampler parameters, strict `FlowParameters`, local
pool paths, multi-GPU solver build). Items from the latter are marked *(sbd-flow-updates)*.

This report follows on from [UHF_CHANGES_REPORT.md](UHF_CHANGES_REPORT.md). That report made
the pipeline open-shell capable. This one lets it **consume a broken-symmetry reference chosen
upstream** (an AVAS active space + spin pattern from `plugins/bs_uno_avas`) and makes sampling
and recovery runs provably use the same reference. Usage is in the tutorial
[Run BS-UHF SQD on ROQUO](tutorials/run_bs_uhf_sqd_roquo.md).
Each layer below states what was wrong, what changed and how it is tested.

---

## 0. The problem in one paragraph

The only broken-symmetry input was `FE4S4_AF_GROUPS`, whose `fe4s4` preset hard-codes the MO
indices of one 36-orbital [Fe4S4(SCH3)4] FCIDUMP. On the 16 / 32-orbital AVAS active spaces of
the 5XTD excerpts it crashed with an uncaught `IndexError`. There was no way to pass the AO
labels, spin pattern or reference RDM: pydantic silently dropped unknown keys. Without a manifest,
the reference was whatever the energy-ordered guess found, and the ox/red minima differ by
~7 mEh by pattern. The UHF cache was keyed by file path, so a regenerated FCIDUMP was served a
stale reference. A saved pool was reloaded without checking it belonged to the current active
space.

## 1. Manifest reader: `qcsc_workflow_utility/active_space.py` (new)

Pure numpy, no Prefect / PySCF.

- `load_active_space_manifest(path, fcidump_override=None) -> ActiveSpaceSpec` validates
  schema v1:
  - required keys, `schema_version == 1`, unknown keys tolerated;
  - per-file sha256 (`SBD_SKIP_MANIFEST_HASH=1` → warning);
  - FCIDUMP header `NORB/NELEC/MS2`, read from the namelist only;
  - labels: Fe2S2 / Fe4S4, 5 × Fe 3d + 3 × S 3p, unique, **any order**;
  - `spin_pattern` ↔ `spin_tag`;
  - RDM shape, symmetry and trace (±1e-2).
  - A `fcidump_override` must be byte-identical to the manifest FCIDUMP.
- `cache_token` = sha256 over the FCIDUMP, labels and RDM contents and the tag. It identifies
  the reference.
- Label helpers (`parse_label`, `atom_groups`, `fe_atoms`, `validate_cluster_labels`) and spin
  helpers (`af_groups_from_labels`, `bs_density_from_labels`, `density_from_af_groups`,
  `atom_spins`, `spin_tag_from_spins`).
- `validate_orbital_indices(af_groups, norb)`: the bounds check behind the `FE4S4_AF_GROUPS` guard.

## 2. Classical reference: `qcsc_workflow_utility/chem.py`

**Bounds guard.** `_check_af_groups` runs before the UHF cache lookup and at the top of
`_converge_broken_symmetry_uhf`. On a 16-orbital FCIDUMP `FE4S4_AF_GROUPS=fe4s4` now gives
`ValueError: FE4S4_AF_GROUPS preset 'fe4s4' expects 36 orbitals; FCIDUMP has NORB=16. Use
active_space_manifest ...` instead of `IndexError`. The AF density fill moved to
`active_space.density_from_af_groups` (identical values, bounds-checked). Non-object JSON in the
env var is now a `ValueError`.

**Manifest reference.** `compute_molecular_integrals_from_fcidump(fcidump_file | None, ...,
active_space_manifest=None)` loads the manifest inside the task, so the persisted task inputs
stay JSON. With UHF it calls `_converge_reference_uhf(mf, spec)`:

- two starts, the manifest RDM and the label guess;
- each converged by `_converge_from_density`: a fresh second-order UHF per step, **internal**
  stability following, keeping the lowest iterate;
- pick the lowest among solutions whose converged tag equals the manifest tag (RDM wins ties);
- compare with `e_active_uhf` (`SBD_BS_E_TOL`);
- log `reference_source`, tag, E, ⟨S²⟩ and per-atom spins;
- `SBD_BS_STRICT=1` → `RuntimeError` on any mismatch.

With RHF the manifest only supplies the FCIDUMP (warning).

*Why not the existing external-stability follow:* `_follow_stability_external` is not reused. On both SCH3
fixtures its `newton().run(mo_external)` collapsed the Fe moments (tag `xx`, +196 / +205 mEh),
because the external mode is a GHF direction. Also `newton()` on an SOSCF object returns `self`,
which overwrote the kept "best" iterate. The legacy path is unchanged.

**Provenance on `ElectronicProperties`.** New optional fields `reference_tag`,
`reference_source`, `fcidump_sha256` (default `None` → RHF / legacy unchanged). They are carried
through both CC refreshes and `orbital_opt.rotate_electronic_properties` via `REFERENCE_FIELDS`
/ `reference_fields()`.

**Content-based UHF cache.** The key is `sha1(v2 | FCIDUMP sha256 | guess token |
nelec)`, with no path. The guess token is `af=<FE4S4_AF_GROUPS spec>` (legacy) or
`manifest=<cache_token>`. Cache files store `nelec` / `tag` / `source` and are written atomically.
On load, inconsistent, unreadable and old v1 files are ignored. A cached manifest reference is
re-validated: the tag is recomputed from the cached orbitals, and on mismatch it is re-converged.

**Helpers.** `_fcidump_uhf` (the plain-UHF construction, extracted unchanged) and
`manifest_reference_summary` (reference rebuild without UCCSD, for launcher dry-runs).

## 3. Flow: `sbd/flow_params.py`, `sbd/main.py`, `sbd/sqd.py`, `sbd/data_io.py`

- *(sbd-flow-updates)* `FlowParameters` / `CircuitParameters` / `DEParameters`:
  `extra="forbid"`. Every params JSON
  and every constructor call in `algorithms/`, `examples/` and the (untracked) `sweep/` was
  checked: none uses an unknown key.
- `fcidump: str | None` and new `active_space_manifest: str | None`. At least one is required.
  The fcidump ↔ manifest sha check runs at load time, so a params JSON validates on a machine
  without the files.
- `main.py` passes `active_space_manifest=` to the integrals task and logs the reference line.
- `walker_sqd` telemetry rows gain `norb`, `nelec`, `reference_tag`, `reference_source`,
  `fcidump_sha256` (`_reference_telemetry`).
- **Saved-pool guard.** Persisted pools carry `ref_meta` (JSON in a `<U2048` array:
  `norb, nelec, fcidump_sha256, reference_tag, method`). On `quantum_source="saved"`,
  `_check_saved_pool` behaves as follows:
  - `num_bits != 2·norb` always raises;
  - a reference-record mismatch raises unless `SBD_ALLOW_POOL_MISMATCH=1`;
  - old pools without a record are accepted with a warning;
  - a `None` sha / tag on either side (legacy run) is not compared.

  `data_io.load_ndarray_optional` is new.

## 4. ROQUO launcher: `algorithms/sbd/roquo/` (new, tracked)

`run_bs.sh check | job | sweep | report` (usage in its header and in
[Run BS-UHF SQD on ROQUO](tutorials/run_bs_uhf_sqd_roquo.md)) runs one SQD flow per manifest,
each task on its own node of one Slurm job:

- `run_bs.toml`: settings shared by every task (`[flow]`, `[sweep]`, `[run]`, `[solver]`).
- `run_task.py`: one task on one node. It writes the blocks and parameters, samples once, then
  recovers per size. The run data goes to `runs/<run_name>/samples/` and
  `runs/<run_name>/d<size>/` (`result.json`, `ckpt/`).
- A checkpoint (`ckpt/source.json`) is resumed only on the same samples, FCIDUMP sha256, spin
  tag, `sqd_dim` and `n_batches`. Otherwise it is moved to `ckpt.prev-<time>/` and recovery
  starts at step 1.
- `task_state.py`: task and size states on the login node (`report`, the suggested action
  `run` / `resume` / `new` / `skip`).
- `preflight.py`, `check_reach.py`: the `check` job. It rebuilds each manifest reference
  (`bs_common.dry_run_report`), registers the blocks, reaches Prefect Cloud and IBM Quantum, and
  launches the solver binary.
- Tasks run with `SBD_BS_STRICT=1`. `result.json` carries `molecule` = run_name,
  `cluster_type`, `spin_tag`, `fcidump_sha256`, the manifest and the sample pool.
- `bs_common.isolate_prefect` refreshes Prefect's settings after import and points
  `SBD_LOCAL_STORAGE_DIR` at the run, so pools land in the run directory.

`sweep/run_fe4s4_*` is unchanged. `.gitignore` gains `algorithms/sbd/roquo/runs/` and `logs/`.

## 5. Tests

| File | # | Covers |
|---|---|---|
| `qcsc_workflow_utility/tests/test_active_space.py` | 80 | labels (incl. `3dz^2`, `3dx2-y2`, permuted order), cluster checks, patterns / tags, densities, bounds guard (incl. the preset on the N2 FCIDUMP through the real task), header / sha / override / trace rules, SCH3 fixtures |
| `sbd/tests/test_bs_manifest_reference.py` | 28 | SCH3 ox/red reproduction (RDM and label start), projected-like RDM, wrong tag → warn / strict raise, energy tolerance, task wiring, selection rules (mocked), field propagation (CC refresh, rotation, JSON), cache key / reuse / revalidation, flow → task wiring, telemetry |
| `sbd/tests/test_flow_params_strict.py` | 27 | unknown keys rejected at all levels, existing JSONs validate, fcidump / manifest rule, example JSON, sampler defaults / options / validation, `create_blocks.py` rejects removed sampling options |
| `sbd/tests/test_saved_pool_guard.py` | 17 | ref_meta format, num_bits always enforced, per-field mismatch + override, legacy pools, real `file://` npz round trip |
| `sbd/tests/test_data_io_local.py` *(sbd-flow-updates)* | 3 | plain local pool paths read as files, missing file, S3 key without the `s3-sqd` block |
| `sbd/tests/test_roquo_launcher_env.py` | 3 | launcher storage isolation takes effect after Prefect import (own home / Cloud profile kept), per-run trace, no removed flags |

Fixture tests use `plugins/bs_uno_avas/workspace/output/N1a_State1_*` in place and skip when absent. They are never
copied into `algorithms/`.

Full suites (`algorithms/sbd/tests` + `algorithms/qcsc_workflow_utility/tests`): 337 passed, all
158 tests above among them. 14 pre-existing failures remain, none caused by this change:
`test_error_aware_layout` ×3 (known, deferred t2 pair-symmetrization fix) and
`test_uhf_spin_unbalanced` ×11. The latter call Prefect tasks directly, which start an ephemeral
server that cannot bind a port under the local sandbox. They fail identically on the base commit.

## 6. Validation (local)

| Check | Result |
|---|---|
| SCH3 ox manifest → reference | `manifest-rdm`, tag `01`, −5067.870133 (Δ < 1e-8 vs `e_active_uhf`) |
| SCH3 red manifest → reference | `manifest-rdm`, tag `10`, −5067.633303 (Δ −2e-9) |
| red, perturbed RDM (Tr 11.995 / 10.995) | converges to the same `10` state |
| red, manifest claiming `01` | label start → `01` at −5067.551131 (+82 mEh) → warning; strict → error |
| red manifest → `ElectronicProperties` + LUCJ (UCCSD, 16 orb) | norb 16, nelec (12, 11), tag `10`, 1058 LUCJ parameters / walker (1 layer, stride 4), 0.6 s |
| `FE4S4_AF_GROUPS=fe4s4` on N2 (NORB 18) | `ValueError` before any SCF |
| ROQUO launcher `--dry-run` (red) | reference reproduced, 32 qubits, no run dir created |

Not run locally: sampling, GPU recovery (ROQUO), and the full Prefect flow (sandbox port).

## 7. Behaviour changes for existing runs

1. *(sbd-flow-updates)* **Strict parameters**: an unknown key in a params JSON / constructor now fails validation.
2. **UHF cache key** no longer contains the path. Existing `~/.cache/qcsc_uhf/uhf_bs_*.npz`
   files are not found, so each legacy run recomputes its reference once (deterministic per BLAS
   / thread setting, as before). `SBD_UHF_CACHE=0` still skips only the lookup.
3. **`FE4S4_AF_GROUPS`** with out-of-range indices now raises `ValueError`. Previously it was an
   `IndexError` in the preset case, or silently wrong for some custom specs.
4. **Saved pools** with the wrong `num_bits` are refused. New pools carry `ref_meta`; readers
   that select keys by name are unaffected.
5. *(sbd-flow-updates)* **Sampling options are flow parameters** (`FlowParameters.sampler`: `shots` default 10 000,
   `n_shot_batches`, `dynamical_decoupling` / `dd_sequence`, `measure_twirling`, `gate_twirling`,
   `saved_samples`). The flow no longer reads the Prefect Variable `sqd_options`, and
   `create_blocks.py` no longer writes it. Its old flags (`--shots`, `--n-shot-batches`,
   `--saved-samples`, `--sqd-options-json`, `--options-variable-name`, DD / twirling) and their
   `SBD_*` / config keys now stop with a pointer to the parameter. A launcher that still passes
   them fails at block creation instead of running with default shots. The tracked params JSONs
   set `sampler.shots = 50000` explicitly; untracked `sweep/` launchers must be updated before
   reuse.
6. **Pool storage**: pools follow `SBD_LOCAL_STORAGE_DIR` *(sbd-flow-updates;
   `data_io._local_storage_dir`, default `$PREFECT_HOME/storage`)*, and a plain local path in
   `saved_samples` is read as a file, not an S3 key. The ROQUO launcher sets it per run
   (`isolate_prefect`, which also refreshes Prefect's settings after import). The recovery trace
   is read for the run's own flow run only (needed on a shared Prefect Cloud workspace).

Without a manifest and without the env vars, the RHF and UHF numerics are unchanged.
