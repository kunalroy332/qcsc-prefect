# `workspace/`: run files of `bs_uno_avas` (not in git)

Everything here except this file is ignored by git. Relative paths in the configs
(`[structure] pdb`, `[output] dir`, `work_dir`) resolve against this folder. To use another
folder, set `BS_UNO_AVAS_WORKSPACE` (scripts and Python both honor it).

| Folder | What | Who writes it |
|---|---|---|
| `data/` | Inputs: `.pdb` and `.fchk` files | you |
| `output/` | One folder per run: FCIDUMP, orbitals, labels, RDM, `*_manifest.json` | `build`, `build-fchk` |
| `work/` | Restart chkfiles (`work/bs_uno_avas/`), PySCF scratch | every SCF |
| `logs/` | Job and submit logs of the `roquo/` scripts | `roquo/*.sh` |

- `work/` lets a re-run skip converged SCFs. Delete a chkfile to recompute it.
- Hand-off to the SQD flow: `ls output/*/*_manifest.json`.
