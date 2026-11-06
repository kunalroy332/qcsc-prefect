# secrets/

Optional private credentials for the ROQUO runs. The default is Prefect Cloud via
`prefect cloud login` and IBM Quantum via the Prefect block `ibm-runner` made in the UI
(`docs/howto/howto_setup_qcsc_prefect_roquo.md`, Steps 5-6); this file is the fallback without a
browser. Everything in this folder is ignored by git except
this README and `secrets.env.example`.

| File | Tracked | Content |
|---|---|---|
| `secrets.env.example` | yes | template with fake values in the real formats |
| `secrets.env` | **no** | your IBM Quantum (and optional Prefect Cloud) credentials, `chmod 600` |

Create and fill in `secrets.env` from the repository root:

```bash
cp secrets/secrets.env.example secrets/secrets.env
chmod 600 secrets/secrets.env
${EDITOR:-vi} secrets/secrets.env  # replace every "EDIT..." value; no nano on ROQUO
```

Used by:

- `algorithms/sbd/roquo/run_bs.sh` (`SECRETS`): if the block `ibm-runner` does not exist, the
  first job creates it from `IBM_API_KEY` / `IBM_CRN` / `IBM_BACKEND`. An existing block is never
  overwritten.

Never commit it, copy it elsewhere or paste it into a ticket. If a key was exposed, revoke it
(IBM Quantum / Prefect Cloud → API keys → delete).
