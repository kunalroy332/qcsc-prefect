"""Command line: ``python -m bs_uno_avas.cli <command>`` (or ``bs-uno-avas <command>``); relative
config paths resolve against the workspace (``paths.workspace``).

prep   --config run.toml [--pdb x.pdb]   structure report, charge/spin per state (no SCF)
build  --config run.toml [--state red]   full pipeline -> artifacts + manifest per state
       [--pdb x.pdb] [--set scf.engine=cpu ...]   PDB / config overrides
       [--cluster-resname XYZ] [--geometry ox]    cluster residue (default: FES / SF4), R label
build  --config run.toml --dry-run       prep + basis size / engine / target check
verify --manifest <file>                 re-run the reference check on an artifact set

From a Gaussian BS-DFT fchk (no PDB, no full-system SCF; see ``fchk_route.py``):
check-fchk --fchk x_BS_OPT.fchk [--no-energy]          conversion check, cluster, Fe spins, gate
build-fchk --fchk x_BS_OPT.fchk --state red            -> output/x_Ered_Rred_S<tag>_BS_OPT/
           [--geometry ox] [--n-fe3 1] [--config c.toml] [--set gate.min_fe=3.0 ...]
"""

import argparse
import json
import sys

from . import manifest as manifest_mod
from . import pipeline
from . import verify as verify_mod

CLUSTER_HELP = "cluster residue name (default: detect FES / SF4)"


def _overrides(args):
    """``--set`` values plus the shortcut options as overrides of the config."""
    out = list(args.set)
    # as TOML strings: a residue name such as 103 or a label such as true stays a string
    if args.cluster_resname:
        out.append(f"structure.cluster_resname={json.dumps(args.cluster_resname)}")
    if getattr(args, "geometry", None):
        out.append(f"output.geometry_label={json.dumps(args.geometry)}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bs_uno_avas")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--config", required=True)
    p.add_argument("--pdb", help="override [structure] pdb")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--cluster-resname", help=CLUSTER_HELP)
    b = sub.add_parser("build")
    b.add_argument("--config", required=True)
    b.add_argument("--pdb", help="override [structure] pdb (one config for many PDBs)")
    b.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set scf.engine=cpu",
    )
    b.add_argument("--state", action="append", help="redox state(s); default: all in config")
    b.add_argument("--cluster-resname", help=CLUSTER_HELP)
    b.add_argument("--geometry", help="R label of the run name (default: [output] geometry_label)")
    b.add_argument("--dry-run", action="store_true")
    b.add_argument("--verbose", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--tol", type=float, default=1e-5)
    for name in ("check-fchk", "build-fchk"):
        f = sub.add_parser(name)
        f.add_argument("--fchk", required=True)
        f.add_argument("--config", help="optional TOML (sections of fchk_route.DEFAULTS)")
        f.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
        f.add_argument("--verbose", type=int, default=0)
        if name == "check-fchk":
            f.add_argument("--no-energy", action="store_true", help="skip the PySCF energy check")
        else:
            f.add_argument("--state", help="redox state (ox, red, ...): run name + e count check")
            f.add_argument("--n-fe3", type=int, help="number of Fe(III) instead of --state")
            f.add_argument("--geometry", help="R label of the run name (default: the state)")
    args = ap.parse_args(argv)

    if args.cmd in ("check-fchk", "build-fchk"):
        from . import fchk_route

        cfg = fchk_route.load_config(args.config, args.set)
        if args.cmd == "check-fchk":
            if args.no_energy:
                cfg["check"]["energy"] = False
            d = fchk_route.load(args.fchk, cfg, args.verbose)
            print(fchk_route.report(d))
            ok = d["sol"].gate and "x" not in d["sol"].tag
            print("check-fchk:", "PASS" if ok else "FAIL (gate / tag)")
            return 0 if ok else 1
        if args.state is None and args.n_fe3 is None:
            ap.error("build-fchk needs --state or --n-fe3")
        fchk_route.run_fchk(
            args.fchk, args.state, args.n_fe3, args.geometry, cfg, verbose=args.verbose
        )
        return 0

    if args.cmd == "prep":
        cfg = pipeline.load_config(args.config, pdb=args.pdb, overrides=_overrides(args))
        model, states = pipeline.prep(cfg)
        from .structure import report

        print(report(model, states))
        return 0
    if args.cmd == "build":
        cfg = pipeline.load_config(args.config, pdb=args.pdb, overrides=_overrides(args))
        if args.dry_run:
            pipeline.dry_run(cfg)
            return 0
        for state in args.state or list(cfg["states"]):
            if state not in cfg["states"]:
                ap.error(f"state {state!r} not in config ({list(cfg['states'])})")
            pipeline.run_state(cfg, state, verbose=args.verbose)
        return 0
    if args.cmd == "verify":
        m, files, labels, rdm = manifest_mod.read_manifest(args.manifest)
        ref = m["bs_reference"]
        ver = verify_mod.verify(
            files["fcidump"], [str(x) for x in labels], rdm, ref["spin_pattern"], ref["spin_tag"]
        )
        ver.pop("rdm_converged")
        ver["label_start"].pop("rdm")
        print(pipeline.dumps(ver))
        # as the SBD flow: lowest converged start (RDM, labels) with the manifest tag
        starts = [(ver["e_active_uhf"], ver["spin_tag"], ver["converged"])]
        starts.append(tuple(ver["label_start"][k] for k in ("e", "spin_tag", "converged")))
        matching = [e for e, t, c in starts if c and t == ref["spin_tag"]]
        ok = bool(matching)
        if ok and ref.get("e_active_uhf") is not None:
            ok &= abs(min(matching) - ref["e_active_uhf"]) <= args.tol
        print("verify:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
