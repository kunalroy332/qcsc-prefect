"""Cross-system SQD recovery figure: % of the UHF->CCSD(T) correlation gap recovered.

Fe2S2 (40q) and Fe4S4 (72q) have very different absolute energies, so we normalize onto one axis:

    %recovered(E) = 100 * (E - E_UHF) / (E_CCSD(T) - E_UHF)

0% = the UHF reference, 100% = the CCSD(T) correlated reference. Fe4S4 is drawn as a recovery
CURVE (per recovery iteration, from its recovery_trace); Fe2S2 is a single result POINT/line
(its Fugaku run has only a final energy, no per-iteration trajectory).

Reads (relative to a runs/ dir, override with RUNS_DIR):
    runs/fe4s4_uhf/recover/*.json        -- Fe4S4 UHF recovery_trace
    runs/fe4s4_refs.json                 -- Fe4S4 UHF/UCCSD/CCSD(T)
    runs/fe2s2_refs.json                 -- Fe2S2 refs
    plus a FE2S2_UHF_E env / arg for the Fe2S2 SQD result energy.

Writes fe_systems_recovery.{png,svg} + fe_systems.csv into runs/.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; set before pyplot
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
# CVD-safe categorical: Fe4S4 = blue (slot 1), Fe2S2 = orange (slot 8).
COL = {"fe4s4": "#2a78d6", "fe2s2": "#eb6834"}
MARK = {"fe4s4": "o", "fe2s2": "s"}
PRETTY = {"fe4s4": "Fe$_4$S$_4$ (72q, 54e/36o)", "fe2s2": "Fe$_2$S$_2$ (40q, 30e/20o)"}


def _runs() -> Path:
    return Path(os.environ.get("RUNS_DIR", str(Path(__file__).resolve().parent / "runs")))


def _refs(mol: str) -> dict:
    for name in ((f"{mol}_refs.json",) if mol != "fe2s2" else ("fe2s2_refs.json", "refs.json")):
        p = _runs() / name
        if p.is_file():
            return json.loads(p.read_text())
    return {}


def _pct(e: float, uhf: float, ccsdt: float) -> float:
    """Fraction (%) of the UHF->CCSD(T) gap recovered. 0% = UHF, 100% = CCSD(T)."""
    denom = ccsdt - uhf
    return 100.0 * (e - uhf) / denom if denom else 0.0


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _fe4s4_trace() -> list[dict]:
    best: list[dict] = []
    for jf in sorted((_runs() / "fe4s4_uhf" / "recover").glob("*.json")):
        d = json.loads(jf.read_text())
        tr = d.get("recovery_trace") or []
        if len(tr) > len(best):
            best = tr
    return best


def main() -> None:
    rows = []
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)

    # --- Fe4S4: recovery curve -----------------------------------------------------------------
    r4 = _refs("fe4s4")
    tr4 = _fe4s4_trace()
    if r4.get("UHF") is not None and r4.get("CCSD(T)") is not None and tr4:
        uhf, ct = r4["UHF"], r4["CCSD(T)"]
        xs = [t["step"] + 1 for t in tr4]
        ys = [_pct(t["energy"], uhf, ct) for t in tr4]
        ax.plot(xs, ys, marker=MARK["fe4s4"], ms=7, lw=2.2, color=COL["fe4s4"],
                label=PRETTY["fe4s4"], markeredgecolor=SURFACE, markeredgewidth=1, zorder=5)
        ax.annotate(f"{ys[-1]:.0f}%", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), color=COL["fe4s4"], fontsize=10, fontweight="bold", va="center")
        for t in tr4:
            rows.append(["fe4s4", t["step"] + 1, round(t["energy"], 6),
                         round(_pct(t["energy"], uhf, ct), 2)])

    # --- Fe2S2: single result point (no trajectory) -------------------------------------------
    r2 = _refs("fe2s2")
    e2 = os.environ.get("FE2S2_UHF_E")
    e2 = float(e2) if e2 else None
    if e2 is not None and r2.get("UHF") is not None and r2.get("CCSD(T)") is not None:
        p2 = _pct(e2, r2["UHF"], r2["CCSD(T)"])
        # horizontal marker line (final result; Fe2S2 has no per-iteration trajectory data)
        xmax = max([t["step"] + 1 for t in tr4], default=5)
        ax.axhline(p2, color=COL["fe2s2"], ls="--", lw=1.8, zorder=4)
        ax.plot([xmax], [p2], marker=MARK["fe2s2"], ms=9, color=COL["fe2s2"],
                markeredgecolor=SURFACE, markeredgewidth=1, zorder=6, label=PRETTY["fe2s2"])
        ax.annotate(f"{p2:.0f}%", (xmax, p2), textcoords="offset points", xytext=(8, 6),
                    color=COL["fe2s2"], fontsize=10, fontweight="bold", va="bottom")
        rows.append(["fe2s2", "final", round(e2, 6), round(p2, 2)])

    # reference bands: 0% (UHF) and 100% (CCSD(T))
    ax.axhline(0, color=MUTED, ls=":", lw=1.2, zorder=2)
    ax.axhline(100, color=INK, ls="-", lw=1.4, zorder=2)
    ax.annotate("UHF (0%)", (0.01, 0), xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", fontsize=8, color=MUTED, fontfamily="monospace")
    ax.annotate("CCSD(T) (100%)", (0.01, 100), xycoords=("axes fraction", "data"), xytext=(0, -12),
                textcoords="offset points", fontsize=8, color=INK, fontfamily="monospace")

    ax.set_xlabel("Configuration-recovery iteration")
    ax.set_ylabel("Correlation recovered  (% of UHF→CCSD(T) gap)")
    ax.set_title("SQD recovery on GB200 GPU: correlation captured vs iteration",
                 color=INK, fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out = _runs()
    for ext in ("png", "svg"):
        fig.savefig(out / f"fe_systems_recovery.{ext}", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    with open(out / "fe_systems.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "iter", "energy_Ha", "pct_corr_recovered"])
        w.writerows(rows)
    print(f"WROTE fe_systems_recovery.png/svg + fe_systems.csv to {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
