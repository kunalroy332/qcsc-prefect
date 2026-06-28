"""Regenerate the SQD_FINDINGS_REPORT figures. Run from anywhere; writes alongside this file.

Data are the local noiseless harness results and the real 50q ibm_fez excitation profile, as
recorded in SQD_FINDINGS_REPORT.md. Re-run after updating those numbers.
"""
import os
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))

# Fig 1: spin-contamination removal (NH 12q, CN 20q) — data from local runs
systems=["NH triplet\n(12q)","CN doublet\n(20q)"]
pure=[2.0,0.75]
uhf=[2.101,0.782]
sqd=[2.011,0.755]
x=np.arange(len(systems)); w=0.35
fig,ax=plt.subplots(figsize=(6,4))
ax.bar(x-w/2,[u-p for u,p in zip(uhf,pure)],w,label="UHF reference",color="#c0392b")
ax.bar(x+w/2,[s-p for s,p in zip(sqd,pure)],w,label="SQD-over-UHF",color="#27ae60")
ax.axhline(0,color="k",lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(systems)
ax.set_ylabel(r"spin contamination  $\langle S^2\rangle - S(S{+}1)$")
ax.set_title("SQD restores the pure-spin state (noiseless)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/spin_contamination.png",dpi=140); print("wrote spin_contamination.png")

# Fig 2: ansatz connectivity vs UCCSD (CN 20q) — the beat-UCCSD result
labels=["heavy-hex\nL=2","FULL\nL=2","heavy-hex\nL=4","FULL\nL=4"]
vs_uccsd=[6.22,-6.20,55.46,55.92]  # mHa
colors=["#e67e22" if v>0 else "#27ae60" for v in vs_uccsd]
fig,ax=plt.subplots(figsize=(6.5,4))
ax.bar(labels,vs_uccsd,color=colors)
ax.axhline(0,color="k",lw=1.0,label="UCCSD")
ax.set_ylabel("E(SQD) - E(UCCSD)  [mHa]")
ax.set_title("CN doublet (20q): FULL UCJ connectivity beats UCCSD")
ax.annotate("BEATS UCCSD",xy=(1,-6.2),xytext=(1,-22),ha="center",color="#27ae60",
            arrowprops=dict(arrowstyle="->",color="#27ae60"))
fig.tight_layout(); fig.savefig(f"{OUT}/ansatz_connectivity.png",dpi=140); print("wrote ansatz_connectivity.png")

# Fig 3: noise wall schematic (excitation profile at 50q from real Fugaku diag)
fig,ax=plt.subplots(figsize=(6,4))
cats=["HF","singles+\ndoubles","higher\n(>2 exc)"]
counts=[1,20,426]  # representative 50q alpha-det breakdown (C4H5)
ax.bar(cats,counts,color=["#2c3e50","#27ae60","#c0392b"])
ax.set_ylabel("determinants in subsampled subspace (~447)")
ax.set_title("50q on ibm_fez: ~95% of the subspace cannot couple to HF")
for i,c in enumerate(counts): ax.text(i,c+8,str(c),ha="center")
fig.tight_layout(); fig.savefig(f"{OUT}/noise_wall_excitations.png",dpi=140); print("wrote noise_wall_excitations.png")
print("DONE")
