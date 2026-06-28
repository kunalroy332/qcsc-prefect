import warnings; warnings.filterwarnings("ignore")
import os, sys
_TESTS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
sys.path.insert(0,_TESTS); sys.path.insert(0,os.path.join(_TESTS,".."))
import numpy as np
import local_sqd_harness as H
import pyscf
from pyscf import cc, fci
atom,spin="C 0 0 0; N 0 0 1.3",1
mol=pyscf.gto.M(atom=atom,basis="sto-3g",spin=spin,verbose=0)
mf=pyscf.scf.UHF(mol).run(verbose=0)
e_uhf=mf.e_tot; e_uccsd=cc.UCCSD(mf).run(verbose=0).e_tot; e_fci=fci.FCI(mf).kernel()[0]
ep=H.build_uhf_props(atom=atom,spin=spin)
print(f"CN 20q: UHF={e_uhf:.5f} UCCSD={e_uccsd:.5f} FCI={e_fci:.5f}",flush=True)
SEEDS=[1,2,3,4,5]

def avg(conn,p,shots,dim,batches,rec):
    es=[]
    for sd in SEEDS:
        mat,probs=H.prepare_state_and_sample(ep,n_lucj_layers=2,shots=shots,seed=sd,connectivity=conn,noise_p=p)
        # run_one_pass uses MODULE_RNG seeded by rng_seed; vary it with sd too
        res=H.run_one_pass(ep,mat,probs,sqd_dim=dim,n_batches=batches,n_recovery_steps=rec,rng_seed=sd)
        if res.energy is not None: es.append(res.energy)
    es=np.array(es); return es.mean(), es.std(), es.min()

def line(name,conn,p,b,r):
    m,s,mn=avg(conn,p,100000,20000,b,r)
    beat="BEST-BEATS" if mn<e_uccsd else ("MEAN-BEATS" if m<e_uccsd else "")
    print(f"{name:28s} mean {(m-e_uccsd)*1000:+7.1f}  std {s*1000:5.1f}  best {(mn-e_uccsd)*1000:+7.1f} mHa  {beat}",flush=True)

print("\n(all vs UCCSD, mHa; 5 seeds averaged)",flush=True)
print("\n-- NOISELESS ceiling --",flush=True)
line("heavy-hex p=0",       "heavy-hex",0.0,3,2)
line("full p=0",            "full",0.0,3,2)
print("\n-- PARAMETRIZATION axis @ p=0.01 --",flush=True)
for c in ["heavy-hex",2,"full"]:
    line(f"conn={c} p=0.01",c,0.01,3,2)
print("\n-- RECOVERY axis @ p=0.01, heavy-hex --",flush=True)
for b,r in [(1,1),(5,1),(5,3),(10,3)]:
    line(f"b={b} rec={r} p=0.01",  "heavy-hex",0.01,b,r)
print("\n-- combined best guess: full + heavy recovery @ p=0.01 --",flush=True)
line("full b=10 rec=3 p=0.01","full",0.01,10,3)
print("\nDONE",flush=True)
