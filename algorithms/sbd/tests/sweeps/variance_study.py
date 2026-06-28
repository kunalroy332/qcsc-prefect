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
mf=pyscf.scf.UHF(mol).run(verbose=0); e_uccsd=cc.UCCSD(mf).run(verbose=0).e_tot
ep=H.build_uhf_props(atom=atom,spin=spin)
SEEDS=range(1,9)  # 8 seeds for tighter stats
def study(conn,p,shots,dim,b,r):
    es=[]
    for sd in SEEDS:
        mat,probs=H.prepare_state_and_sample(ep,n_lucj_layers=2,shots=shots,seed=sd,connectivity=conn,noise_p=p)
        res=H.run_one_pass(ep,mat,probs,sqd_dim=dim,n_batches=b,n_recovery_steps=r,rng_seed=sd)
        if res.energy is not None: es.append((res.energy-e_uccsd)*1000)
    es=np.array(es); return es.mean(),es.std(),es.min(),es.max()
print(f"CN20q vs UCCSD mHa, 8 seeds. Does sqd_dim/shots tame variance? (heavy-hex p=0.01 b=10 rec=3)",flush=True)
print("shots   sqd_dim  mean    std    min     max",flush=True)
for shots,dim in [(100000,20000),(100000,60000),(300000,60000),(300000,120000)]:
    m,s,mn,mx=study("heavy-hex",0.01,shots,dim,10,3)
    print(f"{shots:7d} {dim:7d}  {m:+6.1f}  {s:5.1f}  {mn:+6.1f}  {mx:+6.1f}",flush=True)
print("\n-- and NOISELESS with big dim (is the ceiling itself high-variance, or is it the noise?) --",flush=True)
for shots,dim in [(100000,20000),(300000,120000)]:
    m,s,mn,mx=study("heavy-hex",0.0,shots,dim,10,3)
    print(f"p=0 {shots:7d} {dim:7d}  mean{m:+6.1f} std{s:5.1f} min{mn:+6.1f} max{mx:+6.1f}",flush=True)
print("DONE",flush=True)
