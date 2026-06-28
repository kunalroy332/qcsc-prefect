import warnings; warnings.filterwarnings("ignore")
import os,sys
_T=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tests")
sys.path.insert(0,_T); sys.path.insert(0,os.path.join(_T,".."))
import numpy as np, local_sqd_harness as H, math
import pyscf
from pyscf import cc, fci
for atom,spin,name in [("O 0 0 0; O 0 0 1.2",2,"O2"),("C 0 0 0; N 0 0 1.3",1,"CN")]:
    mol=pyscf.gto.M(atom=atom,basis="sto-3g",spin=spin,verbose=0)
    mf=pyscf.scf.UHF(mol).run(verbose=0)
    e_uccsd=cc.UCCSD(mf).run(verbose=0).e_tot; e_fci=fci.FCI(mf).kernel()[0]
    ep=H.build_uhf_props(atom=atom,spin=spin)
    print(f"\n{name}: UCCSD={e_uccsd:.5f} FCI={e_fci:.5f}",flush=True)
    print("connectivity layers  E_SQD(best/3seed) vs_UCCSD vs_FCI",flush=True)
    for conn in ["heavy-hex","full"]:
        for L in [2,4]:
            best=None
            for sd in [1,2,3]:
                try:
                    mat,probs=H.prepare_state_and_sample(ep,n_lucj_layers=L,shots=50000,seed=sd,connectivity=conn,noise_p=0.0)
                    res=H.run_one_pass(ep,mat,probs,sqd_dim=200000,n_batches=5,n_recovery_steps=3,rng_seed=sd)
                    if res.energy and (best is None or res.energy<best): best=res.energy
                except Exception as e:
                    best=float('nan')
            print(f"{conn:12s} {L:5d}  {best:.5f}  {(best-e_uccsd)*1000:+7.1f} {(best-e_fci)*1000:+7.1f} {'BEATS' if best<e_uccsd else ''}",flush=True)
print("\nDONE",flush=True)
