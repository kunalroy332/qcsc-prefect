import warnings; warnings.filterwarnings("ignore")
import os,sys,time,resource
_T=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tests")
sys.path.insert(0,_T); sys.path.insert(0,os.path.join(_T,".."))
try: resource.setrlimit(resource.RLIMIT_AS,(14*1024**3,14*1024**3))
except Exception: pass
import numpy as np, local_sqd_harness as H
import pyscf, math
from pyscf import cc
# CN 6-31g 18 orb (tractable proxy, full space 3.2e4/spin). Sweep sqd_dim at SEVERAL noise levels
# to see if the monotonic "bigger sqd_dim is better" holds WITH noise (the regime the Fugaku run is in).
atom,spin,basis="C 0 0 0; N 0 0 1.17",1,"6-31g"
mol=pyscf.gto.M(atom=atom,basis=basis,spin=spin,verbose=0)
mf=pyscf.scf.UHF(mol).run(verbose=0); e_uhf=mf.e_tot
e_uccsd=cc.UCCSD(mf).run(verbose=0).e_tot
ep=H.build_uhf_props(atom=atom,spin=spin,basis=basis)
print(f"CN 18orb: UHF={e_uhf:.5f} UCCSD={e_uccsd:.5f}",flush=True)
print(f"{'noise_p':>7} {'sqd_dim':>8} {'dets/spin':>9} {'vs_UCCSD(mHa)':>13} {'vs_UHF(mHa)':>11}",flush=True)
for p in [0.0, 0.01, 0.03]:
    for dim in [30000, 150000, 600000]:
        best=None; t=time.time()
        for sd in [1,2]:
            try:
                mat,probs=H.prepare_state_and_sample(ep,n_lucj_layers=2,shots=80000,seed=sd,connectivity="heavy-hex",noise_p=p)
                res=H.run_one_pass(ep,mat,probs,sqd_dim=dim,n_batches=1,n_recovery_steps=1,rng_seed=sd)
                if res.energy and (best is None or res.energy<best): best=res.energy
            except MemoryError: best=None; break
        if best is None:
            print(f"{p:>7.2f} {dim:>8} {int(dim**0.5):>9} {'OOM':>13}",flush=True)
        else:
            print(f"{p:>7.2f} {dim:>8} {int(dim**0.5):>9} {(best-e_uccsd)*1000:>+12.1f} {(best-e_uhf)*1000:>+10.1f}  ({time.time()-t:.0f}s)",flush=True)
print("DONE",flush=True)
