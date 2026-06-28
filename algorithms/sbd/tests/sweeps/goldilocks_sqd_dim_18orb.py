import warnings; warnings.filterwarnings("ignore")
import os,sys,time,resource
_T=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tests")
sys.path.insert(0,_T); sys.path.insert(0,os.path.join(_T,".."))
try: resource.setrlimit(resource.RLIMIT_AS,(14*1024**3,14*1024**3))
except Exception: pass
import numpy as np, local_sqd_harness as H
import pyscf, math
from pyscf import cc, fci
atom,spin,basis="C 0 0 0; N 0 0 1.17",1,"6-31g"
mol=pyscf.gto.M(atom=atom,basis=basis,spin=spin,verbose=0)
mf=pyscf.scf.UHF(mol).run(verbose=0); e_uhf=mf.e_tot; s2=mf.spin_square()[0]
e_uccsd=cc.UCCSD(mf).run(verbose=0).e_tot
na,nb=mol.nelec; norb=mol.nao_nr(); fa=math.comb(norb,na)
ep=H.build_uhf_props(atom=atom,spin=spin,basis=basis)
print(f"=== GOLDILOCKS SWEEP: CN 6-31g, norb={norb} ({2*norb}q), nelec=({na},{nb}) ===",flush=True)
print(f"UHF={e_uhf:.5f} (<S^2>={s2:.3f})  UCCSD={e_uccsd:.5f}  full/spin={fa:.2e}",flush=True)
print(f"noiseless, heavy-hex L2, best of 3 seeds, rec=1 b=1",flush=True)
print(f"{'sqd_dim':>9} {'dets/spin':>9} {'%full':>7} {'best_E':>12} {'vs_UCCSD':>9} {'<S^2>':>7} {'sec':>5}",flush=True)
# sweep dim widely to bracket the optimum (dets/spin from ~45 up toward full space ~32k)
for dim in [2000,8000,30000,80000,150000,300000,600000,1000000]:
    bestE=None; bestS2=None; t=time.time()
    for sd in [1,2,3]:
        try:
            mat,probs=H.prepare_state_and_sample(ep,n_lucj_layers=2,shots=80000,seed=sd,connectivity="heavy-hex",noise_p=0.0)
            res=H.run_one_pass(ep,mat,probs,sqd_dim=dim,n_batches=1,n_recovery_steps=1,rng_seed=sd)
            if res.energy and (bestE is None or res.energy<bestE): bestE=res.energy; bestS2=res.spin_sq
        except MemoryError:
            bestE=None; break
        except Exception as e:
            print(f"  seed {sd} dim {dim}: {type(e).__name__}",flush=True)
    dps=int(dim**0.5)
    if bestE is None:
        print(f"{dim:>9} {dps:>9} {dps/fa:>6.1%} {'OOM/skip':>12} ({time.time()-t:.0f}s)",flush=True)
    else:
        tag="BEATS" if bestE<e_uccsd else ""
        print(f"{dim:>9} {dps:>9} {dps/fa:>6.1%} {bestE:>12.5f} {(bestE-e_uccsd)*1000:>+8.1f} {bestS2:>7.4f} {time.time()-t:>5.0f} {tag}",flush=True)
print("DONE",flush=True)
