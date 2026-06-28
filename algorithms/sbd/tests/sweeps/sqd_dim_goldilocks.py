import warnings; warnings.filterwarnings("ignore")
import os,sys
_T=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tests")
sys.path.insert(0,_T); sys.path.insert(0,os.path.join(_T,".."))
import numpy as np, local_sqd_harness as H
import pyscf
from pyscf import cc
atom,spin="C 0 0 0; N 0 0 1.3",1
mol=pyscf.gto.M(atom=atom,basis="sto-3g",spin=spin,verbose=0)
mf=pyscf.scf.UHF(mol).run(verbose=0); e_uccsd=cc.UCCSD(mf).run(verbose=0).e_tot
ep=H.build_uhf_props(atom=atom,spin=spin)
print(f"UCCSD={e_uccsd:.5f}",flush=True)
# EXACT earlier config that gave -91.167: ansatz_cmp2.py used shots=100000,seed=3,sqd_dim=20000,b=3,rec=2,full,L=2
print("Reproducing earlier 'beat' config exactly (full, L2, shots100k, seed3, dim20000, b3, rec2):",flush=True)
mat,probs=H.prepare_state_and_sample(ep,n_lucj_layers=2,shots=100000,seed=3,connectivity="full",noise_p=0.0)
res=H.run_one_pass(ep,mat,probs,sqd_dim=20000,n_batches=3,n_recovery_steps=2,rng_seed=0)
print(f"  E={res.energy:.6f}  vs_UCCSD={(res.energy-e_uccsd)*1000:+.2f} mHa  uniq_sampled={len(set(map(bytes,mat)))}",flush=True)
# now the recent config: shots 50000, dim 200000
print("Recent config (full, L2, shots50k, seed1, dim200000, b3, rec3):",flush=True)
mat2,probs2=H.prepare_state_and_sample(ep,n_lucj_layers=2,shots=50000,seed=1,connectivity="full",noise_p=0.0)
res2=H.run_one_pass(ep,mat2,probs2,sqd_dim=200000,n_batches=3,n_recovery_steps=3,rng_seed=1)
print(f"  E={res2.energy:.6f}  vs_UCCSD={(res2.energy-e_uccsd)*1000:+.2f} mHa  uniq_sampled={len(set(map(bytes,mat2)))}",flush=True)
# vary ONLY sqd_dim at the earlier good config
print("Vary sqd_dim at earlier config (shots100k seed3 b3 rec2 full):",flush=True)
for dim in [2000,20000,200000]:
    m,p=H.prepare_state_and_sample(ep,n_lucj_layers=2,shots=100000,seed=3,connectivity="full",noise_p=0.0)
    r=H.run_one_pass(ep,m,p,sqd_dim=dim,n_batches=3,n_recovery_steps=2,rng_seed=0)
    print(f"  dim={dim:7d} dets/spin={int(dim**0.5):4d}  E={r.energy:.6f} vs_UCCSD={(r.energy-e_uccsd)*1000:+.2f}",flush=True)
print("DONE",flush=True)
