import warnings; warnings.filterwarnings("ignore")
import os, sys
_TESTS=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
sys.path.insert(0,_TESTS); sys.path.insert(0,os.path.join(_TESTS,".."))
import numpy as np
import local_sqd_harness as H
import ffsim, pyscf
from pyscf import cc, fci

def build_op(t2_tuple, pairs, L):
    # mirror production _t2_to_ucj_parameters: optimize=True, n_reps=L+1, truncate last rep
    tmp=ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(t2=t2_tuple,n_reps=L+1,interaction_pairs=pairs,
                                                    optimize=True,options={"maxiter":50})
    return ffsim.UCJOpSpinUnbalanced(
        diag_coulomb_mats=tmp.diag_coulomb_mats[:-1],
        orbital_rotations=tmp.orbital_rotations[:-1],
        final_orbital_rotation=tmp.orbital_rotations[-1])

def run(atom,spin,label):
    mol=pyscf.gto.M(atom=atom,basis="sto-3g",spin=spin,verbose=0)
    mf=pyscf.scf.UHF(mol).run(verbose=0); e_uhf=mf.e_tot
    e_uccsd=cc.UCCSD(mf).run(verbose=0).e_tot
    try: e_fci=fci.FCI(mf).kernel()[0]
    except Exception: e_fci=float('nan')
    ep=H.build_uhf_props(atom=atom,spin=spin)
    norb=ep.num_orbitals; na,nb=ep.num_electrons
    t2=(ep.t2,ep.t2_ab,ep.t2_bb)
    aa,ab=H._heavy_hex_indices(norb); bb=H.lucj._default_bb_indices(aa,None)
    print(f"\n=== {label} ({2*norb}q) UHF={e_uhf:.5f} UCCSD={e_uccsd:.5f} FCI={e_fci:.5f} ===",flush=True)
    print("connectivity layers uniq  E_SQD       vs_UCCSD(mHa) vs_FCI <S^2>",flush=True)
    for name,pairs in [("heavy-hex",(aa,ab,bb)),("FULL",None)]:
        for L in [2,4]:
            op=build_op(t2,pairs,L)
            vec=ffsim.hartree_fock_state(norb,(na,nb))
            vec=ffsim.apply_unitary(vec,op,norb=norb,nelec=(na,nb))
            strings=ffsim.sample_state_vector(vec,norb=norb,nelec=(na,nb),shots=100000,seed=np.random.default_rng(3),concatenate=True)
            mat=np.array([[c=="1" for c in s] for s in strings],dtype=bool)
            uniq=len(set(map(bytes,mat)))
            probs=np.ones(len(strings))/len(strings)
            res=H.run_one_pass(ep,mat,probs,sqd_dim=20000,n_batches=3,n_recovery_steps=2)
            beat="BEATS UCCSD" if res.energy<e_uccsd else ""
            print(f"{name:12s} {L:5d} {uniq:5d}  {res.energy:.6f}  {(res.energy-e_uccsd)*1000:+8.2f}  {(res.energy-e_fci)*1000:+7.2f} {res.spin_sq:.4f} {beat}",flush=True)

run("N 0 0 0; H 0 0 1.3",2,"NH triplet")
run("C 0 0 0; N 0 0 1.3",1,"CN doublet")
print("\nDONE",flush=True)
