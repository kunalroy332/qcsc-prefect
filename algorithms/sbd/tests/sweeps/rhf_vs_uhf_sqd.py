import warnings; warnings.filterwarnings("ignore")
import os,sys
_T=os.path.join(os.path.dirname(os.path.abspath(__file__)),"tests")
sys.path.insert(0,_T); sys.path.insert(0,os.path.join(_T,".."))
import numpy as np, local_sqd_harness as H
import ffsim, pyscf
from pyscf import cc, fci
from qiskit_addon_sqd.fermion import solve_fermion

def uccsd_fci(atom,spin,unrestricted):
    mol=pyscf.gto.M(atom=atom,basis="sto-3g",spin=spin,verbose=0)
    mf=(pyscf.scf.UHF(mol) if unrestricted else pyscf.scf.RHF(mol)).run(verbose=0)
    e_cc=(cc.UCCSD(mf) if unrestricted else cc.CCSD(mf)).run(verbose=0).e_tot
    e_fci=fci.FCI(mf).kernel()[0]
    return mf.e_tot,e_cc,e_fci

def rhf_sqd(ep,dim,seed,L=2):
    # build RHF (spin-balanced) LUCJ via optimize=True, sample, subsample_close_shell, solve
    norb=ep.num_orbitals; na,nb=ep.num_electrons
    aa,ab=H._heavy_hex_indices(norb)
    tmp=ffsim.UCJOpSpinBalanced.from_t_amplitudes(t2=ep.t2,n_reps=L+1,interaction_pairs=(aa,ab),optimize=True,options={"maxiter":50})
    op=ffsim.UCJOpSpinBalanced(diag_coulomb_mats=tmp.diag_coulomb_mats[:-1],orbital_rotations=tmp.orbital_rotations[:-1],final_orbital_rotation=tmp.orbital_rotations[-1])
    vec=ffsim.hartree_fock_state(norb,(na,nb)); vec=ffsim.apply_unitary(vec,op,norb=norb,nelec=(na,nb))
    strings=ffsim.sample_state_vector(vec,norb=norb,nelec=(na,nb),shots=100000,seed=np.random.default_rng(seed),concatenate=True)
    mat=np.array([[c=="1" for c in s] for s in strings],dtype=bool)
    probs=np.ones(len(strings))/len(strings)
    H.sqd.MODULE_RNG=np.random.default_rng(0)
    b,pr=H.sqd.recover_configurations.fn(bitstring_matrix=mat,probabilities=probs,avg_occupancies=ep.initial_occupancy,num_elec_a=na,num_elec_b=nb,rand_seed=H.sqd.MODULE_RNG)
    bp,pp=H.sqd.postselect_bitstrings.fn(bitstring_matrix=b,probabilities=pr,hamming_right=na,hamming_left=nb)
    empty=np.empty((0,norb),bool)
    ci=H.sqd.subsample_close_shell.fn(bitstring_matrix=bp,probabilities=pp,carryover=empty,subspace_dim=dim,norb=norb,num_elec_a=na)
    e,_,_,_=solve_fermion((ci,ci),ep.one_body_tensor,ep.two_body_tensor,open_shell=False)
    return e+ep.nuclear_repulsion_energy

atom,spin="N 0 0 0; N 0 0 1.3",0
e_uhf,e_uccsd_u,e_fci=uccsd_fci(atom,spin,True)
e_rhf,e_ccsd_r,_=uccsd_fci(atom,spin,False)
print(f"N2 20q closed-shell: RHF={e_rhf:.5f} CCSD={e_ccsd_r:.5f} | UHF={e_uhf:.5f} UCCSD={e_uccsd_u:.5f} | FCI={e_fci:.5f}",flush=True)
epR=H.build_uhf_props(atom=atom,spin=spin) if False else None
import qcsc_workflow_utility.chem as chem
from unittest import mock; chem.get_run_logger=lambda:mock.MagicMock()
epR=chem.compute_molecular_integrals_from_geometry.fn(atom=atom,basis="sto-3g",unrestricted=False)
epU=chem.compute_molecular_integrals_from_geometry.fn(atom=atom,basis="sto-3g",unrestricted=True,spin=spin)
print("\nNOISELESS, full sqd_dim Goldilocks sweep, L=2, best of 3 seeds:",flush=True)
print("dim    dets/spin  RHF-SQD   vs_CCSD | UHF-SQD   vs_UCCSD",flush=True)
for dim in [5000,20000,80000,200000]:
    bR=bU=None
    for sd in [1,2,3]:
        try:
            r=rhf_sqd(epR,dim,sd)
            if bR is None or r<bR: bR=r
        except Exception as e: bR=float('nan')
        m,p=H.prepare_state_and_sample(epU,n_lucj_layers=2,shots=100000,seed=sd,connectivity="heavy-hex",noise_p=0.0)
        ru=H.run_one_pass(epU,m,p,sqd_dim=dim,n_batches=3,n_recovery_steps=2,rng_seed=0)
        if ru.energy and (bU is None or ru.energy<bU): bU=ru.energy
    print(f"{dim:6d} {int(dim**0.5):8d}  {bR:.5f} {(bR-e_ccsd_r)*1000:+7.1f} | {bU:.5f} {(bU-e_uccsd_u)*1000:+7.1f}",flush=True)
print("DONE",flush=True)
