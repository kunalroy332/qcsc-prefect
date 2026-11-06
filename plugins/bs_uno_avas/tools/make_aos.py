# Free-atom ROHF occupied orbitals used as AVAS targets (bs_uno_avas, target_aos = "file:AOs").
# Run from anywhere: rewrites the package data src/bs_uno_avas/data/AOs/{Fe,S}_AO.npy / _lb.npy
# and the .cube files in tools/AOs/. The reference ROHF orbitals tools/AOs/{Fe,S}_ref.chk are
# only read (mf.chkfile = None), never overwritten.
import os

import numpy as np
from pyscf import gto, scf, tools

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'AOs')  # reference chkfiles + cube files
OUT = os.path.join(HERE, '..', 'src', 'bs_uno_avas', 'data', 'AOs')  # package data
b = 'dzvp'  # PySCF built-in DGauss DZVP (same as the Basis Set Exchange dgauss-dzvp set)

#import numpy as np
#from pyscf import gto,scf,tools,mcscf,mrpt,cc,ao2mo,dft,lo
#from pyscf.data.elements import chemcore
#from pyscf.tools import cubegen
#from pyscf.mcscf import avas

mol = gto.Mole()
mol.build(
    atom     = [['S',(0,0,0)]],
    basis    = b,
    verbose  = 4,
    charge   = 0,
    spin     = 2,
    symmetry = True)

mf = scf.ROHF(mol)
ref_chk = os.path.join(REF, 'S_ref.chk')
mf.chkfile = None  # read-only reference: never write back to ref_chk
dm0 = None
if os.path.exists(ref_chk):
    print(f'Loading reference orbitals from {ref_chk}')
    dm0 = mf.from_chk(ref_chk)
    mf.max_cycle = 0
else:
    print(f'No {ref_chk}: converging from scratch (not saved)')
mf = scf.newton(mf)
mf.kernel(dm0=dm0)

nocc = (mf.mol.nelectron+mf.mol.spin)//2
lb = ['1s','2s','2pz','2px','2py','3s','3py','3pz','3px']
for p in range(nocc):
    tools.cubegen.orbital(mol,os.path.join(REF,'S_AO_%d_%s.cube'%(p,lb[p])),mf.mo_coeff[:,p])
np.save(os.path.join(OUT,'S_AO.npy'),mf.mo_coeff[:,:nocc])
np.save(os.path.join(OUT,'S_lb.npy'),lb)

# -----------------

mol = gto.Mole()
mol.build(
    atom     = [['Fe',(0,0,0)]],
    basis    = b,
    verbose  = 4,
    charge   = 0,
    spin     = 4,
    symmetry = True)

mf = scf.ROHF(mol)
import os
ref_chk = os.path.join(REF, 'Fe_ref.chk')
mf.chkfile = None  # read-only reference: never write back to ref_chk
dm0 = None
if os.path.exists(ref_chk):
    print(f'Loading reference orbitals from {ref_chk}')
    dm0 = mf.from_chk(ref_chk)
    mf.max_cycle = 0
else:
    print(f'No {ref_chk}: converging from scratch (not saved)')
mf = scf.newton(mf)
mf.kernel(dm0=dm0)

nocc = (mf.mol.nelectron+mf.mol.spin)//2
lb = ['1s','2s','2py','2px','2pz','3s','3px','3pz','3py','3dyz','4s','3dxz','3dxy','3dx2-y2','3dz^2']
for p in range(nocc):
    tools.cubegen.orbital(mol,os.path.join(REF,'Fe_AO_%d_%s.cube'%(p,lb[p])),mf.mo_coeff[:,p])
np.save(os.path.join(OUT,'Fe_AO.npy'),mf.mo_coeff[:,:nocc])
np.save(os.path.join(OUT,'Fe_lb.npy'),lb)


