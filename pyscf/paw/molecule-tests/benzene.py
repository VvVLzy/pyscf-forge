from pyscf import gto, scf, dft, sgx
from pyscf.pbc import gto as pgto
import numpy as np
import jax, pyscf
from pyscf.paw import PAW
from Gausslets import MultiSlicing, Deformation1D, HFdenfit, addGTO


jax.config.update("jax_enable_x64", True)
L = 36
atom = f''' C {L/2 +0.000000} {L/2 +1.396792} {L/2+0.000000};
            C {L/2 +0.000000} {L/2 -1.396792} {L/2+0.000000};
            C {L/2 +1.209657} {L/2 +0.698396} {L/2 +0.00000};
            C {L/2 -1.209657} {L/2 -0.698396} {L/2+0.000000};
            C {L/2 -1.209657} {L/2 +0.698396} {L/2+0.000000};
            C {L/2 +1.209657} {L/2 -0.698396} {L/2+0.000000};
            H {L/2 +0.000000} {L/2 +2.484212} {L/2+0.000000};
            H {L/2 +2.151390} {L/2 +1.242106} {L/2+0.000000};
            H {L/2 -2.151390} {L/2 -1.242106} {L/2+0.000000};
            H {L/2 -2.151390} {L/2 +1.242106} {L/2+0.000000};
            H {L/2 +2.151390} {L/2 -1.242106} {L/2+0.000000};
            H {L/2 +0.000000} {L/2 -2.484212} {L/2+0.000000}'''

#benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
#           [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
#           [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
#           [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
#           [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
#           [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
#           [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
#           [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
#           [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
#           [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
#           [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
#           [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]

verbose = 1
ke_cutoff = 50
init_guess = '1e'

cell = pgto.M(
    atom=atom, 
    basis='cc-pvdz', 
    a=np.array([[20,0,0],[0,20,0],[0,0,15]]),
    verbose=verbose,
    unit='B',
    ke_cutoff=ke_cutoff,)


mol = gto.M(atom=cell.atom, basis=cell.basis, verbose=verbose,unit='B')
mf = dft.RKS(mol)
mf.xc = 'pbe'
gaussletints=MultiSlicing.runAtomCalculation(mf, L=L, d=[0.3]*mol.natm, cutoff=100000)

#V2e=np.array(gaussletints["V2e"])
#pos=np.array(gaussletints["pos"])
#aovals=gaussletints["aovals"]

# paw calculation
mf_mol_paw = dft.RKS(mol).density_fit()
mf_mol_paw.xc = 'pbe'
mf_mol_paw.init_guess = init_guess
mydf = PAW.from_mf(mf_mol_paw, cell,gaussletints,
                    PWAccuracy=1e-4,alpha0=5,augRadius=1.5).build()
mf_mol_paw.with_df = mydf
mf_mol_paw.kernel()
print(mf_mol_paw.e_tot)
print(mydf.Times_)
