from pyscf import gto, scf, dft, sgx
from pyscf.pbc import gto as pgto
import numpy as np
import jax, pyscf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

jax.config.update("jax_enable_x64", True)

benzene = [[ 'C'  , ( 4.673795 ,   6.280948 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   5.572311 , 0.00  ) ],
           [ 'C'  , ( 5.901190 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 4.673795 ,   3.446400 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   4.155037 , 0.00  ) ],
           [ 'C'  , ( 3.446400 ,   5.572311 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   7.376888 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   6.120281 , 0.00  ) ],
           [ 'H'  , ( 6.850301 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 4.673795 ,   2.350461 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   3.607068 , 0.00  ) ],
           [ 'H'  , ( 2.497289 ,   6.120281 , 0.00  ) ]]

verbose = 4
ke_cutoff = 50
init_guess = '1e'

cell = pgto.M(
    atom=benzene, 
    basis='cc-pvtz', 
    a=np.array([[20,0,0],[0,20,0],[0,0,20]]),
    verbose=verbose,
    ke_cutoff=ke_cutoff,)


mol = gto.M(atom=cell.atom, basis=cell.basis, verbose=verbose)

# exact calculation
mf_mol_exact = scf.RKS(mol)
mf_mol_exact.xc = 'pbe'
mf_mol_exact.init_guess = init_guess
mf_mol_exact.kernel()


# paw calculation
mf_mol_paw = scf.RKS(mol).density_fit()
mf_mol_paw.xc = 'pbe'
mf_mol_paw.init_guess = init_guess
mydf = PAW.from_mf(mf_mol_paw, cell,
                    alpha0=5, augRadius=None, with_multigrid=2, alpha0_lowmem=True).build()
mf_mol_paw.with_df = mydf
pawnumint = PAWNumInt(mydf, mf_mol_paw, use_merged_multigrid=True)
mf_mol_paw._numint = pawnumint # this works but is no better than the default vxc
mf_mol_paw.kernel()

print(f'Energy difference: {mf_mol_exact.e_tot - mf_mol_paw.e_tot}')
