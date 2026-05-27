from pyscf import gto, scf, dft, sgx
from pyscf.pbc import gto as pgto
import numpy as np
import jax, pyscf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt


jax.config.update("jax_enable_x64", True)

# O2 molecule, triplet ground state
o2 = '''
O 0.0 0.0 0.0
O 0.0 0.0 1.207
'''

verbose = 4
ke_cutoff = 50
init_guess = '1e'

# We set spin=2 for O2 (triplet)
cell = pgto.M(
    atom=o2, 
    basis='cc-pvtz', 
    a=np.array([[10,0,0],[0,10,0],[0,0,10]]),
    spin=2,
    verbose=verbose,
    ke_cutoff=ke_cutoff,
)

mol = gto.M(atom=cell.atom, basis=cell.basis, spin=2, verbose=verbose)

# exact calculation
# UKS is chosen explicitly
mf_mol_exact = scf.UKS(mol)
mf_mol_exact.xc = 'pbe'
mf_mol_exact.init_guess = init_guess
mf_mol_exact.kernel()

# paw calculation
mf_mol_paw = scf.UKS(mol).density_fit()
mf_mol_paw.xc = 'pbe'
mf_mol_paw.init_guess = init_guess
mydf = PAW.from_mf(mf_mol_paw, cell,
                    alpha0=5, augRadius=None, with_multigrid=2, 
                    use_merged_multigrid=True, alpha0_lowmem=True, auto_box=True).build()
mf_mol_paw.with_df = mydf
# pawnumint = PAWNumInt(mydf, mf_mol_paw, use_merged_multigrid=True)
# mf_mol_paw._numint = pawnumint # this works but is no better than the default vxc
mf_mol_paw.kernel()

print(f'Energy difference: {mf_mol_exact.e_tot - mf_mol_paw.e_tot}')