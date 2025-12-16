from pyscf import gto, scf, dft, sgx
from pyscf.pbc import gto as pgto
import numpy as np
import jax, pyscf
from pyscf.paw import PAW
from Gausslets import GaussletGrid, MultiSlicing, Deformation1D, HFdenfit, addGTO

L=14
he=[['He',[L/2,L/2,L/2]]]
alpha0=3

cell = pgto.M(
    atom=he,
    basis='cc-pvdz',
    a=np.array([[L,0,0],[0,L,0],[0,0,L]]),
    verbose=1,
    ke_cutoff=100,
    unit='B')
mol = gto.M(atom=he, basis='cc-pvdz', verbose=1,unit='B')

# paw calculation
mf=dft.RKS(mol)
mf.xc = 'pbe'
mf.kernel()
gausslet=GaussletGrid.GaussletGrid(mf,L,[0.3]*mol.natm,uniform=6,cutoff=0.5,interp=2,spread=2)
gausslet.getGrid()
gausslet.plot2DGrid()
mf_mol_paw = dft.RKS(mol).density_fit()
mf_mol_paw.xc = 'pbe'
mydf = PAW.from_mf(mf_mol_paw, cell,gausslet,alpha0=alpha0,#augRadius=1,
                    PWAccuracy=1e-8).build()
mf_mol_paw.with_df = mydf
mf_mol_paw.kernel()
print("Original SCF: ",mf.e_tot)
print("PAW SCF: ",mf_mol_paw.e_tot)
print("Difference: ",mf_mol_paw.e_tot-mf.e_tot,flush=True)
gausslet=GaussletGrid.GaussletGrid(mf,L,[0.15]*mol.natm,uniform=6,cutoff=0.5,interp=2,spread=2)
gausslet.getGrid()
gausslet.plot2DGrid()
mf_mol_paw = dft.RKS(mol).density_fit()
mf_mol_paw.xc = 'pbe'
mydf = PAW.from_mf(mf_mol_paw, cell,gausslet,alpha0=alpha0,#augRadius=1,
                    PWAccuracy=1e-8).build()
mf_mol_paw.with_df = mydf
mf_mol_paw.kernel()
print("Original SCF: ",mf.e_tot)
print("PAW SCF: ",mf_mol_paw.e_tot)
print("Difference: ",mf_mol_paw.e_tot-mf.e_tot,flush=True)
gausslet=GaussletGrid.GaussletGrid(mf,L,[0.1]*mol.natm,uniform=6,cutoff=0.5,interp=2,spread=2)
gausslet.getGrid()
gausslet.plot2DGrid()
mf_mol_paw = dft.RKS(mol).density_fit()
mf_mol_paw.xc = 'pbe'
mydf = PAW.from_mf(mf_mol_paw, cell,gausslet,alpha0=alpha0,#augRadius=1,
                    PWAccuracy=1e-8).build()
mf_mol_paw.with_df = mydf
mf_mol_paw.kernel()
print("Original SCF: ",mf.e_tot)
print("PAW SCF: ",mf_mol_paw.e_tot)
print("Difference: ",mf_mol_paw.e_tot-mf.e_tot,flush=True)

