from os import truncate
import numpy as np
from pyscf import gto, scf, data, df
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import pyscf, time, PAWutils, jax, HF_PAW
import HF_PAW_new
import coulomb

def calculate():
    a = 1.5
    L = 3
    cell = pgto.M(
        verbose = 3,
        atom = f'He {a} {a} {a}',
        basis = 'cc-pvtz',
        ke_cutoff=50.,
        a = np.eye(3)*L
    )


    #mo0, Energy0 = HF_PAW.HF(mol, Periodic=True, PAWorbitalCutOff=1e-10)
    # mo1, Energy1 = HF_PAW_new.HF(mol, Periodic=False, PAWorbitalCutOff=1e-10)
    # mo2, Energy2 = HF_PAW_new.HF_new(mol, Periodic=False, PAWorbitalCutOff=1e-10) #== GDF == PAW(1e-1)
    # print(Energy1, Energy2, (Energy1+Energy2)/2)


    # consider doing exact pyscf calculation? not sure how
    # GDF
    mf = pscf.RKS(mol)

    # Set the exchange-correlation functional to empty string for no XC functional
    mf.xc = ''

    # Enable Gaussian density fitting for periodic calculations
    # This sets the density fitting method to 'DF' (Gaussian density fitting)
    auxbasis = {'He': 'unc-sto3g'}
    auxbasis = {'He': [[0, (6.36242139, 1)], [0, (1.15892300, 1)], [0, (0.31364979, 1)]]}
    auxbasis = pyscf.df.aug_etb(mol, beta=1.3)
    # mf = mf.density_fit(auxbasis=auxbasis)

    # Run the SCF calculation
    # This will perform the self-consistent field calculation for the periodic system
    energy = mf.kernel()

def compare_energy():
    # basis = {'He': [[0, [8., 1]], [0, [3.77, 1]], [0, [1.24, 1]], [1, [4., 1]], [2, [3.77, 1]]]}
    basis = {'He': [[0, (3.36242139, 1)], [0, (1.15892300, 1)], [0, (0.31364979, 1)]]}
    a = 1.5
    L = 3
    cell = pgto.Cell(
        verbose = 3,
        #atom = atom,
        atom = f'He {a} {a} {a}',
        basis = basis,
        # basis = 'unc-cc-pvdz',
        # basis = 'unc-sto3g',
        #basis = 'unc-gth-dzvp',
        #pseudo = 'gth-hf',
        #basis = {'C': gto.etb([])}
        ke_cutoff=200,
        a = np.eye(3)*L
        #a = np.array([[10,0,0],[0,14,0],[0,0,12]])
    )

    cell.build()

    # consider doing exact pyscf calculation? not sure how
    # GDF
    mf = pscf.RKS(cell)

    # Set the exchange-correlation functional to empty string for no XC functional
    mf.xc = ''

    # Enable Gaussian density fitting for periodic calculations
    # This sets the density fitting method to 'DF' (Gaussian density fitting)
    # auxbasis = {'He': 'unc-sto3g'}
    # auxbasis = {'He': [[0, (6.36242139, 1)], [0, (1.15892300, 1)], [0, (0.31364979, 1)]]}
    auxbasis = pyscf.df.aug_etb(cell, beta=1.3)
    mf = mf.density_fit(auxbasis=auxbasis)

    # Run the SCF calculation
    # This will perform the self-consistent field calculation for the periodic system
    energy = mf.kernel()

    mos = mf.mo_coeff
    S = mf.get_ovlp()

    moOcc = mos[:,:cell.nelec[0]]
    # moOcc = np.array([
    #     [1,0],
    #     [0,0],
    #     [0,0],
    #     [0,1],
    #     [0,0],
    #     [0,0],
    # ])
    dm = moOcc @ moOcc.T
    print('Using Pyscf converged DM to calculate energies')

    nuc = cell.energy_nuc()
    h = mf.get_hcore().reshape((cell.nao, cell.nao))
    kpt = mf.kpt
    kin = mf.cell.pbc_intor('int1e_kin', 1, 1, kpt).reshape((cell.nao,cell.nao))
    nucElec = h-kin

    # pyscf all electron energy
    J1,K1 = mf.get_jk()
    # Hx = h + J1 - 0.5*K1
    Hx = h + J1
    E0 = PAWutils.getE(Hx, dm, h, nuc)
    # import pdb; pdb.set_trace()
    print(f'The pyscf energy with pyscf coulomb (no exchange) is\n{E0}')
    print('')

    # PAW energy (original coulomb)
    PAWdata, mesh, Rgrid, aoOnR, gOnRAll, mol = HF_PAW.getPAWdata(cell, PAWorbitalCutOff=1e-15)
    # HF_PAW.HF(cell)
    J_old = PAWutils.getj_PAW_JAX(mol, jnp.array(moOcc), aoOnR, mesh, PAWdata, Periodic=True )
    # K_old = PAWutils.getk_PAW_JAX(mol, jnp.array(moOcc), aoOnR, mesh, PAWdata, S, Periodic=False )
    # Hx = h + J_old - 0.5*K_old
    Hx = h + J_old
    E1 = PAWutils.getE(Hx, dm, h, nuc)
    print(f'The PAW energy "old" coulomb (no exchange) is\n{E1}')
    print(E0-E1)
    print('')
    # import pdb; pdb.set_trace()

    # PAW energy (charge neutral coulomb)
    PAWdata, mesh, Rgrid, aoOnR, gOnRAll, mol = HF_PAW_new.getPAWdata(cell, PAWorbitalCutOff=1e-15)
    J_new = coulomb.getj_PAW_JAX_new(mol, jnp.array(moOcc), aoOnR, mesh, PAWdata, Periodic=True )
    # J_new1s = coulomb.getj_PAW_new_debug(mol, jnp.array(moOcc), aoOnR, mesh, PAWdata, Periodic=False )
    # K_old = coulomb.getk_PAW_JAX(mol, jnp.array(moOcc), aoOnR, mesh, PAWdata, S, Periodic=False )
    Hx = kin + J_new
    E = coulomb.getE(Hx, dm, h, nuc)
    print(f'The PAW energy with "charge neutral" coulomb (no exchange) is\n{E}')
    print('')
    import pdb; pdb.set_trace()

def main():
    # calculate()
    compare_energy()

if __name__ == '__main__':
    main()
