from pyscf import gto, scf, dft, sgx
from pyscf.pbc import gto as pgto
import numpy as np
import jax, pyscf
from pyscf.paw import NewPAW as PAW
from Gausslets import isdfgrid

jax.config.update("jax_enable_x64", True)

L=30
alpha0=10

# Molecule Definitions
he=[['He',[L/2,L/2,L/2]]]
ne=[['Ne',[L/2,L/2,L/2]]]
he2=[['He',[L/2-3,L/2,L/2]],
['He',[L/2+3,L/2,L/2]]]
ne2=[['Ne',[L/2-3,L/2,L/2]],
['Ne',[L/2+3,L/2,L/2]]]
n2=[
    ['N', [L/2+1.037, L/2, L/2]],
    ['N', [L/2-1.037, L/2, L/2]]
]
methane=[
    ['C', [L/2,           L/2,           L/2          ]],
    ['H', [L/2 + 1.18588, L/2 + 1.18588, L/2 + 1.18588]],
    ['H', [L/2 - 1.18588, L/2 - 1.18588, L/2 + 1.18588]],
    ['H', [L/2 - 1.18588, L/2 + 1.18588, L/2 - 1.18588]],
    ['H', [L/2 + 1.18588, L/2 - 1.18588, L/2 - 1.18588]]
]
ethylene=[
    ['C', [L/2 + 1.265,L/2,        L/2]],
    ['C', [L/2 - 1.265,L/2,        L/2]],
    ['H', [L/2 + 2.329,L/2 + 1.756,L/2]],
    ['H', [L/2 + 2.329,L/2 - 1.756,L/2]],
    ['H', [L/2 - 2.329,L/2 + 1.756,L/2]],
    ['H', [L/2 - 2.329,L/2 - 1.756,L/2]]
]
benzene=[
    # Carbon ring
    ['C', [L/2 + 2.62672, L/2,           L/2]],
    ['C', [L/2 + 1.31336, L/2 + 2.27481, L/2]],
    ['C', [L/2 - 1.31336, L/2 + 2.27481, L/2]],
    ['C', [L/2 - 2.62672, L/2,           L/2]],
    ['C', [L/2 - 1.31336, L/2 - 2.27481, L/2]],
    ['C', [L/2 + 1.31336, L/2 - 2.27481, L/2]],
    
    # Hydrogen ring
    ['H', [L/2 + 4.68652, L/2,           L/2]],
    ['H', [L/2 + 2.34326, L/2 + 4.05865, L/2]],
    ['H', [L/2 - 2.34326, L/2 + 4.05865, L/2]],
    ['H', [L/2 - 4.68652, L/2,           L/2]],
    ['H', [L/2 - 2.34326, L/2 - 4.05865, L/2]],
    ['H', [L/2 + 2.34326, L/2 - 4.05865, L/2]]
]

ddict2={'H':1/2,'He':1/2,'C':1/2,'N':1/2,'Ne':1/2}
ddict3={'H':1/3,'He':1/3,'C':1/3,'N':1/3,'Ne':1/3}
ddict4={'H':1/4,'He':1/4,'C':1/4,'N':1/4,'Ne':1/4}
ddict5={'H':1/5,'He':1/5,'C':1/5,'N':1/5,'Ne':1/5}
ddict6={'H':1/5,'He':1/6,'C':1/6,'N':1/6,'Ne':1/6}
ddict7={'H':1/5,'He':1/7,'C':1/7,'N':1/7,'Ne':1/7}
ddict8={'H':1/5,'He':1/8,'C':1/8,'N':1/8,'Ne':1/8}

# Testing Configuration
for ddict in [ddict8]:
    for atom in [methane]:
        verbose = 1
        cell = pgto.M(
            atom=atom,
            basis='cc-pvtz',
            #basis={'He':gto.parse('He S\n 12.212 1.0\n He S\n 2.212 1.0')}, # Sharp + Smooth
            a=np.array([[L,0,0],[0,L,0],[0,0,L]]),
            verbose=verbose,
            ke_cutoff=50,
            unit='B')

        mol = gto.M(atom=cell.atom, basis=cell.basis, verbose=verbose, unit='B')

        # Baseline RHF to generate consistent initial guess
        mf = scf.RHF(mol)
        mf.kernel()

        # ISDF-PAW setup
        gausslet = isdfgrid.ISDFGrid(mf, L, [ddict[i] for i in mol.elements], uniform=6, spread=0.7, maxgto=alpha0)
        mf_mol_paw = scf.RHF(mol).density_fit()
        mydf = PAW.from_mf(mf_mol_paw, cell, gausslet, alpha0=alpha0, augRadius=1.5).build()

        #from pyscf import lib
        #dm0 = mf.make_rdm1()
        #dm0 = lib.tag_array(dm0, mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ)
        from pyscf import lib
        dm0 = mf_mol_paw.get_init_guess()
        if hasattr(dm0, 'mo_coeff'):
            # Ensure mo_coeff matches nao
            if dm0.mo_coeff.shape[1] != mol.nao:
                dm0 = lib.tag_array(dm0, mo_coeff=None, mo_occ=None)

        mf_mol_paw.with_df = mydf
        mf_mol_paw.kernel(dm0=dm0)

        print("Original SCF: ", mf.e_tot)
        print("PAW SCF:      ", mf_mol_paw.e_tot)
        print("Difference:   ", mf_mol_paw.e_tot - mf.e_tot, flush=True)
        gausslet.print_timing_summary()
        exit()

        print("Smooth JK Differences")
        # Calculate analytic K matrix and E_K for smooth portions
        smooth_mol = gausslet.smooth_mol
        from Gausslets.addGTO import get_ao_radii
        radii = get_ao_radii(smooth_mol)
        nPAWidx = np.where(radii > 0)[0]
        
        DM = mf.make_rdm1()
        
        # Zero out non-smooth AO contributions in DM to isolate smooth exchange
        DM_smooth_full = np.zeros_like(DM)
        DM_smooth_full[np.ix_(nPAWidx, nPAWidx)] = DM[np.ix_(nPAWidx, nPAWidx)]
        
        # Use PySCF's efficient J and K build
        J_anal_full, K_anal_full = pyscf.scf.hf.get_jk(smooth_mol, DM_smooth_full, with_j=True, with_k=True)
        EJ = np.einsum('ij,ij', J_anal_full, DM_smooth_full)
        EK = 0.5 * np.einsum('ij,ij', K_anal_full, DM_smooth_full)

        J = gausslet.compute_coulomb(DM)
        EJ_isdf = np.einsum('ij,ji->', DM, J)
        K = gausslet.compute_exchange(DM)
        EK_isdf = 0.5 * np.einsum('ij,ji->', DM, K)
        print(f"Coulomb Difference: {abs(EJ_isdf - EJ):.4e}")
        print(f"Exchange Difference: {abs(EK_isdf - EK):.4e}")

        # Correct overlap and compute transformed exchange
        S_exact = smooth_mol.intor('int1e_ovlp')
        L_trans, T_trans = gausslet.create_transformation(S_exact)
        K_trans = gausslet.compute_exchange(DM, transform_L=L_trans)
        EK_trans = 0.5 * np.einsum('ij,ji->', DM, K_trans)
        print(f"Transformed Exchange Difference: {abs(EK_trans - EK):.4e}")
