import numpy as np
import pyscf
from pyscf import gto, scf, dft
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt


import json
import os
import time

# Configuration
L = 40
alpha0 = 10
ddict10 = {'H':1/5,'He':1/10,'C':1/10,'N':1/10,'Ne':1/10}
basis_sets = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz']

# Molecule Geometries (centered at L/2)
molecules = {
    'He': [['He', [L/2, L/2, L/2]]],
    'Ne': [['Ne', [L/2, L/2, L/2]]],
    'He2': [['He', [L/2-3, L/2, L/2]], ['He', [L/2+3, L/2, L/2]]],
    'Ne2': [['Ne', [L/2-3, L/2, L/2]], ['Ne', [L/2+3, L/2, L/2]]],
    'N2': [['N', [L/2+1.037, L/2, L/2]], ['N', [L/2-1.037, L/2, L/2]]],
    'Methane': [
        ['C', [L/2,           L/2,           L/2          ]],
        ['H', [L/2 + 1.18588, L/2 + 1.18588, L/2 + 1.18588]],
        ['H', [L/2 - 1.18588, L/2 - 1.18588, L/2 + 1.18588]],
        ['H', [L/2 - 1.18588, L/2 + 1.18588, L/2 - 1.18588]],
        ['H', [L/2 + 1.18588, L/2 - 1.18588, L/2 - 1.18588]]
    ],
    'Ethylene': [
        ['C', [L/2 + 1.265,L/2,        L/2]],
        ['C', [L/2 - 1.265,L/2,        L/2]],
        ['H', [L/2 + 2.329,L/2 + 1.756,L/2]],
        ['H', [L/2 + 2.329,L/2 - 1.756,L/2]],
        ['H', [L/2 - 2.329,L/2 + 1.756,L/2]],
        ['H', [L/2 - 2.329,L/2 - 1.756,L/2]]
    ],
    'Propene': [
        ['C', [L/2 + 0.00000, L/2 + 0.00000, L/2 + 0.00000]],
        ['C', [L/2 + 2.52000, L/2 + 0.00000, L/2 + 0.00000]],
        ['C', [L/2 - 1.43000, L/2 + 2.38000, L/2 + 0.00000]],
        ['H', [L/2 - 0.95000, L/2 - 1.65000, L/2 + 0.00000]],
        ['H', [L/2 + 3.00000, L/2 - 1.65000, L/2 + 0.00000]],
        ['H', [L/2 + 3.00000, L/2 + 1.65000, L/2 + 0.00000]],
        ['H', [L/2 - 0.80000, L/2 + 4.20000, L/2 + 0.00000]],
        ['H', [L/2 - 2.30000, L/2 + 2.20000, L/2 + 1.60000]],
        ['H', [L/2 - 2.30000, L/2 + 2.20000, L/2 - 1.60000]]
    ],
    'Butadiene': [
        ['C', [L/2 - 2.52000, L/2 - 1.34000, L/2 + 0.00000]],
        ['C', [L/2 - 0.00000, L/2 - 0.00000, L/2 + 0.00000]],
        ['C', [L/2 + 2.52000, L/2 - 0.00000, L/2 + 0.00000]],
        ['C', [L/2 + 5.04000, L/2 + 1.34000, L/2 + 0.00000]],
        ['H', [L/2 - 4.20000, L/2 - 0.60000, L/2 + 0.00000]],
        ['H', [L/2 - 2.80000, L/2 - 3.30000, L/2 + 0.00000]],
        ['H', [L/2 - 0.30000, L/2 + 2.00000, L/2 + 0.00000]],
        ['H', [L/2 + 2.80000, L/2 - 2.00000, L/2 + 0.00000]],
        ['H', [L/2 + 5.34000, L/2 + 3.30000, L/2 + 0.00000]],
        ['H', [L/2 + 6.72000, L/2 + 0.60000, L/2 + 0.00000]]
    ],
    'Cyclobutane': [
        ['C', [L/2 + 1.45000, L/2 + 1.45000, L/2 + 0.00000]],
        ['C', [L/2 - 1.45000, L/2 + 1.45000, L/2 + 0.00000]],
        ['C', [L/2 - 1.45000, L/2 - 1.45000, L/2 + 0.00000]],
        ['C', [L/2 + 1.45000, L/2 - 1.45000, L/2 + 0.00000]],
        ['H', [L/2 + 2.50000, L/2 + 2.50000, L/2 + 1.00000]],
        ['H', [L/2 + 2.50000, L/2 + 2.50000, L/2 - 1.00000]],
        ['H', [L/2 - 2.50000, L/2 + 2.50000, L/2 + 1.00000]],
        ['H', [L/2 - 2.50000, L/2 + 2.50000, L/2 - 1.00000]],
        ['H', [L/2 - 2.50000, L/2 - 2.50000, L/2 + 1.00000]],
        ['H', [L/2 - 2.50000, L/2 - 2.50000, L/2 - 1.00000]],
        ['H', [L/2 + 2.50000, L/2 - 2.50000, L/2 + 1.00000]],
        ['H', [L/2 + 2.50000, L/2 - 2.50000, L/2 - 1.00000]]
    ],
    'Benzene': [
        ['C', [L/2 + 2.62672, L/2,           L/2]],
        ['C', [L/2 + 1.31336, L/2 + 2.27481, L/2]],
        ['C', [L/2 - 1.31336, L/2 + 2.27481, L/2]],
        ['C', [L/2 - 2.62672, L/2,           L/2]],
        ['C', [L/2 - 1.31336, L/2 - 2.27481, L/2]],
        ['C', [L/2 + 1.31336, L/2 - 2.27481, L/2]],
        ['H', [L/2 + 4.68652, L/2,           L/2]],
        ['H', [L/2 + 2.34326, L/2 + 4.05865, L/2]],
        ['H', [L/2 - 2.34326, L/2 + 4.05865, L/2]],
        ['H', [L/2 - 4.68652, L/2,           L/2]],
        ['H', [L/2 - 2.34326, L/2 - 4.05865, L/2]],
        ['H', [L/2 + 2.34326, L/2 - 4.05865, L/2]]
    ]
}

basis_sets = ['cc-pvqz']
basis_sets = ['cc-pvdz', 'cc-pvtz','cc-pvqz']

# Molecule Geometries (centered at L/2)
molecules = {
    'Ne': [['Ne', [L/2, L/2, L/2]]],
}

results_file = 'paw_basis_errors.json'
if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        all_results = json.load(f)
else:
    all_results = {}

for basis in basis_sets:
    if basis not in all_results:
        all_results[basis] = {}
        
    for name, geo in molecules.items():
        if name in all_results[basis]:
            print(f"Skipping {name}/{basis} (already done)")
            continue
            
        print(f"\n>>> Running {name} with {basis}...")
        # 1. Analytic Reference (RKS with xc='')
        mol = gto.M(atom=geo, basis=basis, unit='Bohr', verbose=0)
        mf = dft.RKS(mol)
        mf.xc = ''
        mf.init_guess = '1e'
        mf.kernel()
        dm = mf.make_rdm1()
        e_anal = mf.e_tot

        # # 2. Setup ISDF for J/K errors (Grid 1/10)
        # d_vals = [ddict10[mol.atom_pure_symbol(i)] for i in range(mol.natm)]
        # grid = isdfgrid.ISDFGrid(mf, L, d_vals, uniform=6, spread=0.7, maxgto=alpha0, cutoff=10000)

        # 3. PAW Calculation
        from pyscf.pbc import gto as pgto
        cell = pgto.M(atom=geo, basis=basis, a=np.array([[L,0,0],[0,L,0],[0,0,L]]), 
                        unit='B', verbose=4, ke_cutoff=40)
        
        mf_paw = dft.RKS(mol).density_fit()
        mf_paw.xc = ''
        mf.init_guess = '1e'
        mydf = PAW.from_mf(mf_paw, cell, alpha0=None, augRadius=None).build()
        mf_paw.with_df = mydf
        # pawnumint = PAWNumInt(mydf, mf_paw)
        # mf_paw._numint = pawnumint
        mf_paw.kernel()
        e_paw = mf_paw.e_tot
        err_scf = abs(e_paw - e_anal)

        # Calculate analytic reference for smooth part
        # smooth_mol = grid.smooth_mol
        # radii = get_ao_radii(smooth_mol)
        # nPAWidx = np.where(radii > 0)[0]
        # dm_smooth = np.zeros_like(dm)
        # dm_smooth[np.ix_(nPAWidx, nPAWidx)] = dm[np.ix_(nPAWidx, nPAWidx)]
        
        # j_anal, k_anal = pyscf.scf.hf.get_jk(smooth_mol, dm_smooth, with_j=True, with_k=True)
        # ej_ref = 0.5 * np.einsum('ij,ij', j_anal, dm_smooth)
        # ek_ref = 0.25 * np.einsum('ij,ij', k_anal, dm_smooth)

        # # J/K from ISDF
        # j_isdf = grid.compute_coulomb(dm)
        # ej_isdf = 0.5 * np.einsum('ij,ji->', dm, j_isdf)
        # k_isdf = grid.compute_exchange(dm)
        # ek_isdf = 0.25 * np.einsum('ij,ji->', dm, k_isdf)

        # err_j = abs(ej_isdf - ej_ref)
        # err_k = abs(ek_isdf - ek_ref)


        # Save result
        all_results[basis][name] = {
            'err_scf': err_scf,
            # 'err_j': err_j,
            # 'err_k': err_k,
            'e_anal': e_anal,
            'e_paw': e_paw
        }
        # print(f"  {name}/{basis} Done: SCF Error={err_scf:.4e}, J Error={err_j:.4e}, K Error={err_k:.4e}")
        print(f"  {name}/{basis} Done: SCF Error={err_scf:.4e}")

        # except Exception as e:
        #     print(f"  ERROR running {name}/{basis}: {e}")
            
        # Continuous saving
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=4)

print(f"\nFinished. Results saved to {results_file}")
