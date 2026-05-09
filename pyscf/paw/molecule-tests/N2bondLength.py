###
# Compare PAW DFT(PBE) energy with the exact pyscf energy as the bond
# length between two C atoms are varied

import numpy as np
import csv

from pyscf.pbc import gto as pgto
from pyscf import gto, scf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

# specify C2
r1      = 0.    # location of first carbon
r0       = 1.10    # N-N bond length
atom    = f'N {r1} {r1} {r1}; N {r1+r0/3.**0.5} {r1+r0/3.**0.5} {r1+r0/3.**0.5}'
L = 20
rscales = np.linspace(0.7, 1.5, 15)

# other params
verbose     = 4
basis       = "unc-cc-pvdz"
ke_cutoff   = 50.
aa          = L
a           = np.eye(3) * aa

# periodic cell for PAW calculation
cell = pgto.M(
    atom        = atom,
    verbose     = verbose,
    basis       = basis,
    ke_cutoff   = ke_cutoff,
    a           = a,
)

# corresponding mol for pyscf exact calculation
mol = gto.M(
    atom        = atom,
    verbose     = verbose,
    basis       = basis,
)

# mf params
xc = 'pbe'
init_guess = '1e'

def single_point():
    # exact calculation
    mf_mol_exact = scf.RKS(mol)
    mf_mol_exact.xc = xc
    mf_mol_exact.init_guess = init_guess
    mf_mol_exact.kernel()

    # paw calculation
    mf_mol_paw = scf.RKS(mol).density_fit()
    mf_mol_paw.xc = xc
    mf_mol_paw.init_guess = init_guess
    mydf = PAW.from_mf(mf_mol_paw, cell,
                        alpha0=None, augRadius=None, with_multigrid=2).build()
    mf_mol_paw.with_df = mydf
    # pawnumint = PAWNumInt(mydf, mf_mol_paw)
    # mf_mol_paw._numint = pawnumint # this works but is no better than the default vxc
    mf_mol_paw.kernel()

    print(mf_mol_exact.e_tot - mf_mol_paw.e_tot)


def exact_paw_diff():
    results = []
    for rscale in rscales:
        r       = r0*rscale    # rescale C=C bond length
        atom    = f'N {r1} {r1} {r1}; N {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'

        cell.atom = atom
        cell.build()
        mol.atom = atom
        mol.build()

        # exact calculation
        mf_mol_exact = scf.RKS(mol)
        mf_mol_exact.xc = xc
        mf_mol_exact.init_guess = init_guess
        mf_mol_exact.kernel()

        # paw calculation
        mf_mol_paw = scf.RKS(mol).density_fit()
        mf_mol_paw.xc = xc
        mf_mol_paw.init_guess = init_guess
        mydf = PAW.from_mf(mf_mol_paw, cell,
                           PWAccuracy=1e-6).build()
        mf_mol_paw.with_df = mydf
        mf_mol_paw.kernel()

        # append all results in one dict
        results.append({
            "r": r,
            "etot_pyscf": mf_mol_exact.e_tot,
            "etot_paw": mf_mol_paw.e_tot,
            "ediff_per_atm": abs(mf_mol_exact.e_tot - mf_mol_paw.e_tot) / cell.natm,
            "conv_pyscf": mf_mol_exact.converged,
            "conv_paw": mf_mol_paw.converged
        })

    # write to csv
    path = f"data/N2bondLength_{ke_cutoff:.2f}_{int(aa)}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "r", "etot_pyscf", "etot_paw", "ediff_per_atm", "conv_pyscf", "conv_paw"
        ])
        writer.writeheader()
        writer.writerows(results)

def exact_gdf_diff():
    results = []
    for rscale in rscales:
        r       = r0*rscale    # rescale C=C bond length
        atom    = f'N {r1} {r1} {r1}; N {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'

        cell.atom = atom
        cell.build()
        mol.atom = atom
        mol.build()

        # exact calculation
        mf_mol_exact = scf.RKS(mol)
        mf_mol_exact.xc = xc
        mf_mol_exact.init_guess = init_guess
        mf_mol_exact.kernel()

        # gdf calculation
        mf_mol_gdf = scf.RKS(mol).density_fit()
        mf_mol_gdf.xc = xc
        mf_mol_gdf.init_guess = init_guess
        mf_mol_gdf.kernel()

        # append all results in one dict
        results.append({
            "r": r,
            "etot_pyscf": mf_mol_exact.e_tot,
            "etot_gdf": mf_mol_gdf.e_tot,
            "ediff_per_atm": abs(mf_mol_exact.e_tot - mf_mol_gdf.e_tot) / cell.natm,
            "conv_pyscf": mf_mol_exact.converged,
            "conv_gdf": mf_mol_gdf.converged
        })

    # write to csv
    path = f"data/N2bondLength_{ke_cutoff:.2f}_{int(aa)}_gdf .csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "r", "etot_pyscf", "etot_gdf", "ediff_per_atm", "conv_pyscf", "conv_gdf"
        ])
        writer.writeheader()
        writer.writerows(results)

def basis_set_convergence():
    basis = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz']
    data_basis = {}

    for b in basis:
        mol.basis = b
        rs, etot_pyscf, pyscf_converged = [], [], []
        for rscale in rscales:
            r = r0 * rscale
            atom = f'N {r1} {r1} {r1}; N {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'
            mol.atom = atom
            mol.build()

            mf = scf.RKS(mol)
            mf.xc = xc
            mf.init_guess = init_guess
            mf.kernel()

            rs.append(r)
            etot_pyscf.append(mf.e_tot)
            pyscf_converged.append(mf.converged)

        data_basis[b] = {"r": rs, "etot_pyscf": etot_pyscf, "conv_pyscf": pyscf_converged}

    # write to csv in wide format
    path = 'data/N2bondLength_convergence.csv'
    print(f'Calculation done. Saving data to {path}...')
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # --- energies ---
        writer.writerow(['etot_pyscf'])
        writer.writerow(['r'] + data_basis[basis[0]]["r"])
        for b in basis:
            writer.writerow([b] + data_basis[b]["etot_pyscf"])

        writer.writerow([])  # blank row between tables

        # --- convergence flags ---
        writer.writerow(['conv_pyscf'])
        writer.writerow(['r'] + data_basis[basis[0]]["r"])
        for b in basis:
            writer.writerow([b] + data_basis[b]["conv_pyscf"])



def main():
    # basis_set_convergence()
    # import time
    # start = time.time()
    # exact_paw_diff()
    # print(f'Time elapsed: {time.time()-start:.2f}')
    # exact_gdf_diff()
    single_point()

if __name__ == '__main__':
    main()