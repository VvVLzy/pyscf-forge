###
# Compare PAW DFT(PBE) energy with the exact pyscf energy as the bond
# length between two C atoms are varied

import numpy as np
import csv
import json
import argparse

from pyscf.pbc import gto as pgto
from pyscf import gto, scf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

# specify C2
r1      = 0.    # location of first carbon
r0       = 1.037*2    # N-N bond length
atom    = f'N {r1} {r1} {r1}; N {r1+r0} {r1} {r1}'
L = 20
# rscales = np.linspace(0.7, 1.5, 15)
rscales = [1.]

# other params
verbose     = 4
aa          = L
a           = np.eye(3) * aa

xc = 'pbe'
init_guess = '1e'

def get_systems(zeta, ke_cutoff):
    basis = "cc-pv" + zeta
    # periodic cell for PAW calculation
    cell = pgto.M(
        atom        = atom,
        verbose     = verbose,
        basis       = basis,
        ke_cutoff   = ke_cutoff,
        a           = a,
        unit = 'B',
        max_memory = 15000
    )

    # corresponding mol for pyscf exact calculation
    mol = gto.M(
        atom        = atom,
        verbose     = verbose,
        basis       = basis,
        unit='B'
    )
    return mol, cell

def single_point(mol, cell):
    # paw calculation
    mf_mol_paw = scf.RKS(mol).density_fit()
    mf_mol_paw.xc = xc
    mf_mol_paw.init_guess = init_guess
    mydf = PAW.from_mf(mf_mol_paw, cell,
                        alpha0=None, augRadius=None, with_multigrid=2).build()
    mf_mol_paw.with_df = mydf
    print(mydf.alpha0)


def exact_paw_diff(mol, cell, ke_cutoff):
    results = []
    for rscale in rscales:
        r       = r0*rscale    # rescale C=C bond length
        atom_new    = f'N {r1} {r1} {r1}; N {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'

        cell.atom = atom_new
        cell.build()
        mol.atom = atom_new
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

def exact_gdf_diff(mol, cell, ke_cutoff):
    results = []
    for rscale in rscales:
        r       = r0*rscale    # rescale C=C bond length
        atom_new    = f'N {r1} {r1} {r1}; N {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'

        cell.atom = atom_new
        cell.build()
        mol.atom = atom_new
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

def basis_set_convergence(mol):
    basis = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz']
    data_basis = {}

    for b in basis:
        mol.basis = b
        rs, etot_pyscf, pyscf_converged = [], [], []
        for rscale in rscales:
            r = r0 * rscale
            atom_new = f'N {r1} {r1} {r1}; N {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'
            mol.atom = atom_new
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


def alpha0_scan(mol, cell, zeta, ke_cutoff, alpha0_list):
    path = f"data/alpha0_scan_{zeta}_{ke_cutoff}.json"
    # exact calculation
    mf_mol_exact = scf.RKS(mol)
    mf_mol_exact.xc = xc
    mf_mol_exact.init_guess = init_guess
    mf_mol_exact.kernel()
    exact_e_tot = mf_mol_exact.e_tot

    results = []
    for alpha0 in alpha0_list:
        # paw calculation
        mf_mol_paw = scf.RKS(mol).density_fit()
        mf_mol_paw.xc = xc
        mf_mol_paw.init_guess = init_guess
        mydf = PAW.from_mf(mf_mol_paw, cell,
                            alpha0=alpha0, augRadius=None, with_multigrid=2).build()
        mf_mol_paw.with_df = mydf
        mf_mol_paw.kernel()

        radius = mydf.augRadius
        if hasattr(radius, 'tolist'):
            radius = radius.tolist()

        results.append({
            "alpha0": alpha0,
            "radius": radius,
            "etot_pyscf": exact_e_tot,
            "etot_paw": mf_mol_paw.e_tot,
            "ediff": exact_e_tot - mf_mol_paw.e_tot,
            "conv_pyscf": bool(mf_mol_exact.converged),
            "conv_paw": bool(mf_mol_paw.converged)
        })

        # Save on the fly
        print(f"Alpha0={alpha0} done. Saving data to {path}...")
        with open(path, "w") as f:
            json.dump(results, f, indent=4)

def alpha0_j_scan(mol, cell, zeta, ke_cutoff, alpha0_list):
    path = f"data/alpha0_j_scan_{zeta}_{ke_cutoff}.json"
    npz_path = path.replace('.json', '.npz')
    # exact calculation
    mf_mol_exact = scf.RKS(mol)
    mf_mol_exact.xc = xc
    mf_mol_exact.init_guess = init_guess
    mf_mol_exact.kernel()
    
    dm = mf_mol_exact.make_rdm1()
    vj_exact = mf_mol_exact.get_j(dm=dm)

    results = []
    arrays_to_save = {'dm': dm}

    for alpha0 in alpha0_list:
        # paw calculation setup
        mf_mol_paw = scf.RKS(mol).density_fit()
        mf_mol_paw.xc = xc
        mf_mol_paw.init_guess = init_guess
        mydf = PAW.from_mf(mf_mol_paw, cell,
                            alpha0={'He': 10}, augRadius=None, with_multigrid=2).build()
        mf_mol_paw.with_df = mydf
        
        vj_paw = mf_mol_paw.get_j(dm=dm)
        
        j_diff = vj_exact - vj_paw
        max_diff = float(np.max(np.abs(j_diff)))
        norm_diff = float(np.linalg.norm(j_diff))

        radius = mydf.augRadius
        if hasattr(radius, 'tolist'):
            radius = radius.tolist()

        results.append({
            "alpha0": alpha0,
            "radius": radius,
            "j_max_diff": max_diff,
            "j_norm_diff": norm_diff
        })

        arrays_to_save[f'j_diff_{alpha0}'] = j_diff

        # Save on the fly
        print(f"Alpha0={alpha0} done. Saving J scan data to {path} and arrays to {npz_path}...")
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        np.savez(npz_path, **arrays_to_save)

def main():
    parser = argparse.ArgumentParser(description="Run N2 bond length PAW vs PySCF exact calculations.")
    parser.add_argument('--zetas', nargs='+', default=['qz'], help="List of zeta values (e.g. dz tz qz)")
    parser.add_argument('--ke_cutoffs', nargs='+', type=int, default=[2000], help="List of kinetic energy cutoffs")
    parser.add_argument('--alpha0_list', nargs='+', type=float, default=[5, 10, 15, 20, 25, 30, 50], help="List of alpha0 values to scan")
    parser.add_argument('--scan_type', choices=['energy', 'j_matrix', 'both'], default='j_matrix', help="Which scan to perform")
    
    args = parser.parse_args()

    for zeta in args.zetas:
        for ke_cutoff in args.ke_cutoffs:
            print(f"\\n========================================")
            print(f"Setting up system for zeta='{zeta}', ke_cutoff={ke_cutoff}")
            print(f"========================================")
            mol, cell = get_systems(zeta, ke_cutoff)

            if args.scan_type in ['energy', 'both']:
                print("Running energy scan...")
                alpha0_scan(mol, cell, zeta, ke_cutoff, args.alpha0_list)
            
            if args.scan_type in ['j_matrix', 'both']:
                print("Running J-matrix scan...")
                alpha0_j_scan(mol, cell, zeta, ke_cutoff, args.alpha0_list)

if __name__ == '__main__':
    main()