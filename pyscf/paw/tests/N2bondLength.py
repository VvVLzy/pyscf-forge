###
# Compare PAW DFT(PBE) energy with the exact pyscf energy as the bond
# length between two C atoms are varied

import numpy as np
import csv

from pyscf.pbc import gto as pgto
from pyscf import gto, scf
from pyscf.paw import PAW

# specify C2
r1      = 0.    # location of first carbon
r0       = 1.10    # N-N bond length
atom    = f'N {r1} {r1} {r1}; N {r1+r0/3.**0.5} {r1+r0/3.**0.5} {r1+r0/3.**0.5}'
rscales = np.linspace(0.7, 1.5, 15)

# other params
verbose     = 4
basis       = "cc-pvtz"
ke_cutoff   = 50.
a           = np.eye(3) * 20.

# periodic cell for PAW calculation
cell = pgto.M(
    atom        = atom,
    verbose     = verbose,
    basis       = basis,
    ke_cutoff   = ke_cutoff,
    a           = a
)

# corresponding mol for pyscf exact calculation
mol = gto.M(
    atom        = atom,
    verbose     = verbose,
    basis       = basis,
)

# mf params
xc = 'PBE'
init_guess = '1e'

def exact_paw_diff():
    rs, etot_pyscf, etot_paw, ediff_per_atm = [], [], [], []
    pyscf_converged, paw_converged = [], []
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
                        PWAccuracy=1e-4).build()
        mf_mol_paw.with_df = mydf
        mf_mol_paw.kernel()

        # save data
        rs.append(r)
        etot_pyscf.append(mf_mol_exact.e_tot)
        etot_paw.append(mf_mol_paw.e_tot)
        ediff_per_atm.append(np.abs(etot_pyscf[-1]-etot_paw[-1])/cell.natm)
        pyscf_converged.append(mf_mol_exact.converged)
        paw_converged.append(mf_mol_paw.converged)

    # write to csv
    path = f'data/N2bondLength{ke_cutoff: .2f}.csv'
    print(f'Calculation done. Saving data to {path}...')
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['r', "etot_pyscf", "etot_paw", "ediff_per_atm", "conv_pyscf", "conv_paw"])
        for a, b, c, d, e, f in zip(rs, etot_pyscf, etot_paw, ediff_per_atm, pyscf_converged, paw_converged):
            writer.writerow([a, b, c, d, e, f])

def basis_set_convergence():
    basis = ['cc-pvdz', 'cc-pvtz', 'cc-pvqz']

    data_basis = dict()
    for b in basis:
        mol.basis = b
        rs, etot_pyscf, pyscf_converged = [], [], []
        for rscale in rscales:
            r       = r0*rscale    # rescale C=C bond length
            atom    = f'N {r1} {r1} {r1}; N {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'

            mol.atom = atom
            mol.build()

            # exact calculation
            mf_mol_exact = scf.RKS(mol)
            mf_mol_exact.xc = xc
            mf_mol_exact.init_guess = init_guess
            mf_mol_exact.kernel()

            # save data
            rs.append(r)
            etot_pyscf.append(mf_mol_exact.e_tot)
            pyscf_converged.append(mf_mol_exact.converged)
        
        data_basis[b] = [rs, etot_pyscf, pyscf_converged]

    # write to csv
    path = 'data/N2bondLength_convergence.csv'
    print(f'Calculation done. Saving data to {path}...')
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(['etot_pyscf'])
        for i, bas in enumerate(basis):
            if i == 0:
                row = ['r']
                row.extend(data_basis[bas][0])
                writer.writerow(row)

            row = [bas]
            row.extend(data_basis[bas][1])
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(['conv_pyscf'])
        for i, bas in enumerate(basis):
            if i == 0:
                row = ['r']
                row.extend(data_basis[bas][0])
                writer.writerow(row)

            row = [bas]
            row.extend(data_basis[bas][2])
            writer.writerow(row)


def main():
    basis_set_convergence()

if __name__ == '__main__':
    main()