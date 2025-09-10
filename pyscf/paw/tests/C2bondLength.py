###
# Compare PAW DFT(PBE) energy with the exact pyscf energy as the bond
# length between two C atoms are varied

import numpy as np

from pyscf.pbc import gto as pgto
from pyscf import gto, scf
from pyscf.paw import PAW

# specify C2
r1      = 0.    # location of first carbon
r       = 1.33    # C=C bond length
atom    = f'C {r1} {r1} {r1}; C {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'

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

rs, etot_pyscf, etot_paw, ediff_per_atm = [], [], [], []
pyscf_converged, paw_converged = [], []
for rscale in np.linspace(0.7, 1.5, 15):
    r       = 1.33*rscale    # rescale C=C bond length
    atom    = f'C {r1} {r1} {r1}; C {r1+r/3.**0.5} {r1+r/3.**0.5} {r1+r/3.**0.5}'

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
import csv

path = 'data/C2bondLength.csv'
print(f'Calculation done. Saving data to {path}...')
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['r', "etot_pyscf", "etot_paw", "ediff_per_atm", "conv_pyscf", "conv_paw"])
    for a, b, c, d, e, f in zip(rs, etot_pyscf, etot_paw, ediff_per_atm, pyscf_converged, paw_converged):
        writer.writerow([a, b, c, d, e, f])
