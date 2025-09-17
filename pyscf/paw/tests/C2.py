import numpy as np

from pyscf.pbc import gto as pgto
from pyscf import gto, scf
from pyscf.paw import PAW

# specify C2
r1      = 0.    # location of first carbon
r0       = 1.33    # C=C bond length
atom    = f'C {r1} {r1} {r1}; C {r1+r0/3.**0.5} {r1+r0/3.**0.5} {r1+r0/3.**0.5}'

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

init_guess = '1e'

# paw calculation
mf_mol_paw = scf.RHF(mol).density_fit()
mf_mol_paw.init_guess = init_guess
mydf = PAW.from_mf(mf_mol_paw, cell,
                    PWAccuracy=1e-4).build()
mf_mol_paw.with_df = mydf
mf_mol_paw.kernel()