import numpy
import pyscf

from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import PAW


L = 3.  # box size
# x = L/2 # atom in the center
x = 0

atom =  f"""
He      {x} {x} {x}
"""
basis = "ccpvtz"
verbose = 3
a = numpy.eye(3) * L
ke_cutoff = 100

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = 1e-5
)

# TODO: the default minao guess usually give mo that is not (cell.nao, cell.nao)
# and thus will give an error
init_guess = '1e'
xc = ''
auxbasis = pyscf.df.aug_etb(cell, beta=1.3)

# Reference GDF calculation
mf_per_rks = pscf.RKS(cell)
mf_per_rks.init_guess = init_guess
mf_per_rks.xc = xc
mf_per_rks = mf_per_rks.density_fit(auxbasis=auxbasis)
mf_per_rks.kernel()
e_ref = mf_per_rks.e_tot/cell.natm

# PAW calculation
results = []
for ke_cut in numpy.arange(25, 201, 25):
    print(f'ke cutoff: {ke_cut}')
    cell.ke_cutoff = ke_cut
    cell.build()

    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8).build()
    mf_per_rks_paw.with_df = mydf

    mf_per_rks_paw.kernel()
    e_paw = mf_per_rks_paw.e_tot/cell.natm

    results.append({
            "ke_cut (Ha)": ke_cut,
            "etot_paw (Ha)": e_paw
        })
    
# write to csv
import csv
path = "/Users/vvv_lzy/GitHub/pyscf-forge/pyscf/paw/periodic-tests/data/He_cutoff.csv"
print(f"Calculation done. Saving data to {path}...")
with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "ke_cut (Ha)", "etot_paw (Ha)"
    ])
    writer.writeheader()
    writer.writerows(results)
