import numpy
import pyscf
import csv

from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import PAW


L = 3.  # box size

# specify He2
r1      = 0.    # location of first carbon
r0      = 1.    # N-N bond length
atom    = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
rscales = numpy.linspace(0.7, 1.5, 15)
print(rscales)

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

def exact_paw_diff():
    results = []
    for rscale in rscales:
        r       = r0*rscale    # rescale C=C bond length
        atom    = atom    = f'He {r1} {r1} {r1}; He {r1+r} {r1} {r1}'

        cell.atom = atom
        cell.build()


        # exact calculation
        mf_per_rks = pscf.RKS(cell)
        mf_per_rks.init_guess = init_guess
        mf_per_rks.xc = xc
        mf_per_rks = mf_per_rks.density_fit(auxbasis=auxbasis)
        mf_per_rks.kernel()

        # paw calculation
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8).build()
        mf_per_rks_paw.with_df = mydf
        mf_per_rks_paw.kernel()

        # append all results in one dict
        results.append({
            "r": r,
            "etot_pyscf": mf_per_rks.e_tot,
            "etot_paw": mf_per_rks_paw.e_tot,
            "conv_pyscf": mf_per_rks.converged,
            "conv_paw": mf_per_rks_paw.converged
        })

    # write to csv
    prefix = '/Users/vvv_lzy/GitHub/pyscf-forge/pyscf/paw/periodic-tests/'
    path = prefix + f"data/He2bondLength_{ke_cutoff:.2f}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "r", "etot_pyscf", "etot_paw", "conv_pyscf", "conv_paw"
        ])
        writer.writeheader()
        writer.writerows(results)

def main():
    # basis_set_convergence()
    import time
    start = time.time()
    exact_paw_diff()
    print(f'Time elapsed: {time.time()-start:.2f}')

if __name__ == '__main__':
    main()
