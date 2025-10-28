from copy import deepcopy
import numpy
import pyscf
import csv

from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import PAW


L = 3.  # box size
# x = L/2 # atom in the center for now, need to fix the code later
x = 0

atom =  f"""
He      {x} {x} {x}
"""
zeta = 'tz'
basis = "ccpv"+zeta
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


def autoMode():
    '''
    determine alpha0 and aug sphere automatically
    '''
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
    path = f"/Users/vvv_lzy/GitHub/pyscf-forge/pyscf/paw/periodic-tests/data/He_cutoff_{zeta}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ke_cut (Ha)", "etot_paw (Ha)"
        ])
        writer.writeheader()
        writer.writerows(results)


def manualMode(alpha0=None, Rb=None):
    '''
    manually define alpha0 and aug sphere radius, for debugging
    can use auto mode to first determine reasonable guess.
    '''
    # PAW calculation
    results = []
    for ke_cut in numpy.arange(25, 201, 25):
        print(f'ke cutoff: {ke_cut}')
        cell.ke_cutoff = ke_cut
        cell.build()

        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
        mf_per_rks_paw.with_df = mydf

        mf_per_rks_paw.kernel()
        e_paw = mf_per_rks_paw.e_tot/cell.natm

        results.append({
                "ke_cut (Ha)": ke_cut,
                "etot_paw (Ha)": e_paw
            })
        
    # write to csv
    path = f"/Users/vvv_lzy/GitHub/pyscf-forge/pyscf/paw/periodic-tests/data/He_cutoff_a_{alpha0}_r_{Rb}_{zeta}_edge_2.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ke_cut (Ha)", "etot_paw (Ha)"
        ])
        writer.writeheader()
        writer.writerows(results)

def reference_gdf():
    # Reference GDF calculation
    mf_per_rks = pscf.RKS(cell)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc
    mf_per_rks = mf_per_rks.density_fit(auxbasis=auxbasis)
    mf_per_rks.kernel()
    e_ref = mf_per_rks.e_tot/cell.natm

def reference_fftdf():
    # Reference GDF calculation
    cell_fftdf = deepcopy(cell).build(
        ke_cutoff=1000,
        verbose=3)
    print(cell_fftdf.ke_cutoff)
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc
    mf_per_rks.kernel()
    e_ref = mf_per_rks.e_tot/cell.natm

def main():
    # reference_gdf()
    reference_fftdf()
    print(cell.ke_cutoff)
    # autoMode()
    manualMode(alpha0=10., Rb=1.5)

if __name__ == '__main__':
    main()