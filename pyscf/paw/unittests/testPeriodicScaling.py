import numpy
import argparse
import sys
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc import tools
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

# This script is adapted from molecule-tests/candidacy/periodic_timing/pyscf/Si1/Si1.py
# It performs a single point PAW calculation on a Silicon cell, with support for supercells.

def main():
    parser = argparse.ArgumentParser(description='Periodic scaling test for PAW')
    parser.add_argument('ncopy', type=int, nargs='*', default=[1, 1, 1],
                        help='Number of supercells in each dimension (1 or 3 integers). If 1 integer is provided, it is used for the X dimension [n, 1, 1].')
    
    args = parser.parse_args()
    
    if not args.ncopy:
        ncopy = [1, 1, 1]
    elif len(args.ncopy) == 1:
        ncopy = [args.ncopy[0], 1, 1]
    elif len(args.ncopy) == 3:
        ncopy = args.ncopy
    else:
        print("Error: ncopy must be 1 or 3 integers")
        sys.exit(1)

    L = 5.443702372939453  # box size

    atom = '''
    Si 4.082777 4.082777 1.360926
    Si -0.000000 2.721851 2.721851
    Si 4.082777 1.360926 4.082777
    Si 0.000000 0.000000 0.000000
    Si 1.360926 4.082777 4.082777
    Si 2.721851 2.721851 0.000000
    Si 1.360926 1.360926 1.360926
    Si 2.721851 0.000000 2.721851
    '''

    basis = "ccpvdz"
    verbose = 4
    a = numpy.eye(3) * L
    ke_cutoff = 200
    precision = 1e-5

    cell = pgto.M(
        atom        = atom,
        basis       = basis,
        verbose     = verbose,
        a           = a,
        ke_cutoff   = ke_cutoff,
        precision   = precision
    )

    if any(n > 1 for n in ncopy):
        cell = tools.super_cell(cell, ncopy)

    init_guess = '1e'
    xc = 'pbe'

    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 1
    
    # alpha0=10, Rb=1.5, with_multigrid=2 as in the original file
    mydf = PAW.from_mf(mf_per_rks_paw, alpha0=10, augRadius=1.5, with_multigrid=2).build()
    mf_per_rks_paw.with_df = mydf
    pawnumint = PAWNumInt(mydf, mf_per_rks_paw)
    mf_per_rks_paw._numint = pawnumint
    
    print(f"Running PAW calculation with ncopy={ncopy}")
    mf_per_rks_paw.kernel()

if __name__ == '__main__':
    main()
