from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

import numpy

from pyscf.paw import NewPAW


L = 3  # box size

# specify He2
r1      = 0.    # location of first carbon
r0      = 0.7    # N-N bond length
atom    = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
# atom    = f'He 0 0 0'

zeta = 'dz'
basis = "ccpv"+zeta

verbose = 3
a = numpy.eye(3) * L
ke_cutoff = 100
precision = 1e-8

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision
)

# cell_fftdf = pgto.M(
#     atom        = atom,
#     basis       = basis,
#     verbose     = 3,
#     a           = a,
#     ke_cutoff   = 2000,
#     precision   = precision
# )

mf_per_rks_paw = pscf.RKS(cell)
mf_per_rks_paw.init_guess = '1e'
mf_per_rks_paw.xc = ''
mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=10, augRadius=1.5).build()
mf_per_rks_paw.with_df = mydf

def testmakeWignerSeitz():
    from PAWutilsJax import makeWignerSeitz as old
    from PAWutilsNumpy import makeWignerSeitz as new
    result = old(mydf.Rgrid, mydf.cell, mydf.Periodic)
    result1 = new(mydf.Rgrid, mydf.cell, mydf.Periodic)

    if isinstance(result, tuple):
        for i in range(len(result)):
            print(numpy.max(numpy.abs(result[i]-result1[i])))
            assert(numpy.max(numpy.abs(result[i]-result1[i])))
    else:
        print(numpy.max(numpy.abs(result-result1)))
        assert(numpy.max(numpy.abs(result-result1)))

def testmakeWignerSeitz():
    from PAWutilsJax import compensatingCharge as old
    from PAWutilsNumpy import compensatingCharge as new
    result = old(mydf.Rgrid, mydf.cell, mydf.Periodic)
    result1 = new(mydf.Rgrid, mydf.cell, mydf.Periodic)

    if isinstance(result, tuple):
        for i in range(len(result)):
            print(numpy.max(numpy.abs(result[i]-result1[i])))
            assert(numpy.max(numpy.abs(result[i]-result1[i])))
    else:
        print(numpy.max(numpy.abs(result-result1)))
        assert(numpy.max(numpy.abs(result-result1)))
            

def main():
    testmakeWignerSeitz()

if __name__ == '__main__':
    main()