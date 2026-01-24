import numpy
import pyscf
import csv
import time


from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import PAW, NewPAW

L = 3.  # box size

# specify He2
r1      = 0.    # location of first carbon
r0      = 1.    # N-N bond length
atom    = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'

zeta = 'dz'
basis = "ccpv"+zeta
verbose = 3
a = numpy.eye(3) * L
ke_cutoff = 800
precision = 1e-8

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision
)

cell_fftdf = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = 3,
    a           = a,
    ke_cutoff   = 2000,
    precision   = precision
)

# TODO: the default minao guess usually give mo that is not (cell.nao, cell.nao)
# and thus will give an error
init_guess = '1e'
xc = ''

mf_per_rks = pscf.RKS(cell_fftdf)
mf_per_rks.init_guess = init_guess
mf_per_rks.xc = xc

alpha0=10
Rb=1.5
mf_per_rks_paw = pscf.RKS(cell)
mf_per_rks_paw.init_guess = init_guess
mf_per_rks_paw.xc = xc
mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
mf_per_rks_paw.with_df = mydf

mf_per_rks_paw_new = pscf.RKS(cell)
mf_per_rks_paw_new.init_guess = init_guess
mf_per_rks_paw_new.xc = xc
mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
mf_per_rks_paw_new.with_df = mydf

# compare different ways of getting Js using the converged dm
mf_per_rks.kernel()
dm = mf_per_rks.make_rdm1()

J_fftdf = mf_per_rks.get_j(dm=dm)
J_paw = mf_per_rks_paw.get_j(dm=dm)
J_paw_new = mf_per_rks_paw_new.get_j(dm=dm)

print(numpy.max(numpy.abs(J_fftdf-J_paw)))
print(numpy.max(numpy.abs(J_fftdf-J_paw_new)))

import pdb; pdb.set_trace()