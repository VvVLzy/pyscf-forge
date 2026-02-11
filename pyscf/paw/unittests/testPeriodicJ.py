import numpy
import pyscf
import csv
import time

from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import PAW, NewPAW

L = 3  # box size

# [[7.00000000e-01 8.96619360e-05 1.84464241e-04]
#  [7.57142857e-01 9.53693946e-05 1.80626570e-04]
#  [8.14285714e-01 9.76563076e-05 1.75774843e-04]
#  [8.71428571e-01 9.90000841e-05 1.72130454e-04]
#  [9.28571429e-01 9.81983584e-05 1.67659724e-04]
#  [9.85714286e-01 9.62802266e-05 1.63038058e-04]
#  [1.04285714e+00 9.39782091e-05 1.58654935e-04]
#  [1.10000000e+00 9.18535170e-05 1.54877196e-04]
#  [1.15714286e+00 9.01609802e-05 1.51866558e-04]
#  [1.21428571e+00 8.80977227e-05 1.48664411e-04]
#  [1.27142857e+00 8.58370086e-05 1.45351668e-04]
#  [1.32857143e+00 8.41949384e-05 1.42936479e-04]
#  [1.38571429e+00 8.34147339e-05 1.41731463e-04]
#  [1.44285714e+00 8.25938594e-05 1.40553877e-04]
#  [1.50000000e+00 8.26227517e-05 1.40550343e-04]]

# specify He2
r1      = 0.    # location of first carbon
r0      = 1.5    # N-N bond length
atom    = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
# atom    = f'He 0 0 0'

zeta = 'dz'
basis = "ccpv"+zeta
# basis = {'He': gto.basis.parse('''
# He    S
#      38.3600000              0.0238090        
#       5.7700000              0.1548910        
#       1.2400000              0.4699870        
# He    S
#       0.2976000              1.0000000        
# He    P
#       1.2750000              1.0000000 
# ''')}
verbose = 3
a = numpy.eye(3) * L
ke_cutoff = 200
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

init_guess = '1e'
xc = ''

rscales = numpy.linspace(0.7, 1.5, 15)

def check_J_scan(rscales):
    results = numpy.zeros((rscales.shape[0], 3))
    for i, rscale in enumerate(rscales):
        r       = r0*rscale    # rescale C=C bond length
        atom    = atom    = f'He {r1} {r1} {r1}; He {r1+r} {r1} {r1}'

        cell.atom = atom
        cell.build()

        cell_fftdf.atom = atom
        cell_fftdf.build()

        # fftdf
        mf_per_rks = pscf.RKS(cell_fftdf)
        mf_per_rks.init_guess = init_guess
        mf_per_rks.xc = xc

        # old coulomb
        alpha0=10
        Rb=1.5
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
        mf_per_rks_paw.with_df = mydf

        # new coulomb
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
        results[i, 0] = r
        results[i, 1] = numpy.max(numpy.abs(J_fftdf-J_paw))
        results[i, 2] = numpy.max(numpy.abs(J_fftdf-J_paw_new))
    print(results)

def check_J():
    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # old coulomb
    alpha0=10
    Rb=1.5
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    mf_per_rks_paw.with_df = mydf

    # new coulomb
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

    # import pdb; pdb.set_trace()
    hcore_fftdf = mf_per_rks.get_hcore()
    hcore_paw = mf_per_rks_paw.get_hcore()
    hcore_paw_new = mf_per_rks_paw_new.get_hcore()

    fock_fftdf = mf_per_rks.get_fock(dm=dm)
    fock_paw = mf_per_rks_paw.get_fock(dm=dm)
    fock_paw_new = mf_per_rks_paw_new.get_fock(dm=dm)

    print(numpy.max(numpy.abs(J_fftdf-J_paw)))
    print(numpy.max(numpy.abs(J_fftdf-J_paw_new)))

    import pdb; pdb.set_trace()

def check_nuc():
    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # rsdf nuc
    alpha0=30
    Rb=5
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    mf_per_rks_paw.with_df = mydf

    # new nuc
    mf_per_rks_paw_new = pscf.RKS(cell)
    mf_per_rks_paw_new.init_guess = init_guess
    mf_per_rks_paw_new.xc = xc
    mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    mf_per_rks_paw_new.with_df = mydf

    nuc_fftdf = mf_per_rks.with_df.get_nuc()
    nuc_rsdf = mf_per_rks_paw.with_df.get_nuc()
    nuc_paw = mf_per_rks_paw_new.with_df.get_nuc()

    print(numpy.max(numpy.abs(nuc_fftdf-nuc_rsdf)))
    print(numpy.max(numpy.abs(nuc_paw-nuc_rsdf)))
    # TODO: this is a problem nuc_rsdf should be close to nuc_paw even for 2 atoms (if reasonably separated)

    nuc1_paw = mf_per_rks_paw_new.with_df.get_nuc1()
    nuc2_paw = mf_per_rks_paw_new.with_df.get_nuc2()
    nuc3_paw = mf_per_rks_paw_new.with_df.get_nuc3()

    nuc11_paw = mf_per_rks_paw_new.with_df.get_nuc1(atomI=0)
    nuc21_paw = mf_per_rks_paw_new.with_df.get_nuc2(atomI=0)
    nuc31_paw = mf_per_rks_paw_new.with_df.get_nuc3(atomI=0)

    nuc12_paw = mf_per_rks_paw_new.with_df.get_nuc1(atomI=1)
    nuc22_paw = mf_per_rks_paw_new.with_df.get_nuc2(atomI=1)
    nuc32_paw = mf_per_rks_paw_new.with_df.get_nuc3(atomI=1)


    import pdb; pdb.set_trace()
def main():
    check_J()
    # check_J_scan(rscales)
    # check_nuc()

if __name__ == '__main__':
    main()