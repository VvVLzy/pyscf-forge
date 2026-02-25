import numpy
import pyscf
import csv
import time

from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import PAW, NewPAW

L = 3  # box size

# specify He2
r1      = 0.    # location of first carbon
r0      = 0.7    # N-N bond length
atom    = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
# atom    = f'He 0 0 0'

zeta = 'dz'
basis = "ccpv"+zeta

## ccpvdz
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

init_guess = '1e'
xc = ''

def debugNumpyJ(alpha0=20, Rb=1.5):
    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # paw
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    mf_per_rks_paw.with_df = mydf

    # compare different ways of getting Js using the converged dm
    mf_per_rks.kernel()
    dm = mf_per_rks.make_rdm1()

    J1 = mf_per_rks_paw.with_df.getJ1(dm=dm)
    J1Numpy = mf_per_rks_paw.with_df.getJ1Numpy(dm=dm)
    print(f'Difference: {numpy.max(numpy.abs(J1-J1Numpy))}')
    print(f'Max of jax: {numpy.max(numpy.abs(J1))}')
    print(f'Max of npy: {numpy.max(numpy.abs(J1Numpy))}')

    J2 = mf_per_rks_paw.with_df.getJ2(dm=dm)
    J2Numpy = mf_per_rks_paw.with_df.getJ2Numpy(dm=dm)
    print(f'Difference: {numpy.max(numpy.abs(J2-J2Numpy))}')
    print(f'Max of jax: {numpy.max(numpy.abs(J2))}')
    print(f'Max of npy: {numpy.max(numpy.abs(J2Numpy))}')

    J3 = mf_per_rks_paw.with_df.getJ3(dm=dm)
    J3Numpy = mf_per_rks_paw.with_df.getJ3Numpy(dm=dm)
    print(f'Difference: {numpy.max(numpy.abs(J3-J3Numpy))}')
    print(f'Max of jax: {numpy.max(numpy.abs(J3))}')
    print(f'Max of npy: {numpy.max(numpy.abs(J3Numpy))}')

    import pdb; pdb.set_trace()

def debugRb(Rb1, Rb2, alpha0=20):
    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # with Rb1
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb1).build()
    mf_per_rks_paw.with_df = mydf

    # with Rb2
    mf_per_rks_paw_new = pscf.RKS(cell)
    mf_per_rks_paw_new.init_guess = init_guess
    mf_per_rks_paw_new.xc = xc
    mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb2).build()
    mf_per_rks_paw_new.with_df = mydf

    PAWdata1 = mf_per_rks_paw.with_df.PAWdata
    PAWdata2 = mf_per_rks_paw_new.with_df.PAWdata

    # compare different ways of getting Js using the converged dm
    mf_per_rks.kernel()
    dm = mf_per_rks.make_rdm1()

    J_fftdf = mf_per_rks.get_j(dm=dm)
    J_paw = mf_per_rks_paw.get_j(dm=dm)
    J_paw_new = mf_per_rks_paw_new.get_j(dm=dm)
    print(numpy.max(numpy.abs(J_paw-J_paw_new)))

    J1_paw = mf_per_rks_paw.with_df.getJ1Numpy(dm=dm)
    J2_paw = mf_per_rks_paw.with_df.getJ2Numpy(dm=dm)
    J3_paw = mf_per_rks_paw.with_df.getJ3Numpy(dm=dm)

    J1_paw_new = mf_per_rks_paw_new.with_df.getJ1Numpy(dm=dm)
    J2_paw_new = mf_per_rks_paw_new.with_df.getJ2Numpy(dm=dm)
    J3_paw_new = mf_per_rks_paw_new.with_df.getJ3Numpy(dm=dm)

    print(numpy.max(numpy.abs(J1_paw-J1_paw_new)))
    print(numpy.max(numpy.abs(J2_paw-J2_paw_new)))
    print(numpy.max(numpy.abs(J3_paw-J3_paw_new)))

    import pdb; pdb.set_trace()


def debug_J2(alpha0=20, Rb=1.5):
    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # old coulomb
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

    J1_paw_new = mf_per_rks_paw_new.with_df.getJ1(dm=dm)
    J2_paw_new = mf_per_rks_paw_new.with_df.getJ2(dm=dm)
    J3_paw_new = mf_per_rks_paw_new.with_df.getJ3(dm=dm)

    print(J2_paw_new - J_fftdf)
    print(numpy.max(numpy.abs(J2_paw_new - J_fftdf))) # TODO: RSDF get_eri eventually needs to be fixed
    print(numpy.unravel_index(numpy.argmax(numpy.abs(J2_paw_new - J_fftdf)), J_fftdf.shape))

    import pdb; pdb.set_trace()

def check_J_scan(alpha0=20, Rb=1.5):
    rscales = numpy.linspace(0.7, 1.5, 10)
    results = numpy.zeros((rscales.shape[0], 3))
    r0 = 1.
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
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        mydf = PAW.from_mf(mf_per_rks_paw, alpha0=alpha0, augRadius=Rb).build()
        # mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, PAWorbitalCutOff=1e-8).build()
        # mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, PAWorbitalCutOff=1e-8).build()
        mf_per_rks_paw.with_df = mydf

        # new coulomb
        mf_per_rks_paw_new = pscf.RKS(cell)
        mf_per_rks_paw_new.init_guess = init_guess
        mf_per_rks_paw_new.xc = xc
        mydf = NewPAW.from_mf(mf_per_rks_paw_new, alpha0=alpha0, augRadius=Rb).build()
        # mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, PAWorbitalCutOff=1e-8).build()
        # mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, PAWorbitalCutOff=1e-8).build()
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

        del mf_per_rks
        del mf_per_rks_paw
        del mf_per_rks_paw_new
        del mydf

    print(results)

def check_J(alpha0=10, Rb=1.5):
    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # old coulomb
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    # mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, PAWorbitalCutOff=1e-8).build()
    mf_per_rks_paw.with_df = mydf

    # new coulomb
    mf_per_rks_paw_new = pscf.RKS(cell)
    mf_per_rks_paw_new.init_guess = init_guess
    mf_per_rks_paw_new.xc = xc
    mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    # mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, PAWorbitalCutOff=1e-8).build()
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

#PWcut: 200
# with Numpy
# [[7.00000000e-01 1.01288658e-04 4.18760258e-05]
#  [7.88888889e-01 1.05158958e-04 3.90737203e-05]
#  [8.77777778e-01 1.04520931e-04 3.53078435e-05]
#  [9.66666667e-01 1.00762154e-04 3.11806371e-05]
#  [1.05555556e+00 9.62367356e-05 2.63836144e-05]
#  [1.14444444e+00 9.19873748e-05 2.13465306e-05]
#  [1.23333333e+00 8.78218812e-05 1.64856668e-05]
#  [1.32222222e+00 8.43851283e-05 1.17293834e-05]
#  [1.41111111e+00 8.26710738e-05 9.05100209e-06]
#  [1.50000000e+00 8.21945332e-05 7.97939740e-06]]

# with Jax
# [[7.00000000e-01 1.01288655e-04 4.18760093e-05]
#  [7.88888889e-01 1.05158958e-04 3.90737202e-05]
#  [8.77777778e-01 1.04520931e-04 3.53078435e-05]
#  [9.66666667e-01 1.00762154e-04 3.11806371e-05]
#  [1.05555556e+00 9.62367356e-05 2.63836144e-05]
#  [1.14444444e+00 9.19873748e-05 2.13465306e-05]
#  [1.23333333e+00 8.78218812e-05 1.64856668e-05]
#  [1.32222222e+00 8.43851283e-05 1.17293834e-05]
#  [1.41111111e+00 8.26710738e-05 9.05100209e-06]
#  [1.50000000e+00 3.86514369e-04 1.02021616e-03]]


def main():
    debugNumpyJ(alpha0=20, Rb=1.5)
    debugNumpyJ(alpha0=20, Rb=3.0)
    # debugRb(1.5, 3.0, alpha0=20)
    # debug_J2(alpha0=20, Rb=1.5)
    # debug_J(alpha0=20, Rb=3.0)
    # check_J()
    # check_J_scan(alpha0=10, Rb=1.5)
    # check_nuc()

if __name__ == '__main__':
    main()