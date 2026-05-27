import numpy
import pyscf
import csv
import time

from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import PAW, NewPAW
from pyscf.paw.vxc import PAWNumInt

L = 3  # box size
a = numpy.eye(3) * L

# specify He2
r1      = 0.    # location of first carbon
r0      = 1.0    # N-N bond length
atom    = f'C {r1} {r1} {r1}; C {r1+r0} {r1} {r1}'
# atom    = f'He 0 0 0'

a = numpy.array([[ 2.18050236,  0.,          1.25891346],
                 [ 0.72683412,  2.05579703,  1.25891346],
                 [-0.,          0.,          2.51782692]])

atom = '''
C 2.543919 1.798822 4.406197
C 0.363417 0.256975 0.629457
'''

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
xc = 'lda'

def check_J(alpha0=10, Rb=1.5):
    # fftdf
    # mf_per_rks = pscf.RKS(cell_fftdf)
    # mf_per_rks.init_guess = init_guess
    # mf_per_rks.xc = xc

    # gdf
    from pyscf.pbc import df
    mf_per_rks_gdf = pscf.RKS(cell)
    mf_per_rks_gdf.init_guess = init_guess
    mf_per_rks_gdf.xc = xc
    mydf = df.GDF(cell)
    # mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=alpha0, PAWorbitalCutOff=1e-8).build()
    mf_per_rks_gdf.with_df = mydf

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
    mf_per_rks_paw_new.verbose = 4
    mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    pawnumint = PAWNumInt(mydf)
    mf_per_rks_paw_new._numint = pawnumint
    # mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, PAWorbitalCutOff=1e-8).build()
    mf_per_rks_paw_new.with_df = mydf

    # compare different ways of getting Js using the converged dm
    mf_per_rks_gdf.kernel()
    dm = mf_per_rks_gdf.make_rdm1()

    J_fftdf = mf_per_rks.get_j(dm=dm)
    J_paw = mf_per_rks_paw.get_j(dm=dm)
    J_gdf = mf_per_rks_gdf.get_j(dm=dm)
    J_paw_new = mf_per_rks_paw_new.get_j(dm=dm)

    print(numpy.max(numpy.abs(J_paw_new-J_gdf)))
    # print(numpy.max(numpy.abs(J_fftdf-J_paw_new)))

    mf_per_rks_paw_new.kernel()

    import pdb; pdb.set_trace()

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
    rscales = numpy.linspace(0.7, 1.5, 9)
    results = []
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
        mydf = PAW.from_mf(mf_per_rks_paw, alpha0=alpha0, augRadius=Rb, use_new_comp_charge=False).build()
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
        results.append({
            'r': r,
            'Before': numpy.max(numpy.abs(J_fftdf-J_paw)),
            'After': numpy.max(numpy.abs(J_fftdf-J_paw_new))
        })
        # results[i, 0] = r
        # results[i, 1] = numpy.max(numpy.abs(J_fftdf-J_paw))
        # results[i, 2] = numpy.max(numpy.abs(J_fftdf-J_paw_new))

        del mf_per_rks
        del mf_per_rks_paw
        del mf_per_rks_paw_new
        del mydf

    # print(results)
    # write to csv
    prefix = './'
    path = prefix + f"data/J_scan_{ke_cutoff:.2f}_a_{alpha0}_r_{Rb}_{zeta}_{precision:.0e}_{xc}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "r", "Before", "After"
        ])
        writer.writeheader()
        writer.writerows(results)

def check_J_random_position(alpha0=20, Rb=1.5):
    results = []
    r0 = 1.
    r       = r0*1.27    # rescale C=C bond length
    atom    = f'He {r1} {r1} {r1}; He {r1+r} {r1} {r1}'

    cell.atom = atom
    cell.build()

    cell_fftdf.atom = atom
    cell_fftdf.build()

    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # compare different ways of getting Js using the converged dm
    mf_per_rks.kernel()
    dm = mf_per_rks.make_rdm1()
    J_fftdf = mf_per_rks.get_j(dm=dm)

    xyzs = numpy.random.random((10, 3)) * L
    # xyzs = numpy.array([
    #     [0.,0.,0.],
    #     [1.5,1.5,1.5]
    # ])
    for xyz in xyzs:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]

        atom = f'He {x} {y} {z}; He {x+r} {y} {z}'
        cell.atom = atom
        cell.build()

        # print(cell._atom)

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

        # import pdb; pdb.set_trace()
        

        J_paw = mf_per_rks_paw.get_j(dm=dm)
        J_paw_new = mf_per_rks_paw_new.get_j(dm=dm)

        print(numpy.max(numpy.abs(J_fftdf-J_paw)))
        print(numpy.max(numpy.abs(J_fftdf-J_paw_new)))
        results.append({
            'r': numpy.sqrt(numpy.sum(xyz**2)),
            'Before': numpy.max(numpy.abs(J_fftdf-J_paw)),
            'After': numpy.max(numpy.abs(J_fftdf-J_paw_new))
        })
        # results[i, 0] = r
        # results[i, 1] = numpy.max(numpy.abs(J_fftdf-J_paw))
        # results[i, 2] = numpy.max(numpy.abs(J_fftdf-J_paw_new))

        del mf_per_rks_paw
        del mf_per_rks_paw_new
        del mydf

    # print(results)
    # write to csv
    prefix = './'
    path = prefix + f"data/J_random_pos_{ke_cutoff:.2f}_a_{alpha0}_r_{Rb}_{zeta}_{precision:.0e}_{xc}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "r", "Before", "After"
        ])
        writer.writeheader()
        writer.writerows(results)

def main():
    # debugNumpyJ(alpha0=20, Rb=1.5)
    # debugNumpyJ(alpha0=20, Rb=3.0)
    # debugRb(1.5, 3.0, alpha0=20)
    # debug_J2(alpha0=20, Rb=1.5)
    # debug_J(alpha0=20, Rb=3.0)
    # check_J(alpha0=10, Rb=1.5)
    check_J_scan(alpha0=10, Rb=1.5)
    # check_J_random_position(alpha0=10, Rb=1.5)
    # check_nuc()

if __name__ == '__main__':
    main()