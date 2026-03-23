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
ke_cutoff = 400
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

def check_nuc_scan(alpha0=10, Rb=1.5):
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

        # new nuc
        mf_per_rks_paw_new = pscf.RKS(cell)
        mf_per_rks_paw_new.init_guess = init_guess
        mf_per_rks_paw_new.xc = xc
        mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
        mf_per_rks_paw_new.with_df = mydf

        nuc_fftdf = mf_per_rks.with_df.get_nuc()
        nuc_rsdf = mf_per_rks_paw_new.with_df.get_nuc_rsdf()
        nuc_paw = mf_per_rks_paw_new.with_df.get_nuc()

        print(numpy.max(numpy.abs(nuc_fftdf-nuc_rsdf)))
        print(numpy.max(numpy.abs(nuc_paw-nuc_rsdf)))

        results.append({
            'r': r,
            'fft-rs': numpy.max(numpy.abs(nuc_fftdf-nuc_rsdf)),
            'paw-rs': numpy.max(numpy.abs(nuc_paw-nuc_rsdf))
        })
        # results[i, 0] = r
        # results[i, 1] = numpy.max(numpy.abs(J_fftdf-J_paw))
        # results[i, 2] = numpy.max(numpy.abs(J_fftdf-J_paw_new))

        del mf_per_rks
        del mf_per_rks_paw_new
        del mydf

    prefix = './'
    path = prefix + f"data/Nuc_scan_{ke_cutoff:.2f}_a_{alpha0}_r_{Rb}_{zeta}_{precision:.0e}_{xc}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "r", "fft-rs", "paw-rs"
        ])
        writer.writeheader()
        writer.writerows(results)

def check_nuc(alpha0=10, Rb=1.5):
    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    # new nuc
    mf_per_rks_paw_new = pscf.RKS(cell)
    mf_per_rks_paw_new.init_guess = init_guess
    mf_per_rks_paw_new.xc = xc
    mydf = NewPAW.from_mf(mf_per_rks_paw_new, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb).build()
    mf_per_rks_paw_new.with_df = mydf

    nuc_fftdf = mf_per_rks.with_df.get_nuc()
    nuc_rsdf = mf_per_rks_paw_new.with_df.get_nuc_rsdf()
    nuc_paw = mf_per_rks_paw_new.with_df.get_nuc()

    print(numpy.max(numpy.abs(nuc_fftdf-nuc_rsdf)))
    print(numpy.max(numpy.abs(nuc_paw-nuc_rsdf)))
    import pdb; pdb.set_trace()
    # # TODO: this is a problem nuc_rsdf should be close to nuc_paw even for 2 atoms (if reasonably separated)

    nuc1_paw = mf_per_rks_paw_new.with_df.get_nuc_paw1()
    nuc2_paw = mf_per_rks_paw_new.with_df.get_nuc_paw2()
    nuc3_paw = mf_per_rks_paw_new.with_df.get_nuc_paw3()

    # # nuc11_paw = mf_per_rks_paw_new.with_df.get_nuc1(atomI=0)
    # # nuc21_paw = mf_per_rks_paw_new.with_df.get_nuc2(atomI=0)
    # # nuc31_paw = mf_per_rks_paw_new.with_df.get_nuc3(atomI=0)

    # # nuc12_paw = mf_per_rks_paw_new.with_df.get_nuc1(atomI=1)
    # # nuc22_paw = mf_per_rks_paw_new.with_df.get_nuc2(atomI=1)
    # # nuc32_paw = mf_per_rks_paw_new.with_df.get_nuc3(atomI=1)
    return nuc1_paw, nuc2_paw, nuc3_paw, mf_per_rks_paw_new.with_df

def main():
    # nuc1, nuc2, nuc3, df = check_nuc(alpha0=10, Rb=1.5)
    # nuc11, nuc21, nuc31, df1 = check_nuc(alpha0=10, Rb=2)
    # print(numpy.max(numpy.abs(nuc1-nuc11)))
    # print(numpy.max(numpy.abs(nuc2-nuc21)))
    # print(numpy.max(numpy.abs(nuc3-nuc31)))
    # import pdb; pdb.set_trace()

    # check_nuc_scan(alpha0=20, Rb=1.5)
    check_nuc(alpha0=10, Rb=1.5)

if __name__ == '__main__':
    main()