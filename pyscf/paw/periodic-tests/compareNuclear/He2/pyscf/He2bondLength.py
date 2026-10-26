import sys
import csv
import gc
import numpy

from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

# He2 bond stretch in a periodic box; settings chosen to match the CP2K GAPW
# inputs in ../cp2k/template.inp
L = 3.            # box size (A), CP2K ABC 3.0 3.0 3.0
r1 = 0.           # location of first He
r0 = 1.           # reference bond length (A)
rscales = numpy.round(numpy.arange(0.7, 1.51, 0.1), 1)   # 0.7, 0.8, ..., 1.5

zeta = 'dz'
basis = 'ccpv' + zeta            # CP2K BASIS_SET CC-PVDZ
verbose = 4
a = numpy.eye(3) * L
ke_cutoff = 200                  # Ha; CP2K CUTOFF 400 Ry, NGRIDS 1
ke_cutoff_exact = 2000           # Ha; exact all-electron FFTDF reference
precision = 1e-8
alpha0 = 12.5                    # CP2K ALPHA0_HARD 12.5
Rb = 1.7                         # bohr; CP2K HARD_EXP_RADIUS 1.7
gdfNuc = False                   # True: nuclear attraction via GDF instead of the PAW nuclear terms
xc = ''                          # no XC (Hartree only); CP2K XC_FUNCTIONAL NONE
init_guess = '1e'

def make_atom(r):
    return f'He {r1} {r1} {r1}; He {r1+r} {r1} {r1}'

def make_cell(r, cutoff):
    return pgto.M(
        atom        = make_atom(r),
        basis       = basis,
        verbose     = verbose,
        a           = a,
        ke_cutoff   = cutoff,
        precision   = precision
    )

def run_exact(r, cutoff=ke_cutoff_exact):
    mf = pscf.RKS(make_cell(r, cutoff))
    mf.init_guess = init_guess
    mf.xc = xc
    mf.kernel()
    return mf

def run_paw(r):
    mf = pscf.RKS(make_cell(r, ke_cutoff))
    mf.init_guess = init_guess
    mf.xc = xc
    mydf = PAW.from_mf(mf, PWAccuracy=1e-8, alpha0=alpha0, augRadius=Rb, gdfNuc=gdfNuc).build()
    mf.with_df = mydf
    mf._numint = PAWNumInt(mydf, mf)
    mf.kernel()
    return mf

def exact_cutoff_convergence(rscale=rscales[0], cutoffs=(1000, 1500, 2000, 2500, 3000)):
    '''Check the FFTDF reference converges with ke_cutoff (shortest bond is hardest).'''
    r = r0 * rscale
    results = []
    for cut in cutoffs:
        mf = run_exact(r, cut)
        results.append({"ke_cutoff": cut, "etot_exact": mf.e_tot, "conv_exact": mf.converged})
        print(f'ke_cutoff {cut}: E = {mf.e_tot:.10f}')
        del mf
        gc.collect()

    path = f"data/He2_exact_cutoff_convergence_r_{r:.2f}_{zeta}_{precision:.0e}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ke_cutoff", "etot_exact", "conv_exact"])
        writer.writeheader()
        writer.writerows(results)

def exact_paw_diff(tag=''):
    results = []
    for rscale in rscales:
        r = r0 * rscale
        mf_exact = run_exact(r)
        mf_paw = run_paw(r)

        results.append({
            "r": r,
            "etot_exact": mf_exact.e_tot,
            "etot_paw": mf_paw.e_tot,
            "conv_exact": mf_exact.converged,
            "conv_paw": mf_paw.converged
        })

        # CLEANUP
        del mf_exact
        del mf_paw
        gc.collect()

    path = f"data/He2bondLength_{ke_cutoff:.2f}_a_{alpha0}_r_{Rb}_{zeta}_{precision:.0e}_gdfNuc_{gdfNuc}_exact_{ke_cutoff_exact}{tag}.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "r", "etot_exact", "etot_paw", "conv_exact", "conv_paw"
        ])
        writer.writeheader()
        writer.writerows(results)

def number(s):
    '''10 -> 10, 12.5 -> 12.5, so file names match the CP2K folders (a_10_r_1.5)'''
    v = float(s)
    return int(v) if v.is_integer() else v

def main():
    # usage: python He2bondLength.py [scan|convergence] [--gdfNuc True|False] [--alpha0 A] [--Rb R] [--tag T]
    import argparse
    global gdfNuc, alpha0, Rb
    parser = argparse.ArgumentParser()
    parser.add_argument('task', nargs='?', default='scan', choices=['scan', 'convergence'])
    parser.add_argument('--gdfNuc', default=str(gdfNuc), choices=['True', 'False'])
    parser.add_argument('--alpha0', type=number, default=alpha0)
    parser.add_argument('--Rb', type=number, default=Rb)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    gdfNuc = args.gdfNuc == 'True'
    alpha0, Rb = args.alpha0, args.Rb
    if args.task == 'convergence':
        exact_cutoff_convergence()
    else:
        exact_paw_diff(args.tag)

if __name__ == '__main__':
    main()
