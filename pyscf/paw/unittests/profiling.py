import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

import pyscf
from pyscf.paw import NewPAW

L = 5.443702372939453  # box size

atom    = '''
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
    ke_cutoff   = 100,
    precision   = precision
)

def profileobtainLocalFns():
    from PAWutilsNumpy import obtainLocalFns, modifyMolBasis

    # function args
    pmol, ctr_coeff = cell.decontract_basis()
    pmol._basis = modifyMolBasis(pmol._basis)
    mol = cell
    mf = pyscf.scf.RKS(pmol)
    mf.grids.build()
    grids = mf.grids
    alpha0 = 10
    epsilon = 1e-5
    Rb = 1.5
    Periodic = True

    # func to profile
    obtainLocalFns(pmol, mol, ctr_coeff, grids, alpha0, epsilon=epsilon, Rb=Rb, Periodic=Periodic)

def profilegetj_PAW():
    from PAWutilsNumpy import getj_PAW
    from pyscf import lib

    init_guess = '1e'
    xc = ''
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=10, augRadius=1.5).build()
    mf_per_rks_paw.with_df = mydf

    cell = mydf.cell
    nk = 1
    nao = cell.nao

    # fftdf
    mf_per_rks = pscf.RKS(cell_fftdf)
    mf_per_rks.init_guess = init_guess
    mf_per_rks.xc = xc

    mf_per_rks.kernel()
    dm = mf_per_rks.make_rdm1()
    mo_coeff = numpy.asarray(dm.mo_coeff).reshape(-1, nk, nao, nao)
    mo_occ = numpy.asarray(dm.mo_occ).reshape(-1, nk, nao)
    dm = lib.tag_array(dm.reshape(-1, nk, nao, nao), mo_coeff=mo_coeff, mo_occ=mo_occ)

    getj_PAW(cell, dm, mydf.aoOnR_tilde, mydf.mesh, mydf.PAWdata, Periodic=True)

def profileobtainLocal2e():
    from PAWutilsNumpy import obtainLocal2e, modifyMolBasis
    pmol, ctr_coeff = cell.decontract_basis()
    pmol._basis = modifyMolBasis(pmol._basis)
    mol = cell
    Periodic = True
    obtainLocal2e(mol, pmol, Periodic)


def main():
    # profileobtainLocalFns()
    # profilegetj_PAW()
    profileobtainLocal2e()
    print('Done')

if __name__ == '__main__':
    main()