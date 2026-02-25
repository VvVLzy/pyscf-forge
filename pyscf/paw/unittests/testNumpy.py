from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

import numpy

from pyscf.paw import NewPAW

import time


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

mf_per_rks_paw = pscf.RKS(cell)
mf_per_rks_paw.init_guess = init_guess
mf_per_rks_paw.xc = xc
mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=10, augRadius=1.5).build()
mf_per_rks_paw.with_df = mydf

def testmakeWignerSeitz():
    from PAWutilsJax import makeWignerSeitz as old
    from PAWutilsNumpy import makeWignerSeitz as new

    start = time.time()
    result = old(mydf.Rgrid, mydf.cell, mydf.Periodic)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(mydf.Rgrid, mydf.cell, mydf.Periodic)
    print(f'Npytime: {time.time()-start: .10f}')

    for i in range(len(result)):

        # print(numpy.max(numpy.abs(result[i]-result1[i])))
        assert(numpy.isclose(result[i], result1[i]).all())

def testcompensatingCharge():
    from PAWutilsJax import compensatingCharge as old
    from PAWutilsNumpy import compensatingCharge as new
    
    start = time.time()
    result = old(mydf.pcell, mydf.cell, mydf.alpha0, mydf.Rgrid,
                 mydf.PAWorbitalCutOff,Periodic=True)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(mydf.pcell, mydf.cell, mydf.alpha0, mydf.Rgrid,
                 mydf.PAWorbitalCutOff,Periodic=True)
    print(f'Npytime: {time.time()-start: .10f}')
    for i in range(5):
        # print(i)
        for atom in range(len(result[i])):
            # print(numpy.max(numpy.abs(result[i][atom]-result1[i][atom])))
            assert(numpy.isclose(result[i][atom], result1[i][atom]).all())
            
def testcompensatingChargeSph():
    from PAWutilsJax import compensatingChargeSph as old
    from PAWutilsNumpy import compensatingChargeSph as new

    from PAWutilsNumpy import makeWignerSeitz, makeAugmentationSphere, getAlphaAtomsL

    pmol = mydf.pcell
    alpha0 = mydf.alpha0

    alpha, atoms, L, M = getAlphaAtomsL(pmol._bas, pmol._env)

    ##introducing the compensating charge basis set
    gmax = int(L.max()*2)
    gbas = {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        gbas[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]
    gmol = pgto.M(atom=pmol.atom, basis=gbas, a=pmol.a, unit=pmol.unit) ##if periodic then cartesian functions
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmol._bas, gmol._env, cart=gmol.cart)
    WignerSeitzData = makeWignerSeitz(mydf.Rgrid, mydf.pcell, Periodic=False)
    gridIdx = makeAugmentationSphere(WignerSeitzData, mydf.pcell, LG, alpha0, Rb=mydf.augRadius, epsilon=mydf.PAWorbitalCutOff)[0]

    start = time.time()
    result = old(mydf.pcell, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmol, mydf.Rgrid, gridIdx)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(mydf.pcell, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmol, mydf.Rgrid, gridIdx)
    print(f'Npytime: {time.time()-start: .10f}')
    for i in range(5):
        print(i)
        for atom in range(len(result[i])):
            # print(numpy.max(numpy.abs(result[i][atom]-result1[i][atom])))
            assert(numpy.isclose(result[i][atom], result1[i][atom]).all())

def testgetMPQLarray():
    from PAWutilsJax import getMPQLarray as old
    from PAWutilsNumpy import getMPQLarray as new

    from PAWutilsNumpy import getAlphaAtomsL

    pmol = mydf.pcell
    alpha0 = mydf.alpha0

    alpha, atoms, L, M = getAlphaAtomsL(pmol._bas, pmol._env)

    ##introducing the compensating charge basis set
    gmax = int(L.max()*2)
    gbas = {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        gbas[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]
    gmol = pgto.M(atom=pmol.atom, basis=gbas, a=pmol.a, unit=pmol.unit) ##if periodic then cartesian functions
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmol._bas, gmol._env, cart=gmol.cart)

    start = time.time()
    result = old(pmol, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(pmol, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax)
    print(f'Npytime: {time.time()-start: .10f}')
    for atom in range(len(result)):
        # print(numpy.max(numpy.abs(result[atom]-result1[atom])))
        assert(numpy.isclose(result[atom], result1[atom]).all())

def testgetjSmoothPW():
    from PAWutilsJax import getjSmoothPW as old
    from PAWutilsNumpy import getjSmoothPW as new

    from pyscf import lib
    
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

    start = time.time()
    result = old(cell, dm, mydf.aoOnR_tilde, mydf.mesh, mydf.PAWdata, Periodic=True)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(cell, dm, mydf.aoOnR_tilde, mydf.mesh, mydf.PAWdataNumpy, Periodic=True)
    print(f'Npytime: {time.time()-start: .10f}')
    assert(numpy.isclose(result, result1).all())
    # print(numpy.max(numpy.abs(result-result1)))
    # import pdb; pdb.set_trace()
    # ## debugging intermediates
    # for atom in range(len(result)):
    #     print(numpy.max(numpy.abs(result[atom]-result1[atom])))

def testgetjSharpLocal():
    from PAWutilsJax import getjSharpLocal as old
    from PAWutilsNumpy import getjSharpLocal as new

    from pyscf import lib
    
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

    start = time.time()
    result = old(cell, dm, mydf.aoOnR_tilde, mydf.mesh, mydf.PAWdata, Periodic=True)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(cell, dm, mydf.aoOnR_tilde, mydf.mesh, mydf.PAWdataNumpy, Periodic=True)
    print(f'Npytime: {time.time()-start: .10f}')
    assert(numpy.isclose(result, result1).all())

def testgetjSmoothLocal():
    from PAWutilsJax import getjSmoothLocal as old
    from PAWutilsNumpy import getjSmoothLocal as new

    from pyscf import lib
    
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

    start = time.time()
    result = old(cell, dm, mydf.aoOnR_tilde, mydf.mesh, mydf.PAWdata, Periodic=True)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(cell, dm, mydf.aoOnR_tilde, mydf.mesh, mydf.PAWdataNumpy, Periodic=True)
    print(f'Npytime: {time.time()-start: .10f}')
    assert(numpy.isclose(result, result1).all())

def testobtainLocalFns():
    from PAWutilsJax import obtainLocalFns as old
    from PAWutilsNumpy import obtainLocalFns as new
    pmol = mydf.pcell
    mol = mydf.cell
    ctr_coeff = mydf.ctr_coeff
    grids = mydf.BeckeGrid
    alpha0 = mydf.alpha0
    epsilon = mydf.PAWorbitalCutOff
    Rb = mydf.augRadius
    Periodic = mydf.Periodic

    start = time.time()
    result = old(pmol, mol, ctr_coeff, grids, alpha0, epsilon=epsilon, Rb=Rb, Periodic=Periodic, rtol=1e-10)
    print(f'Jaxtime: {time.time()-start: .10f}')
    start = time.time()
    result1 = new(pmol, mol, ctr_coeff, grids, alpha0, epsilon=epsilon, Rb=Rb, Periodic=Periodic, rtol=1e-10)
    print(f'Npytime: {time.time()-start: .10f}')
    for i in range(len(result)):
        # print(i)
        for atom in range(len(result[i])):
            # print(numpy.max(numpy.abs(result[i][atom]-result1[i][atom])))
            assert(numpy.isclose(result[i][atom], result1[i][atom]).all())

def main():
    testmakeWignerSeitz()
    testcompensatingCharge()
    testcompensatingChargeSph()
    testgetMPQLarray()
    testgetjSmoothPW()
    testgetjSharpLocal()
    testgetjSmoothLocal()
    testobtainLocalFns()

if __name__ == '__main__':
    main()