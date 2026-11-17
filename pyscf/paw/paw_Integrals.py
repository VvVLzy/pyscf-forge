import numpy

import jax
jax.config.update("jax_enable_x64",True)
jax.config.update('jax_platform_name', 'cpu')

import pyscf
from pyscf.pbc import gto as pgto
from pyscf import __config__
from pyscf.pbc.df.rsdf_builder import _RSNucBuilder
from pyscf.pbc.lib.kpts_helper import unique

from .paw_helper import *

from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

def getVLLVPQLarray(mol, pmol, gmol_cart, gmax):
    V_LLarray, V_PQLarray = [], []

    atomBass = {}
    atomVlls, atomVpqls = {}, {}
    def getCalcIdx(atomInfo, atomBas):
        for i, info in enumerate(atomBas):
            if (atomInfo==info).all():
                return i
        return None

    for atomI in range(pmol._atm.shape[0]):
        idx = numpy.where(pmol._bas[:,0] == atomI)[0]
        
        # check if redundant
        atomZ = pmol._atm[atomI, 0]
        atomL = pmol._bas[idx][:, 1]
        atomExp = pmol._env[pmol._bas[idx][:, -3]]
        atomInfo = numpy.concatenate((atomL, atomExp))
        atomBas = atomBass.get(atomZ, [])
        i = getCalcIdx(atomInfo, atomBas)
        if i is not None:
            V_LLarray.append(atomVlls[atomZ][i])
            V_PQLarray.append(atomVpqls[atomZ][i])
            continue

        VLLatom, VPQLatom = getVLLVPQLarrayAtom(atomI, mol, pmol, gmol_cart, gmax)
        V_LLarray.append(VLLatom)
        V_PQLarray.append(VPQLatom)

        # update calculated
        atomBas.append(atomInfo)
        atomBass[atomZ] = atomBas

        atomVpql = atomVpqls.get(atomZ, [])
        atomVpql.append(V_PQLarray[-1])
        atomVpqls[atomZ] = atomVpql

        atomVll = atomVlls.get(atomZ, [])
        atomVll.append(V_LLarray[-1])
        atomVlls[atomZ] = atomVll

    return V_LLarray, V_PQLarray

def getVLLVPQLarrayAtom(atomI, mol, pmol, gmol_cart, gmax):
    pmolAtom = buildPmolAtom(mol, pmol, atomI, True)
    gmolAtom = pgto.M(atom = [gmol_cart._atom[atomI]], basis = gmol_cart.basis, a = pmol.lattice_vectors(), unit='B', cart = True)
    mydf = pyscf.pbc.df.RSDF(pmolAtom)
    mydf.auxbasis = gmolAtom.basis
    mydf.build()
    dfbuilder = pyscf.pbc.df.rsdf_builder._RSGDFBuilder(pmolAtom, gmolAtom).build()

    # (g|g')
    VLL = dfbuilder.get_2c2e(numpy.zeros((1, 3)))[0]
    j2c_cd, j2c_negative, j2ctag = dfbuilder.decompose_j2c(VLL)
    assert(j2c_negative is None)
    assert(j2ctag == 'CD')

    # (PQ|g)
    # TODO: gamma point only
    eri_3d = numpy.vstack([Lpq[0].copy() for Lpq in mydf.sr_loop(compact=False)])
    eri_3d = smart_einsum('LM, Mp -> pL', j2c_cd, eri_3d)
    VPQL = eri_3d.reshape((pmolAtom.nao, pmolAtom.nao, gmolAtom.nao))

    if gmax < 2:
        idx = [0, 1, 4, 6] # indices correspond to S00, x^2,y^2, z^2
        VLLatom = VLL[numpy.ix_(idx, idx)]
        VPQLatom = VPQL[:, :, idx]
    else:
        # transform all but the four cart basis to sph basis
        car2sph = gmolAtom.cart2sph_coeff()
        assert(car2sph.shape[0] == VLL.shape[0])
        maskCart = numpy.zeros(car2sph.shape[0], dtype=bool)
        maskCart[[0, 4, 7, 9]] = True
        maskSph = numpy.zeros(car2sph.shape[1], dtype=bool)
        maskSph[[0, 6, 8]] = True
        car2sph = car2sph[~maskCart][:, ~maskSph]

        VLL_cc = VLL[maskCart][:, maskCart]
        VLL_cs = smart_einsum('LM, Mm -> Lm', VLL[maskCart][:, ~maskCart], car2sph)
        VLL_sc = numpy.transpose(VLL_cs)
        VLL_ss = smart_einsum('LM, Ll, Mm -> lm', VLL[~maskCart][:, ~maskCart], car2sph, car2sph)

        VLLatom = numpy.block([
                [VLL_cc, VLL_cs],
                [VLL_sc, VLL_ss]
            ])

        assert(pmol.cart == False) # comp charge doesn't really work cleanly with cartesian basis
        VPQLatom = smart_einsum('PQL, Pp, Qq->pqL', numpy.block([
                VPQL[:, :, maskCart],
                smart_einsum('PQL, Ll -> PQl', VPQL[:, :, ~maskCart], car2sph)
            ]), pmolAtom.cart2sph_coeff(), pmolAtom.cart2sph_coeff())

    return VLLatom, VPQLatom

def getGOnR(pmol, gmolCart, gmolSph, Rgrid, gridIdx, gmax):
    gOnR = []
    for atomI in range(pmol._atm.shape[0]):
        shellsB = numpy.where(gmolCart._bas[:,0] == atomI)[0]
        gmolCartAtom = pgto.M(atom = [gmolCart._atom[atomI]], basis = gmolCart.basis, a = gmolCart.lattice_vectors(), cart = True, unit='B')
        gmolSphAtom = pgto.M(atom = [gmolSph._atom[atomI]], basis = gmolSph.basis, a = gmolSph.lattice_vectors(), cart = False, unit='B')
        if gmax < 2:
            gOnR.append(
                gmolCartAtom.pbc_eval_gto(
                    'GTOval', Rgrid[gridIdx[atomI]]
                    )[:, [0, 1, 4, 6]]
            )
        else:
            # evaluate S00 cart AO
            s00OnR = gmolCartAtom.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(0, 1))

            # evaluate l=2 cart AO
            l2OnR = gmolCartAtom.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(2, 3))[:, [0, 3, 5]]

            # evaluate sph AO
            sphOnR = gmolSphAtom.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]])
            maskSph = numpy.zeros(sphOnR.shape[-1], dtype=bool)
            maskSph[[0, 6, 8]] = True
            sphOnR = sphOnR[:, ~maskSph]
            
            gOnR.append(numpy.block([
                s00OnR, l2OnR, sphOnR
            ]))

    return gOnR

def getNucShortRange(cell, omega, kpts=numpy.zeros((1,3))):
    '''Short-range nuclear attraction; omega is a number or one value per atom.

    With per-atom omega, atoms are grouped by omega and each group is computed
    with the charges of all other atoms set to zero (the result is linear in Z).
    '''
    omega = numpy.asarray(omega, dtype=float)
    if omega.ndim == 0:
        return _getNucShortRange(cell, float(omega), kpts)
    assert omega.shape == (cell.natm,)
    vsr = 0
    for w in numpy.unique(omega):
        sub = cell.copy(deep=False)
        sub._atm = cell._atm.copy()
        sub._atm[omega != w, 0] = 0   # CHARGE_OF
        vsr = vsr + _getNucShortRange(sub, float(w), kpts)
    return vsr

def _getNucShortRange(cell, omega, kpts=numpy.zeros((1,3))):
    '''Short-range nuclear attraction  sum_C <mu| -Z_C erfc(omega|r-R_C|)/|r-R_C| |nu>
    for a periodic cell, analytic 3-center integrals (PySCF RS nuclear builder).

    The G=0 part is removed with the same convention as _RSNucBuilder.get_nuc, so
    get_nuc() = getNucShortRange(omega) + long-range erf(omega r)/r part.
    Returns an array of shape (nkpts, nao, nao).
    '''
    from pyscf.pbc.df import aft, ft_ao
    from pyscf.pbc.df.rsdf_builder import estimate_rcut
    kpts = numpy.reshape(kpts, (-1, 3))
    builder = _RSNucBuilder(cell, kpts)
    builder.exclude_dd_block = False   # all AO pairs through the analytic SR integrals
    builder.build(omega=omega)
    # build() rounds omega to its PW mesh; the SR integrals need exactly this omega
    builder.omega = omega
    fakenuc = aft._fake_nuc(cell, with_pseudo=False)
    rcut_sr = estimate_rcut(builder.rs_cell, fakenuc, omega, exclude_dd_block=False)
    supmol = ft_ao.ExtendedMole.from_cell(builder.rs_cell, builder.bvk_kmesh, rcut_sr.max(), builder.verbose)
    supmol.omega = -omega
    builder.supmol = supmol.strip_basis(rcut_sr)
    vsr = builder._int_nuc_vloc(fakenuc)
    return numpy.asarray([pyscf.lib.unpack_tril(v) for v in vsr])

def obtainLocal2e(mol, pmol, Periodic, alpha0=None):
    '''One-center 2e integrals (PQ|RS) per atom; index 0 is the nucleus.

    alpha0: compensation exponent (number or {element label: exponent}). If given,
    the periodic nuclear row (Z|RS) is only the long-range part of the point-nucleus
    attraction, i.e. the interaction with a Gaussian core charge of exponent
    a = alpha0[element] (erf(sqrt(a) r)/r); the short-range erfc part is then added
    analytically for the whole cell in PAW.get_nuc. None: point nucleus.
    '''
    VPQRSArr = []

    # store calculated atom-basis to avoid redundant calculation
    atomBass = {}
    atom2eInts = {}
    def getCalcIdx(atomInfo, atomBas):
        for i, info in enumerate(atomBas):
            if (atomInfo==info).all():
                return i
        return None

    for atomI in range(mol._atm.shape[0]):
        # local 4-index integral
        idx = numpy.where(pmol._bas[:,0] == atomI)[0]
        
        # check if redundant
        atomZ = mol._atm[atomI, 0]
        atomL = pmol._bas[idx][:, 1]
        atomExp = pmol._env[pmol._bas[idx][:, -3]]
        if alpha0 is None:
            coreExp = None
        else:
            coreExp = alpha0[mol._atom[atomI][0]] if isinstance(alpha0, dict) else alpha0
        # the nuclear row depends on the core exponent too, so it is part of the key
        atomInfo = numpy.concatenate((atomL, atomExp, [-1. if coreExp is None else coreExp]))
        atomBas = atomBass.get(atomZ, [])
        i = getCalcIdx(atomInfo, atomBas)
        if i is not None:
            VPQRSArr.append(atom2eInts[atomZ][i])
            continue
        # calculate integrals
        # auxbasis = getAuxbasis(pmol)
        if Periodic :
            molAtom = buildPmolAtom(mol, pmol, atomI, pmol.cart)
            mydf = pyscf.pbc.df.RSDF(molAtom)
            # mydf.auxbasis = auxbasis # TODO: this gets large quadratically
            mydf.build()
            # mydf = pyscf.pbc.df.FFTDF(molAtom)
            # TODO: get this by GDF 2c2e
            # import pdb; pdb.set_trace()
            VPQRS = mydf.get_eri(compact=False).reshape((molAtom.nao, molAtom.nao, molAtom.nao, molAtom.nao))

            VPQRSWithNuc = numpy.zeros([VPQRS.shape[0]+1]*len(VPQRS.shape))
            VPQRSWithNuc[1:, 1:, 1:, 1:] = VPQRS

            # nuclear
            dfbuilder = _RSNucBuilder(molAtom, kpts=numpy.zeros((1,3))).build()
            vnuc = dfbuilder.get_nuc()[0]
            if coreExp is not None: # keep only the long-range erf part
                vnuc = vnuc - getNucShortRange(molAtom, numpy.sqrt(coreExp))[0]
            VPQRSWithNuc[0, 0, 1:, 1:] = -vnuc/pmol._atm[atomI, 0]
            VPQRSArr.append(VPQRSWithNuc)
        else:
            # molecules: nuclear attraction is analytic in PySCF, the PAW nuclear
            # row is not computed (left zero)
            molAtom = buildPmolAtom(mol, pmol, atomI, pmol.cart)
            # VPQRS = pmol.intor('int2e', shls_slice=(idx[0], idx[-1]+1,idx[0], idx[-1]+1,idx[0], idx[-1]+1,idx[0], idx[-1]+1))
            VPQRS = molAtom.intor('int2e')

            VPQRSWithNuc = numpy.zeros([VPQRS.shape[0]+1]*len(VPQRS.shape))
            VPQRSWithNuc[1:, 1:, 1:, 1:] = VPQRS
            VPQRSArr.append(VPQRSWithNuc)

        # update calculated
        atomBas.append(atomInfo)
        atomBass[atomZ] = atomBas
        atom2eInt = atom2eInts.get(atomZ, [])
        atom2eInt.append(VPQRSArr[-1])
        atom2eInts[atomZ] = atom2eInt

    return VPQRSArr