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
        if gmax < 2:
            gOnR.append(
                gmolCart.pbc_eval_gto(
                    'GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[-1]+1)
                    )[:, [0, 1, 4, 6]]
            )
        else:
            # evaluate S00 cart AO
            s00OnR = gmolCart.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[0]+1))

            # evaluate l=2 cart AO
            l2OnR = gmolCart.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[2], shellsB[2]+1))[:, [0, 3, 5]]

            # evaluate sph AO
            sphOnR = gmolSph.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[-1]+1))
            maskSph = numpy.zeros(sphOnR.shape[-1], dtype=bool)
            maskSph[[0, 6, 8]] = True
            sphOnR = sphOnR[:, ~maskSph]
            
            gOnR.append(numpy.block([
                s00OnR, l2OnR, sphOnR
            ]))

    return gOnR

def obtainLocal2e(mol, pmol, Periodic):
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
        atomInfo = numpy.concatenate((atomL, atomExp))
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
            VPQRSWithNuc[0, 0, 1:, 1:] = -dfbuilder.get_nuc()[0]/pmol._atm[atomI, 0]
            VPQRSArr.append(VPQRSWithNuc)
        else:
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