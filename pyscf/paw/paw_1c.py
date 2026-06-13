import numpy
import scipy

import jax
jax.config.update("jax_enable_x64",True)
jax.config.update('jax_platform_name', 'cpu')


import pyscf
from pyscf.pbc import gto as pgto
from pyscf import __config__
from pyscf.pbc.lib.kpts_helper import unique
from pyscf.lib import logger

from . import ClebschGordan
from .paw_helper import *
from .paw_Integrals import *

from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

def RadialNorm(alpha, l):
    return 1./2./alpha**(0.5*(l+1.)) * scipy.special.gamma(0.5*(l+1.))

def gammaBar(n, l, a):
    """Vectorized Gamma calculation"""
    L_val = n + 2 + l
    LL = (L_val + 1) / 2
    # Use scipy.special.gamma instead of jsp.special.gamma
    return scipy.special.gamma(LL) / (2 * a**(LL))

def I_func(alpha_val, x, y, z):
    """Computes Integral I (Vectorized or Scalar)"""
    xyz = numpy.array([x, y, z])
    n = (xyz + 1) / 2
    # alpha_val can be scalar or array; ensure broadcasting works
    Ixyz = scipy.special.gamma(n) / (alpha_val**n)
    return numpy.prod(Ixyz)

def get_L_index(l, m):
    """Helper for flattened CG indices"""
    return l**2 + l + m

def Basisnorm(alpha, l):
    # L calculation works for both scalar and array inputs
    L = l * 2 + 2
    n = 0.5 * (L + 1)
    
    # Use scipy.special.gamma instead of jsp.special.gamma
    gamma_n = scipy.special.gamma(n)
    
    # Calculate the normalization factor
    # Corresponds to: 1 / sqrt( (1/2) * (1/(2*alpha)^n) * gamma(n) )
    denominator = (1.0 / 2.0 / (2.0 * alpha)**n) * gamma_n
    
    return 1.0 / numpy.sqrt(denominator)

def getmpql(L, M, alpha, LG, MG, alphaG, idx, idxg):
    CG = ClebschGordan.RealCG
    # --- Prepare Inputs for Broadcasting ---
    # Group 1 vars (j1, m1, alpha1) map to Axis 1 -> Shape (1, N, 1)
    j1 = L[idx][None, :, None]
    m1 = M[idx][None, :, None]
    a1 = alpha[idx][None, :, None]

    # Group 2 vars (j2, m2, alpha2) map to Axis 0 -> Shape (N, 1, 1)
    # Note: These come from the same arrays as Group 1, but we reshape them differently
    # so they broadcast orthogonally to j1.
    j2 = L[idx][:, None, None]
    m2 = M[idx][:, None, None]
    a2 = alpha[idx][:, None, None]

    # Group 3 vars (j3, m3, alpha3) map to Axis 2 -> Shape (1, 1, M)
    j3 = LG[idxg][None, None, :]
    m3 = MG[idxg][None, None, :]
    a3 = alphaG[idxg][None, None, :]

    # --- Vectorized Calculation ---
    
    # Calculate indices for the CG array
    # This will broadcast to shape (N, N, M) automatically
    idx_cg1 = j1*j1 + m1 + j1
    idx_cg2 = j2*j2 + m2 + j2
    idx_cg3 = j3*j3 + m3 + j3
    
    # Retrieve CG coefficients using advanced indexing
    cg_term = CG[idx_cg1, idx_cg2, idx_cg3]

    # Compute the normalization terms
    # Assuming RadialNorm and Basisnorm can handle array inputs
    num_term = RadialNorm(a1 + a2, j1 + j2 + j3 + 2) * Basisnorm(a1, j1) * Basisnorm(a2, j2)
    den_term = RadialNorm(a3, 2*j3 + 2) * Basisnorm(a3, j3)

    M_PQL = cg_term * num_term / den_term

    return M_PQL

def getMPQLarray(pmol, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax):
    # Retrieve external helpers
    CG = ClebschGordan.RealCG
    
    M_PQLarray = []

    atomBass = {}
    atomMpqls = {}
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
            M_PQLarray.append(atomMpqls[atomZ][i])
            continue

        M_PQLarray.append(getMPQLarrayAtom(atomI, CG, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax))

        # update calculated
        atomBas.append(atomInfo)
        atomBass[atomZ] = atomBas
        atomMpql = atomMpqls.get(atomZ, [])
        atomMpql.append(M_PQLarray[-1])
        atomMpqls[atomZ] = atomMpql

    return M_PQLarray


def getMPQLarrayAtom(atomI, CG, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax):
    # Boolean masks
    idx = (atoms == atomI)
    idxg = (atomsG == atomI)
    
    # Store indices for this atom

    # --- Prepare Data Subsets ---
    # "Left" inputs (correspond to l1, m1, alpha1)
    sub_L = L[idx]
    sub_M = M[idx]
    sub_alpha = alpha[idx]
    
    # Gauge inputs (correspond to l3, m3, alpha3 for Sph)
    sub_LG = LG[idxg]
    sub_MG = MG[idxg]
    sub_alphaG = alphaG[idxg]
    
    N_subset = len(sub_L)
    
    # =================================================================
    # PART A: MPQLCart Calculation (Equivalent to double vmap)
    # =================================================================
    
    # 1. Setup Matrix A (Constant for this atomI block)
    # In JAX code, alpha3 was alphaG[0]
    alpha3_cart = alphaG[0] 
    
    N0 = Basisnorm(alpha3_cart, 0)
    N2 = Basisnorm(alpha3_cart, 2)
    I0  = I_func(alpha3_cart, 0, 0, 0)
    I2  = I_func(alpha3_cart, 2, 0, 0)
    I4  = I_func(alpha3_cart, 4, 0, 0)
    I22 = I_func(alpha3_cart, 2, 2, 0)

    sq_4pi = numpy.sqrt(4 * numpy.pi)
    
    A0 = N0 * I2 / sq_4pi
    A1 = N2 * I4
    A2 = N2 * I22
    A3 = N0 * I0 / sq_4pi
    A4 = N2 * I2

    # Shape (4, 4)
    A = numpy.array([
        [A0, A1, A2, A2],
        [A0, A2, A1, A2],
        [A0, A2, A2, A1],
        [A3, A4, A4, A4]
    ])

    # 2. Setup Vector b (Varies pairwise)
    # We use broadcasting to create (N, N) matrices
    
    # Column vectors (N, 1) -> corresponds to index 'i'
    l1 = sub_L[:, None]
    m1 = sub_M[:, None]
    a1 = sub_alpha[:, None]
    
    # Row vectors (1, N) -> corresponds to index 'j'
    l2 = sub_L[None, :]
    m2 = sub_M[None, :]
    a2 = sub_alpha[None, :]

    NP = Basisnorm(a1, l1)
    NQ = Basisnorm(a2, l2)

    # Gamma terms broadcast to (N, N)
    gamma0 = gammaBar(0, l1 + l2, a1 + a2)
    gamma2 = gammaBar(2, l1 + l2, a1 + a2)

    N00 = numpy.sqrt(4 * numpy.pi)
    N20 = 4 * numpy.sqrt(numpy.pi / 5)
    N22 = 4 * numpy.sqrt(numpy.pi / 15)

    # CG Indexing
    idx_1 = get_L_index(l1, m1)
    idx_2 = get_L_index(l2, m2)
    idx_00 = get_L_index(0, 0) # Scalar
    idx_20 = get_L_index(2, 0) # Scalar
    idx_22 = get_L_index(2, 2) # Scalar

    # Fetch CG coefficients (Broadcasting handles the lookup)
    NC00 = N00 * CG[idx_1, idx_2, idx_00]
    NC20 = N20 * CG[idx_1, idx_2, idx_20]
    NC22 = N22 * CG[idx_1, idx_2, idx_22]

    # Construct b components. Resulting shape for each is (N, N)
    b0 = (1/6)*NP*NQ*gamma2*(2 + 3*NC22 - NC20)
    b1 = (1/6)*NP*NQ*gamma2*(2 - 3*NC22 - NC20)
    b2 = (1/3)*NP*NQ*gamma2*(1 + NC20)
    b3 = NP*NQ*gamma0*NC00

    # Stack to (N, N, 4)
    b_vec = numpy.stack([b0, b1, b2, b3], axis=-1)
    
    # Solve Ax = b for every pixel in (N, N).
    # Optimization: Flatten to (N*N, 4), transpose to (4, N*N) for solve, then reshape.
    # This solves A * x_i = b_i efficiently.
    MPQLCart_flat = numpy.linalg.solve(A, b_vec.reshape(-1, 4).T).T
    MPQLCart = MPQLCart_flat.reshape(N_subset, N_subset, 4)

    # =================================================================
    # PART B: MPQLSph Calculation (Equivalent to triple vmap)
    # =================================================================
    
    # We need a 3D tensor result: (N, N, K)
    # Reshape inputs for 3D broadcasting:
    # i (axis 0): (N, 1, 1)
    # j (axis 1): (1, N, 1)
    # g (axis 2): (1, 1, K)

    j1_3d = sub_L[:, None, None]
    m1_3d = sub_M[:, None, None]
    a1_3d = sub_alpha[:, None, None]

    j2_3d = sub_L[None, :, None]
    m2_3d = sub_M[None, :, None]
    a2_3d = sub_alpha[None, :, None]

    j3_3d = sub_LG[None, None, :]
    m3_3d = sub_MG[None, None, :]
    a3_3d = sub_alphaG[None, None, :]

    # Indices for CG
    cg_idx1 = (j1_3d*j1_3d + (m1_3d+j1_3d)).astype(int)
    cg_idx2 = (j2_3d*j2_3d + (m2_3d+j2_3d)).astype(int)
    cg_idx3 = (j3_3d*j3_3d + (m3_3d+j3_3d)).astype(int)

    # Compute Terms
    term_cg = CG[cg_idx1, cg_idx2, cg_idx3]
    
    num = term_cg * RadialNorm(a1_3d + a2_3d, j1_3d + j2_3d + j3_3d + 2) * \
            Basisnorm(a1_3d, j1_3d) * Basisnorm(a2_3d, j2_3d)
            
    den = RadialNorm(a3_3d, 2*j3_3d + 2) * Basisnorm(a3_3d, j3_3d)

    MPQLSph = num / den

    # =================================================================
    # PART C: Combine Results
    # =================================================================

    if gmax < 2:
        # M_PQLarray.append(MPQLCart)
        M_PQL = MPQLCart
    else:
        # Mask logic from original code
        maskSph = numpy.zeros(MPQLSph.shape[-1], dtype=bool)
        # Ensure indices are within bounds (assuming K >= 9 based on code snippet)
        # If K is small, this might need a check, but copying logic directly:
        maskSph[[0, 6, 8]] = True
        
        # Combine Cart (N, N, 4) with Sph subset (N, N, K_subset)
        M_PQL = numpy.concatenate([
            MPQLCart, 
            MPQLSph[:, :, ~maskSph]
        ], axis=-1)
        
        # M_PQLarray.append(combined)
    # do analytical M_00L
    alpha2 = alphaG[0]
    alpha1 = 2*alpha[idx][0]
    NN2 = (alpha2/numpy.pi)**(3/2)
    # C0_num = -NN2 * (3*alpha2/2/alpha1 - 5/2)
    C0 = NN2 * 5/2
    # C0_num1 = M_PQL[0, 0, 0] * Basisnorm(alpha2, 0) / numpy.sqrt(4*numpy.pi)
    # C2_num = -NN2 * (alpha2 - alpha2**2/alpha1)
    C2 = -NN2 * alpha2
    # C2_num1 = M_PQL[0, 0, 1] * Basisnorm(alpha2, 2)
    M_000 = C0 / Basisnorm(alpha2, 0) * numpy.sqrt(4*numpy.pi)
    M_002 = C2 / Basisnorm(alpha2, 2)
    M_PQL[0, 0, 0] = M_000
    M_PQL[0, 1:4, 1:4] = M_002
    return M_PQL
    
    # # do analytical M_00L
    # alpha2 = alphaG[0]
    # alpha1 = 2*alpha[idx][0]
    # NN2 = (alpha2/numpy.pi)**(3/2)
    # # C0 = -NN2 * (3*alpha2/2/alpha1 - 5/2)
    # C0 = NN2 * 5/2
    # # C2 = -NN2 * (alpha2 - alpha2**2/alpha1)
    # C2 = -NN2 * alpha2
    # M_000 = C0 / Basisnorm(alpha2, 0) * numpy.sqrt(4*numpy.pi)
    # M_002 = C2 / Basisnorm(alpha2, 2)
    # print(M_000, M_PQLarray[atomI][0, 0, 0] )
    # print(M_002, M_PQLarray[atomI][0, 0, 1] )
    # M_PQLarray[atomI][0, 0, 0] = M_000
    # M_PQLarray[atomI][0, 1:4, 1:4] = M_002


def compensatingCharge(pmol, mol, alpha0, Rgrid, epsilon, Rb=None, Periodic = False):
    pmol = addSharpGTO2Atom(pmol)
    # assert(type(pmolNuc.basis) == dict)
    # assert(pmolNuc.basis == modifyMolBasis(pmolNuc.basis))
    # alpha, atoms, L, M = getAlphaAtomsL(pmolNuc._bas, pmolNuc._env)
    alpha, atoms, L, M = getAlphaAtomsL(pmol._bas, pmol._env)

    ##introducing the compensating charge basis set
    gmax = int(L.max()*2)
    gbas = {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        alpha0_val = alpha0[elem] if isinstance(alpha0, dict) else alpha0
        gbas[elem] = [ [l, [alpha0_val, 1.]] for l in range(gmax+1)]

    gmol = pgto.M(atom=pmol.atom, basis=gbas, a=pmol.a, unit=pmol.unit) ##if periodic then cartesian functions
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmol._bas, gmol._env, cart=gmol.cart)

    logger.debug(mol, 'Making augmentation sphere for uniform grid:')
    # WignerSeitzData = makeWignerSeitz(Rgrid, pmol, Periodic=Periodic)
    # gridIdx, masked_gridIdx, masks, Rs = makeAugmentationSphere(WignerSeitzData, pmol, LG, alpha0, Rb=Rb, epsilon=epsilon)
    # gridIdx, masked_gridIdx, masks, Rs = makeAugmentationSphere1(Rgrid, mol, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=Periodic)
    gridIdx, masked_gridIdx, masks, Rs = makeAugmentationSphere_Fast(Rgrid, mol, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=Periodic)

    M_PQLarray, V_PQLarray, V_LLarray, gOnR = [], [], [], []
    for atomI in range(pmol._atm.shape[0]):
        idx  = (atoms==atomI)
        idxg = (atomsG==atomI)
        lmax = numpy.max(L[idx])
        M_PQLarray.append(getmpql(L, M, alpha, LG, MG, alphaG, idx, idxg))

        shellsA, shellsB = numpy.where(pmol._bas[:,0] == atomI)[0], numpy.where(gmol._bas[:,0] == atomI)[0]

        if (not Periodic):
            VPQL = intor_cross('int3c2e', pmol, gmol,  
                                shls_slice=(shellsA[0], shellsA[-1]+1, shellsA[0], shellsA[-1]+1, pmol.nbas + shellsB[0], pmol.nbas + shellsB[-1]+1))
            V_PQLarray.append(VPQL)

            VLL = gmol.intor('int2c2e', shls_slice=(shellsB[0], shellsB[-1]+1,shellsB[0], shellsB[-1]+1))
            V_LLarray.append(VLL)


        else:
            pmolAtom = buildPmolAtom(mol, pmol, atomI, False)
            gmolAtom = pgto.M(atom = [gmol._atom[atomI]], basis = gmol.basis, a = gmol.lattice_vectors(), cart = False, unit='B')
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

            V_LLarray.append(VLL)
            V_PQLarray.append(VPQL)

        gOnR.append(gmol.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[-1]+1)))

    return M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol, Rs

def compensatingChargeSph(pmol, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmol, Rgrid, gridIdx):
    import time
    CG = ClebschGordan.RealCG
    M_PQLarray, V_PQLarray, V_LLarray, gOnR = [], [], [], []

    atomBass = {}
    atomData = {}
    
    def getCalcIdx(atomInfo, atomBas):
        for i, info in enumerate(atomBas):
            if (atomInfo==info).all():
                return i
        return None

    t_mpql, t_vpql, t_vll, t_gonr = 0, 0, 0, 0

    for atomI in range(pmol._atm.shape[0]):
        shellsA, shellsB = numpy.where(pmol._bas[:,0] == atomI)[0], numpy.where(gmol._bas[:,0] == atomI)[0]
        gmolAtom = pgto.M(atom = [gmol._atom[atomI]], basis = gmol.basis, a = gmol.lattice_vectors(), cart = False, unit='B')
        pmolAtom = buildPmolAtom(pmol, pmol, atomI, cart=False)

        # check if redundant (cache key)
        idx_bas = numpy.where(pmol._bas[:,0] == atomI)[0]
        atomZ = pmol._atm[atomI, 0]
        atomL = pmol._bas[idx_bas][:, 1]
        atomExp = pmol._env[pmol._bas[idx_bas][:, -3]]
        atomInfo = numpy.concatenate((atomL, atomExp))
        
        atomBas = atomBass.get(atomZ, [])
        i = getCalcIdx(atomInfo, atomBas)
        
        if i is not None:
            # Use cached
            mpql, vpql, vll = atomData[atomZ][i]
            M_PQLarray.append(mpql)
            V_PQLarray.append(vpql)
            V_LLarray.append(vll)
        else:
            # Calculate and cache
            idx  = (atoms==atomI)
            idxg = (atomsG==atomI)
            
            t0 = time.time()
            mpql = getmpql(L, M, alpha, LG, MG, alphaG, idx, idxg)
            t_mpql += time.time() - t0

            t0 = time.time()
            # VPQL = intor_cross('int3c2e', pmol, gmol,  
                                # shls_slice=(shellsA[0], shellsA[-1]+1, shellsA[0], shellsA[-1]+1, pmol.nbas + shellsB[0], pmol.nbas + shellsB[-1]+1))
            VPQL = intor_cross('int3c2e', pmolAtom, gmolAtom,  
                                shls_slice=(0, pmolAtom.nbas, 0, pmolAtom.nbas, pmolAtom.nbas, pmolAtom.nbas+gmolAtom.nbas))
            t_vpql += time.time() - t0

            t0 = time.time()
            # VLL = gmol.intor('int2c2e', shls_slice=(shellsB[0], shellsB[-1]+1, shellsB[0], shellsB[-1]+1))
            VLL = gmolAtom.intor('int2c2e', shls_slice=(0, gmolAtom.nbas, 0, gmolAtom.nbas))
            t_vll += time.time() - t0

            M_PQLarray.append(mpql)
            V_PQLarray.append(VPQL)
            V_LLarray.append(VLL)

            # Update cache
            atomBas.append(atomInfo)
            atomBass[atomZ] = atomBas
            data_list = atomData.get(atomZ, [])
            data_list.append((mpql, VPQL, VLL))
            atomData[atomZ] = data_list

        # gOnR depends on Rgrid which is globally shifted relative to the atom, must compute per atom
        t0 = time.time()
        # gOnR.append(gmol.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[-1]+1)))
        gOnR.append(gmolAtom.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]]))
        t_gonr += time.time() - t0

    print(f"      [Time breakdown] mpql: {t_mpql:.4f}s, vpql: {t_vpql:.4f}s, vll: {t_vll:.4f}s, gonr: {t_gonr:.4f}s")
    return M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol

def mergeCompensatingCharge(pmol, mol, alpha0, Rgrid, epsilon, Rb=None, Periodic = False):
    import time
    print("\n--- Starting mergeCompensatingCharge ---")
    t0 = time.time()
    
    pmolNuc = addSharpGTO2Atom(pmol)
    assert(type(pmolNuc.basis) == dict)
    assert(pmolNuc.basis == modifyMolBasis(pmolNuc.basis))
    alpha, atoms, L, M = getAlphaAtomsL(pmolNuc._bas, pmolNuc._env)
    
    t1 = time.time()
    print(f"  [Time] Setup & getAlphaAtomsL (pmolNuc): {t1 - t0:.4f}s")

    ##introducing the compensating charge basis set
    # need both spherical and cartesian
    gmax = int(L.max()*2)
    gbasCart, gbasSph = {}, {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        alpha0_val = alpha0[elem] if isinstance(alpha0, dict) else alpha0
        gbasSph[elem] = [ [l, [alpha0_val, 1.]] for l in range(gmax+1)]
        if gmax < 2:
            assert(gmax == 0)
            gbasCart[elem] = [ [0, [alpha0_val, 1.]], [2, [alpha0_val, 1.]]]
        else:
            gbasCart[elem] = [ [l, [alpha0_val, 1.]] for l in range(gmax+1)]

    gmolCart = pgto.M(atom=pmol.atom, basis=gbasCart, a=pmol.a, unit=pmol.unit, cart=True)
    gmolSph = pgto.M(atom=pmol.atom, basis=gbasSph, a=pmol.a, unit=pmol.unit, cart=False)
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmolSph._bas, gmolSph._env, cart=gmolSph.cart)

    t2 = time.time()
    print(f"  [Time] Compensating charge basis setup (gmolCart/Sph): {t2 - t1:.4f}s")

    logger.debug(mol, 'Making augmentation sphere for uniform grid:')
    # WignerSeitzData = makeWignerSeitz(Rgrid, gmolSph, Periodic=Periodic)
    Lmax = numpy.array([max(LG.max(), 2)]*len(LG))
    # gridIdx, _, _, Rs = makeAugmentationSphere(WignerSeitzData, gmolSph, Lmax, alpha0, Rb=Rb, epsilon=epsilon)
    # gridIdx, _, _, Rs = makeAugmentationSphere1(Rgrid, mol, L, alpha0, Rb=Rb, epsilon=epsilon, Periodic=Periodic)
    gridIdx, _, _, Rs = makeAugmentationSphere_Fast(Rgrid, mol, L, alpha0, Rb=Rb, epsilon=epsilon, Periodic=Periodic)
    
    t3 = time.time()
    print(f"  [Time] makeAugmentationSphere1: {t3 - t2:.4f}s")

    if Periodic:
        t_periodic_start = time.time()
        M_PQLarray = getMPQLarray(pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax)
        t_mpql = time.time()
        print(f"    [Time] getMPQLarray: {t_mpql - t_periodic_start:.4f}s")
        
        V_LLarray, V_PQLarray = getVLLVPQLarray(mol, pmolNuc, gmolCart, gmax)
        t_vll_vpql = time.time()
        print(f"    [Time] getVLLVPQLarray: {t_vll_vpql - t_mpql:.4f}s")
        
        gOnR = getGOnR(pmolNuc, gmolCart, gmolSph, Rgrid, gridIdx, gmax)
        t_gonr = time.time()
        print(f"    [Time] getGOnR: {t_gonr - t_vll_vpql:.4f}s")
        
        gmol = gmolCart
    else:
        t_non_periodic_start = time.time()
        M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol = compensatingChargeSph(
            # pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, Periodic, gmolSph, Rgrid, gridIdx
            pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmolSph, Rgrid, gridIdx
        )
        print(f"    [Time] compensatingChargeSph (Non-Periodic): {time.time() - t_non_periodic_start:.4f}s")

    t_end = time.time()
    print(f"--- Finished mergeCompensatingCharge (Total: {t_end - t0:.4f}s) ---\n")

    return M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol, Rs

def obtainLocalFnsNewer(pmol, mol, ctr_coeff, alpha0, epsilon=1.e-5, Rs=None, Periodic=False, neighbor_list=None, rtol=1e-8):
    '''
    Fit AO and diffuse AO directly, much faster
    set F_Pmu directly to contraction coefficient when P=mu
    new projectors!!
    '''
    import time
    print("\n--- Starting obtainLocalFnsNewer ---")
    t0 = time.time()

    labels = labelShellWithIdx(ctr_coeff, mol)

    alpha, atoms, L, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    atomsAO, _, _ = getAtomsL(mol._bas, mol._env)
    
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = [], [], [], [], []

    t1 = time.time()
    print(f"  [Time] Setup & label/getAtoms: {t1 - t0:.4f}s")
    
    aoslice = mol.aoslice_by_atom()

    logger.debug(mol, 'Building projectors...')
    for atomI in range(mol._atm.shape[0]):
        Rb = Rs[atomI] if Rs is not None else None
        logger.debug(mol, 'Atom %d:', atomI)
        elem = mol._atom[atomI][0]
        alpha0_val = alpha0[elem] if isinstance(alpha0, dict) else alpha0
        # fit all local primitives with atom centered primitives
        ACenteredId = numpy.where(atoms == atomI)[0]
        ACenteredAOId = numpy.where(atomsAO == atomI)[0]

        # construct projectors
        numPrim = len(ACenteredId)
        Latom = L[ACenteredId]
        alphaAtom = alpha[ACenteredId]

        primCounts = {}
        for l in Latom:
            count = primCounts.get(l, 0)
            primCounts[l] = count + 1
        
        projBasis = []
        for l, c in primCounts.items():
            # nAlpha = c//(2*l+1) # assume spherical GTO
            alphaProj = (l * numpy.log(Rb) - numpy.log(epsilon)) / Rb**2
            alphaAtomL = alphaAtom[Latom==l]
            isoPrim = alphaAtomL > alpha0_val
            numPrimL = len(alphaAtomL) // (2*l+1)
            numIsoPrimL = sum(isoPrim) // (2*l+1)
            assert(c == numPrimL*(2*l+1))

            # smallest exponent nees to be contained in the sphere
            # cp2k way of determining the ratio
            x = (80.0/alphaProj)**(1.0/max(1, float(numPrimL - numIsoPrimL - 1))) if numPrimL - numIsoPrimL - 1 > 2 else 2.0
            if x > 2.0: x = 2.0
            alphasProj = numpy.zeros(numPrimL); zetval = alphaProj
            for i in range(numPrimL-1, -1, -1):
                if not isoPrim[i]: alphasProj[i] = zetval; zetval *= x
            for i in range(numPrimL-1, -1, -1):
                if isoPrim[i]: alphasProj[i] = zetval; zetval *= x

            projBasis4L = [[l, [a, 1.]] for a in alphasProj]
            logger.debug(mol, 'L=%d: projExp %s', l, alphasProj)
            projBasis.extend(projBasis4L)


        projElem = pmol._atom[atomI][0] + '!' # make sure this doens't coincide with existing elements

        # construct proj-prim overlap 
        projPrimBasis = pmol._basis.copy()
        projPrimBasis[projElem] = projBasis
        projPrimAtom = [pmol._atom[atomI], (projElem, pmol._atom[atomI][1])]
        projPrimMol = pgto.M(atom=projPrimAtom, basis=projPrimBasis, a = pmol.lattice_vectors(), unit='B', cart = pmol.cart)
        projPrimOvlp = projPrimMol.pbc_intor('int1e_ovlp')[numPrim:][:, :numPrim]

        # isolated projectors
        sharpAlphaId = numpy.where(alphaAtom > alpha0_val)[0] # could use a different threshold?
        for i in sharpAlphaId:
            projPrimOvlp[i, :] = 0
            projPrimOvlp[:, i] = 0

        # ppoinv = numpy.linalg.pinv(projPrimOvlp, rtol=rtol)
        U, s, Vh = numpy.linalg.svd(projPrimOvlp)
        s_inv = numpy.zeros_like(s)
        s_inv[s>rtol] = 1/s[s>rtol]
        ppoinv = Vh.T@numpy.diag(s_inv)@U.T

        # construct proj-ao overlap using NEIGHBOR CLUSTER
        numProj = projPrimOvlp.shape[0]
        neighbors = neighbor_list[atomI] if neighbor_list is not None else range(mol._atm.shape[0])
        
        clusterAOAtom = []
        clusterAOBasis = {}
        global_ao_indices = []
        
        for j in neighbors:
            clusterAOAtom.append(mol._atom[j])
            elem = mol._atom[j][0]
            clusterAOBasis[elem] = mol._basis[elem]
            global_ao_indices.extend(range(aoslice[j, 2], aoslice[j, 3]))
            
        global_ao_indices = numpy.array(global_ao_indices)
        
        clusterAOAtom.append((projElem, mol._atom[atomI][1]))
        clusterAOBasis[projElem] = projBasis
        
        projAOMol = pgto.M(atom=clusterAOAtom, basis=clusterAOBasis, a=mol.lattice_vectors(), unit='B', cart=mol.cart)
        projAOOvlp_cluster = projAOMol.pbc_intor('int1e_ovlp')[-numProj:][:, :-numProj]
        # determine local functions
        local_locId = numpy.where(numpy.max(numpy.abs(projAOOvlp_cluster), axis=0) > epsilon)[0]
        locId = global_ao_indices[local_locId]
        
        # assert(numpy.isin(ACenteredAOId, locId).all()) # this must be true
        if not numpy.isin(ACenteredAOId, locId).all():
            logger.warn(mol, f"Warning: ACenteredAOId not fully contained in locId for atom {atomI}. Check threshold.")
            
        idxToFit = numpy.where(~numpy.isin(locId, ACenteredAOId))[0]
        idxToSet = numpy.where(numpy.isin(locId, ACenteredAOId))[0]

        # fit the functions
        projAOOvlp_active = projAOOvlp_cluster[:, local_locId][:, idxToFit]
        # F_fitted, residuals, rank, s = numpy.linalg.lstsq(projPrimOvlp, projAOOvlp, rcond=None)
        F_fitted = ppoinv @ projAOOvlp_active

        # update arrays
        F_Pmu = numpy.zeros((len(ACenteredId), len(locId)))
        Ftilde_Pmu = numpy.zeros_like(F_Pmu)
        F_Pmu[:, idxToFit] = F_fitted
        Ftilde_Pmu[:, idxToFit] = F_fitted

        # set local, a-centered function directly as contraction coefficient (no fitting)
        diffuseAcenteredPrimMask = alpha[ACenteredId] < alpha0_val
        C_Pa = getContMatFromID(ACenteredAOId, ACenteredId, ctr_coeff, labels, mol)
        Ctilde_Pa = numpy.zeros_like(C_Pa)
        Ctilde_Pa[diffuseAcenteredPrimMask, :] = C_Pa[diffuseAcenteredPrimMask, :]
        assert(C_Pa.shape[0] == F_Pmu.shape[0])
        assert(Ctilde_Pa.shape[0] == Ftilde_Pmu.shape[0])
        assert(numpy.isin(ACenteredAOId, locId).all())
        F_Pmu[:, idxToSet] = C_Pa
        Ftilde_Pmu[:, idxToSet] = Ctilde_Pa

        # update array
        localIdx.append(locId)
        SArr.append(idxToFit)
        F_PmuArr.append(F_Pmu)
        Ftilde_PmuArr.append(Ftilde_Pmu)

    t2 = time.time()
    print(f"  [Time] Atom loop (projectors & fitting): {t2 - t1:.4f}s")

    VPQRSArr = obtainLocal2e(mol, pmol, Periodic)

    t3 = time.time()
    print(f"  [Time] obtainLocal2e: {t3 - t2:.4f}s")
    print(f"--- Finished obtainLocalFnsNewer (Total: {t3 - t0:.4f}s) ---\n")

    return localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr