import itertools
import numpy
import scipy, time

import jax
jax.config.update("jax_enable_x64",True)
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
from jax import vmap, jit, lax
import jax.scipy as jsp

import pyscf
from pyscf import gto, lib
from pyscf.pbc import gto as pgto
from pyscf.pbc.dft.multigrid import multigrid_pair
from pyscf.pbc.dft.multigrid import _backend_c as backend
from pyscf import __config__
from pyscf.pbc.df.rsdf_builder import _RSNucBuilder
from pyscf.pbc.lib.kpts_helper import unique
from pyscf.pbc.df import GDF, incore
from pyscf.lib import logger

from . import ClebschGordan

from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

def makeAugmentationRadius(mol):
    # Create the 3x3x3 supercell to account for periodic boundary conditions
    repMol = pyscf.pbc.tools.pbc.cell_plus_imgs(mol, [1, 1, 1])
    
    # PySCF provides a direct way to get atomic coordinates as an (N, 3) numpy array
    atomPos = repMol.atom_coords() 
    
    # Calculate pairwise distance matrix using NumPy broadcasting
    # (N, 1, 3) - (1, N, 3) results in an (N, N, 3) array, then we take the norm
    diff = atomPos[:, numpy.newaxis, :] - atomPos[numpy.newaxis, :, :]
    atomAtomDistance = numpy.linalg.norm(diff, axis=-1)

    # Mask self-distances (and any perfectly overlapping atoms)
    atomAtomDistance = numpy.where(atomAtomDistance < 1.e-5, 1.e11, atomAtomDistance)
    
    # Minimum distance for each atom, halved
    Rb = numpy.min(atomAtomDistance, axis=0) / 2.0
    
    return numpy.min(Rb) ####This needs a proper fix


def pickalpha0(mol, Rb, epsilon = 1.e-5, maxL = 6):
    def solve_alpha(r_c, l, epsilon):
        """
        Solves for alpha in the equation:
        sqrt((2*(2*alpha)**(l+1.5))/gamma(l+1.5)) * r_c**l * exp(-alpha * r_c**2) = epsilon
        """
        
        # 1. Define the substituted variables to keep the code clean
        p = (l + 1.5) / 2.0
        
        # Calculate the constant C
        numerator_C = 2**(l + 2.5)
        denominator_C = scipy.special.gamma(l + 1.5)
        C = (r_c**l) * numpy.sqrt(numerator_C / denominator_C)
        
        # 2. Calculate z (the argument for the Lambert W function)
        z = -(r_c**2 / p) * (epsilon / C)**(1.0 / p)
        
        # 3. Apply the Lambert W function
        # We use the k=-1 branch because we expect a small epsilon and a large positive alpha.
        # The argument z will be small and negative (between -1/e and 0).
        W_val = scipy.special.lambertw(z, k=-1)
        
        # 4. Solve for alpha
        alpha = -(p / r_c**2) * W_val
        
        # The Lambert W function returns a complex number type in SciPy.
        # The physical solution we want is purely real, so we return the real part.
        return alpha.real

    alpha0 = []
    for i in range(mol._atm.shape[0]):
        basI = mol._bas[mol._bas[:,0]==i]
        maxl = min(2*numpy.max(basI[:,1]), maxL) ##cannot go beyond 6 for practical reasons
        alpha = solve_alpha(Rb, maxl, epsilon)
        alpha0.append(-numpy.log(epsilon/max(1., Rb**maxl))/Rb**2)
        # alpha0.append(alpha)

    return numpy.max(alpha0)

def prepareMolForPAW(mol):
    ##move the atoms so they are in the center of the cell
    atomPos = numpy.asarray([mol._atom[i][1] for i in range(len(mol._atom))])
    center = numpy.sum(atomPos, axis=0)/atomPos.shape[0]    


    displacement = mol.lattice_vectors().diagonal()/2. - center

    atomPos = [(atom[0], [(atom[1][0]+displacement[0])*pyscf.data.nist.BOHR, 
                          (atom[1][1]+displacement[1])*pyscf.data.nist.BOHR, 
                          (atom[1][2]+displacement[2])*pyscf.data.nist.BOHR]) for atom in mol._atom]

    if mol.unit[0].capitalize()=='A':
        k=mol.a
    else:
        k=pyscf.data.nist.BOHR*mol.a

    return pgto.M(atom = atomPos, basis = mol.basis, a = k, ke_cutoff = mol.ke_cutoff,unit='A')

def addSharpGTO2Atom(mol):
    mol = mol.copy()
    # 2. Define the new s-type Gaussian primitive to add
    # Here, we'll add a single s-type primitive with an exponent and a coefficient.
    new_s_gaussian = [0, [1e12, 1.0]]  # Angular momentum = 0, exponent = 1e12, coefficient = 1.0

    # 3. Create a new basis set dictionary
    custom_basis = {}

    # Iterate through each atom in the molecule and modify its basis
    for atm_symbol, basis_def in mol._basis.items():
        # Make a copy of the current basis definition for the atom
        modified_basis_def = [new_s_gaussian]
        
        # Add the new s-type Gaussian to the list of basis functions
        modified_basis_def.extend(list(basis_def))
        
        # Update the custom basis dictionary
        custom_basis[atm_symbol] = modified_basis_def

    # 4. Modify the molecule's basis set with the new custom basis
    mol.basis = custom_basis

    # 5. Rebuild the molecule object to apply the new basis set
    mol.build()

    return mol

def makeWignerSeitz(Rgrid, mol, Periodic=False):
    def makeWignerSeitz4Grid(grid):
        grid = numpy.array(grid) # (N, 3)
        atomPos = numpy.asarray([mol._atom[i][1] for i in range(len(mol._atom))]) # [A, 3]
        atomGridDistance = numpy.sqrt(numpy.sum((grid-atomPos[:, None])**2, axis=-1))

        return atomGridDistance

    if Periodic:
        nx = (-1, 0, 1)
        ny = (-1, 0, 1)
        nz = (-1, 0, 1)
        # consider all neighboring unit cells
        prod = list(itertools.product(nx, ny, nz))
        L = mol.lattice_vectors() # in Bohr

        atomGridDistances = []
        for coeff in prod:
            shift = numpy.dot(coeff, L)
            Rgrid_shifted = Rgrid + shift
            atomGridDistances.append(makeWignerSeitz4Grid(Rgrid_shifted))

        atomGridDistance = numpy.min(atomGridDistances, axis=0) # (Na, Ng)
    else:
        atomGridDistance = makeWignerSeitz4Grid(Rgrid)

    closestAtomToGrid = numpy.argmin(atomGridDistance, axis=0)

    return closestAtomToGrid, atomGridDistance, Rgrid

def getAtomsL(bas, env, cart=False):
    l, atmId, nexp = bas[:,1], bas[:,0], bas[:,3]

    if not cart :
        L     = numpy.concatenate([(2*l[i]+1)*[l[i]]*nexp[i] for i in range(len(l))])
        M     = numpy.concatenate([[1,-1,0]*nexp[i] if l[i] == 1 else list(range(-l[i],l[i]+1))*nexp[i] for i in range(len(l)) ])
        atoms = numpy.concatenate([(2*l[i]+1)*[atmId[i]]*nexp[i] for i in range(len(l))])
    else :
        L     = numpy.concatenate([(l[i]+1)*(l[i]+2)//2*[l[i]]*nexp[i] for i in range(len(l))])
        M     = numpy.concatenate([[1,2,3]*nexp[i] if l[i] == 1 else list(range(tillL(l[i]-1),tillL(l[i]-1)+(l[i]+1)*(l[i]+2)//2))*nexp[i] for i in range(len(l)) ])
        atoms = numpy.concatenate([(l[i]+1)*(l[i]+2)//2*[atmId[i]]*nexp[i] for i in range(len(l))])
    return atoms, L, M

def getAlphaAtomsL(bas, env, cart=False):
    # should only use for primitive basis!!
    assert(len(numpy.unique(bas[:, 2])) == 1)
    assert(numpy.unique(bas[:, 2])[0] == 1)
    l, atmId = bas[:,1], bas[:,0]
    # alpha    = numpy.concatenate([env[bas[i,5]:bas[i,6]] for i in range(bas.shape[0])])
    alpha    = numpy.concatenate([env[bas[i,5]:bas[i,5]+1] for i in range(bas.shape[0])])

    if not cart :
        alpha = numpy.concatenate([(2*li+1)*[a] for li,a in zip(l, alpha)])
        L     = numpy.concatenate([(2*li+1)*[li] for li,a in zip(l, alpha)])
        M     = numpy.concatenate([[1,-1,0] if li == 1 else list(range(-li,li+1)) for li,a in zip(l, alpha) ])
        atoms = numpy.concatenate([(2*li+1)*[a] for li,a in zip(l, atmId)])
    else :
        alpha = numpy.concatenate([(li+1)*(li+2)//2*[a] for li,a in zip(l, alpha)])
        L     = numpy.concatenate([(li+1)*(li+2)//2*[li] for li,a in zip(l, alpha)])
        M     = numpy.concatenate([[1,2,3] if li == 1 else list(range(tillL(li-1),tillL(li-1)+(li+1)*(li+2)//2)) for li,a in zip(l, alpha) ])
        atoms = numpy.concatenate([(li+1)*(li+2)//2*[a] for li,a in zip(l, atmId)])

    return alpha, atoms, L.astype(numpy.int64), M.astype(numpy.int64)

def RadialNorm(alpha, l):
    return 1./2./alpha**(0.5*(l+1.)) * scipy.special.gamma(0.5*(l+1.))

def basisnorm(alpha, l):
    L = l*2+2
    n = 0.5*(L+1)
    return 1./(1./2./(2.*alpha)**n * scipy.special.gamma(n))**0.5

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
    # Assuming RadialNorm and basisnorm can handle array inputs
    num_term = RadialNorm(a1 + a2, j1 + j2 + j3 + 2) * basisnorm(a1, j1) * basisnorm(a2, j2)
    den_term = RadialNorm(a3, 2*j3 + 2) * basisnorm(a3, j3)

    M_PQL = cg_term * num_term / den_term

    return M_PQL

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
        gbas[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]

    gmol = pgto.M(atom=pmol.atom, basis=gbas, a=pmol.a, unit=pmol.unit) ##if periodic then cartesian functions
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmol._bas, gmol._env, cart=gmol.cart)

    print('Making augmentation sphere for uniform grid:')
    WignerSeitzData = makeWignerSeitz(Rgrid, pmol, Periodic=Periodic)
    gridIdx = makeAugmentationSphere(WignerSeitzData, pmol, LG, alpha0, Rb=Rb, epsilon=epsilon)[0]

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

    return M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol

def compensatingChargeSph(pmol, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmol, Rgrid, gridIdx):
    CG = ClebschGordan.RealCG
    M_PQLarray, V_PQLarray, V_LLarray, gOnR = [], [], [], []
    for atomI in range(pmol._atm.shape[0]):
        idx  = (atoms==atomI)
        idxg = (atomsG==atomI)
        M_PQLarray.append(getmpql(L, M, alpha, LG, MG, alphaG, idx, idxg))

        shellsA, shellsB = numpy.where(pmol._bas[:,0] == atomI)[0], numpy.where(gmol._bas[:,0] == atomI)[0]

        # assert(Periodic)
        VPQL = intor_cross('int3c2e', pmol, gmol,  
                            shls_slice=(shellsA[0], shellsA[-1]+1, shellsA[0], shellsA[-1]+1, pmol.nbas + shellsB[0], pmol.nbas + shellsB[-1]+1))
        V_PQLarray.append(VPQL)

        VLL = gmol.intor('int2c2e', shls_slice=(shellsB[0], shellsB[-1]+1,shellsB[0], shellsB[-1]+1))
        V_LLarray.append(VLL)

        gOnR.append(gmol.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[-1]+1)))

    return M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol

def buildPmolAtom(mol, pmol, atomI, cart):
    # if mol.nao != pmol.nao and isinstance(pmol.basis, str): # mol uses a contracted basis
    #         pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = 'unc-'+pmol.basis, a = pmol.lattice_vectors(), unit='B', cart = cart) 
    # elif mol.nao != pmol.nao or isinstance(pmol.basis, dict):
    #     pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = pmol._basis, a = pmol.lattice_vectors(), unit='B', cart = cart)
    if mol.nao != pmol.nao:
        pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = pmol._basis, a = pmol.lattice_vectors(), unit='B', cart = cart)
    else:
        assert(mol.nao == pmol.nao)
        pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = pmol.basis, a = pmol.lattice_vectors(), unit='B',  cart = cart)

    return pmolAtom

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

# --- 1. Define Helper Functions (NumPy versions) ---
    
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
    # C0_num1 = M_PQL[0, 0, 0] * basisnorm(alpha2, 0) / numpy.sqrt(4*numpy.pi)
    # C2_num = -NN2 * (alpha2 - alpha2**2/alpha1)
    C2 = -NN2 * alpha2
    # C2_num1 = M_PQL[0, 0, 1] * basisnorm(alpha2, 2)
    M_000 = C0 / basisnorm(alpha2, 0) * numpy.sqrt(4*numpy.pi)
    M_002 = C2 / basisnorm(alpha2, 2)
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
    # M_000 = C0 / basisnorm(alpha2, 0) * numpy.sqrt(4*numpy.pi)
    # M_002 = C2 / basisnorm(alpha2, 2)
    # print(M_000, M_PQLarray[atomI][0, 0, 0] )
    # print(M_002, M_PQLarray[atomI][0, 0, 1] )
    # M_PQLarray[atomI][0, 0, 0] = M_000
    # M_PQLarray[atomI][0, 1:4, 1:4] = M_002
        

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

def mergeCompensatingCharge(pmol, mol, alpha0, Rgrid, epsilon, Rb=None, Periodic = False):
    pmolNuc = addSharpGTO2Atom(pmol)
    assert(type(pmolNuc.basis) == dict)
    assert(pmolNuc.basis == modifyMolBasis(pmolNuc.basis))
    alpha, atoms, L, M = getAlphaAtomsL(pmolNuc._bas, pmolNuc._env)

    ##introducing the compensating charge basis set
    # need both spherical and cartesian
    gmax = int(L.max()*2)
    gbasCart, gbasSph = {}, {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        gbasSph[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]
        if gmax < 2:
            assert(gmax == 0)
            gbasCart[elem] = [ [0, [alpha0, 1.]], [2, [alpha0, 1.]]]
        else:
            gbasCart[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]

    gmolCart = pgto.M(atom=pmol.atom, basis=gbasCart, a=pmol.a, unit=pmol.unit, cart=True)
    gmolSph = pgto.M(atom=pmol.atom, basis=gbasSph, a=pmol.a, unit=pmol.unit, cart=False)
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmolSph._bas, gmolSph._env, cart=gmolSph.cart)
    assert((alphaG == alphaG[0]).all())

    print('Making augmentation sphere for uniform grid:')
    WignerSeitzData = makeWignerSeitz(Rgrid, gmolSph, Periodic=Periodic)
    Lmax = numpy.array([max(LG.max(), 2)]*len(LG))
    gridIdx, _, _, Rs = makeAugmentationSphere(WignerSeitzData, gmolSph, Lmax, alpha0, Rb=Rb, epsilon=epsilon)

    if Periodic:
        M_PQLarray = getMPQLarray(pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax)
        V_LLarray, V_PQLarray = getVLLVPQLarray(mol, pmolNuc, gmolCart, gmax)
        gOnR = getGOnR(pmolNuc, gmolCart, gmolSph, Rgrid, gridIdx, gmax)
        gmol = gmolCart
    else:
        M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol = compensatingChargeSph(
            # pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, Periodic, gmolSph, Rgrid, gridIdx
            pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmolSph, Rgrid, gridIdx
        )

    return M_PQLarray, V_PQLarray, V_LLarray, gridIdx, gOnR, gmol, Rs

def intor_cross(intor, mol1, mol2, shls_slice, comp=None):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atmc, basc, envc = gto.mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                mol2._atm, mol2._bas, mol2._env)


    if (intor.endswith('_sph') or intor.startswith('cint') or
        intor.endswith('_spinor') or intor.endswith('_cart')):
        return gto.moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart == mol2.cart:
        intor = mol1._add_suffix(intor)
        return gto.moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart:
        mat = gto.moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return gto.numpy.dot(mat, mol2.cart2sph_coeff())
    else:
        mat = gto.moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return gto.numpy.dot(mol1.cart2sph_coeff().T, mat)

def get_Gv(nmesh,reciprocal_vecs):
    rx = numpy.fft.fftfreq(nmesh[0], 1./nmesh[0])
    ry = numpy.fft.fftfreq(nmesh[1], 1./nmesh[1])
    rz = numpy.fft.fftfreq(nmesh[2], 1./nmesh[2])
    return numpy.dot(pyscf.lib.cartesian_prod((rx,ry,rz)), reciprocal_vecs).astype(numpy.float64)

def getFormFactor_Truncated(nmesh,cell):
    L = cell.lattice_vectors().max()/2.
    G2 = get_Gv(nmesh,cell.reciprocal_vectors())**2
    G2 = numpy.sum( G2, axis = 1)

    idx = G2 < 1.e-10
    G2[idx] = 1.e-8    
    G = G2**0.5
    FF = 2 * (numpy.sin(G * L /2.)/G)**2
    FF[idx] = L**2/2.
    return FF * 4 *numpy.pi

def getFormFactor(nmesh,cell):
    G2 = get_Gv(nmesh,cell.reciprocal_vectors())**2
    G2 = numpy.sum( G2, axis = 1)
    FF = numpy.zeros((G2.shape[0]), numpy.float64)
    idx = numpy.greater( G2, 0.)
    FF[idx] = 4. * numpy.pi /G2[idx]
    return FF

def getFullK_fromoccRI(Kimu, occMO, S):
    Kij = Kimu @ occMO

    SC = S @ occMO
    K = SC @ Kimu
    return (K + K.T - SC @ Kij @ SC.T) 

tillL = lambda l : int( (l+1) * (l+2) * (l+3)//6 )


###
# Modified functions for contracted basis
####

def partitionAOs(mol, pmol, Rgrid, ctr_coeff, alpha0):
    aoOnR = mol.pbc_eval_gto('GTOval', Rgrid)

    aoOnR_prim = pmol.pbc_eval_gto('GTOval', Rgrid)
    alpha, atoms, L, M = getAlphaAtomsL(pmol._bas, pmol._env)

    assert(aoOnR_prim.shape[1] == len(alpha))

    def reconstructAOFromPrim():
        count_prim = 0
        count_contracted = 0
        # aoOnR_from_prim = numpy.zeros_like(aoOnR)
        aoOnR_tilde = numpy.zeros_like(aoOnR)
        for coeff in ctr_coeff:
            nprim = coeff.shape[0]
            ncontracted = coeff.shape[1]
            
            alphaa = alpha[count_prim: count_prim+nprim]
            diffuseIdx = numpy.where(alphaa < alpha0)[0]

            aoOnR_p = aoOnR_prim[:, count_prim: count_prim+nprim]

            # assertions
            LL = numpy.unique(L[count_prim: count_prim+nprim])
            aa = numpy.unique(atoms[count_prim: count_prim+nprim])
            assert(len(LL) == 1)
            assert(len(aa) == 1)
            ###

            # aoOnR_from_prim[:, count_contracted: count_contracted+ncontracted] = aoOnR_p @ coeff
            aoOnR_tilde[:, count_contracted: count_contracted+ncontracted] = aoOnR_p[:, diffuseIdx] @ coeff[diffuseIdx, :]

            count_prim += nprim
            count_contracted += ncontracted

        assert(count_prim == aoOnR_prim.shape[1])
        assert(count_contracted == aoOnR.shape[1])

        return aoOnR, aoOnR_tilde
    
    # test differences
    # print(numpy.max(numpy.abs(aoOnR_from_prim - aoOnR)))


    return reconstructAOFromPrim()

def getj_PAW(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    J1 = getjSmoothPW(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=Periodic)
    J2 = getjSharpLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=Periodic)
    J3 = getjSmoothLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=Periodic)

    return J1+J2+J3

def getjSmoothPW(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    # TODO: generalize to multiple k-points
    # use list of numpy arrays instead of jax arrays
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    occ_cut = 1e-12 # TODO: is this reasonable?
    dm, mo_coeff, mo_occ = dm[0, 0, :, :]/2, dm.mo_coeff[0, 0, :, :], dm.mo_occ[0, 0, :]
    occMo = mo_coeff[:,numpy.abs(mo_occ)>occ_cut] * mo_occ[numpy.abs(mo_occ)>occ_cut]/2
    ###

    Ng = numpy.prod(mesh)
    dv = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    moOnR = smart_einsum('ra,am->rm', aoOnR_tilde, occMo)
    density = smart_einsum('rm,rm->r', moOnR.conj(), moOnR)

    for atomI in range(cell._atm.shape[0]):
        submat = dm[localIdx[atomI]][:, localIdx[atomI]]
        DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
        DPQtilde = smart_einsum('Pm,Qn,mn->PQ', Ftilde_Pmu[atomI], Ftilde_Pmu[atomI], submat)
        zg = smart_einsum('PQ,PQL->L', DPQ-DPQtilde, M_PQLarr[atomI])
        numpy.add.at(density, gridIdx[atomI], smart_einsum('L,rL->r', zg, gOnR[atomI]))
    potential = numpy.fft.ifftn( FF * numpy.fft.fftn(density.reshape(mesh))).real.flatten()

    J = smart_einsum('ra,r,rb->ab', aoOnR_tilde.conj(), potential, aoOnR_tilde)*dv
    assert(J.shape[0] == cell.nao)
    assert(J.shape[1] == cell.nao)

    for atomI in range(cell._atm.shape[0]):
        # el+comp-compOnA
        zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*dv
        GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
        numpy.add.at(
            J, numpy.ix_(localIdx[atomI], localIdx[atomI]),
             smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
            -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
        )
    
    return J*2

def get_vxc_and_j_smooth(cell, dm, mg_ni, mesh, PAWdata, xc_code, Periodic=False, hermi=1, kpt=None):
    GGA_METHOD = getattr(__config__, 'pbc_dft_multigrid_gga_method', 'FFT')

    if kpt is None:
        kpt = numpy.zeros((1,3))
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    # Use total DM to ensure correct non-linear XC potential evaluation
    # dm expected as (nset, nkpts, nao, nao)
    dm_val = dm[0, 0, :, :]
    
    xctype = mg_ni._xc_type(xc_code)
    if xctype in (None, 'LDA', 'HF'):
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        raise NotImplementedError(f'xctype {xctype} not implemented in PAW multigrid')
        
    # Pass 1: AO -> Grid (Smooth Density)
    rhoG = multigrid_pair._eval_rhoG(mg_ni, dm, hermi=hermi, kpts=kpt, deriv=deriv)
    
    Ng = numpy.prod(mesh)
    dv = cell.vol / Ng
    weight = dv
    
    # XC Evaluation on pseudo-density only
    rhoR = numpy.fft.ifftn(rhoG.reshape(-1, *mesh), axes=(1,2,3)).real.reshape(-1, Ng) * (1./weight)
    
    if xctype == 'HF': # No XC part
        nelec = numpy.sum(rhoR[0]) * weight
        excsum = 0
        vxc_G = numpy.zeros_like(rhoG[0])
    else:
        # Evaluate XC on total density
        exc, vxcR = mg_ni.eval_xc_eff(xc_code, rhoR, deriv=1, xctype=xctype)[:2]
        nelec = numpy.sum(rhoR[0]) * weight
        excsum = numpy.dot(rhoR[0], exc) * weight
        vxc_G = numpy.fft.fftn(vxcR.reshape(-1, *mesh), axes=(1,2,3)).reshape(-1, Ng)
        
        if xctype == 'GGA' and GGA_METHOD.upper() == 'FFT':
            Gv = cell.get_Gv(mesh)
            vxc_G = backend.get_gga_vrho_gs(vxc_G[:1], vxc_G[1:4], Gv, weight, Ng, 1.)
        else:
            vxc_G *= weight

    # Hartree Evaluation on pseudo-density + compensating charges
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    
    rhoR_J = rhoR[0].copy()
    for atomI in range(cell._atm.shape[0]):
        submat = dm_val[localIdx[atomI]][:, localIdx[atomI]]
        # Use @ for faster matrix multiplication
        DPQ = F_Pmu[atomI] @ submat @ F_Pmu[atomI].T
        DPQtilde = Ftilde_Pmu[atomI] @ submat @ Ftilde_Pmu[atomI].T
        
        # zg = (DPQ - DPQtilde) : M_PQL
        diff_DPQ = (DPQ - DPQtilde).ravel()
        zg = diff_DPQ @ M_PQLarr[atomI].reshape(-1, M_PQLarr[atomI].shape[-1])
        
        # rhoR_J += zg @ gOnR^T
        rhoR_J[gridIdx[atomI]] += gOnR[atomI] @ zg

    rhoG_J = numpy.fft.fftn(rhoR_J.reshape(mesh)).flatten()
    vG_J = rhoG_J * FF.flatten()
    
    # Use real-space integration for ecoul to ensure consistent scaling
    potential_J = numpy.fft.ifftn(vG_J.reshape(mesh)).real.flatten()
    ecoul = 0.5 * numpy.dot(rhoR_J, potential_J) * dv
    
    # Pass 2: Grid -> Matrix
    # vxc matrix
    if xctype == 'GGA' and GGA_METHOD.upper() != 'FFT':
        vxc_mat = multigrid_pair._get_gga_pass2(mg_ni, vxc_G, kpts=kpt, hermi=hermi)
    else:
        vxc_mat = multigrid_pair._get_j_pass2(mg_ni, vxc_G, kpts=kpt, hermi=hermi)
    
    # vj matrix
    vj_mat = multigrid_pair._get_j_pass2(mg_ni, vG_J * weight, kpts=kpt, hermi=hermi)
    
    # Squeeze J and Vxc
    while vxc_mat.ndim > 2: vxc_mat = vxc_mat[0]
    while vj_mat.ndim > 2: vj_mat = vj_mat[0]
    
    # Local Hartree Corrections
    for atomI in range(cell._atm.shape[0]):
        # zg2 = \int v(r) g_L(r) dr
        zg2 = potential_J[gridIdx[atomI]] @ gOnR[atomI] * dv
        
        # GL = zg2 : M_PQL
        # GL_RS = \sum_L zg2_L * M_RSL
        GL = (M_PQLarr[atomI].reshape(-1, M_PQLarr[atomI].shape[-1]) @ zg2).reshape(M_PQLarr[atomI].shape[:-1])
        
        # vj_mat += F @ GL @ F^T
        vj_mat[numpy.ix_(localIdx[atomI], localIdx[atomI])] += \
            F_Pmu[atomI].T @ GL @ F_Pmu[atomI] - \
            Ftilde_Pmu[atomI].T @ GL @ Ftilde_Pmu[atomI]

    return nelec, excsum, ecoul, vxc_mat, vj_mat

def getjSmoothPW2(cell, dm, mg_ni, mesh, PAWdata, Periodic=False):
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    # Use alpha-DM to match getjSmoothPW logic (which returns J*2)
    dm_alpha = dm / 2
    dm_val = dm_alpha[0, 0, :, :]

    # Pass 1: AO -> Grid (Smooth Density)
    # Use multigrid for efficient collocation on the smooth basis
    rhoG = multigrid_pair._eval_rhoG(mg_ni, dm_alpha, hermi=1, kpts=numpy.zeros((1,3)), deriv=0)
    
    Ng = numpy.prod(mesh)
    dv = cell.vol / Ng
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    # Transform back to the fine grid for adding local compensating charges
    # rhoG from _eval_rhoG is already scaled by weight (Vol/Ng)
    rhoR = numpy.fft.ifftn(rhoG.reshape(mesh)).real.flatten() * (1./dv)
    
    for atomI in range(cell._atm.shape[0]):
        submat = dm_val[localIdx[atomI]][:, localIdx[atomI]]
        DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
        DPQtilde = smart_einsum('Pm,Qn,mn->PQ', Ftilde_Pmu[atomI], Ftilde_Pmu[atomI], submat)
        zg = smart_einsum('PQ,PQL->L', DPQ-DPQtilde, M_PQLarr[atomI])
        numpy.add.at(rhoR, gridIdx[atomI], smart_einsum('L,rL->r', zg, gOnR[atomI]))

    # Poisson in G-space
    # Standard FFT (unscaled)
    rhoG_total = numpy.fft.fftn(rhoR.reshape(mesh))
    vG = rhoG_total * FF
    
    # Pass 2: Grid -> Matrix (Smooth Potential Integration)
    # Note: _get_j_pass2 expects vG * weight
    wv_freq = vG * dv
    J = multigrid_pair._get_j_pass2(mg_ni, wv_freq, kpts=numpy.zeros((1,3)), hermi=1)
    while J.ndim > 2:
        J = J[0]
    
    # Local Matrix Corrections (zg2 terms)
    potential = numpy.fft.ifftn(vG).real.flatten()
    for atomI in range(cell._atm.shape[0]):
        # el+comp-compOnA
        zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*dv
        GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
        numpy.add.at(
            J, numpy.ix_(localIdx[atomI], localIdx[atomI]),
             smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
            -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
        )
    
    return J*2

def getjSharpLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    # TODO: generalize to multiple k-points
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    dm = dm[0, 0, :, :]/2
    ###


    J = numpy.zeros((cell.nao, cell.nao))
    assert(J.shape[0] == cell.nao)
    assert(J.shape[1] == cell.nao)

    for atomI in range(cell._atm.shape[0]):
        submat = dm[localIdx[atomI]][:,localIdx[atomI]]
        DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
        numpy.add.at(
            J, numpy.ix_(localIdx[atomI], localIdx[atomI]),
            smart_einsum('PQ,PQRS,Rm,Sn->mn', DPQ, VPQRSarray[atomI], F_Pmu[atomI], F_Pmu[atomI])
        )
    # TODO: no errors in this scan, so error solely comes from adding density
    return J*2

def getjSmoothLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    # TODO: generalize to multiple k-points
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    dm = dm[0, 0, :, :]/2
    ###

    J = numpy.zeros((cell.nao, cell.nao))
    assert(J.shape[0] == cell.nao)
    assert(J.shape[1] == cell.nao)

    for atomI in range(cell._atm.shape[0]):
        submat = dm[localIdx[atomI]][:,localIdx[atomI]]
        DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
        DPQtilde = smart_einsum('Pm,Qn,mn->PQ', Ftilde_Pmu[atomI], Ftilde_Pmu[atomI], submat)
        zg = smart_einsum('PQ,PQL->L', DPQ-DPQtilde, M_PQLarr[atomI])

        # smooth RHS
        # smoothel-smoothel
        A = -smart_einsum('PQ,PQRS->RS', DPQtilde, VPQRSarray[atomI])
        # smoothel-smoothcomp
        A += smart_einsum('PQ,PQL,RSL->RS', DPQtilde, V_PQLarr[atomI], M_PQLarr[atomI])
        # comp-smoothel
        A -= smart_einsum('L,RSL->RS', zg, V_PQLarr[atomI])
        # comp-smoothcomp
        A += smart_einsum('L,LM,RSM->RS', zg, V_LMarr[atomI], M_PQLarr[atomI])

        # sharp RHS
        # smoothel-sharpcomp
        B = -smart_einsum('PQ,PQL,RSL->RS', DPQtilde, V_PQLarr[atomI], M_PQLarr[atomI])
        # comp-sharpcomp
        B -= smart_einsum('L,LM,RSM->RS', zg, V_LMarr[atomI], M_PQLarr[atomI])

        numpy.add.at(
            J, numpy.ix_(localIdx[atomI], localIdx[atomI]),
            smart_einsum('RS,Rm,Sn->mn', A, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])+\
            smart_einsum('RS,Rm,Sn->mn', B, F_Pmu[atomI], F_Pmu[atomI])
        )

    return numpy.asarray(J*2)

## below I try to fix the jax problem of getk

# this is the original get k
def getk_PAW_JAX_old(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    print("entering exchange")
    t0 = time.time()

    # TODO: generalize to multiple k-points
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gOnR = PAWdata
    
    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    occ_cut = 1e-12 # TODO: is this reasonable?
    dm, mo_coeff, mo_occ = jnp.array(dm[0, 0, :, :]/2), dm.mo_coeff[0, 0, :, :], dm.mo_occ[0, 0, :]
    occMo = jnp.array(mo_coeff[:,numpy.abs(mo_occ)>occ_cut] * mo_occ[numpy.abs(mo_occ)>occ_cut]/2)
    ###

    nao, nmo = occMo.shape[0], occMo.shape[1]
    print(f'nmo: {nmo}')
    Ng = numpy.prod(mesh)
    f = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    FF = jnp.array(FF)

    Kimu = 0.*occMo.T  ##occRI exchange i, mu


    G = vmap(lambda f, l: f @ occMo[l], (0,0))(F_Pmu, localIdx)             # shape: (atom, P, MO)
    Gtilde = vmap(lambda f, l: f @ occMo[l], (0,0))(Ftilde_Pmu, localIdx)
    natom = localIdx.shape[0]

    start1 = time.time()
    @jit
    def loopOverj(carry, xs):
        occMoj, Gj, Gtildej = xs

        kimu, phi_i, Gi, Gtildei = carry

        phi_j = occMoj @ aoOnR_tilde.T # this is pretty slow
        Rho_ij = phi_i * phi_j

        Rho_ij, _ = lax.scan(updateRho, Rho_ij, (gridIdx, Gi, Gj, Gtildei, Gtildej, M_PQLarr, gOnR)) # looping over atom

        potential_ij = jnp.fft.ifftn( FF * jnp.fft.fftn(Rho_ij.reshape(mesh))).real.flatten()

        kimu = jnp.einsum('r,r,rm->m', potential_ij, phi_j, aoOnR_tilde) * f +\
                kimu #
        carry, _ = lax.scan(updateKimu, (kimu, potential_ij, f), (gridIdx, Gj, Gtildej, M_PQLarr, gOnR, localIdx, F_Pmu, Ftilde_Pmu)) # 19 seconds

        return (carry[0], phi_i, Gi, Gtildei), None

    Gj , Gtildej = jnp.transpose(G, (2,0,1)), jnp.transpose(Gtilde, (2,0,1)) # shape: (MO, atom, P)
    for i in range(nmo):
        phi_i = occMo[:,i] @ aoOnR_tilde.T
        # start10 = time.time()
        carry = (Kimu[i], phi_i, G[:,:,i], Gtilde[:,:,i])
        # test = (Kimu[i], aoOnR_tilde, phi_i, gridIdx, M_PQLarr, gOnR, FF, F_Pmu, Ftilde_Pmu, localIdx, f, G[:,:,i], Gtilde[:,:,i])
        # test = (Kimu[i], aoOnR_tilde, phi_i, gridIdx, M_PQLarr, gOnR, FF, F_Pmu, Ftilde_Pmu, localIdx, f, G[:,:,i], Gtilde[:,:,i])
        # end10 = time.time()

        # start11 = time.time()
        carry, _ = lax.scan(loopOverj, carry, (occMo.T, Gj, Gtildej)) 
        # end11 = time.time()

        start12 = time.time()
        Kimu = Kimu.at[i].set(carry[0]+Kimu[i])
        # end12 = time.time()
        # print(f'matmul, loopoverj, setting', (end10-start10), (end11-start11), (end12-start12), flush=True)
        # print(f'Sum of three: {(end10-start10)+(end11-start11)+(end12-start12)}', flush=True)
    end1 = time.time()
    print(f'PW part: {end1-start1}')

    start2 = time.time()
    K = jnp.zeros((nao,nao))
    @jit
    def localAtomContribution(k, xs):
        fpmu, ftildepmu, localidx, vpqrs, vpql, mpql, vlm = xs
        DPQ      = jnp.einsum('Pm, Qn, mn->PQ',      fpmu,      fpmu, dm[localidx][:,localidx])
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, ftildepmu, dm[localidx][:,localidx])

        # sharp-sharp
        Katom = jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS',vpqrs, DPQ), fpmu)

        ##diffuse-diffuse
        B = jnp.einsum('PQg, RSg->PQRS', vpql, mpql)
        PQRS = vpqrs - B - B.T + jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql) # 4 terms
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde), ftildepmu)

        # mixed terms
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, fpmu, dm[localidx][:,localidx])
        PQRS = B - jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Ktemp   = jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde),  fpmu)
        Katom -= Ktemp + Ktemp.T # 4 terms

        PQRS = jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQ),  fpmu) # 1 term

        return k.at[jnp.ix_(localidx, localidx)].set(
            k[localidx][:, localidx]  + Katom
            ) , None

    K, _ = lax.scan(localAtomContribution, K, (F_Pmu, Ftilde_Pmu, localIdx, VPQRSarray, V_PQLarr, M_PQLarr, V_LMarr)) 
    end2 = time.time()

    K = K + getFullK_fromoccRI(Kimu, occMo, S)
    print("Atom part: ",(end2-start2))

    return numpy.asarray(K*2)


# update Kimu for each i one time instead of j times
def getk_PAW_JAX_new(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    print("entering exchange")
    t0 = time.time()

    # TODO: generalize to multiple k-points
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gOnR = PAWdata
    
    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    occ_cut = 1e-12 # TODO: is this reasonable?
    dm, mo_coeff, mo_occ = jnp.array(dm[0, 0, :, :]/2), dm.mo_coeff[0, 0, :, :], dm.mo_occ[0, 0, :]
    occMo = jnp.array(mo_coeff[:,numpy.abs(mo_occ)>occ_cut] * mo_occ[numpy.abs(mo_occ)>occ_cut]/2)
    ###

    nao, nmo = occMo.shape[0], occMo.shape[1]
    print(f'nmo: {nmo}')
    Ng = numpy.prod(mesh)
    f = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    FF = jnp.array(FF)

    Kimu = 0.*occMo.T  ##occRI exchange i, mu


    G = vmap(lambda f, l: f @ occMo[l], (0,0))(F_Pmu, localIdx)             # shape: (atom, P, MO)
    Gtilde = vmap(lambda f, l: f @ occMo[l], (0,0))(Ftilde_Pmu, localIdx)
    natom = localIdx.shape[0]

    start1 = time.time()
    @jit
    def loopOverj(carry, xs):
        occMoj, Gj, Gtildej = xs

        kimu, phi_i, Gi, Gtildei = carry

        phi_j = occMoj @ aoOnR_tilde.T
        Rho_ij = phi_i * phi_j

        Rho_ij, _ = lax.scan(updateRho, Rho_ij, (gridIdx, Gi, Gj, Gtildei, Gtildej, M_PQLarr, gOnR)) # looping over atom

        potential_ij = jnp.fft.ifftn( FF * jnp.fft.fftn(Rho_ij.reshape(mesh))).real.flatten()

        potential_phi_i = potential_ij*phi_j

        carry, _ = lax.scan(updateKimu, (kimu, potential_ij, f), (gridIdx, Gj, Gtildej, M_PQLarr, gOnR, localIdx, F_Pmu, Ftilde_Pmu))

        return (carry[0], phi_i, Gi, Gtildei), (potential_phi_i,)

    Gj , Gtildej = jnp.transpose(G, (2,0,1)), jnp.transpose(Gtilde, (2,0,1)) # shape: (MO, atom, P)
    for i in range(nmo):
        phi_i = occMo[:,i] @ aoOnR_tilde.T
        start10 = time.time()
        carry = (Kimu[i], phi_i, G[:,:,i], Gtilde[:,:,i])
        # test = (Kimu[i], aoOnR_tilde, phi_i, gridIdx, M_PQLarr, gOnR, FF, F_Pmu, Ftilde_Pmu, localIdx, f, G[:,:,i], Gtilde[:,:,i])
        # test = (Kimu[i], aoOnR_tilde, phi_i, gridIdx, M_PQLarr, gOnR, FF, F_Pmu, Ftilde_Pmu, localIdx, f, G[:,:,i], Gtilde[:,:,i])
        end10 = time.time()

        start11 = time.time()
        carry, ys = lax.scan(loopOverj, carry, (occMo.T, Gj, Gtildej))
        potential_phi_i = jnp.sum(ys[0], axis=0)
        Kimu = Kimu.at[i].set(jnp.einsum('r,rm->m', potential_phi_i, aoOnR_tilde) * f +\
                carry[0])
        end11 = time.time()

        start12 = time.time()
        # Kimu = Kimu.at[i].set(carry[0]+Kimu[i])
        end12 = time.time()
        print(f'matmul, loopoverj, setting', (end10-start10), (end11-start11), (end12-start12), flush=True)
        print(f'Sum of three: {(end10-start10)+(end11-start11)+(end12-start12)}', flush=True)
    end1 = time.time()
    print(f'PW part: {end1-start1}')

    start2 = time.time()
    K = jnp.zeros((nao,nao))
    @jit
    def localAtomContribution(k, xs):
        fpmu, ftildepmu, localidx, vpqrs, vpql, mpql, vlm = xs
        DPQ      = jnp.einsum('Pm, Qn, mn->PQ',      fpmu,      fpmu, dm[localidx][:,localidx])
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, ftildepmu, dm[localidx][:,localidx])

        # sharp-sharp
        Katom = jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS',vpqrs, DPQ), fpmu)

        ##diffuse-diffuse
        B = jnp.einsum('PQg, RSg->PQRS', vpql, mpql)
        PQRS = vpqrs - B - B.T + jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql) # 4 terms
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde), ftildepmu)

        # mixed terms
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, fpmu, dm[localidx][:,localidx])
        PQRS = B - jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Ktemp   = jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde),  fpmu)
        Katom -= Ktemp + Ktemp.T # 4 terms

        PQRS = jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQ),  fpmu) # 1 term

        return k.at[jnp.ix_(localidx, localidx)].set(
            k[localidx][:, localidx]  + Katom
            ) , None

    K, _ = lax.scan(localAtomContribution, K, (F_Pmu, Ftilde_Pmu, localIdx, VPQRSarray, V_PQLarr, M_PQLarr, V_LMarr)) 
    end2 = time.time()

    K = K + getFullK_fromoccRI(Kimu, occMo, S)
    print("Atom part: ",(end2-start2))

    return numpy.asarray(K*2)

# test function for fft only
def getk_PAW_loop(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    print("entering exchange")
    # TODO: rewrite this whole thing in jax
    # aoOnR_tilde = jnp.array(aoOnR_tilde)
    # TODO: generalize to multiple k-points
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gOnR = PAWdata
    
    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    occ_cut = 1e-12 # TODO: is this reasonable?
    dm, mo_coeff, mo_occ = jnp.array(dm[0, 0, :, :]/2), dm.mo_coeff[0, 0, :, :], dm.mo_occ[0, 0, :]
    occMo = jnp.array(mo_coeff[:,numpy.abs(mo_occ)>occ_cut] * mo_occ[numpy.abs(mo_occ)>occ_cut]/2)
    aoOnR_tilde = jnp.array(aoOnR_tilde.T)
    ###

    nao, nmo = occMo.shape[0], occMo.shape[1]
    print(f'nmo: {nmo}')
    Ng = numpy.prod(mesh)
    f = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    FF = jnp.array(FF)

    Kimu = 0.*occMo.T  ##occRI exchange i, mu

    G = vmap(lambda f, l: f @ occMo[l], (0,0))(F_Pmu, localIdx)             # shape: (atom, P, MO)
    Gtilde = vmap(lambda f, l: f @ occMo[l], (0,0))(Ftilde_Pmu, localIdx)
    natom = localIdx.shape[0]

    def loopOverj(carry, xs):
        occMoj, Gj, Gtildej = xs

        kimu, phi_i, Gi, Gtildei = carry

        phi_j = occMoj @ aoOnR_tilde
        Rho_ij = phi_i * phi_j

        Rho_ij, _ = lax.scan(updateRho, Rho_ij, (gridIdx, Gi, Gj, Gtildei, Gtildej, M_PQLarr, gOnR)) # looping over atom

        potential_ij = jnp.fft.ifftn( FF * jnp.fft.fftn(Rho_ij.reshape(mesh))).real.flatten()

        potential_phi_i = potential_ij*phi_j

        carry, _ = lax.scan(updateKimu, (kimu, potential_ij, f), (gridIdx, Gj, Gtildej, M_PQLarr, gOnR, localIdx, F_Pmu, Ftilde_Pmu))

        return (carry[0], phi_i, Gi, Gtildei), (potential_phi_i,)
    
    Gj , Gtildej = jnp.transpose(G, (2,0,1)), jnp.transpose(Gtilde, (2,0,1)) # shape: (MO, atom, P)
    start1 = time.time()
    for i in range(nmo):
        phi_i = occMo[:,i] @ aoOnR_tilde
        carry = (Kimu[i], phi_i, G[:,:,i], Gtilde[:,:,i])

        ys_loop = (numpy.zeros((nmo, Ng)), ) # works for additive things...
        for j in range(nmo):
            carry, y_loop = loopOverj(carry, (occMo.T[j], Gj[j], Gtildej[j]))
            for k in range(len(y_loop)):
                ys_loop[k][j, :] = y_loop[k]
        potential_phi_i = numpy.sum(ys_loop[0], axis=0)
        Kimu = Kimu.at[i].set(jnp.einsum('r,mr->m', potential_phi_i, aoOnR_tilde) * f +\
                carry[0])
    end1 = time.time()
    print(f'PW Part: {end1-start1}')

    start2 = time.time()
    K = jnp.zeros((nao,nao))
    @jit
    def localAtomContribution(k, xs):
        fpmu, ftildepmu, localidx, vpqrs, vpql, mpql, vlm = xs
        DPQ      = jnp.einsum('Pm, Qn, mn->PQ',      fpmu,      fpmu, dm[localidx][:,localidx])
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, ftildepmu, dm[localidx][:,localidx])

        # sharp-sharp
        Katom = jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS',vpqrs, DPQ), fpmu)

        ##diffuse-diffuse
        B = jnp.einsum('PQg, RSg->PQRS', vpql, mpql)
        PQRS = vpqrs - B - B.T + jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql) # 4 terms
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde), ftildepmu)

        # mixed terms
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, fpmu, dm[localidx][:,localidx])
        PQRS = B - jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Ktemp   = jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde),  fpmu)
        Katom -= Ktemp + Ktemp.T # 4 terms

        PQRS = jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQ),  fpmu) # 1 term

        return k.at[jnp.ix_(localidx, localidx)].set(
            k[localidx][:, localidx]  + Katom
            ) , None

    K, _ = lax.scan(localAtomContribution, K, (F_Pmu, Ftilde_Pmu, localIdx, VPQRSarray, V_PQLarr, M_PQLarr, V_LMarr)) 
    end2 = time.time()

    K = K + getFullK_fromoccRI(Kimu, occMo, S)
    print("Atom part: ",(end2-start2))

    return numpy.asarray(K*2)

def getk_PAW_JAX(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    print("entering exchange")
    # TODO: rewrite this whole thing
    # some thing fishy going on that makes matmul very slow
    aoOnR_tilde = jnp.array(aoOnR_tilde.T)
    # TODO: generalize to multiple k-points
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gOnR = PAWdata
    
    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    occ_cut = 1e-12 # TODO: is this reasonable?
    dm, mo_coeff, mo_occ = jnp.array(dm[0, 0, :, :]/2), dm.mo_coeff[0, 0, :, :], dm.mo_occ[0, 0, :]
    occMo = jnp.array(mo_coeff[:,numpy.abs(mo_occ)>occ_cut] * mo_occ[numpy.abs(mo_occ)>occ_cut]/2)
    ###

    nao, nmo = occMo.shape[0], occMo.shape[1]
    print(f'nmo: {nmo}')
    Ng = numpy.prod(mesh)
    f = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    FF = jnp.array(FF)

    Kimu = 0.*occMo.T  ##occRI exchange i, mu


    G = vmap(lambda f, l: f @ occMo[l], (0,0))(F_Pmu, localIdx)             # shape: (atom, P, MO)
    Gtilde = vmap(lambda f, l: f @ occMo[l], (0,0))(Ftilde_Pmu, localIdx)
    natom = localIdx.shape[0]

    start1 = time.time()
    @jit
    def loopOverj(carry, xs):
        occMoj, Gj, Gtildej = xs

        kimu, phi_i, Gi, Gtildei = carry

        phi_j = occMoj @ aoOnR_tilde
        # Rho_ij = phi_i * phi_j

        Rho_ij = phi_j

        potential_phi_i = jnp.fft.ifftn( FF * jnp.fft.fftn(Rho_ij.reshape(mesh))).real.flatten()

        return (kimu, phi_i, Gi, Gtildei), (potential_phi_i,)

    Gj , Gtildej = jnp.transpose(G, (2,0,1)), jnp.transpose(Gtilde, (2,0,1)) # shape: (MO, atom, P)
    for i in range(nmo):
        phi_i = occMo[:,i] @ aoOnR_tilde
        # start10 = time.time()
        carry = (Kimu[i], phi_i, G[:,:,i], Gtilde[:,:,i])
        # test = (Kimu[i], aoOnR_tilde, phi_i, gridIdx, M_PQLarr, gOnR, FF, F_Pmu, Ftilde_Pmu, localIdx, f, G[:,:,i], Gtilde[:,:,i])
        # test = (Kimu[i], aoOnR_tilde, phi_i, gridIdx, M_PQLarr, gOnR, FF, F_Pmu, Ftilde_Pmu, localIdx, f, G[:,:,i], Gtilde[:,:,i])
        # end10 = time.time()

        # start11 = time.time()
        carry, ys = lax.scan(loopOverj, carry, (occMo.T, Gj, Gtildej))
        # potential_phi_i = jnp.sum(ys[0], axis=0)
        # Kimu = Kimu.at[i].set(jnp.einsum('r,rm->m', potential_phi_i, aoOnR_tilde) * f +\
        #         carry[0])
        # end11 = time.time()

        # start12 = time.time()
        # Kimu = Kimu.at[i].set(carry[0]+Kimu[i])
        # end12 = time.time()
        # print(f'matmul, loopoverj, setting', (end10-start10), (end11-start11), (end12-start12), flush=True)
        # print(f'Sum of three: {(end10-start10)+(end11-start11)+(end12-start12)}', flush=True)
    end1 = time.time()
    print(f'matmul(jax): {end1-start1}')

    occMoj = occMo.T[0,:]
    start1 = time.time()
    for _ in range(nmo**2):
        phi_j = occMoj @ aoOnR_tilde
        Rho_ij = phi_j
        # startfft = time.time()
        potential_phi_i = jnp.fft.ifftn( FF * jnp.fft.fftn(Rho_ij.reshape(mesh))).real.flatten()
        # endfft = time.time()
        # print(f'single fft time: {endfft-startfft}')
    end1 = time.time()
    print(f'matmul(numpy): {end1-start1}')

    start2 = time.time()
    K = jnp.zeros((nao,nao))
    @jit
    def localAtomContribution(k, xs):
        fpmu, ftildepmu, localidx, vpqrs, vpql, mpql, vlm = xs
        DPQ      = jnp.einsum('Pm, Qn, mn->PQ',      fpmu,      fpmu, dm[localidx][:,localidx])
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, ftildepmu, dm[localidx][:,localidx])

        # sharp-sharp
        Katom = jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS',vpqrs, DPQ), fpmu)

        ##diffuse-diffuse
        B = jnp.einsum('PQg, RSg->PQRS', vpql, mpql)
        PQRS = vpqrs - B - B.T + jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql) # 4 terms
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde), ftildepmu)

        # mixed terms
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftildepmu, fpmu, dm[localidx][:,localidx])
        PQRS = B - jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Ktemp   = jnp.einsum('Pm,PQ,Qn->mn', ftildepmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQtilde),  fpmu)
        Katom -= Ktemp + Ktemp.T # 4 terms

        PQRS = jnp.einsum('PQg, gf, RSf->PQRS', mpql, vlm, mpql)
        Katom -= jnp.einsum('Pm,PQ,Qn->mn', fpmu, jnp.einsum('PQRS, QR->PS', PQRS, DPQ),  fpmu) # 1 term

        return k.at[jnp.ix_(localidx, localidx)].set(
            k[localidx][:, localidx]  + Katom
            ) , None

    K, _ = lax.scan(localAtomContribution, K, (F_Pmu, Ftilde_Pmu, localIdx, VPQRSarray, V_PQLarr, M_PQLarr, V_LMarr)) 
    end2 = time.time()

    K = K + getFullK_fromoccRI(Kimu, occMo, S)
    print("Atom part: ",(end2-start2))

    return numpy.asarray(K*2)

########


def makeAugmentationSphere(WignerSeitzData, mol, L, alpha0, Rb=None, epsilon=1.e-5):
    ##this is useful for finding points that are close to atom
    ##while taking into account the periodic images
    '''
    use highest L to find the radius of sphere, include all points within the radius, return both overlapped and not overlapped
    '''
    grid2Atom, atomGridDist, Rgrid = WignerSeitzData
    Sharpbas = {}
    for atomI in range(mol._atm.shape[0]):
        elem = mol._atom[atomI][0]
        Sharpbas[elem] = [ [L.max(), [alpha0, 1.]] ]
    print(f'L max is :{L.max()}')

    sharpMol = pgto.M(atom=mol.atom, basis=Sharpbas, a = mol.a, unit=mol.unit) 
    sharpAOonR = sharpMol.pbc_eval_gto('GTOval_sph', Rgrid,
                                       Ls=numpy.zeros((1,3))) # don't include periodic image here
    Sharpalpha, Sharpatoms, SharpL, _ = getAlphaAtomsL(sharpMol._bas, sharpMol._env)

    gridIdx, masked_gridIdx, masks, Rs = [], [], [], []
    for atomI in range(mol._atm.shape[0]):
        idx = numpy.where(numpy.max(numpy.abs(sharpAOonR[:, Sharpatoms==atomI]), axis=1) > epsilon)[0]
        maxR = numpy.max(atomGridDist[atomI, idx]) if Rb==None else Rb
        # print(f'The augmentation radius is: {maxR}')
        allIdx = numpy.where(atomGridDist[atomI, :] < maxR)[0]
        mask = grid2Atom[allIdx] == atomI
        masked_idx = allIdx[mask] # ensure the grids are in the aug sphere

        gridIdx.append(allIdx)
        masked_gridIdx.append(masked_idx)
        masks.append(mask)
        Rs.append(maxR)

    print(f'The max augmentation radius is: {max(Rs)} Bohr')
    
    return gridIdx, masked_gridIdx, masks, Rs


def getPrimIdxFromAOIdx(mol, pmol):
    ao2prim = []
    count = 0
    for bas in mol._bas:
        l = bas[1]
        nprim = bas[2]
        nao = bas[3]

        primIdx = numpy.arange((2*l+1)*nprim) + count
        ao2prim.extend([primIdx for _ in range((2*l+1)*nao)])

        count = primIdx[-1] + 1

    assert(len(ao2prim) == mol.nao)
    assert(ao2prim[-1][-1] == pmol.nao-1)

    return ao2prim

def getContMatFromID(idx, primIdx, ctr_coeff, labels, mol):
    '''
    return N_prim x N_ao matrix
    '''
    Nao = len(idx)
    NPrim = len(primIdx)
    C_mua = numpy.zeros((NPrim, Nao))

    # determine which shell a given AO is in
    binId = numpy.digitize(idx, labels)

    # determine the relative index with in the shell of a given AO
    labels_buff = numpy.append(labels, 0)
    aoIdInBin = idx - labels_buff[binId-1]

    count = 0 # adding AOs one by one
    NPrimStart = 0
    binLast = binId[0]
    for bI, aobI in zip(binId, aoIdInBin):
        binNew = binId[count]
        NPrimStart = NPrimStart if binNew == binLast else NPrimStart + nPrim
        binLast = binNew

        submat = ctr_coeff[bI][:, aobI]
        nPrim = submat.shape[0]
        C_mua[NPrimStart: NPrimStart+nPrim, count] = submat

        count += 1
        
    assert(count == Nao)

    return C_mua

def labelShellWithIdx(ctr_coeff, mol):
    labels = numpy.zeros((len(ctr_coeff),), dtype=int)

    count = 0
    for i, c in enumerate(ctr_coeff):
        count += c.shape[1]
        labels[i] = count

    assert(count == mol.nao)

    return labels

def obtainLocalFns(pmol, mol, ctr_coeff, grids, alpha0, epsilon=1.e-5, Rb=None, Periodic = False, rtol=1e-10):
    '''
    Fit AO and diffuse AO directly, much faster
    set F_Pmu directly to contraction coefficient when P=mu
    '''

    labels = labelShellWithIdx(ctr_coeff, mol)


    BeckeCoords = grids.coords
    alpha, atoms, L, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    atomsAO, _, _ = getAtomsL(mol._bas, mol._env)
    
    print('Making augmentation sphere for Becke grid:')
    WignerSeitzData = makeWignerSeitz(BeckeCoords, mol, Periodic=False) # Becke grid should not be interpreted periodically
    gridIdx = makeAugmentationSphere(WignerSeitzData, mol, L, alpha0, Rb=Rb, epsilon=epsilon)[1]

    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = [], [], [], [], []

    for atomI in range(mol._atm.shape[0]):
        idx = gridIdx[atomI]
        coordsA = BeckeCoords[idx]
        wtsA = grids.weights[idx]

        ##evaluate the value of the functions on atom A grid
        _, aoOnA_tilde = partitionAOs(mol, pmol, coordsA, ctr_coeff, alpha0)

        # fit all local primitives with atom centered primitives
        ACenteredId = numpy.where(atoms == atomI)[0]
        ACenteredAOId = numpy.where(atomsAO == atomI)[0]

        locId = numpy.where(numpy.sum(abs(aoOnA_tilde) > epsilon, axis=0) > 0)[0]
        locId = numpy.union1d(locId, ACenteredAOId)

        idxToFit = numpy.where(~numpy.isin(locId, ACenteredAOId))[0]
        idxToSet = numpy.where(numpy.isin(locId, ACenteredAOId))[0]

        SrP  = pmol.eval_gto('GTOval_sph', coordsA)[:, ACenteredId]
        Sra_tilde = aoOnA_tilde[:, locId][:, idxToFit]

        ##least square minimization which tries to fit the local function in terms of atom centered function
        ## on the local grid
        F_Pmu = numpy.zeros((len(ACenteredId), len(locId)))
        Ftilde_Pmu = numpy.zeros_like(F_Pmu)
        SPQ_inv = numpy.linalg.pinv(smart_einsum('rP,r,rQ->PQ', SrP, wtsA, SrP), rtol=rtol)
        F_fitted = SPQ_inv @ smart_einsum('rP,r,rQ->PQ', SrP, wtsA, Sra_tilde)
        F_Pmu[:, idxToFit] = F_fitted
        Ftilde_Pmu[:, idxToFit] = F_fitted

        # set local, a-centered function directly as contraction coefficient (no fitting)
        diffuseAcenteredPrimMask = alpha[ACenteredId] < alpha0
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
        SArr.append(smart_einsum('rP,r,rQ->PQ', SrP, wtsA, SrP))
        F_PmuArr.append(F_Pmu)
        Ftilde_PmuArr.append(Ftilde_Pmu)

        print(f'a{atomI}: numLoc {F_Pmu.shape[1]}; numPrim {F_Pmu.shape[0]}')

    VPQRSArr = obtainLocal2e(mol, pmol, Periodic)

    return localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr

def smoothCell(mol, alpha0, projAOMol=False):
    smoothMol = mol.copy()
    env2Mod = smoothMol._env.copy()
    for shell in smoothMol._bas:
        # import pdb; pdb.set_trace()
        if projAOMol and shell[0] == mol._atm.shape[0]-1:
            continue
        nprim = shell[2]

        # locate sharp alphas
        alphaStart = shell[-3]
        alphas = smoothMol._env[alphaStart:alphaStart+nprim]
        sharpAlphaId = numpy.where(alphas > alpha0)[0]

        # set sharp contraction coeff to 0
        coeffStart = shell[-2]
        coeffs = smoothMol._env[coeffStart:coeffStart+nprim]
        env2Mod[sharpAlphaId+coeffStart] = 0
        # print(env2Mod)
    smoothMol._env = env2Mod
    return smoothMol

def obtainLocalFnsNew(pmol, mol, ctr_coeff, grids, alpha0, r=2, epsilon=1.e-5, Rb=None, Periodic = False, rtol=1e-10):
    '''
    Fit AO and diffuse AO directly, much faster
    set F_Pmu directly to contraction coefficient when P=mu
    new projectors!!
    '''

    labels = labelShellWithIdx(ctr_coeff, mol)


    # BeckeCoords = grids.coords
    alpha, atoms, L, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    atomsAO, _, _ = getAtomsL(mol._bas, mol._env)

    if hasattr(grids, 'coords'):
        coords = grids.coords
    else:
        coords = grids
    dv = mol.vol/coords.shape[0]
    
    print('Making augmentation sphere for uniform grid:')
    WignerSeitzData = makeWignerSeitz(coords, mol, Periodic=True)
    gridIdx = makeAugmentationSphere(WignerSeitzData, mol, L, alpha0, Rb=Rb, epsilon=epsilon)[0]

    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = [], [], [], [], []

    for atomI in range(mol._atm.shape[0]):
        # fit all local primitives with atom centered primitives
        ACenteredId = numpy.where(atoms == atomI)[0]
        ACenteredAOId = numpy.where(atomsAO == atomI)[0]

        # construct projectors
        numPrim = len(ACenteredId)
        Latom = L[ACenteredId]
        projElem = pmol._atom[atomI][0] + '!' # make sure this doens't coincide with existing elements
        alphaProj = [alpha0*r**i for i in range(numPrim)]
        gmax = Latom.max()
        projBasis = [[l, [alpha, 1.]] for l in range(gmax+1) for alpha in alphaProj]

        # construct proj-prim overlap 
        projPrimBasis = pmol._basis.copy()
        projPrimBasis[projElem] = projBasis
        projPrimAtom = [pmol._atom[atomI], (projElem, pmol._atom[atomI][1])]
        projPrimMol = pgto.M(atom=projPrimAtom, basis=projPrimBasis, a = pmol.lattice_vectors(), unit='B', cart = pmol.cart)
        projPrimOvlp = projPrimMol.pbc_intor('int1e_ovlp')[numPrim:][:, :numPrim]

        # construct proj-ao overlap
        numProj = projPrimOvlp.shape[0]
        projAOBasis = mol._basis.copy()
        projAOBasis[projElem] = projBasis
        projAOAtom = mol._atom.copy()
        # coord = projAOAtom.pop(atomI)[1]
        projAOAtom.append((projElem, projAOAtom[atomI][1])) # adds projector to last so indexing is easy
        projAOMol = pgto.M(atom=projAOAtom, basis=projAOBasis, a=mol.lattice_vectors(), unit='B', cart = mol.cart)
        # projAOMol = smoothCell(projAOMol, alpha0, projAOMol=True)
        projAOOvlp = projAOMol.pbc_intor('int1e_ovlp')[-numProj:][:, :-numProj]

        # determine local functions
        locId = numpy.where(numpy.max(numpy.abs(projAOOvlp), axis=0) > epsilon)[0]
        # print(f'LocId: {locId}')
        assert(numpy.isin(ACenteredAOId, locId).all()) # this must be true
        idxToFit = numpy.where(~numpy.isin(locId, ACenteredAOId))[0]
        idxToSet = numpy.where(numpy.isin(locId, ACenteredAOId))[0]

        # fit the functions
        projAOOvlp = projAOOvlp[:, locId][:, idxToFit]
        F_fitted, residuals, rank, s = numpy.linalg.lstsq(projPrimOvlp, projAOOvlp, rcond=None)
        # print(F_fitted[:, 0])
        # import pdb; pdb.set_trace()

        # update arrays
        F_Pmu = numpy.zeros((len(ACenteredId), len(locId)))
        Ftilde_Pmu = numpy.zeros_like(F_Pmu)
        F_Pmu[:, idxToFit] = F_fitted
        Ftilde_Pmu[:, idxToFit] = F_fitted

        ### check normalization of fitted AO and original AO
        gridOnA = coords[gridIdx[atomI]]
        AOOnA = mol.pbc_eval_gto('GTOval', gridOnA)[:, locId][:, idxToFit]
        primOnA = pmol.pbc_eval_gto('GTOval', gridOnA)[:, ACenteredId]
        fAOOnA = smart_einsum('rP,Pm->rm', primOnA, F_fitted)
        print(AOOnA.sum(axis=0)*dv)
        print(fAOOnA.sum(axis=0)*dv)
        print(numpy.max(numpy.abs(AOOnA.sum(axis=0)-fAOOnA.sum(axis=0))*dv))

        # set local, a-centered function directly as contraction coefficient (no fitting)
        diffuseAcenteredPrimMask = alpha[ACenteredId] < alpha0
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

    VPQRSArr = obtainLocal2e(mol, pmol, Periodic)

    return localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr

def obtainLocalFnsNewer(pmol, mol, ctr_coeff, alpha0, epsilon=1.e-5, Rb=None, Periodic = False, rtol=1e-8):
    '''
    Fit AO and diffuse AO directly, much faster
    set F_Pmu directly to contraction coefficient when P=mu
    new projectors!!
    '''

    labels = labelShellWithIdx(ctr_coeff, mol)

    alpha, atoms, L, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    atomsAO, _, _ = getAtomsL(mol._bas, mol._env)
    
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = [], [], [], [], []

    print('Building projectors...')
    for atomI in range(mol._atm.shape[0]):
        print(f'Atom {atomI}:')
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
            isoPrim = alphaAtomL > alpha0
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
            print(f'L={l}: projExp {alphasProj}')
            projBasis.extend(projBasis4L)


        projElem = pmol._atom[atomI][0] + '!' # make sure this doens't coincide with existing elements

        # construct proj-prim overlap 
        projPrimBasis = pmol._basis.copy()
        projPrimBasis[projElem] = projBasis
        projPrimAtom = [pmol._atom[atomI], (projElem, pmol._atom[atomI][1])]
        projPrimMol = pgto.M(atom=projPrimAtom, basis=projPrimBasis, a = pmol.lattice_vectors(), unit='B', cart = pmol.cart)
        projPrimOvlp = projPrimMol.pbc_intor('int1e_ovlp')[numPrim:][:, :numPrim]

        # isolated projectors
        sharpAlphaId = numpy.where(alphaAtom > alpha0)[0] # could use a different threshold?
        for i in sharpAlphaId:
            projPrimOvlp[i, :] = 0
            projPrimOvlp[:, i] = 0

        # ppoinv = numpy.linalg.pinv(projPrimOvlp, rtol=rtol)
        U, s, Vh = numpy.linalg.svd(projPrimOvlp)
        s_inv = numpy.zeros_like(s)
        s_inv[s>rtol] = 1/s[s>rtol]
        ppoinv = Vh.T@numpy.diag(s_inv)@U.T

        # construct proj-ao overlap
        numProj = projPrimOvlp.shape[0]
        projAOBasis = mol._basis.copy()
        projAOBasis[projElem] = projBasis
        projAOAtom = mol._atom.copy()
        projAOAtom.append((projElem, projAOAtom[atomI][1])) # adds projector to last so indexing is easy
        projAOMol = pgto.M(atom=projAOAtom, basis=projAOBasis, a=mol.lattice_vectors(), unit='B', cart = mol.cart)
        # projAOMol = smoothCell(projAOMol, alpha0, projAOMol=True)
        projAOOvlp = projAOMol.pbc_intor('int1e_ovlp')[-numProj:][:, :-numProj]

        # determine local functions
        locId = numpy.where(numpy.max(numpy.abs(projAOOvlp), axis=0) > epsilon)[0]
        # print(f'LocId: {locId}')
        assert(numpy.isin(ACenteredAOId, locId).all()) # this must be true
        idxToFit = numpy.where(~numpy.isin(locId, ACenteredAOId))[0]
        idxToSet = numpy.where(numpy.isin(locId, ACenteredAOId))[0]

        # fit the functions
        projAOOvlp = projAOOvlp[:, locId][:, idxToFit]
        # F_fitted, residuals, rank, s = numpy.linalg.lstsq(projPrimOvlp, projAOOvlp, rcond=None)
        F_fitted = ppoinv @ projAOOvlp

        # update arrays
        F_Pmu = numpy.zeros((len(ACenteredId), len(locId)))
        Ftilde_Pmu = numpy.zeros_like(F_Pmu)
        F_Pmu[:, idxToFit] = F_fitted
        Ftilde_Pmu[:, idxToFit] = F_fitted

        # set local, a-centered function directly as contraction coefficient (no fitting)
        diffuseAcenteredPrimMask = alpha[ACenteredId] < alpha0
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

    VPQRSArr = obtainLocal2e(mol, pmol, Periodic)

    return localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr

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
        else:
            molAtom = buildPmolAtom(mol, pmol, atomI, pmol.cart)
            VPQRS = pmol.intor('int2e', shls_slice=(idx[0], idx[-1]+1,idx[0], idx[-1]+1,idx[0], idx[-1]+1,idx[0], idx[-1]+1))

        VPQRSWithNuc = numpy.zeros([VPQRS.shape[0]+1]*len(VPQRS.shape))
        VPQRSWithNuc[1:, 1:, 1:, 1:] = VPQRS

        # nuclear
        dfbuilder = _RSNucBuilder(molAtom, kpts=numpy.zeros((1,3))).build()
        VPQRSWithNuc[0, 0, 1:, 1:] = -dfbuilder.get_nuc()[0]/pmol._atm[atomI, 0]
        VPQRSArr.append(VPQRSWithNuc)

        # update calculated
        atomBas.append(atomInfo)
        atomBass[atomZ] = atomBas
        atom2eInt = atom2eInts.get(atomZ, [])
        atom2eInt.append(VPQRSArr[-1])
        atom2eInts[atomZ] = atom2eInt

    return VPQRSArr

def getIntegralDiff(dmol, L, Rgrid, mesh, FF, Ng, f, Periodic=False, diag=True, gridIdx=None):
    '''
    Calculate the difference between analytical and numerical 2c2e integral
    given a mol object
    '''
    aoOnR = dmol.pbc_eval_gto('GTOval', Rgrid)

    # calculate numerical integral
    density_L = aoOnR.T
    potential_L = numpy.fft.ifftn(density_L.reshape(L, *mesh), axes=(-3, -2, -1))
    potential_L = numpy.fft.fftn(potential_L * FF, axes=(-3,-2,-1))
    potential_L = potential_L.real.reshape(L, Ng)

    if gridIdx is not None:
        potential_L = potential_L[gridIdx]
    

    if diag:
        VLL_numeric = smart_einsum('Gr, rG->G', potential_L, aoOnR) * f
    else:
        VLL_numeric = smart_einsum('Gr, rL->GL', potential_L, aoOnR) * f

    # calculate analytical integral
    if not Periodic:
        integral = dmol.intor('int2c2e')
        VLL_analytical = numpy.diag(integral) if diag else integral
    else:
        dfbuilder = pyscf.pbc.df.rsdf_builder._RSGDFBuilder(dmol, dmol).build()
        # not sure if this is the right way to do periodic integral
        j2c = dfbuilder.get_2c2e(numpy.zeros((1, 3)))[0]
        VLL_analytical = numpy.diag(j2c) if diag else j2c

    return numpy.abs(VLL_numeric - VLL_analytical)

def getBoundaryAlphaFromBasis(alpha, pmol, Rgrid, mesh, FF, Ng, f, tol=1e-3, alpha0_cutoff=(1,100), Periodic=False):
    '''
    Return the biggest exponent in the basis set that can be accurately represented
    given Rgrid up to tolerance, also return the next biggest exponent
    '''
    # sort all exponents in ascending order
    alphaSorted = numpy.unique(alpha)
    # the range 1 to 10 should cover all practical alpha's in reality
    alphaSorted = alphaSorted[(alpha0_cutoff[0] < alphaSorted) & (alphaSorted < alpha0_cutoff[1])]

    # initialize dummy atom at the center of unit cell
    center_coords = numpy.sum(pmol.a, axis=0) * 0.5
    atom = f'He {center_coords[0]} {center_coords[1]} {center_coords[2]}'

    # s-type function for each exponent
    bas = {'He': [ [0, [a*2, 1.]] for a in alphaSorted]}
    dmol = pgto.M(atom=atom, basis=bas, a=pmol.a, unit=pmol.unit)

    VLL_diff = getIntegralDiff(dmol, len(alphaSorted), Rgrid, mesh, FF, Ng, f, Periodic=Periodic)

    # determine which exponents are too sharp to be represented by PWs
    sharpIdx = numpy.where(VLL_diff > tol)[0]
    if len(sharpIdx) > 0:
        minSharpIdx = sharpIdx[0]
        assert(len(sharpIdx) == len(alphaSorted)-minSharpIdx) # this should be true ideally
        # assert(sharpIdx[0]+1 == sharpIdx[1])
        maxSoftIdx = minSharpIdx-1 if minSharpIdx !=0 else None
    else:
        minSharpIdx = None
        maxSoftIdx = len(VLL_diff) - 1

    # determine the value of the min sharp exponent and the max soft exponent
    maxSoftAlpha = alpha0_cutoff[0]*2 if maxSoftIdx is None else alphaSorted[maxSoftIdx]*2
    minSharpAlpha = alpha0_cutoff[1]*2 if minSharpIdx is None else alphaSorted[minSharpIdx]*2

    return maxSoftAlpha, minSharpAlpha

def fineGrainAlpha0(maxSoftAlpha, minSharpAlpha, pmol, Rgrid, mesh, FF, Ng, f, tol=1e-3, nsteps=20, Periodic=False):
    '''
    Fine grain the exponent given the boundaries exponents
    We want to use the result of this as our exponent for compensating charge
    '''
    # fine grain the compensating charge exponent
    alphas = numpy.linspace(maxSoftAlpha, minSharpAlpha, nsteps)

    # initialize dummy atom at the center of unit cell
    center_coords = numpy.sum(pmol.a, axis=0) * 0.5
    atom = f'He {center_coords[0]} {center_coords[1]} {center_coords[2]}'

    # s-type function for each exponent
    bas = {'He': [ [0, [a, 1.]] for a in alphas]}
    dmol = pgto.M(atom=atom, basis=bas, a=pmol.a, unit=pmol.unit)
    VLL_diff_fine = getIntegralDiff(dmol, nsteps, Rgrid, mesh, FF, Ng, f, Periodic)
    # pick the biggest one for compensating charge exponent
    softIdx = numpy.where(VLL_diff_fine < tol)[0]
    maxSoftIdx = softIdx[-1] if len(softIdx) > 0 else 0
    alpha0 = alphas[maxSoftIdx]
    alpha0_wf = alpha0 / 2

    return alpha0, alpha0_wf + 1e-5 # just to make sure the edge case works

def getAlpha0(pmol, Rgrid, mesh, Periodic=False, tol=1e-3):
    '''
    Get alpha0 (compensating charge), and the threshold (alpha0_wf) below which
    the GTOs are treated as diffuse (on PWs)
    '''
    alpha, atoms, _, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    FF = getFormFactor(mesh, pmol).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, pmol).reshape(mesh)
    Ng = numpy.prod(mesh)
    f = (pmol.vol/Ng)

    maxSoftAlpha, minSharpAlpha = getBoundaryAlphaFromBasis(alpha, pmol, Rgrid, mesh, FF, Ng, f,
                                                            tol=tol, Periodic=Periodic)
    alpha0, alpha0_wf = fineGrainAlpha0(maxSoftAlpha, minSharpAlpha, pmol, Rgrid, mesh, FF, Ng, f,
                                        tol=tol, Periodic=Periodic)

    print(f'Recommended alpha0 for the given PW cutoff {alpha0}')
    return alpha0, alpha0_wf


def make_natural_orbitals(cell, kpts, dms):
    """
    Construct natural orbitals from density matrix.

    This is a pure mathematical operation that performs eigenvalue decomposition
    of density matrices to obtain natural orbitals and their occupations.

    Parameters:
    -----------
    cell : Cell
        PySCF cell object
    kpts : ndarray
        K-point coordinates
    dms : ndarray
        Density matrices with shape (nset, nk, nao, nao)

    Returns:
    --------
    ndarray
        Tagged density matrix array with mo_coeff and mo_occ attributes
    """
    nk = kpts.shape[0]
    nao = cell.nao
    nset = dms.shape[0]

    # Compute k-point dependent overlap matrices
    sk = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    if abs(dms.imag).max() < 1.0e-6:
        sk = [s.real.astype(numpy.float64) for s in sk]

    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros((nset, nk, nao), numpy.float64)

    for i, dm in enumerate(dms):
        for k, s in enumerate(sk):
            # Diagonalize the DM in AO basis: S^{1/2} * DM * S^{1/2}
            A = lib.reduce(numpy.dot, (s, dm[k], s))
            w, v = scipy.linalg.eigh(A, b=s)

            # Sort eigenvalues/eigenvectors in descending order
            mo_occ[i][k] = numpy.flip(w)
            mo_coeff[i][k] = numpy.flip(v, axis=1)

    return lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)

def modifyMolBasis(bas):
    '''
    uncontract mol.basis in the case of a dictionary
    '''
    result = {}
    for atom, basis in bas.items():
        result[atom] = list()
        for shell in basis:
            if len(shell) > 2:
                # this is a contracted shell
                l = shell[0]
                for pgto in shell[1:]:
                    result[atom].append([l, [pgto[0], 1.]])
            else:
                result[atom].append(shell)
    
    return result

def separateNuclearElectron(pmol, VPQRSArr, M_PQLarr, V_PQLarr, V_LMarr):
    VPQRSArrElec, VPQRSArrNuc = [], []
    M_PQLarrElec, M_PQLarrNuc = [], []
    V_PQLarrElec = []
    ZNucArr = []
    for atomI in range(pmol._atm.shape[0]):
    
        VPQRSArrElec.append(VPQRSArr[atomI][1:, 1:, 1:, 1:])
        VPQRSArrNuc.append(VPQRSArr[atomI][0, 0, 1:, 1:])

        M_PQLarrElec.append(M_PQLarr[atomI][1:, 1:, :])
        M_PQLarrNuc.append(M_PQLarr[atomI][0, 0, :])

        V_PQLarrElec.append(V_PQLarr[atomI][1:, 1:, :])

        ZNucArr.append(-pmol._atm[atomI][0])

    V_LMarrElec = V_LMarr # no new compensating charge is added

    result = (VPQRSArrElec, M_PQLarrElec, V_PQLarrElec, V_LMarrElec,
              VPQRSArrNuc, M_PQLarrNuc, ZNucArr
    )

    return result

def getAuxbasis(pmol):
    auxbas = {}
    for atom, basOnA in pmol._basis.items():
        shellAtom = []
        for i in range(len(basOnA)):
            shell1 = basOnA[i]
            l1, exp1 = shell1[0], shell1[1][0]
            assert(shell1[1][1] == 1) # should be true for primitive mol
            for j in range(i, len(basOnA)):
                shell2 = basOnA[j]
                l2, exp2 = shell2[0], shell2[1][0]
                assert(shell2[1][1] == 1)
                lmax = l1+l2+1
                lmin = numpy.abs(l1-l2)
                shell = [[l, [exp1+exp2, 1.]] for l in range(lmin, lmax)]
                shellAtom.extend(shell)
        auxbas[atom] = shellAtom

    return auxbas

# PAW nuclear
def getnuc_PAW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, mg_ni, Periodic=False, with_multigrid=2):
    # cell, self.mesh, self.aoOnR_tilde, self.PAWdata, self.PAWNucdata, self.Periodic
    # nuc1 = getNucPAWSmoothPW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic=Periodic)
                            #  cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic
    if with_multigrid > 0:
        nuc1 = getNucPAWSmoothPW2(cell, mesh, mg_ni, PAWElecdata, PAWNucdata, Periodic)
    else:
        assert(with_multigrid == 0)
        nuc1 = getNucPAWSmoothPW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic)
    nuc2 = getNucPAWSharpLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic=Periodic)
    nuc3 = getNucPAWSmoothLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic=Periodic)

    return nuc1+nuc2+nuc3


def getNucPAWSmoothPW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic):
    # TODO: generalize to multiple k-points
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    Ng = numpy.prod(mesh)
    f = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    density = numpy.zeros((Ng,))

    for atomI in range(cell._atm.shape[0]):
        update = smart_einsum('g,rg->r', M_PQLarrNuc[atomI], gOnR[atomI])*ZNucArr[atomI]
        numpy.add.at(density, gridIdx[atomI], update)
    potential = numpy.fft.ifftn( FF * numpy.fft.fftn(density.reshape(mesh))).real.flatten()

    nuc = smart_einsum('ra,r,rb->ab', aoOnR_tilde, potential, aoOnR_tilde)*f
    assert(nuc.shape[0] == cell.nao)
    assert(nuc.shape[1] == cell.nao)

    for atomI in range(cell._atm.shape[0]):
        # compnuc-compOnA
        zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*f
        GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
             smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
            -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
        )
    
    return numpy.array([nuc])

def getNucPAWSmoothPW2(cell, mesh, mg_ni, PAWElecdata, PAWNucdata, Periodic):
    # TODO: generalize to multiple k-points
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    Ng = numpy.prod(mesh)
    dv = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    rhoR = numpy.zeros((Ng,))

    for atomI in range(cell._atm.shape[0]):
        update = smart_einsum('g,rg->r', M_PQLarrNuc[atomI], gOnR[atomI])*ZNucArr[atomI]
        numpy.add.at(rhoR, gridIdx[atomI], update)

    rhoG_total = numpy.fft.fftn(rhoR.reshape(mesh))
    vG = rhoG_total * FF
    
    # Pass 2: Grid -> Matrix (Smooth Potential Integration)
    # Note: _get_j_pass2 expects vG * weight
    wv_freq = vG * dv
    nuc = multigrid_pair._get_j_pass2(mg_ni, wv_freq, kpts=numpy.zeros((1,3)), hermi=1)
    while nuc.ndim > 2:
        nuc = nuc[0]

    potential = numpy.fft.ifftn(vG).real.flatten()
    for atomI in range(cell._atm.shape[0]):
        # compnuc-compOnA
        zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*dv
        GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
             smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
            -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
        )
    
    return numpy.array([nuc])

def getNucPAWSharpLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic):
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    nao = cell.nao
    nuc = numpy.zeros((nao, nao))

    for atomI in range(cell._atm.shape[0]):
        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
            smart_einsum('RS,Rm,Sn->mn', VPQRSArrNuc[atomI]*ZNucArr[atomI], F_Pmu[atomI], F_Pmu[atomI])
        )
    
    return numpy.array([nuc])

def getNucPAWSmoothLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic):
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    nao = cell.nao
    nuc = numpy.zeros((nao, nao))

    for atomI in range(cell._atm.shape[0]):
        # smooth RHS
        # compnuc-smoothel
        A = -smart_einsum('g,PQg->PQ', M_PQLarrNuc[atomI], V_PQLarr[atomI])*ZNucArr[atomI]
        # compnuc-smoothcomp
        A += smart_einsum('g, gf, PQf->PQ', M_PQLarrNuc[atomI], V_LMarr[atomI], M_PQLarr[atomI])*ZNucArr[atomI]

        # sharp RHS
        # compnuc-sharpcomp
        B = -smart_einsum('g,gf, PQf->PQ', M_PQLarrNuc[atomI], V_LMarr[atomI], M_PQLarr[atomI])*ZNucArr[atomI]

        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
            smart_einsum('RS,Rm,Sn->mn', A, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])+\
            smart_einsum('RS,Rm,Sn->mn', B, F_Pmu[atomI], F_Pmu[atomI])
        )

    return numpy.array([nuc])

import os

class PAWDF(GDF):
    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        if j_only is not None:
            self._j_only = j_only
        if self.kpts_band is not None:
            self.kpts_band = numpy.reshape(self.kpts_band, (-1,3))
        if kpts_band is not None:
            kpts_band = numpy.reshape(kpts_band, (-1,3))
            if self.kpts_band is None:
                self.kpts_band = kpts_band
            else:
                self.kpts_band = unique(numpy.vstack((self.kpts_band,kpts_band)))[0]

        self.check_sanity()
        self.dump_flags()

        self.auxcell = incore.make_auxcell(self.cell, self.auxbasis)

        if with_j3c and self._cderi_to_save is not None:
            if isinstance(self._cderi_to_save, str):
                cderi = self._cderi_to_save
            else:
                cderi = self._cderi_to_save.name
            if isinstance(self._cderi, str):
                if self._cderi == cderi and os.path.isfile(cderi):
                    logger.warn(self, 'File %s (specified by ._cderi) is '
                                'overwritten by GDF initialization.', cderi)
                    os.remove(cderi)
                else:
                    logger.warn(self, 'Value of ._cderi is ignored. '
                                'DF integrals will be saved in file %s .', cderi)
            self._cderi = cderi
            t1 = (logger.process_clock(), logger.perf_counter())
            self._make_j3c(self.cell, self.auxcell, None, cderi)
            t1 = logger.timer_debug1(self, 'j3c', *t1)
        return self
    

### Jax section

def get_PAW_inUsefulFormForJax(PAWdata):
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    ##whenever tensor are smaller than these then pad tensor with zeros
    maxlocIdx = max([len(loc) for loc in localIdx])
    maxP      = max([F_PmuArr[i].shape[0] for i in range(len(F_PmuArr))])
    maxP_tild = max([Ftilde_PmuArr[i].shape[0] for i in range(len(Ftilde_PmuArr))])
    assert(maxP == maxP_tild)
    maxg      = max([M_PQLarr[i].shape[2] for i in range(len(M_PQLarr))])
    maxGrid   = max([gridIdx[i].shape[0] for i in range(len(gridIdx))])


    natom = len(localIdx)
    localIdxJax = jnp.zeros((natom, maxlocIdx), dtype=int)
    F_PmuJax    = jnp.zeros((natom, maxP, maxlocIdx))
    Ftilde_PmuJax= jnp.zeros((natom, maxP, maxlocIdx))
    M_PQLarrJax = jnp.zeros((natom, maxP, maxP, maxg))
    V_PQLarrJax = jnp.zeros((natom, maxP, maxP, maxg))
    V_LMarrJax  = jnp.zeros((natom, maxg, maxg))
    VPQRSarrayJax = jnp.zeros((natom, maxP, maxP, maxP, maxP))
    gridIdxJax  = jnp.zeros((natom, maxGrid), dtype=int)
    gOnRJax     = jnp.zeros((natom, maxGrid, maxg))

    for i in range(natom):
        row1       = jnp.asarray([i])
        rowIdx     = jnp.arange(len(localIdx[i]))
        rowP       = jnp.arange(F_PmuArr[i].shape[0])
        rowg       = jnp.arange(V_LMarr[i].shape[0])
        rowGrid    = jnp.arange(gOnR[i].shape[0])

        localIdxJax = localIdxJax.at[jnp.ix_(row1, rowIdx)].set(localIdx[i])
        F_PmuJax    = F_PmuJax   .at[jnp.ix_(row1, rowP, rowIdx)].set(F_PmuArr[i])
        Ftilde_PmuJax = Ftilde_PmuJax.at[jnp.ix_(row1, rowP, rowIdx)].set(Ftilde_PmuArr[i])
        M_PQLarrJax = M_PQLarrJax.at[jnp.ix_(row1, rowP, rowP, rowg)].set(M_PQLarr[i])
        V_PQLarrJax = V_PQLarrJax.at[jnp.ix_(row1, rowP, rowP, rowg)].set(V_PQLarr[i])
        V_LMarrJax  = V_LMarrJax .at[jnp.ix_(row1, rowg, rowg)].set(V_LMarr[i])
        VPQRSarrayJax = VPQRSarrayJax.at[jnp.ix_(row1, rowP, rowP, rowP, rowP)].set(VPQRSArr[i])
        gridIdxJax  = gridIdxJax .at[jnp.ix_(row1, rowGrid)].set(gridIdx[i])
        gOnRJax     = gOnRJax    .at[jnp.ix_(row1, rowGrid, rowg)].set(gOnR[i])

        # locDiffIdx = jnp.where(jnp.isin(localIdx[i], localDiffuseIdx[i]))[0]
        #Ftilde_Pmu.append(0.*numpy.array(F_Pmu[i]))
        #Ftilde_Pmu[i][:,locDiffIdx] = F_Pmu[i][:,locDiffIdx]

    return localIdxJax, F_PmuJax, Ftilde_PmuJax, VPQRSarrayJax, M_PQLarrJax, V_PQLarrJax, V_LMarrJax, gridIdxJax, gOnRJax