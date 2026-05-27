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

def makeWignerSeitz1(Rgrid, mol, Periodic=False):
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

def obtainLocalFns(pmol, mol, ctr_coeff, grids, alpha0, epsilon=1.e-5, Rb=None, Periodic = False, rtol=1e-10):
    '''
    Fit AO and diffuse AO directly, much faster
    set F_Pmu directly to contraction coefficient when P=mu
    '''

    labels = labelShellWithIdx(ctr_coeff, mol)


    BeckeCoords = grids.coords
    alpha, atoms, L, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    atomsAO, _, _ = getAtomsL(mol._bas, mol._env)
    
    logger.debug(mol, 'Making augmentation sphere for Becke grid:')
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

        logger.debug(mol, 'a%d: numLoc %d; numPrim %d', atomI, F_Pmu.shape[1], F_Pmu.shape[0])

    VPQRSArr = obtainLocal2e(mol, pmol, Periodic)

    return localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr

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
    
    logger.debug(mol, 'Making augmentation sphere for uniform grid:')
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
        logger.debug(mol, 'a%d AOOnA sum: %s', atomI, AOOnA.sum(axis=0)*dv)
        logger.debug(mol, 'a%d fAOOnA sum: %s', atomI, fAOOnA.sum(axis=0)*dv)
        logger.debug(mol, 'a%d max diff: %s', atomI, numpy.max(numpy.abs(AOOnA.sum(axis=0)-fAOOnA.sum(axis=0))*dv))

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

## K related

@jit
def updateRho(rho, xs):
    grididx, gi, gj, gtildei, gtildej, mpql, gonr = xs
    return rho.at[grididx].set(
        rho[grididx] +
        jnp.einsum('P, Q, PQg->g', gi, gj, mpql) @ gonr.T  -
        jnp.einsum('P, Q, PQg->g', gtildei, gtildej, mpql) @ gonr.T ) , None
@jit
def updateKimu(carry, xs):
    grididx, g, gtilde, mpql, gonr, localidx, fpmu, fpmutilde = xs
    K, potential_ij, f = carry
    zg = jnp.einsum('r,rg->g', potential_ij[grididx], gonr) * f
    A  = jnp.einsum('P,PQg,g->Q', g,  mpql, zg)
    B  = jnp.einsum('P,PQg,g->Q', gtilde, mpql, zg)
    return (K.at[localidx].set(K[localidx] + fpmu.T @ A - fpmutilde.T @ B), potential_ij, f), None

def getFullK_fromoccRI(Kimu, occMO, S):
    Kij = Kimu @ occMO

    SC = S @ occMO
    K = SC @ Kimu
    return (K + K.T - SC @ Kij @ SC.T)

## below I try to fix the jax problem of getk

# this is the original get k
def getk_PAW_JAX_old(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    logger.debug(cell, "entering exchange")
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
    logger.debug(cell, 'nmo: %d', nmo)
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
    logger.debug(cell, 'PW part: %s', end1-start1)

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
    logger.debug(cell, "Atom part: %s", end2-start2)

    return numpy.asarray(K*2)


# update Kimu for each i one time instead of j times
def getk_PAW_JAX_new(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    logger.debug(cell, "entering exchange")
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
    logger.debug(cell, 'nmo: %d', nmo)
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
        logger.debug(cell, 'matmul, loopoverj, setting %s %s %s', end10-start10, end11-start11, end12-start12)
        logger.debug(cell, 'Sum of three: %s', (end10-start10)+(end11-start11)+(end12-start12))
    end1 = time.time()
    logger.debug(cell, 'PW part: %s', end1-start1)

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
    logger.debug(cell, "Atom part: %s", end2-start2)

    return numpy.asarray(K*2)

# test function for fft only
def getk_PAW_loop(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    logger.debug(cell, "entering exchange")
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
    logger.debug(cell, 'nmo: %d', nmo)
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
    logger.debug(cell, 'PW Part: %s', end1-start1)

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
    logger.debug(cell, "Atom part: %s", end2-start2)

    return numpy.asarray(K*2)

def getk_PAW_JAX(cell, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
    logger.debug(cell, "entering exchange")
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
    logger.debug(cell, 'nmo: %d', nmo)
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
    logger.debug(cell, 'matmul(jax): %s', end1-start1)

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
    logger.debug(cell, 'matmul(numpy): %s', end1-start1)

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
    logger.debug(cell, "Atom part: %s", end2-start2)

    return numpy.asarray(K*2)

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
