import itertools
import numpy
import scipy, time

import jax
import jax.numpy as jnp
from jax import vmap, jit, lax
import jax.scipy as jsp
jax.config.update("jax_enable_x64",True)

import pyscf
from pyscf import gto, lib
from pyscf.pbc import gto as pgto

from . import ClebschGordan
import time


def prepareMolForPAW(mol):
    ##move the atoms so they are in the center of the cell
    atomPos = jnp.asarray([mol._atom[i][1] for i in range(len(mol._atom))])
    center = jnp.sum(atomPos, axis=0)/atomPos.shape[0]    


    displacement = mol.lattice_vectors().diagonal()/2. - center

    atomPos = [(atom[0], [(atom[1][0]+displacement[0])*pyscf.data.nist.BOHR, 
                          (atom[1][1]+displacement[1])*pyscf.data.nist.BOHR, 
                          (atom[1][2]+displacement[2])*pyscf.data.nist.BOHR]) for atom in mol._atom]

    if mol.unit[0].capitalize()=='A':
        k=mol.a
    else:
        k=pyscf.data.nist.BOHR*mol.a

    return pgto.M(atom = atomPos, basis = mol.basis, a = k, ke_cutoff = mol.ke_cutoff,unit='A')


def makeWignerSeitz(Rgrid, mol, Periodic=False):
    def makeWignerSeitz4Grid(grid):
        grid = jnp.array(grid)
        atomPos = jnp.asarray([mol._atom[i][1] for i in range(len(mol._atom))])
        distAtom = lambda Grids, pos : jnp.sum( (Grids- pos)**2, axis=-1)**0.5

        atomGridDistance = vmap(distAtom, (None, 0))(grid, atomPos)

        return atomGridDistance

    if Periodic:
        nx = (-1, 0, 1)
        ny = (-1, 0, 1)
        nz = (-1, 0, 1)
        # consider all neighboring unit cells
        prod = list(itertools.product(nx, ny, nz))
        L = mol.lattice_vectors() # in Bohr

        Rgrids = []
        for coeff in prod:
            shift = numpy.dot(coeff, L)
            Rgrid_shifted = Rgrid + shift
            Rgrids.append(Rgrid_shifted)
        Rgrids = numpy.stack(Rgrids) # (27, Ng, 3)

        atomGridDistances = makeWignerSeitz4Grid(Rgrids) # (Na, 27, Ng)

        atomGridDistance = atomGridDistances.min(axis=1) # (Na, Ng)
        idx = atomGridDistances.argmin(axis=1) # (Na, Ng)
        Ng = idx.shape[1]
        # WignerSeitzRgrid = Rgrids[idx, numpy.arange(Ng), :] # (Na, Ng, 3) # turns out we don't need this
        # WignerSeitzRgrid = Rgrid
    else:
        atomGridDistance = makeWignerSeitz4Grid(Rgrid)
        # WignerSeitzRgrid = Rgrid

    closestAtomToGrid = jnp.argmin(atomGridDistance, axis=0)

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

    return alpha, atoms, L, M

@jit
def RadialNorm(alpha, l):
    return 1./2./alpha**(0.5*(l+1.)) * jsp.special.gamma(0.5*(l+1.))

@jit
def basisnorm(alpha, l):
    L = l*2+2
    n = 0.5*(L+1)
    return 1./(1./2./(2.*alpha)**n * jsp.special.gamma(n))**0.5

def compensatingCharge(pmol, mol, alpha0, Rgrid, epsilon, Rb=None, Periodic = False):
    alpha, atoms, L, M = getAlphaAtomsL(pmol._bas, pmol._env)

    print('Making augmentation sphere for uniform grid:')
    WignerSeitzData = makeWignerSeitz(Rgrid, pmol, Periodic=Periodic)
    gridIdx = makeAugmentationSphere(WignerSeitzData, pmol, L, alpha0, Rb=Rb, epsilon=epsilon)[0]

    ##introducing the compensating charge basis set
    gmax = int(L.max()*2)
    gbas = {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        gbas[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]


    # gmol = pgto.M(atom=pmol.atom, basis=gbas, a=pmol.a, cart=Periodic) ##if periodic then cartesian functions
    gmol = pgto.M(atom=pmol.atom, basis=gbas, a=pmol.a, unit=pmol.unit) ##if periodic then cartesian functions
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmol._bas, gmol._env, cart=gmol.cart)


    CG = ClebschGordan.RealCG
    M_PQLarray, V_PQLarray, V_LLarray, gIdx, gOnR = [], [], [], [], []
    for atomI in range(pmol._atm.shape[0]):
        idx  = (atoms==atomI)
        idxg = (atomsG==atomI)
        lmax = numpy.max(L[idx])
        gIdx.append(idxg)

        @jit
        def getmpql(j1, m1, j2, m2, j3, m3, alpha1, alpha2, alpha3) : 
            return CG[(j1*j1 + (m1+j1)), (j2*j2 + (m2+j2)), (j3*j3 + (m3+j3))] * \
                RadialNorm(alpha1 + alpha2, j1+j2+j3+2) * basisnorm(alpha1, j1) * basisnorm(alpha2, j2) /\
                (RadialNorm(alpha3, 2*j3+2) * basisnorm(alpha3, j3))

        M_PQL = vmap(vmap(vmap(getmpql, (None,None,None,None,0,0,None,None,0)), (0,0,None,None,None,None,0,None,None)), (None,None,0,0,None,None,None,0,None)) (L[idx], M[idx], L[idx], M[idx], LG[idxg], MG[idxg], alpha[idx], alpha[idx], alphaG[idxg])
        M_PQLarray.append(M_PQL)

        shellsA, shellsB = numpy.where(pmol._bas[:,0] == atomI)[0], numpy.where(gmol._bas[:,0] == atomI)[0]

        if (not Periodic):
            VPQL = intor_cross('int3c2e', pmol, gmol,  
                                shls_slice=(shellsA[0], shellsA[-1]+1, shellsA[0], shellsA[-1]+1, pmol.nbas + shellsB[0], pmol.nbas + shellsB[-1]+1))
            V_PQLarray.append(VPQL)

            VLL = gmol.intor('int2c2e', shls_slice=(shellsB[0], shellsB[-1]+1,shellsB[0], shellsB[-1]+1))
            V_LLarray.append(VLL)


        else:
            if mol.nao != pmol.nao:
                gmolAtom = pgto.M(atom = [gmol._atom[atomI]], basis = 'unc-'+pmol.basis, a = gmol.a, cart = pmol.cart) 
                # gmolAtom = pgto.M(atom = [gmol._atom[atomI]], basis = 'unc-'+pmol.basis, a = gmol.a, cart = pmol.cart)
            else:
                gmolAtom = pgto.M(atom = [gmol._atom[atomI]], basis = pmol.basis, a = gmol.a, cart = pmol.cart) 
            gmolAtomAux = pgto.M(atom = [gmol._atom[atomI]], basis = gmol.basis, a = gmol.a, cart = gmol.cart) 

            dfbuilder = pyscf.pbc.df.rsdf_builder._RSGDFBuilder(gmolAtom, gmolAtomAux).build()
            j2c = dfbuilder.get_2c2e(numpy.zeros((1, 3)))[0]
            V_LLarray.append(j2c)

            mydf = pyscf.pbc.df.RSDF(gmolAtom)
            mydf.auxbasis = gmol.basis

            eri_3d = numpy.vstack([Lpq[0].copy() for Lpq in mydf.sr_loop(compact=False)])
            eri_3d = jnp.einsum('Pp,PQ->pQ', eri_3d, jnp.linalg.cholesky(j2c, upper=False))
            V_PQLarray.append(eri_3d.reshape((gmolAtom.nao, gmolAtom.nao, gmolAtomAux.nao)))

        gOnR.append(gmol.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[-1]+1)))
        # gOnR.append(gmol.pbc_eval_gto('GTOval', Rgrid[gridIdx[atomI]], shls_slice=(shellsB[0], shellsB[-1]+1)))

        #     mydf = pyscf.pbc.df.MDF(pmol)
        #     nao = pmol.nao
        #     eri = mydf.get_eri(compact=False).reshape((nao,nao,nao,nao))



    return M_PQLarray, V_PQLarray, V_LLarray, gIdx, gridIdx, gmol, gOnR


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

def getE(Hx, dm, CoreH, nuc):
    energy = jnp.einsum('ij,ij',Hx+CoreH, dm)
    energy += nuc
    return energy

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

def get_PAW_inUsefulFormForJax(PAWdata):
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gOnR = PAWdata

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
    gIdxJax     = jnp.zeros((natom, maxg), dtype=int)
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
        gIdxJax     = gIdxJax    .at[jnp.ix_(row1, rowg)].set(jnp.arange(gIdx[i].shape[0])[gIdx[i]])
        VPQRSarrayJax = VPQRSarrayJax.at[jnp.ix_(row1, rowP, rowP, rowP, rowP)].set(VPQRSArr[i])
        gridIdxJax  = gridIdxJax .at[jnp.ix_(row1, rowGrid)].set(gridIdx[i])
        gOnRJax     = gOnRJax    .at[jnp.ix_(row1, rowGrid, rowg)].set(gOnR[i])

        # locDiffIdx = jnp.where(jnp.isin(localIdx[i], localDiffuseIdx[i]))[0]
        #Ftilde_Pmu.append(0.*numpy.array(F_Pmu[i]))
        #Ftilde_Pmu[i][:,locDiffIdx] = F_Pmu[i][:,locDiffIdx]

    return localIdxJax, F_PmuJax, Ftilde_PmuJax, VPQRSarrayJax, M_PQLarrJax, V_PQLarrJax, V_LMarrJax, gIdxJax, gridIdxJax, gOnRJax

def getj_PAW_JAX(cell, dm, gaussgrid, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    start=time.time()
    # TODO: generalize to multiple k-points
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gOnR = PAWdata

    ### the factors of two come from RHF
    # TODO: generalize to beyond RHF
    occ_cut = 1e-12 # TODO: is this reasonable?
    dm, mo_coeff, mo_occ = jnp.array(dm[0, 0, :, :]/2), dm.mo_coeff[0, 0, :, :], dm.mo_occ[0, 0, :]
    occMo = jnp.array(mo_coeff[:,numpy.abs(mo_occ)>occ_cut] * mo_occ[numpy.abs(mo_occ)>occ_cut]/2)
    ###

    Ng = numpy.prod(mesh)
    f = 1.0  # weights absorbed into sparse potential
    Ng=len(aoOnR_tilde)

    def dens(carry, occmoi):
        carry += (occmoi @ aoOnR_tilde.T)**2
        return carry, None


    ##diffuse density
    density = jnp.zeros((Ng,))
    density, _ = lax.scan(dens, density, occMo.T)
    # density = numpy.array(density)

    @jit
    def atomContribution(density, xs):
        f_pmu, ftilde_pmu, m_pql, gonr, idxAtom, grididx = xs
        subMat = dm[idxAtom][:,idxAtom]
        DPQ      = jnp.einsum('Pm, Qn, mn',      f_pmu,      f_pmu, subMat)
        DPQtilde = jnp.einsum('Pm, Qn, mn', ftilde_pmu, ftilde_pmu, subMat)
        zg = jnp.einsum('PQ, PQl->l', DPQ-DPQtilde, m_pql)
        update = jnp.einsum('g,rg->r',zg, gonr) + density[grididx]
        return density.at[grididx].set( update ), None

    ##add the compensating change to the diffuse density
    density, _ = lax.scan(atomContribution, density, (F_Pmu, Ftilde_Pmu, M_PQLarr, gOnR, localIdx, gridIdx))

    # potential = jnp.fft.ifftn( FF * jnp.fft.fftn(density.reshape(mesh))).real.flatten()
    # Using sparse ISDF grid potential solver
    potential = jnp.asarray(gaussgrid.get_potential_sparse(numpy.asarray(density)))

    # Removed the uniform volume element `f` because sparse potential absorbs weights
    J = jnp.einsum('ra,r,rb->ab', aoOnR_tilde, potential, aoOnR_tilde).block_until_ready()
    #jax.debug.print('{x}',x=J)
    assert(J.shape[0] == cell.nao)
    assert(J.shape[1] == cell.nao)

    natom = len(localIdx)

    @jit
    def atomContributionToJ(J, xs):
        f_pmu, ftilde_pmu, m_pql, gonr, idxAtom, grididx, vpqrs, vpql, vlm = xs
        subMat = dm[idxAtom][:,idxAtom]
        DPQ      = jnp.einsum('Pm, Qn, mn->PQ',      f_pmu,      f_pmu, subMat)
        #jax.debug.print('dpq {x}',x=DPQ)
        DPQtilde = jnp.einsum('Pm, Qn, mn->PQ', ftilde_pmu, ftilde_pmu, subMat)
        #jax.debug.print('dpqr {x}',x=DPQtilde)
        zg  = jnp.einsum('PQ, PQl->l', DPQ-DPQtilde, m_pql)
        #jax.debug.print('zg {x}',x=zg)
        zg2 = jnp.einsum('r,rg->g', potential[grididx], gonr)
        #jax.debug.print('zg2 {x}',x=zg2)
        GL  = jnp.einsum('g,PQg', zg2, m_pql)  ##global-local term
        #jax.debug.print('gl {x}',x=GL)

        ##local-local
        A = -jnp.einsum('PQRS, PQ->RS', vpqrs, DPQtilde) + jnp.einsum('PQ, PQg, RSg->RS', DPQtilde, vpql, m_pql)
        #jax.debug.print('a1 {x}',x=A)
        A +=-jnp.einsum('g,PQg->PQ', zg, vpql) + jnp.einsum('g,gf, PQf->PQ', zg, vlm, m_pql)
        #jax.debug.print('a2 {x}',x=A)
        ##gloal-local
        A -= GL
        #jax.debug.print('a3 {x}',x=A)

        B  = jnp.einsum('PQRS, PQ->RS', vpqrs, DPQ) #sharp-sharp
        #jax.debug.print('b1 {x}',x=B)
        B += -jnp.einsum('PQ, PQg, RSg->RS', DPQtilde, vpql, m_pql)
        #jax.debug.print('b2 {x}',x=B)
        B += -jnp.einsum('g,gf, PQf->PQ', zg, vlm, m_pql)
        #jax.debug.print('b3 {x}',x=B)
        ##gloal-local
        B += GL
        #jax.debug.print('b4 {x}',x=B)

        return J.at[jnp.ix_(idxAtom,idxAtom)].set( \
                jnp.einsum('RS, Rm, Sn->nm', B, f_pmu, f_pmu) +
                jnp.einsum('RS, Rm, Sn->nm', A, ftilde_pmu, ftilde_pmu) +
                J[idxAtom][:,idxAtom]), None

    xs = (F_Pmu, Ftilde_Pmu, M_PQLarr, gOnR, localIdx, gridIdx, VPQRSarray, V_PQLarr, V_LMarr)
    J , _ = lax.scan(atomContributionToJ, J, xs)
    #jax.debug.print('{x}',x=J)

    print("Finished J: ",time.time()-start)
    return numpy.asarray(J*2)

## below I try to fix the jax problem of getk

# this is the original get k
def getk_PAW_JAX_old(cell, dm, gaussgrid, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
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
    f = 1.0  # weights absorbed into sparse potential
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

        import jax.experimental
        import numpy
        potential_ij = jax.pure_callback(
            lambda x: numpy.asarray(gaussgrid.get_potential_sparse(numpy.asarray(x))),
            jax.ShapeDtypeStruct(Rho_ij.shape, Rho_ij.dtype),
            Rho_ij
        )

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
        # import pdb; pdb.set_trace()
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
def getk_PAW_JAX_new(cell, dm, gaussgrid, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
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
    f = 1.0  # weights absorbed into sparse potential
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

        import jax.experimental
        import numpy
        potential_ij = jax.pure_callback(
            lambda x: numpy.asarray(gaussgrid.get_potential_sparse(numpy.asarray(x))),
            jax.ShapeDtypeStruct(Rho_ij.shape, Rho_ij.dtype),
            Rho_ij
        )

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
        # import pdb; pdb.set_trace()
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
def getk_PAW_loop(cell, dm, gaussgrid, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
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
    f = 1.0  # weights absorbed into sparse potential
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

        import jax.experimental
        import numpy
        potential_ij = jax.pure_callback(
            lambda x: numpy.asarray(gaussgrid.get_potential_sparse(numpy.asarray(x))),
            jax.ShapeDtypeStruct(Rho_ij.shape, Rho_ij.dtype),
            Rho_ij
        )

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
    # import pdb;pdb.set_trace()

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

def getk_PAW_JAX(cell, dm, gaussgrid, aoOnR_tilde, mesh, PAWdata, S, Periodic = False):
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
    f = 1.0  # weights absorbed into sparse potential
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
        # import pdb; pdb.set_trace()
        # Rho_ij = phi_i * phi_j

        Rho_ij = phi_j

        import jax.experimental
        import numpy
        potential_phi_i = jax.pure_callback(
            lambda x: numpy.asarray(gaussgrid.get_potential_sparse(numpy.asarray(x))),
            jax.ShapeDtypeStruct(Rho_ij.shape, Rho_ij.dtype),
            Rho_ij
        )

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
        # import pdb; pdb.set_trace()
    end1 = time.time()
    print(f'matmul(jax): {end1-start1}')

    occMoj = occMo.T[0,:]
    start1 = time.time()
    for _ in range(nmo**2):
        phi_j = occMoj @ aoOnR_tilde
        Rho_ij = phi_j
        # startfft = time.time()
        import jax.experimental
        import numpy
        potential_phi_i = jax.pure_callback(
            lambda x: numpy.asarray(gaussgrid.get_potential_sparse(numpy.asarray(x))),
            jax.ShapeDtypeStruct(Rho_ij.shape, Rho_ij.dtype),
            Rho_ij
        )
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
        idx = jnp.where(numpy.max(numpy.abs(sharpAOonR[:, Sharpatoms==atomI]), axis=1) > epsilon)[0]
        maxR = jnp.max(atomGridDist[atomI, idx]) if Rb==None else Rb
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

def obtainLocalFns1(pmol, mol, ctr_coeff, grids, alpha0, epsilon=1.e-5, Periodic = False, rtol=1e-10):
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
    gridIdx = makeAugmentationSphere(WignerSeitzData, mol, L, alpha0, epsilon=epsilon)[1]

    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = [], [], [], [], []

    for atomI in range(mol._atm.shape[0]):
        idx = gridIdx[atomI]
        coordsA = BeckeCoords[idx]
        wtsA = grids.weights[idx]

        ##evaluate the value of the functions on atom A grid
        _, aoOnA_tilde = partitionAOs(mol, pmol, coordsA, ctr_coeff, alpha0)

        # fit all local primitives with atom centered primitives
        ACenteredId = jnp.where(atoms == atomI)[0]
        ACenteredAOId = jnp.where(atomsAO == atomI)[0]

        locId = jnp.where(jnp.sum(abs(aoOnA_tilde) > epsilon, axis=0) > 0)[0]
        locId = jnp.union1d(locId, ACenteredAOId)

        idxToFit = jnp.where(~jnp.isin(locId, ACenteredAOId))[0]
        idxToSet = jnp.where(jnp.isin(locId, ACenteredAOId))[0]

        SrP  = pmol.eval_gto('GTOval_sph', coordsA)[:, ACenteredId]
        Sra_tilde = aoOnA_tilde[:, locId][:, idxToFit]

        ##least square minimization which tries to fit the local function in terms of atom centered function
        ## on the local grid
        F_Pmu = jnp.zeros((len(ACenteredId), len(locId)))
        Ftilde_Pmu = jnp.zeros_like(F_Pmu)
        SPQ_inv = jnp.linalg.pinv(jnp.einsum('rP,r,rQ->PQ', SrP, wtsA, SrP), rtol=rtol)
        F_fitted = SPQ_inv @ jnp.einsum('rP,r,rQ->PQ', SrP, wtsA, Sra_tilde)
        F_Pmu = F_Pmu.at[:, idxToFit].set(F_fitted)
        Ftilde_Pmu = Ftilde_Pmu.at[:, idxToFit].set(F_fitted)

        # set local, a-centered function directly as contraction coefficient (no fitting)
        diffuseAcenteredPrimMask = alpha[ACenteredId] < alpha0
        C_Pa = getContMatFromID(ACenteredAOId, ACenteredId, ctr_coeff, labels, mol)
        Ctilde_Pa = jnp.zeros_like(C_Pa)
        Ctilde_Pa = Ctilde_Pa.at[diffuseAcenteredPrimMask, :].set(C_Pa[diffuseAcenteredPrimMask, :])
        assert(C_Pa.shape[0] == F_Pmu.shape[0])
        assert(Ctilde_Pa.shape[0] == Ftilde_Pmu.shape[0])
        assert(numpy.isin(ACenteredAOId, locId).all())
        F_Pmu = F_Pmu.at[:, idxToSet].set(C_Pa)
        Ftilde_Pmu = Ftilde_Pmu.at[:, idxToSet].set(Ctilde_Pa)

        # update array
        localIdx.append(locId)
        SArr.append(jnp.einsum('rP,r,rQ->PQ', SrP, wtsA, SrP))
        F_PmuArr.append(F_Pmu)
        Ftilde_PmuArr.append(Ftilde_Pmu)

        # local 4-index integral
        idx = numpy.where(pmol._bas[:,0] == atomI)[0]

        if Periodic :
            if mol.nao != pmol.nao: # mol uses a contracted basis
                molAtom = pgto.M(atom = [pmol._atom[atomI]], basis = 'unc-'+pmol.basis, a = pmol.a)
            else:
                molAtom = pgto.M(atom = [pmol._atom[atomI]], basis = pmol.basis, a = pmol.a)
            mydf = pyscf.pbc.df.RSDF(molAtom)
            VPQRSArr.append(mydf.get_eri(compact=False).reshape((molAtom.nao, molAtom.nao, molAtom.nao, molAtom.nao)))
        else:
            VPQRSArr.append(pmol.intor('int2e', shls_slice=(idx[0], idx[-1]+1,idx[0], idx[-1]+1,idx[0], idx[-1]+1,idx[0], idx[-1]+1)))
    # import pdb; pdb.set_trace()
    return localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr

def getIntegralDiff(dmol, L, Rgrid, mesh, FF, Ng, f, Periodic=False, diag=True, gridIdx=None):
    '''
    Calculate the difference between analytical and numerical 2c2e integral
    given a mol object
    '''
    aoOnR = dmol.pbc_eval_gto('GTOval', Rgrid)

    # calculate numerical integral
    density_L = aoOnR.T
    potential_L = jnp.fft.ifftn(density_L.reshape(L, *mesh), axes=(-3, -2, -1))
    potential_L = jnp.fft.fftn(potential_L * FF, axes=(-3,-2,-1))
    potential_L = potential_L.real.reshape(L, Ng)

    if gridIdx is not None:
        potential_L = potential_L[gridIdx]
    

    if diag:
        VLL_numeric = jnp.einsum('Gr, rG->G', potential_L, aoOnR) * f
    else:
        VLL_numeric = jnp.einsum('Gr, rL->GL', potential_L, aoOnR) * f

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

def getBoundaryAlphaFromBasis(alpha, pmol, Rgrid, mesh, FF, Ng, f, tol=1e-3, alpha0_cutoff=(1,20), Periodic=False):
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
