import numpy
import scipy

import itertools

from pyscf.pbc import gto as pgto
from pyscf import pbc
import pyscf

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
                if not pmol.cart:
                    lmax = l1+l2+1
                    lmin = numpy.abs(l1-l2)
                    shell = [[l, [exp1+exp2, 1.]] for l in range(lmin, lmax)]
                else:
                    shell = [[l1+l2, [exp1+exp2, 1.]]]
                shellAtom.extend(shell)
        auxbas[atom] = shellAtom

    return auxbas

tillL = lambda l : int( (l+1) * (l+2) * (l+3)//6 )

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

def basisnorm(alpha, l):
    L = l*2+2
    n = 0.5*(L+1)
    return 1./(1./2./(2.*alpha)**n * scipy.special.gamma(n))**0.5

# def pbc_eval_exp(mol, alpha, Rgrid, periodicImg=0):
#     nx = [i for i in range(-periodicImg, periodicImg+1)]
#     ny = [i for i in range(-periodicImg, periodicImg+1)]
#     nz = [i for i in range(-periodicImg, periodicImg+1)]
#     # consider all neighboring unit cells
#     prod = list(itertools.product(nx, ny, nz))
#     L = mol.lattice_vectors() # in Bohr

#     result = numpy.zeros((Rgrid.shape[0]))
#     for coeff in prod:
#         shift = numpy.dot(coeff, L)
#         Rgrid_shifted = Rgrid + shift
#         r2 = numpy.sum(Rgrid_shifted**2, axis=-1)
#         result += numpy.exp(-alpha * r2)

#     return result

def pbc_eval_exp(mol, alpha, Rgrid, periodicImg=0):
    # 1. Generate all image coefficients (nx, ny, nz) as an (M, 3) array
    rng = numpy.arange(-periodicImg, periodicImg + 1)
    # Using numpy.mgrid is faster than itertools for creating the coordinate grid
    coeffs = numpy.stack(numpy.meshgrid(rng, rng, rng), axis=-1).reshape(-1, 3)
    
    # 2. Convert coefficients to spatial shifts: (M, 3)
    L = mol.lattice_vectors()
    shifts = coeffs @ L  # Matrix multiply to get actual Bohr shifts
    
    # 3. Precompute squared terms for the expansion (R+S)^2 = R^2 + S^2 + 2RS
    # R2: (N, 1), S2: (1, M), RS: (N, M)
    R2 = numpy.sum(Rgrid**2, axis=-1, keepdims=True)
    S2 = numpy.sum(shifts**2, axis=-1, keepdims=True).T
    RS = Rgrid @ shifts.T
    
    # 4. Combine terms into the exponent
    # r2_all has shape (N, M) representing distance to every image for every point
    r2_all = R2 + S2 + 2 * RS
    
    # 5. Calculate and sum across the image dimension (axis 1)
    result = numpy.sum(numpy.exp(-alpha * r2_all), axis=1)
    
    return result

L = 3  # box size

# specify He2
atom    = f'He 0 0 0'
# atom    = f'He 1.5 1.5 1.5'

basis = 'unc-sto3g'
# basis = 'ccpvdz'

verbose = 3
a = numpy.eye(3) * L
ke_cutoff = 2000
precision = 1e-8

pcell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision,
    cart = True,
)

auxbas = getAuxbasis(pcell)
auxcell = pgto.M(
    atom = atom,
    # atom = [pcell._atom[0]], # TODO: unit inconsistency here!
    basis = auxbas,
    a = pcell.a,
    ke_cutoff=pcell.ke_cutoff,
    precision=pcell.precision,
    cart = pcell.cart,
)

print(f'pcell basis: {pcell._basis}')
print(f'auxcell: {auxcell._basis}')

mesh = pyscf.pbc.tools.cutoff_to_mesh(auxcell.lattice_vectors(), auxcell.ke_cutoff)
Rgrid = auxcell.get_uniform_grids(mesh=mesh, wrap_around=False)
dv = auxcell.vol/Rgrid.shape[0]

# pcell AOs
paoOnR = pcell.pbc_eval_gto('GTOval', Rgrid)
pao0 = paoOnR[:, 0]
pao2 = paoOnR[:, 2]
alphaP, _, LP, _  = getAlphaAtomsL(pcell._bas, pcell._env, pcell.cart)
NP0 = basisnorm(alphaP[0], LP[0]) / numpy.sqrt(4*numpy.pi)
NP2 = basisnorm(alphaP[2], LP[2]) / numpy.sqrt(4*numpy.pi)

# auxcell aos
aoOnR = auxcell.pbc_eval_gto('GTOval', Rgrid)
ao0 = aoOnR[:, 0]
ao2 = aoOnR[:, 2]
alphaaux, _, Laux, _ = getAlphaAtomsL(auxcell._bas, auxcell._env, auxcell.cart)
Naux0 = basisnorm(alphaaux[0], Laux[0]) / numpy.sqrt(4*numpy.pi)
Naux2 = basisnorm(alphaaux[2], Laux[2]) / numpy.sqrt(4*numpy.pi)


# product->evalongrid should be the same as evalongrid->product
# 0x0 = 0 -> make sense
ao0dnorm = ao0 / Naux0
pao0dnorm = pao0 / NP0
print(numpy.max(numpy.abs(pao0dnorm - pbc_eval_exp(pcell, alphaP[0], Rgrid, 1))))
print('###########################')
print('Test case 1')
print(ao0dnorm - pao0dnorm*pao0dnorm)
print(numpy.max(numpy.abs(ao0dnorm - pao0dnorm*pao0dnorm))) # densities are same
print(numpy.min(numpy.abs(ao0dnorm - pao0dnorm*pao0dnorm)))
print(numpy.abs((ao0dnorm - pao0dnorm*pao0dnorm).sum()*dv)) # sum over unitcell also same
print('###########################')

# 0x2 != 2 -> why??
ao2dnorm = ao2 / Naux2
pao0dnorm = pao0 / NP0
pao2dnorm = pao2 / NP2
print(numpy.max(numpy.abs(pao0dnorm - pbc_eval_exp(pcell, alphaP[0], Rgrid, 1))))
print(numpy.max(numpy.abs(pao2dnorm - pbc_eval_exp(pcell, alphaP[2], Rgrid, 2))))
print(numpy.max(numpy.abs(pao0dnorm*pao2dnorm - pbc_eval_exp(pcell, alphaP[0], Rgrid, 2)*pbc_eval_exp(pcell, alphaP[2], Rgrid, 2))))
print(numpy.max(numpy.abs(pao0dnorm*pao2dnorm - pbc_eval_exp(pcell, alphaP[0]*alphaP[2]/(alphaP[0]+alphaP[2]), Rgrid, 2))))
import pdb; pdb.set_trace()
print('###########################')
print('Test case 2')
print(ao2dnorm - pao0dnorm*pao2dnorm)
print(numpy.max(numpy.abs(ao2dnorm - pao0dnorm*pao2dnorm))) # densities are different
print(numpy.min(numpy.abs(ao2dnorm - pao0dnorm*pao2dnorm)))
print(numpy.abs((ao2dnorm - pao0dnorm*pao2dnorm).sum()*dv)) # sum over unitcell also different