import numpy

from pyscf.pbc import gto as pgto
from pyscf import pbc
import pyscf

from PAWutilsNumpy import modifyMolBasis, getAlphaAtomsL, basisnorm

L = 3  # box size

# specify He2
# atom    = f'He 0 0 0'
atom    = f'He 1.5 1.5 1.5'

basis = 'sto3g'
# basis = 'ccpvdz'

verbose = 3
a = numpy.eye(3) * L
ke_cutoff = 1000
precision = 1e-8

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision,
    cart = True,
)
# atom    = f'He 0 0 0'

pcell, _ = cell.decontract_basis()
pcell._basis = modifyMolBasis(pcell._basis)

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

def rsdferi():
    mydf = pyscf.pbc.df.RSDF(pcell)
    mydf.auxbasis = getAuxbasis(pcell)
    mydf.build()
    return mydf.get_eri(compact=False).reshape((pcell.nao, pcell.nao, pcell.nao, pcell.nao))

def fftdferi():
    mydf = pyscf.pbc.df.FFTDF(pcell)
    return mydf.get_eri(compact=False).reshape((pcell.nao, pcell.nao, pcell.nao, pcell.nao))

def neweri():
    auxbas = getAuxbasis(pcell)
    auxcell = pgto.M(atom = [pcell._atom[0]], basis = auxbas, a = pcell.a, cart = pcell.cart)
    dfbuilder = pyscf.pbc.df.rsdf_builder._RSGDFBuilder(pcell, auxcell).build()
    VLM = dfbuilder.get_2c2e(numpy.zeros((1, 3)))[0]
    a2pq = aux2PQ(pcell, auxcell)
    import pdb; pdb.set_trace()
    Naux, NPQ = normalizations(pcell, auxcell)
    VLM = VLM / (Naux[:,None] * Naux[None,:])
    VPQRS = VLM[a2pq[:,:,None,None], a2pq[None,None,:,:]]
    VPQRS = VPQRS * (NPQ[:,:,None,None]*NPQ[None,None,:,:])
    return VPQRS

def aux2PQ(pcell, auxcell):
    # get a PQ matrix with index to aux aos
    alpha, _, L, M = getAlphaAtomsL(pcell._bas, pcell._env, cart=pcell.cart)
    alphaaux, _, Laux, Maux = getAlphaAtomsL(auxcell._bas, auxcell._env, cart=auxcell.cart)

    result = numpy.zeros((pcell.nao, pcell.nao), dtype=numpy.int64)
    for P in range(pcell.nao):
        cartP = CartFromIdx(M[P])
        for Q in range(P, pcell.nao):
            # get corresponding auxcell index for a given shell pair
            alphaPQ = alpha[P] + alpha[Q]
            LPQ = L[P] + L[Q]
            idx = numpy.flatnonzero((Laux==LPQ) & (alphaPQ==alphaaux))
            assert(len(idx) == (LPQ+2)*(LPQ+1)//2)

            cartQ = CartFromIdx(M[Q])
            cartPQ = (cartP[0]+cartQ[0], cartP[1]+cartQ[1], cartP[2]+cartQ[2])
            result[P, Q] = idx[Cartindex(*cartPQ)-tillL(LPQ-1)]
            result[Q, P] = result[P, Q]
    
    # TODO: test this
    # aostring = []
    # for P in range(pmol.nao):
    #     row = []
    #     for Q in range(pmol.nao):
    #         row.append(auxcell.ao_labels()[result[P,Q]])
    #     aostring.append(row)
    # aostring = numpy.array(aostring)
    # import pdb; pdb.set_trace()
    return result

def normalizations(pcell, auxcell):
    # TODO
    alphaaux, _, Laux, _ = getAlphaAtomsL(auxcell._bas, auxcell._env, cart=auxcell.cart)
    Naux = basisnorm(alphaaux, Laux)
    correction = 1/numpy.sqrt(4*numpy.pi) * (Laux==0) + (Laux!=0)
    Naux *= correction
    import pdb; pdb.set_trace()

    alphaP, _, LP, _ = getAlphaAtomsL(pcell._bas, pcell._env, cart=pcell.cart)
    NP = basisnorm(alphaP, LP)
    correction = 1/numpy.sqrt(4*numpy.pi) * (LP==0) + (LP!=0)
    NP *= correction
    NPQ = NP[:, None] * NP[None, :]
    return Naux, NPQ

def test2c2e():
    auxbas = getAuxbasis(pcell)
    auxcell = pgto.M(atom = [pcell._atom[0]], basis = auxbas, a = pcell.a, cart = pcell.cart, ke_cutoff=pcell.ke_cutoff, precision=pcell.precision)

    mesh = pyscf.pbc.tools.cutoff_to_mesh(auxcell.lattice_vectors(), auxcell.ke_cutoff)
    Rgrid = auxcell.get_uniform_grids(mesh=mesh, wrap_around=False)
    dv = auxcell.vol/Rgrid.shape[0]

    # auxcell aos
    aoOnR = auxcell.pbc_eval_gto('GTOval', Rgrid)
    ao0 = aoOnR[:, 0]
    ao2 = aoOnR[:, 2]
    alphaaux, _, Laux, _ = getAlphaAtomsL(auxcell._bas, auxcell._env, auxcell.cart)
    Naux0 = basisnorm(alphaaux[0], Laux[0]) / numpy.sqrt(4*numpy.pi)
    Naux2 = basisnorm(alphaaux[2], Laux[2]) / numpy.sqrt(4*numpy.pi)

    # pcell AOs
    paoOnR = pcell.pbc_eval_gto('GTOval', Rgrid)
    pao0 = paoOnR[:, 0]
    pao2 = paoOnR[:, 2]
    alphaP, _, LP, _  = getAlphaAtomsL(pcell._bas, pcell._env, pcell.cart)
    NP0 = basisnorm(alphaP[0], LP[0]) / numpy.sqrt(4*numpy.pi)
    NP2 = basisnorm(alphaP[2], LP[2]) / numpy.sqrt(4*numpy.pi)
    import pdb; pdb.set_trace()

    # this should be the same
    ao2dnorm = ao2 / Naux2
    pao0dnorm = pao0 / NP0
    pao2dnorm = pao2 / NP2
    print(ao2dnorm - pao0dnorm*pao2dnorm)
    print(numpy.max(numpy.abs(ao2dnorm - pao0dnorm*pao2dnorm)))


    def get_Gv(nmesh,reciprocal_vecs):
        rx = numpy.fft.fftfreq(nmesh[0], 1./nmesh[0])
        ry = numpy.fft.fftfreq(nmesh[1], 1./nmesh[1])
        rz = numpy.fft.fftfreq(nmesh[2], 1./nmesh[2])
        return numpy.dot(pyscf.lib.cartesian_prod((rx,ry,rz)), reciprocal_vecs).astype(numpy.float64)

    def getFormFactor(nmesh,cell):
        G2 = get_Gv(nmesh,cell.reciprocal_vectors())**2
        G2 = numpy.sum( G2, axis = 1)
        FF = numpy.zeros((G2.shape[0]), numpy.float64)
        idx = numpy.greater( G2, 0.)
        FF[idx] = 4. * numpy.pi /G2[idx]
        return FF

    # (0|2)
    dfbuilder = pyscf.pbc.df.rsdf_builder._RSGDFBuilder(pcell, auxcell).build()
    VLM = dfbuilder.get_2c2e(numpy.zeros((1, 3)))[0]

    FF = getFormFactor(mesh, auxcell).reshape(mesh)
    v0 = numpy.fft.ifftn((numpy.fft.fftn((ao0).reshape(mesh)) * FF)).flatten()
    v02 = (numpy.dot(ao2, v0)*dv).real
    print(numpy.abs(VLM[0,2]-v02))

    # convert to (00|02)
    VPQRS = fftdferi()
    print(numpy.abs(v02/(Naux0*Naux2)*(NP0**3*NP2) - VPQRS[0,0,0,2]))




## these might be useful
tillL = lambda l : int( (l+1) * (l+2) * (l+3)//6 )

def Cartindex( x, y, z ): 
    L = x+y+z
    return tillL(L-1) + (L-x) * (L-x+1) // 2 + (L-x-y)

def CartFromIdx(I):
    L = 0
    while I >= tillL(L):
        L += 1

    for x in range(L,-1,-1):
        for y in range(L-x,-1,-1):
            if Cartindex(x, y, L-x-y) == I:
                return (x,y,L-x-y)

def CartCartProd(maxL):

    C = numpy.zeros((tillL(maxL/2), tillL(maxL/2), tillL(maxL)))
    for i in range(tillL(maxL//2)):
        for j in range(tillL(maxL//2)):
            Cart1, Cart2 = CartFromIdx(i), CartFromIdx(j)
            outCart = (Cart1[0] + Cart2[0], Cart1[1] + Cart2[1], Cart1[2] + Cart2[2])
            C[i,j, Cartindex( *outCart)] = 1.

    return C

###

def main():
    test2c2e()
    eri_fft = fftdferi()
    # eri_rsdf = rsdferi()
    # print(numpy.max(numpy.abs(eri_fft-eri_rsdf)))
    # import pdb; pdb.set_trace()
    eri_new = neweri()
    print(numpy.max(numpy.abs(eri_fft-eri_new)))
    import pdb; pdb.set_trace()
    # aux2PQ()

if __name__ == '__main__':
    main()