import numpy
import pyscf
import scipy
from DF import PAWDF

from pyscf.pbc import gto as pgto

# cell params
L = 10
x = L/2
ke_cutoff = 500

# primitive pmol (contain AOs)
alpha1 = 9 # sharp alpha
pbasis = {'He': [[0, [alpha1, 1.]], [1, [alpha1, 1.]]]}

# spherical
pmol_sph = pgto.M(
    atom = f'He {x} {x} {x}',
    basis = pbasis,
    ke_cutoff=ke_cutoff,
    a = numpy.eye(3)*L,
    cart = False
)

# cartesian
pmol_cart = pgto.M(
    atom = f'He {x} {x} {x}',
    basis = pbasis,
    ke_cutoff=ke_cutoff,
    a = numpy.eye(3)*L,
    cart = True
)

# compensating pmol
alpha2 = 4 # compensating alpha
cbasis = {'He': [[0, [alpha2, 1.]], [2, [alpha2, 1.]]]}

# cartesian
cmol = pgto.M(
    atom = pmol_cart.atom,
    basis = cbasis,
    ke_cutoff=pmol_cart.ke_cutoff,
    a = numpy.eye(3)*L,
    cart = True,
)

# spherical
smol = pgto.M(
    atom = pmol_sph.atom,
    basis = cbasis,
    ke_cutoff=pmol_sph.ke_cutoff,
    a = numpy.eye(3)*L,
    cart = False,
)

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

mesh = pyscf.pbc.tools.cutoff_to_mesh(pmol_sph.lattice_vectors(), pmol_sph.ke_cutoff)
Rgrid = pmol_sph.get_uniform_grids(mesh=mesh, wrap_around=False)
dv = pmol_sph.vol/Rgrid.shape[0]
FF = getFormFactor(mesh, pmol_sph).reshape(mesh)

def getVPQL(pmol, gmol):
    # (g|g')
    mydf = PAWDF(pmol)
    # mydf = pyscf.pbc.df.GDF(pmol)
    mydf.auxbasis = gmol.basis
    mydf.build()
    dfbuilder = pyscf.pbc.df.rsdf_builder._RSGDFBuilder(pmol, gmol).build()
    j2c = dfbuilder.get_2c2e(numpy.zeros((1, 3)))[0]
    j2c_cd, j2c_negative, j2ctag = dfbuilder.decompose_j2c(j2c)
    assert(j2c_negative is None)
    assert(j2ctag == 'CD')

    # (PQ|g)
    # TODO: gamma point only
    eri_3d = numpy.vstack([Lpq[0].copy() for Lpq in mydf.sr_loop(compact=False)])
    eri_3d = numpy.einsum('LM, Mp -> pL', j2c_cd, eri_3d)
    # eri_3d = numpy.einsum('Pp,PQ->pQ', eri_3d, numpy.linalg.cholesky(j2c, upper=False))
    vpql = eri_3d.reshape((pmol.nao, pmol.nao, gmol.nao))
    # eri_3d = numpy.einsum('Pp,PQ->pQ', eri_3d, numpy.linalg.cholesky(j2c, upper=False))
    # eri_3d = numpy.einsum('PQ,Qp->pP', numpy.linalg.cholesky(j2c, upper=False), eri_3d)
    # vpql = eri_3d.reshape((pmol.nao, pmol.nao, gmol.nao))
    # eri_3d1 = numpy.vstack([Lpq[1].copy() for Lpq in mydf.sr_loop(compact=False)])
    # import pdb; pdb.set_trace()
    # eri_3d = numpy.vstack([Lpq.copy() for Lpq in mydf.loop()])
    # eri_3d = numpy.einsum('Pp,PQ->pQ', eri_3d, numpy.linalg.cholesky(j2c, upper=False))
    # eri_3d = numpy.transpose(eri_3d.reshape(gmol.nao, pmol.nao, pmol.nao), (1,2,0))
    # eri_from_3d = numpy.einsum('PQL,RSL->PQRS', eri_3d, eri_3d)
    # eri = mydf.get_eri(compact=False).reshape([pmol.nao]*4)
    # print(numpy.max(numpy.abs(eri_from_3d-eri)))

    return vpql, j2c

def gammaBar(n, l1, l2, alpha1, alpha2):
    L = n + 2 + l1 + l2
    LL = (L+1)/2
    a = alpha1 + alpha2
    return scipy.special.gamma(LL) / (2*a**(LL))

VPQL_cart, LL_cart = getVPQL(pmol_cart, cmol)
print(VPQL_cart.shape)

VPQL_sph, LL_sph = getVPQL(pmol_sph, smol)
print(VPQL_sph.shape)

# eval aos
aoOnR_pmol_sph = pmol_sph.pbc_eval_gto('GTOval', Rgrid)
aoOnR_pmol_cart = pmol_cart.pbc_eval_gto('GTOval', Rgrid)
aoOnR_smol = smol.pbc_eval_gto('GTOval', Rgrid)
aoOnR_cmol = cmol.pbc_eval_gto('GTOval', Rgrid)

pmol_s00_cart   = aoOnR_pmol_cart[:, 0]
cmol_xx_cart    = aoOnR_cmol[:, 1]
cmol_s00_cart   = aoOnR_cmol[:, 0]
# cmol_s00_cart   = aoOnR_cmol[:, 0]*7.14595073/(aoOnR_cmol[:, 0].sum() * dv)

# calculate (0 | 0) by hand
print('2c2e integral from DF is correct')
print(LL_cart[0, 0])
v_0_cart = numpy.fft.ifftn((numpy.fft.fftn((cmol_s00_cart).reshape(mesh)) * FF)).flatten()
print((numpy.dot(v_0_cart, cmol_s00_cart)*dv).real)

# calculate (0 0 | 0) by hand
# N = gammaBar(0, 0, 0, alpha2, 0)* numpy.sqrt(4*numpy.pi) / numpy.sqrt(gammaBar(0, 0, 0, alpha2, alpha2)) 
# N1 = 7.1459507339040185/5.092958178940651
print('Examine first entry in (PQ|L), that is (0 0 | 0)')
print(f'DF without correction: {VPQL_cart[0, 0, 0]}')
v_s00_cart = numpy.fft.ifftn((numpy.fft.fftn((pmol_s00_cart**2).reshape(mesh)) * FF)).flatten()
print(f'Hand calculate (FFTDF): {(numpy.dot(v_s00_cart, cmol_s00_cart)*dv).real}')

# calculate (0 0 | 1) by hand
# N = gammaBar(0, 0, 0, alpha2, 0)* numpy.sqrt(4*numpy.pi) / numpy.sqrt(gammaBar(0, 0, 0, alpha2, alpha2)) 
# N1 = 7.1459507339040185/5.092958178940651
print('Examine first entry in (PQ|L), that is (0 0 | 1)')
print(f'DF without correction: {VPQL_cart[0, 0, 1]}')
v_s00_cart = numpy.fft.ifftn((numpy.fft.fftn((pmol_s00_cart**2).reshape(mesh)) * FF)).flatten()
print(f'Hand calculate (FFTDF): {(numpy.dot(v_s00_cart, cmol_xx_cart)*dv).real}')

# # calculate (0 0 | x^2) by hand
# print(VPQL_cart[:, :, 1])
# v_s00_cart = numpy.fft.ifftn((numpy.fft.fftn((pmol_s00_cart**2).reshape(mesh)) * FF)).flatten()
# print(numpy.dot(v_s00_cart, cmol_xx_cart)*dv)