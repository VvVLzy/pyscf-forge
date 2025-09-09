import numpy
import time
from pyscf import lib
from pyscf.lib import logger
from . import PAWutils
# from pyscf.paw import PAWutils
import pyscf

def get_jk(mol, dm, S, aoOnR_tilde, mesh, PAWdata,
           hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None):
    '''Compute J, K matrices for all input density matrices

    Args:
        mol : an instance of :class:`Mole`

        dm : ndarray or list of ndarrays
            A density matrix or a list of density matrices

    Kwargs:
        hermi : int
            Whether J, K matrix is hermitian

            | 0 : not hermitian and not symmetric
            | 1 : hermitian or symmetric
            | 2 : anti-hermitian

        vhfopt :
            A class which holds precomputed quantities to optimize the
            computation of J, K matrices

        with_j : boolean
            Whether to compute J matrices

        with_k : boolean
            Whether to compute K matrices

        omega : float
            Parameter of range-separated Coulomb operator.
            When omega is 0 (or None), integrals are computed with the full-range Coulomb potential.
            When it is larger than zero, integrals are evaluated with the long-range
            Coulomb potential erf( omega * r12 ) / r12. When omega is smaller
            than 0, short-range Coulomb potential erfc( omega * r12 ) / r12 is applied.

    Returns:
        Depending on the given dm, the function returns one J and one K matrix,
        or a list of J matrices and a list of K matrices, corresponding to the
        input density matrices.

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf.scf import _vhf
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> dms = numpy.random.random((3,mol.nao_nr(),mol.nao_nr()))
    >>> j, k = scf.hf.get_jk(mol, dms, hermi=0)
    >>> print(j.shape)
    (3, 2, 2)
    '''
    dm = numpy.asarray(dm, order='C')
    dm_shape = dm.shape
    dm_dtype = dm.dtype
    nao = dm_shape[-1]

    if dm_dtype == numpy.complex128:
        dm = numpy.vstack((dm.real, dm.imag)).reshape(-1,nao,nao)
        hermi = 0

    with mol.with_range_coulomb(omega):
        vj = vk = None
        if with_j:
            vj = PAWutils.getj_PAW_JAX(mol, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False)
        if with_k:
            vk = PAWutils.getk_PAW_JAX(mol, dm, aoOnR_tilde, mesh, PAWdata, S, Periodic = False)

    if dm_dtype == numpy.complex128:
        if with_j:
            vj = vj.reshape((2,) + dm_shape)
            vj = vj[0] + vj[1] * 1j
        if with_k:
            vk = vk.reshape((2,) + dm_shape)
            vk = vk[0] + vk[1] * 1j
    else:
        if with_j:
            vj = vj.reshape(dm_shape)
        if with_k:
            vk = vk.reshape(dm_shape)
    return vj, vk

def getPAWdata(mol, PAWorbitalCutOff=1.e-5, PWAccuracy=1e-4, printLevel = 1, Periodic = False, alpha0=None):

    mol.build()
    if (not Periodic):
        mol                          = PAWutils.prepareMolForPAW(mol)
    # uncontract basis
    pmol, ctr_coeff = mol.decontract_basis()

    # initialize grids
    mf = pyscf.scf.RKS(pmol)
    mf.grids.build()
    mesh = pyscf.pbc.tools.cutoff_to_mesh(pmol.lattice_vectors(), pmol.ke_cutoff)
    Rgrid = pmol.get_uniform_grids(mesh=mesh, wrap_around=False)

    # get alpha0
    if alpha0 is None:
        alpha0, alpha0_wf = PAWutils.getAlpha0(pmol, Rgrid, mesh, Periodic=Periodic, tol=PWAccuracy)
    else:
        _, _ = PAWutils.getAlpha0(pmol, Rgrid, mesh, Periodic=Periodic, tol=PWAccuracy)

    alpha0_wf = alpha0 # it seems that this works better in practice

    # PAWData
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = PAWutils.obtainLocalFns1(pmol, mol, ctr_coeff, mf.grids, alpha0_wf, epsilon=PAWorbitalCutOff, Periodic=Periodic, rtol=1e-9)
    M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gmol, gOnR    = PAWutils.compensatingCharge(pmol, alpha0,  Rgrid, PAWorbitalCutOff, Periodic = Periodic) 

    # Evaluate AOs on uniform grid
    aoOnR, aoOnR_tilde = PAWutils.partitionAOs(mol, pmol, Rgrid, ctr_coeff, alpha0_wf)
    gOnRAll = gmol.pbc_eval_gto('GTOval', Rgrid)
    

    # JAX version of PAWdata
    PAWdata = (localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gOnR)
    PAWdataJAX = PAWutils.get_PAW_inUsefulFormForJax(PAWdata)

    

    if (printLevel > 0):
        # print ("Rb          : {0:<10.2f}".format(Rb))
        print ("alpha0      : {0:<10.2f}".format(alpha0))
        print ("alpha0_wf      : {0:<10.2f}".format(alpha0_wf))
    return PAWdataJAX, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, pmol

Times_ ={"Diagonalize":0.,
         "Exchange"   :0.,
         "Direct"     :0.,
         "1e-orbs"    :0.,
         "PAWinit"    :0.,
         "AOs"        :0.,
         "Fock"       :0.}

class PAWSCF(pyscf.scf.hf.SCF):
    def __init__(self, mol,
                 printLevel,
                 PAWorbitalCutOff,
                 PWAccuracy,
                 alpha0):
        super().__init__(mol)

        self.printLevel = printLevel
        self.PAWorbitalCutOff = PAWorbitalCutOff
        self.PWAccuracy = PWAccuracy
        self.Periodic = False
        self.alpha0 = alpha0
        
        self.initPAW()

    def initPAW(self):
        t0 = time.time()
        PAWdata, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, pmol = getPAWdata(self.mol,
                                                                            printLevel=self.printLevel,
                                                                            PAWorbitalCutOff=self.PAWorbitalCutOff,
                                                                            PWAccuracy=self.PWAccuracy,
                                                                            Periodic=self.Periodic,
                                                                            alpha0=self.alpha0)

        
        Times_["PAWinit"] += time.time()-t0

        nelec, nao = self.mol.nelectron, self.mol.nao
        # nocc = nelec//2
        # madelung = pyscf.pbc.tools.pbc.madelung(self.mol, self.mol.make_kpts([1,1,1])) if self.Periodic else 0.

        mf = pyscf.pbc.scf.RHF(self.mol).rs_density_fit() if self.Periodic else pyscf.scf.RHF(pyscf.gto.M(atom = self.mol.atom, basis = self.mol.basis))

        t0 = time.time()
        # hcore = mf.get_hcore().reshape((nao,nao))
        S = mf.get_ovlp().reshape((nao,nao))
        # X = get_transformation_matrix(S)
        # nuc = mf.energy_nuc()
        Times_["1e-orbs"] += time.time()-t0

        self.S = S
        self.aoOnR_tilde = aoOnR_tilde
        self.mesh = mesh
        self.PAWdata = PAWdata
        
        if (self.printLevel > 0):
            print ("Nelection   : {0:<10d}".format(nelec))
            print ("Ngrid points: {0:<10d}".format(numpy.prod(self.mol.mesh)))
            print ("delta-a     : {0:<10.2f}".format((self.mol.vol/numpy.prod(self.mol.mesh))**(1./3.)))


    @lib.with_doc(get_jk.__doc__)
    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if self.direct_scf and self._opt.get(omega) is None:
            # Be careful that opt has to be initialized with a proper setting of
            # omega. opt of regular ERI and SR ERI are incompatible since cint 5.4.0
            with mol.with_range_coulomb(omega):
                self._opt[omega] = self.init_direct_scf(mol)
        vhfopt = self._opt.get(omega)

        if with_j and with_k:
            vj, vk = get_jk(mol, dm, self.S, self.aoOnR_tilde, self.mesh, self.PAWdata,
                            hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None)
        else:
            if with_j:
                prescreen = 'CVHFnrs8_vj_prescreen'
            else:
                prescreen = 'CVHFnrs8_vk_prescreen'
            with lib.temporary_env(vhfopt, prescreen=prescreen):
                vj, vk = get_jk(mol, dm, self.S, self.aoOnR_tilde, self.mesh, self.PAWdata,
                                hermi=1, vhfopt=None, with_j=True, with_k=True, omega=None)

        logger.timer(self, 'vj and vk', *cpu0)
        return vj, vk