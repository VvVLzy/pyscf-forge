from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pyscf import lib
from pyscf import __config__
from pyscf.df import df_jk
from pyscf.pbc.df.gdf_builder import _CCNucBuilder
from pyscf.pbc.df.rsdf_builder import _RSNucBuilder

import pyscf
import numpy

from . import PAWutils

import time


def getPAWdata(mol,
               PAWorbitalCutOff=1.e-5,
               PWAccuracy=1e-5,
               printLevel = 1,
               Periodic = False,
               alpha0=None,
               augRadius=None):

    mol.build()
    if (not Periodic):
        mol                          = PAWutils.prepareMolForPAW(mol)
    # mol                          = PAWutils.prepareMolForPAW(mol) # do this regardless give better result?
    # uncontract basis
    pmol, ctr_coeff = mol.decontract_basis()
    print(pmol._atom)

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
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = PAWutils.obtainLocalFns1(
        pmol, mol, ctr_coeff, mf.grids, alpha0_wf, epsilon=PAWorbitalCutOff, Periodic=Periodic, rtol=1e-9)
    M_PQLarr, V_PQLarr, V_LMarr, gIdx, gridIdx, gmol, gOnR    = PAWutils.compensatingCharge(
        pmol, mol, alpha0, Rgrid, PAWorbitalCutOff, Rb=augRadius, Periodic = Periodic)

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
    return PAWdataJAX, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, mol

def tag_dm(mydf, dm, cell, kpts, nk, nao):
    if mydf.scf_iter == 0:
        dm = numpy.asarray(dm)
    if getattr(dm, 'mo_coeff', None) is None:
        dm = PAWutils.make_natural_orbitals(cell, kpts,
                                            dm.reshape(-1, nk, nao, nao))
    else:
        mo_coeff = numpy.asarray(dm.mo_coeff).reshape(-1, nk, nao, nao)
        mo_occ = numpy.asarray(dm.mo_occ).reshape(-1, nk, nao)
        dm = lib.tag_array(dm.reshape(-1, nk, nao, nao), mo_coeff=mo_coeff, mo_occ=mo_occ)
    return dm

def get_jk_periodic(mydf, dm, hermi=1, kpts=None, kpts_band=None,
                    with_j=True, with_k=True, omega=None, exxdiv=None):

    # TODO: this might be problematic
    if omega is not None:  # J/K for RSH functionals
        with mydf.range_coulomb(omega) as rsh_df:
            return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                    omega=None, exxdiv=exxdiv)

    kpts, is_single_kpt = _check_kpts(mydf, kpts)

    # recreate occMo from DM if not available
    cell = mydf.cell
    if isinstance(dm, list):
        dm = numpy.asarray(dm)
    nk = mydf.kpts.shape[0] # what is kpt default for no kpts (kpt=None)?
    nao = cell.nao
    # if with_k:
    if with_j or with_k:
        dm = tag_dm(mydf, dm, cell, kpts, nk, nao)

    if is_single_kpt:
        # TODO: test paw jk here
        vj = vk = None
        if with_j:
            vj = PAWutils.getj_PAW_JAX(cell, dm, 
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        Periodic=mydf.Periodic)
        if with_k:
            vk = PAWutils.getk_PAW_JAX(cell, dm,
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        mydf.S,
                                        Periodic=mydf.Periodic)
    else:
        # TODO: implement this
        raise NotImplementedError('get J and K for kpts not implemented.')
        vj = vk = None
        if with_k:
            vk = fft_jk.get_k_kpts(mydf, dm, hermi, kpts, kpts_band, exxdiv)
        if with_j:
            vj = fft_jk.get_j_kpts(mydf, dm, hermi, kpts, kpts_band)

    mydf.scf_iter += 1

    return vj, vk

def get_jk_molecule(mydf, dm, hermi=1, with_j=True, with_k=True,
           direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
           omega=None):
    # TODO: this might be problematic
    # A temporary treatment for RSH-DF integrals
    if omega is not None:
        with mydf.range_coulomb(omega) as rsh_df:
            return df_jk.get_jk(rsh_df, dm, hermi, with_j, with_k, direct_scf_tol)

    kpts, is_single_kpt = _check_kpts(mydf, mydf.kpts)

    assert(is_single_kpt) # only makes sense for molecular case

    # recreate occMo from DM if not available
    cell = mydf.cell
    if isinstance(dm, list):
        dm = numpy.asarray(dm)
    nk = kpts.shape[0]
    nao = cell.nao
    # if with_k:
    if with_j or with_k:
        dm = tag_dm(mydf, dm, cell, kpts, nk, nao)

    vj = vk = None
    if with_j:
        vj = PAWutils.getj_PAW_JAX(cell, dm, 
                                    mydf.aoOnR_tilde,
                                    mydf.mesh,
                                    mydf.PAWdata,
                                    Periodic=mydf.Periodic)
    if with_k:
        # TODO: get k here has some problems
        # different nmo than GAPW
        vk = PAWutils.getk_PAW_loop(cell, dm,
                                    mydf.aoOnR_tilde,
                                    mydf.mesh,
                                    mydf.PAWdata,
                                    mydf.S,
                                    Periodic=mydf.Periodic)
    return vj, vk

class PAW(FFTDF):
    def __init__(
            self,
            cell,
            kpts=None,
            printLevel=1,
            PAWorbitalCutOff=1e-5,
            PWAccuracy=1e-5,
            Periodic=False,
            alpha0=None,
            augRadius=None,
    ):
        
        self.scf_iter = 0
        self.cell = cell

        # not sure what this part do
        if kpts is None:
            self.kpts = numpy.zeros(3, numpy.float64)
            self.kmesh = [1, 1, 1]
        else:
            self.kmesh = kpts_to_kmesh(self.cell, kpts, precision=None, rcut=None)
            self.kpts = self.cell.make_kpts(
                self.kmesh,
                space_group_symmetry=False,
                time_reversal_symmetry=False,
                wrap_around=True,
            )
        super().__init__(cell=self.cell, kpts=self.kpts)

        # PAW init
        self.printLevel = printLevel
        self.PAWorbitalCutOff = PAWorbitalCutOff
        self.PWAccuracy = PWAccuracy
        self.Periodic = Periodic
        self.alpha0 = alpha0
        self.augRadius = augRadius
        self.Times_ = {
            "Diagonalize":0.,
            "Exchange"   :0.,
            "Direct"     :0.,
            "1e-orbs"    :0.,
            "PAWinit"    :0.,
            "AOs"        :0.,
            "Fock"       :0.
        }

        self.initPAW(cell)

        

        # update get_jk
        if self.Periodic:
            self.get_jk = get_jk_periodic.__get__(self, self.__class__)
        else:
            self.get_jk = get_jk_molecule.__get__(self, self.__class__)
        ###

    def initPAW(self, cell):
        t0 = time.time()
        PAWdata, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, cell = getPAWdata(
            cell,
            printLevel=self.printLevel,
            PAWorbitalCutOff=self.PAWorbitalCutOff,
            PWAccuracy=self.PWAccuracy,
            Periodic=self.Periodic,
            alpha0=self.alpha0,
            augRadius=self.augRadius
        )

        
        self.Times_["PAWinit"] += time.time()-t0

        nelec, nao = cell.nelectron, cell.nao
        # nocc = nelec//2
        # madelung = pyscf.pbc.tools.pbc.madelung(self.cell, self.cell.make_kpts([1,1,1])) if self.Periodic else 0.

        mf = pyscf.pbc.scf.RHF(cell).rs_density_fit() if self.Periodic else pyscf.scf.RHF(pyscf.gto.M(atom = cell.atom, basis = cell.basis))

        t0 = time.time()
        # hcore = mf.get_hcore().reshape((nao,nao))
        S = mf.get_ovlp().reshape((nao,nao))
        # X = get_transformation_matrix(S)
        # nuc = mf.energy_nuc()
        self.Times_["1e-orbs"] += time.time()-t0

        self.S = S
        self.aoOnR_tilde = aoOnR_tilde
        self.mesh = mesh
        self.PAWdata = PAWdata
        
        if (self.printLevel > 0):
            print ("Nelection   : {0:<10d}".format(nelec))
            print ("Ngrid points: {0:<10d}".format(numpy.prod(self.cell.mesh)))
            print ("delta-a     : {0:<10.2f}".format((self.cell.vol/numpy.prod(self.cell.mesh))**(1./3.)))

    def get_nuc(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        # if self._prefer_ccdf or cell.omega > 0:
        #     # For long-range integrals _CCGDFBuilder is the only option
        #     dfbuilder = _CCNucBuilder(cell, kpts).build()
        # else:
        #     dfbuilder = _RSNucBuilder(cell, kpts).build()
        dfbuilder = _RSNucBuilder(cell, kpts).build()
        nuc = dfbuilder.get_nuc()
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    

    @classmethod
    def from_mf(cls, mf, cell=None, **kwargs):
        """Create OCCRI instance from mean-field object

        Parameters
        ----------
        mf : pyscf mean-field object
            Mean-field instance (RHF, UHF, RKS, UKS, etc.)
        disable_c : bool, optional
            If True, use pure Python implementation
        **kwargs
            Additional arguments passed to OCCRI constructor

        Returns
        -------
        OCCRI
            OCCRI density fitting instance configured for the given mean-field methods


        Example
        -------
        from pyscf.pbc import gto, scf
        from pyscf.occri import OCCRI

        # Set up cell
        cell = gto.Cell()
        cell.atom = 'H 0 0 0; H 0 0 1'
        cell.basis = 'sto3g'
        cell.build()

        # Create mean-field object
        mf = scf.RHF(cell)

        # Use factory method to create OCCRI
        mf.with_df = OCCRI.from_mf(mf)

        # Run calculation
        energy = mf.kernel()

        Alternative direct construction:

        # You can also create OCCRI directly if you have cell and kpts
        # However! It will default to the incore algo if memory is sufficient
        occri = OCCRI(cell, kpts=None)  # Direct construction
        mf.with_df = occri
        """
        # Validate mean-field instance
        mf._is_mem_enough = lambda: False

        # Extract method information
        method = mf.__module__.rsplit('.', 1)[-1]
        # assert method in [
        #     'hf',
        #     'uhf',
        #     'khf',
        #     'kuhf',
        #     'rks',
        #     'uks',
        #     'krks',
        #     'kuks',
        # ], f'Unsupported mean-field method: {method}'

        # Create OCCRI instance
        if getattr(mf, 'cell', False):
            # Periodic
            assert(cell is None)
            occri = cls(mf.cell, mf.kpts, Periodic=True, **kwargs) ## change this
        else:
            assert(getattr(mf, 'mol'))
            assert(cell is not None)
            # molecular
            occri = cls(cell, Periodic=False, **kwargs)
        occri.method = method

        # cell will be modified for molecular case
        

        return occri