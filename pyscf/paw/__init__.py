from pyscf.pbc.df.fft import FFTDF
from pyscf.pbc.df.aft import _check_kpts
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh
from pyscf.pbc import gto as pgto
from pyscf import lib
from pyscf import __config__
from pyscf.df import df_jk
from pyscf.pbc.df.gdf_builder import _CCNucBuilder
from pyscf.pbc.df.rsdf_builder import _RSNucBuilder
from pyscf.lib import logger

from .vxc import smoothCell2
from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridNumInt as MultiGridNumInt2
from pyscf.pbc.dft.multigrid import MultiGridNumInt

import pyscf
import numpy

from . import PAWutils

import time


def getPAWdataNew(mol,
               gaussgrid,
               PAWorbitalCutOff=1.e-5,
               PWAccuracy=1e-5,
               Periodic = False,
               alpha0=None,
               augRadius=None,
               with_multigrid=2,
               alpha0_lowmem=True,
               auto_box=False):
    mol.build()
    if (not Periodic):
        mol                          = PAWutils.prepareMolForPAW(mol, auto_box=auto_box)

    # uncontract basis
    pmol, ctr_coeff = mol.decontract_basis()
    pmol._basis = PAWutils.modifyMolBasis(pmol._basis)

    # initialize grids
    mf = pyscf.scf.RKS(pmol)
    # mf.grids.level = 9
    #mf.grids.build()
    #print(f'DFT grid size: {mf.grids.coords.shape[0]}')
    #mesh = pyscf.pbc.tools.cutoff_to_mesh(pmol.lattice_vectors(), pmol.ke_cutoff)
    #Rgrid = pmol.get_uniform_grids(mesh=mesh, wrap_around=False)
    #TODO just hacking gmol in here early to set up isdf but this could be done better
    alpha, atoms, L, M = PAWutils.getAlphaAtomsL(pmol._bas, pmol._env)
    gmax = int(L.max()*2)
    gbas = {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        gbas[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]

    gmol = pgto.M(atom=pmol.atom, basis=gbas, a=pmol.a, unit=pmol.unit) ##if periodic then cartesian functions
    gaussgrid.setup_isdf(gmol=gmol)
    Rgrid=gaussgrid.get_sparse_grid()
    mesh = (Rgrid.shape[0],)
    # mf.grids.build()

    # get alpha0
    if alpha0_lowmem:
        # Mini-cell implementation: Use a small box with same resolution (ke_cutoff)
        test_cell = pgto.M(
            atom='He 5 5 5',
            basis=pmol.basis,
            a=numpy.eye(3) * 10.0,
            unit='B',
            ke_cutoff=pmol.ke_cutoff,
            verbose=0
        )
        t_pmol, _ = test_cell.decontract_basis()
        t_pmol._basis = PAWutils.modifyMolBasis(t_pmol._basis)
        t_mesh = t_pmol.mesh
        t_Rgrid = t_pmol.get_uniform_grids(mesh=t_mesh)
        if alpha0 is None:
            alpha0, alpha0_wf = PAWutils.getAlpha0(t_pmol, t_Rgrid, t_mesh, Periodic=Periodic, tol=PWAccuracy)
        else:
            _, _ = PAWutils.getAlpha0(t_pmol, t_Rgrid, t_mesh, Periodic=Periodic, tol=PWAccuracy)
            alpha0_wf = alpha0/2
    else:
        if alpha0 is None:
            alpha0, alpha0_wf = PAWutils.getAlpha0(pmol, Rgrid, mesh, Periodic=Periodic, tol=PWAccuracy)
        else:
            #_, _ = PAWutils.getAlpha0(pmol, Rgrid, mesh, Periodic=Periodic, tol=PWAccuracy)
            alpha0_wf = alpha0/2

    alpha0_wf = alpha0 # must be this!

    # PAWData
    M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR, gmol, Rs = PAWutils.mergeCompensatingCharge(
        pmol, mol, alpha0, Rgrid, PAWorbitalCutOff, Rb=augRadius, Periodic = Periodic)
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = PAWutils.obtainLocalFnsNewer(
        pmol, mol, ctr_coeff, alpha0, epsilon=PAWorbitalCutOff, Rb=numpy.max(Rs), Periodic = Periodic, rtol=1e-8)

    # Evaluate AOs on uniform grid
    start = time.time()
    if with_multigrid == 0: # store AO incore to speed up J
        #aoOnR, aoOnR_tilde = PAWutils.partitionAOs(mol, pmol, Rgrid, ctr_coeff, alpha0_wf)
        #No longer needed for gausslets
        aoOnR, aoOnR_tilde = 0, 0
    else:
        aoOnR, aoOnR_tilde = 0, 0
    logger.info(mol, 'AO eval on all grids take %.2e seconds', time.time() - start)
    gOnRAll = None

    # Prepare PAW data
    result = PAWutils.separateNuclearElectron(
        pmol, VPQRSArr, M_PQLarr, V_PQLarr, V_LMarr)
    PAWdata = (localIdx, F_PmuArr, Ftilde_PmuArr, *result[:4], gridIdx, gOnR)
    PAWNucdata = result[-3:]

    return PAWdata, PAWNucdata, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, mol, pmol, gmol, ctr_coeff, mf.grids, alpha0, Rs

def getPAWdata(mol,
               PAWorbitalCutOff=1.e-8,
               PWAccuracy=1e-5,
               Periodic = False,
               alpha0=None,
               augRadius=None,
               with_multigrid=2,
               alpha0_lowmem=True,
               auto_box=False):
    raise ValueError()
    mol.build()
    if (not Periodic):
        mol                          = PAWutils.prepareMolForPAW(mol, auto_box=auto_box)

    # uncontract basis
    pmol, ctr_coeff = mol.decontract_basis()
    pmol._basis = PAWutils.modifyMolBasis(pmol._basis)

    # initialize grids
    mf = pyscf.scf.RKS(pmol)
    # mf.grids.level = 9
    # mf.grids.build()
    # logger.debug(mol, 'DFT grid size: %d', mf.grids.coords.shape[0])
    mesh = pyscf.pbc.tools.cutoff_to_mesh(pmol.lattice_vectors(), pmol.ke_cutoff)
    Rgrid = pmol.get_uniform_grids(mesh=mesh, wrap_around=False)

    # get alpha0
    if alpha0_lowmem:
        # Mini-cell implementation: Use a small box with same resolution (ke_cutoff)
        test_cell = pgto.M(
            atom='He 5 5 5',
            basis=pmol.basis,
            a=numpy.eye(3) * 10.0,
            unit='B',
            ke_cutoff=pmol.ke_cutoff,
            verbose=0
        )
        t_pmol, _ = test_cell.decontract_basis()
        t_pmol._basis = PAWutils.modifyMolBasis(t_pmol._basis)
        t_mesh = t_pmol.mesh
        t_Rgrid = t_pmol.get_uniform_grids(mesh=t_mesh)
        if alpha0 is None:
            alpha0, alpha0_wf = PAWutils.getAlpha0(t_pmol, t_Rgrid, t_mesh, Periodic=Periodic, tol=PWAccuracy)
        else:
            _, _ = PAWutils.getAlpha0(t_pmol, t_Rgrid, t_mesh, Periodic=Periodic, tol=PWAccuracy)
            alpha0_wf = alpha0/2
    else:
        if alpha0 is None:
            alpha0, alpha0_wf = PAWutils.getAlpha0(pmol, Rgrid, mesh, Periodic=Periodic, tol=PWAccuracy)
        else:
            _, _ = PAWutils.getAlpha0(pmol, Rgrid, mesh, Periodic=Periodic, tol=PWAccuracy)
            alpha0_wf = alpha0/2

    alpha0_wf = alpha0 # must be this!

    # PAWData
    M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR, gmol, Rs = PAWutils.compensatingCharge(
        pmol, mol, alpha0, Rgrid, PAWorbitalCutOff, Rb=augRadius, Periodic = Periodic)
    localIdx, F_PmuArr, Ftilde_PmuArr, VPQRSArr, SArr = PAWutils.obtainLocalFnsNewer(
        pmol, mol, ctr_coeff, alpha0, epsilon=PAWorbitalCutOff, Rb=numpy.max(Rs), Periodic = Periodic, rtol=1e-8)

    # Evaluate AOs on uniform grid
    start = time.time()
    if with_multigrid == 0: # store AO incore to speed up J
        aoOnR, aoOnR_tilde = PAWutils.partitionAOs(mol, pmol, Rgrid, ctr_coeff, alpha0_wf)
    else:
        aoOnR, aoOnR_tilde = 0, 0
    logger.info(mol, 'AO eval on all grids take %.2e seconds', time.time() - start)
    gOnRAll = None

    # Prepare PAW data
    result = PAWutils.separateNuclearElectron(
        pmol, VPQRSArr, M_PQLarr, V_PQLarr, V_LMarr)
    PAWdata = (localIdx, F_PmuArr, Ftilde_PmuArr, *result[:4], gridIdx, gOnR)
    PAWNucdata = result[-3:]

    return PAWdata, PAWNucdata, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, mol, pmol, gmol, ctr_coeff, mf.grids, alpha0, Rs

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
            cpu0 = (logger.process_clock(), logger.perf_counter())
            if mydf.with_multigrid > 0:
                raise ValueError('Not implemented multigrid yet')
                # Check if vj1 is already cached from the XC pass
                cached_dm = getattr(mydf, '_cached_vj1_dm', None)
                if cached_dm is not None and numpy.allclose(dm, cached_dm):
                    vj1 = mydf._cached_vj1
                else:
                    vj1 = PAWutils.getjSmoothPW2(cell, dm,
                                                 mydf.mg_ni,
                                                mydf.mesh,
                                                mydf.PAWdata,
                                                Periodic=mydf.Periodic)
            else:
                assert(mydf.with_multigrid == 0)

                #vj1 = PAWutils.getjSmoothPW(cell, dm, 
                #                            mydf.aoOnR_tilde,
                #                            mydf.mesh,
                #                            mydf.PAWdata,
                #                            Periodic=mydf.Periodic)
                vj1 = PAWutils.getjSmoothISDF(cell, dm, mydf.gaussgrid,
                                            mydf.aoOnR_tilde,
                                            mydf.mesh,
                                            mydf.PAWdata,
                                            Periodic=mydf.Periodic)
            logger.timer(mydf, 'vj PW', *cpu0)
            cpu0 = (logger.process_clock(), logger.perf_counter())
            vj2 = PAWutils.getjSharpLocal(cell, dm, 
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        Periodic=mydf.Periodic)
            vj3 = PAWutils.getjSmoothLocal(cell, dm, 
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        Periodic=mydf.Periodic)
            logger.timer(mydf, 'vj atom', *cpu0)
            vj = vj1 + vj2 + vj3
        if with_k:
            is_unrestricted = getattr(dm, 'ndim', 0) == 4 and dm.shape[0] == 2
            if is_unrestricted:
                vk = []
                for spin in range(2):
                    dm_spin = dm[spin] * 2.0
                    mo_occ_spin = dm.mo_occ[spin] * 2.0
                    mo_coeff_spin = dm.mo_coeff[spin]
                    dm_k = lib.tag_array(numpy.array([dm_spin]), mo_coeff=numpy.array([mo_coeff_spin]), mo_occ=numpy.array([mo_occ_spin]))
                    vk_spin = PAWutils.getk_PAW_JAX(cell, dm_k,
                                                mydf.aoOnR_tilde,
                                                mydf.mesh,
                                                mydf.PAWdata,
                                                mydf.S,
                                                Periodic=mydf.Periodic)
                    vk.append(vk_spin / 2.0)
                vk = numpy.array(vk)
            else:
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
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if getattr(mydf, 'with_multigrid', 0) > 0:
            raise ValueError('Not implemented with multigrid yet')
            # Check if vj1 is already cached from the XC pass
            cached_dm = getattr(mydf, '_cached_vj1_dm', None)
            if cached_dm is not None and numpy.allclose(dm, cached_dm):
                vj1 = mydf._cached_vj1
            else:
                vj1 = PAWutils.getjSmoothPW2(cell, dm,
                                             mydf.mg_ni,
                                             mydf.mesh,
                                             mydf.PAWdata,
                                             Periodic=mydf.Periodic)
        else:
            assert(mydf.with_multigrid == 0)

            vj1 = PAWutils.getjSmoothISDF(cell, dm, mydf.gaussgrid,
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        Periodic=mydf.Periodic)
        logger.timer(mydf, 'vj PW', *cpu0)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        vj2 = PAWutils.getjSharpLocal(cell, dm, 
                                    mydf.aoOnR_tilde,
                                    mydf.mesh,
                                    mydf.PAWdata,
                                    Periodic=mydf.Periodic)
        vj3 = PAWutils.getjSmoothLocal(cell, dm, 
                                    mydf.aoOnR_tilde,
                                    mydf.mesh,
                                    mydf.PAWdata,
                                    Periodic=mydf.Periodic)
        logger.timer(mydf, 'vj atom', *cpu0)
        vj = vj1 + vj2 + vj3
    if with_k:
        is_unrestricted = getattr(dm, 'ndim', 0) == 4 and dm.shape[0] == 2
        if is_unrestricted:
            raise ValueError("Unrestricted not implemented yet")
        else:
            cpu0 = (logger.process_clock(), logger.perf_counter())
            vk1 = PAWutils.getkSmoothISDF(cell, dm, mydf.gaussgrid,
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        Periodic=mydf.Periodic)
            logger.timer(mydf, 'vk smooth', *cpu0)
            cpu0 = (logger.process_clock(), logger.perf_counter())
            vk2 = PAWutils.getkSharpLocal(cell, dm, 
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        Periodic=mydf.Periodic)
            vk3 = PAWutils.getkSmoothLocal(cell, dm, 
                                        mydf.aoOnR_tilde,
                                        mydf.mesh,
                                        mydf.PAWdata,
                                        Periodic=mydf.Periodic)
            logger.timer(mydf, 'vk atom', *cpu0)
            vk = vk1 + vk2 + vk3
    return vj, vk

class PAW(FFTDF):
    def __init__(
            self,
            cell,
            kpts=None,
            PAWorbitalCutOff=1e-8,
            PWAccuracy=1e-8,
            Periodic=False,
            alpha0=None,
            augRadius=None,
            gdfNuc=False,
            with_multigrid=2,
            use_merged_multigrid=True,
            alpha0_lowmem=True,
            auto_box=True
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
        self.stdout = cell.stdout
        self.verbose = cell.verbose

        # PAW init
        self.PAWorbitalCutOff = PAWorbitalCutOff
        self.PWAccuracy = PWAccuracy
        self.Periodic = Periodic
        self.augRadius = augRadius
        self.gdfNuc = gdfNuc
        self.with_multigrid = with_multigrid
        self.use_merged_multigrid = use_merged_multigrid
        self.alpha0_lowmem = alpha0_lowmem
        self.auto_box = auto_box
        self.Times_ = {
            "Diagonalize":0.,
            "Exchange"   :0.,
            "Direct"     :0.,
            "1e-orbs"    :0.,
            "PAWinit"    :0.,
            "AOs"        :0.,
            "Fock"       :0.
        }

        self.initPAW(cell, alpha0)

        if self.with_multigrid > 0:
            # Initialize Multigrid for optimized Hartree/XC integration
            self.smoothCell = smoothCell2(self.cell, self.alpha0)
            if self.with_multigrid == 1:
                # self.mg_ni = MultiGridNumInt(self.smoothCell)
                # self.mg_ni.mesh = self.grids.mesh
                # self.mg_ni.xc_with_j = False
                # self.mg_ni.build()
                raise NotImplementedError('MultiGridNumInt1 not integrated with PAW yet')
            elif self.with_multigrid == 2:
                self.smoothCell.precision = min(self.cell.precision, 1e-10)
                self.mg_ni = MultiGridNumInt2(self.smoothCell)
                self.mg_ni.mesh = self.grids.mesh
                self.mg_ni.xc_with_j = False
                self.mg_ni.ntasks = 5 # consider making this a user input
                self.mg_ni.build()
        else:
            self.mg_ni = None

        # update get_jk
        if self.Periodic:
            self.get_jk = get_jk_periodic.__get__(self, self.__class__)
        else:
            self.get_jk = get_jk_molecule.__get__(self, self.__class__)
        ###

    def initPAW(self, cell, alpha0):
        t0 = time.time()
        PAWdata, PAWNucdata, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, mol, pmol, gmol, ctr_coeff, BeckeGrid, alpha0, Rs = getPAWdata(
            cell,
            gaussgrid,
            PAWorbitalCutOff=self.PAWorbitalCutOff,
            PWAccuracy=self.PWAccuracy,
            Periodic=self.Periodic,
            alpha0=alpha0,
            augRadius=self.augRadius,
            with_multigrid=self.with_multigrid,
            alpha0_lowmem=self.alpha0_lowmem,
            auto_box=self.auto_box
        )

        self.augRadius = Rs

        
        self.Times_["PAWinit"] += time.time()-t0

        nelec, nao = cell.nelectron, cell.nao
        # nocc = nelec//2
        # madelung = pyscf.pbc.tools.pbc.madelung(self.cell, self.cell.make_kpts([1,1,1])) if self.Periodic else 0.

        # mf = pyscf.pbc.scf.RHF(cell).rs_density_fit() if self.Periodic else pyscf.scf.RHF(pyscf.gto.M(atom = cell.atom, basis = cell.basis, unit=cell.unit))

        t0 = time.time()
        # hcore = mf.get_hcore().reshape((nao,nao))
        # S = mf.get_ovlp().reshape((nao,nao))
        # X = get_transformation_matrix(S)
        # nuc = mf.energy_nuc()
        self.Times_["1e-orbs"] += time.time()-t0

        # self.S = S
        self.aoOnR_tilde = aoOnR_tilde
        self.mesh = mesh
        self.PAWdata = PAWdata
        self.PAWNucdata = PAWNucdata
        self.Rgrid = Rgrid
        self.cell = mol
        self.pcell = pmol
        self.gcell = gmol
        self.ctr_coeff = ctr_coeff
        self.BeckeGrid = BeckeGrid
        self.alpha0 = alpha0
        
        logger.info(self, "alpha0      : %10.2f", alpha0)
        logger.info(self, "Rb          : %10.2f", numpy.max(Rs))
        logger.info(self, "Nelection   : %10d", nelec)
        logger.info(self, "Ngrid points: %10d", numpy.prod(self.cell.mesh))
        logger.info(self, "delta-a     : %10.2f", (self.cell.vol/numpy.prod(self.cell.mesh))**(1./3.))

    def get_nuc(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        if self.gdfNuc:
            dfbuilder = _RSNucBuilder(cell, kpts).build()
            nuc = dfbuilder.get_nuc()
        else:
            nuc = PAWutils.getnuc_PAW(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                      self.PAWNucdata, self.mg_ni, self.Periodic, self.with_multigrid)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_paw1(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        nuc = PAWutils.getNucPAWSmoothPW(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                 self.PAWNucdata, self.Periodic)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_paw2(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        nuc = PAWutils.getNucPAWSharpLocal(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                 self.PAWNucdata, self.Periodic)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_paw3(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        nuc = PAWutils.getNucPAWSmoothLocal(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                 self.PAWNucdata, self.Periodic)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_rsdf(self, kpts=None):
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
    
    def getJ1(self, dm, hermi=1, kpts=None, kpts_band=None,
              with_j=True, with_k=True, omega=None, exxdiv=None):
        # TODO: this might be problematic
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                        omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)

        # recreate occMo from DM if not available
        cell = self.cell
        if isinstance(dm, list):
            dm = numpy.asarray(dm)
        nk = kpts.shape[0] # what is kpt default for no kpts (kpt=None)?
        nao = cell.nao
        # if with_k:
        if with_j or with_k:
            dm = tag_dm(self, dm, cell, kpts, nk, nao)
        J1 = PAWutils.getjSmoothPW2(self.cell, dm, self.mg_ni, self.mesh, self.PAWdata, Periodic=self.Periodic)
        
        return J1
    
    def getJ2(self, dm, hermi=1, kpts=None, kpts_band=None,
              with_j=True, with_k=True, omega=None, exxdiv=None):
        # TODO: this might be problematic
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                        omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)

        # recreate occMo from DM if not available
        cell = self.cell
        if isinstance(dm, list):
            dm = numpy.asarray(dm)
        nk = kpts.shape[0] # what is kpt default for no kpts (kpt=None)?
        nao = cell.nao
        # if with_k:
        if with_j or with_k:
            dm = tag_dm(self, dm, cell, kpts, nk, nao)
        J2 = PAWutils.getjSharpLocal(self.cell, dm, self.aoOnR_tilde, self.mesh,
                                   self.PAWdata, self.Periodic)
        
        return J2
    
    def getJ3(self, dm, hermi=1, kpts=None, kpts_band=None,
              with_j=True, with_k=True, omega=None, exxdiv=None):
        # TODO: this might be problematic
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                        omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)

        # recreate occMo from DM if not available
        cell = self.cell
        if isinstance(dm, list):
            dm = numpy.asarray(dm)
        nk = kpts.shape[0] # what is kpt default for no kpts (kpt=None)?
        nao = cell.nao
        # if with_k:
        if with_j or with_k:
            dm = tag_dm(self, dm, cell, kpts, nk, nao)
        J3 = PAWutils.getjSmoothLocal(self.cell, dm, self.aoOnR_tilde, self.mesh,
                                   self.PAWdata, self.Periodic)
        
        return J3
    
    @classmethod
    def from_mf(cls, mf, cell=None, gaussgrid=None, **kwargs):
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
            occri = cls(cell, gaussgrid, Periodic=False, **kwargs)
        occri.method = method

        # cell will be modified for molecular case
        

        return occri


class NewPAW(FFTDF):
    def __init__(
            self,
            cell,
            gaussgrid,
            kpts=None,
            PAWorbitalCutOff=1e-8,
            PWAccuracy=1e-8,
            Periodic=False,
            alpha0=None,
            augRadius=None,
            gdfNuc=False,
            with_multigrid=0,
            use_merged_multigrid=False,
            alpha0_lowmem=False,
            auto_box=False
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
        self.stdout = cell.stdout
        self.verbose = cell.verbose

        # PAW init
        self.PAWorbitalCutOff = PAWorbitalCutOff
        self.PWAccuracy = PWAccuracy
        self.Periodic = Periodic
        self.augRadius = augRadius
        self.gdfNuc = gdfNuc
        self.with_multigrid = with_multigrid
        self.use_merged_multigrid = use_merged_multigrid
        self.alpha0_lowmem = alpha0_lowmem
        self.auto_box = auto_box
        self.Times_ = {
            "Diagonalize":0.,
            "Exchange"   :0.,
            "Direct"     :0.,
            "1e-orbs"    :0.,
            "PAWinit"    :0.,
            "AOs"        :0.,
            "Fock"       :0.
        }

        self.initPAW(cell, gaussgrid, alpha0)
        self.gaussgrid=gaussgrid

        if self.with_multigrid > 0:
            # Initialize Multigrid for optimized Hartree/XC integration
            self.smoothCell = smoothCell2(self.cell, self.alpha0)
            if self.with_multigrid == 1:
                # self.mg_ni = MultiGridNumInt(self.smoothCell)
                # self.mg_ni.mesh = self.grids.mesh
                # self.mg_ni.xc_with_j = False
                # self.mg_ni.build()
                raise NotImplementedError('MultiGridNumInt1 not integrated with PAW yet')
            elif self.with_multigrid == 2:
                self.smoothCell.precision = min(self.cell.precision, 1e-10)
                self.mg_ni = MultiGridNumInt2(self.smoothCell)
                self.mg_ni.mesh = self.grids.mesh
                self.mg_ni.xc_with_j = False
                self.mg_ni.ntasks = 5 # consider making this a user input
                self.mg_ni.build()
        else:
            self.mg_ni = None

        # update get_jk
        if self.Periodic:
            self.get_jk = get_jk_periodic.__get__(self, self.__class__)
        else:
            self.get_jk = get_jk_molecule.__get__(self, self.__class__)
        ###

    def initPAW(self, cell, gaussgrid, alpha0):
        t0 = time.time()
        PAWdata, PAWNucdata, mesh, Rgrid, aoOnR, aoOnR_tilde, gOnRAll, mol, pmol, gmol, ctr_coeff, BeckeGrid, alpha0, Rs = getPAWdataNew(
            cell,
            gaussgrid,
            PAWorbitalCutOff=self.PAWorbitalCutOff,
            PWAccuracy=self.PWAccuracy,
            Periodic=self.Periodic,
            alpha0=alpha0,
            augRadius=self.augRadius,
            with_multigrid=self.with_multigrid,
            alpha0_lowmem=self.alpha0_lowmem,
            auto_box=self.auto_box
        )

        self.augRadius = Rs

        
        self.Times_["PAWinit"] += time.time()-t0

        nelec, nao = cell.nelectron, cell.nao
        # nocc = nelec//2
        # madelung = pyscf.pbc.tools.pbc.madelung(self.cell, self.cell.make_kpts([1,1,1])) if self.Periodic else 0.

        # mf = pyscf.pbc.scf.RHF(cell).rs_density_fit() if self.Periodic else pyscf.scf.RHF(pyscf.gto.M(atom = cell.atom, basis = cell.basis, unit=cell.unit))

        t0 = time.time()
        # hcore = mf.get_hcore().reshape((nao,nao))
        # S = mf.get_ovlp().reshape((nao,nao))
        # X = get_transformation_matrix(S)
        # nuc = mf.energy_nuc()
        self.Times_["1e-orbs"] += time.time()-t0

        # self.S = S
        self.aoOnR_tilde = aoOnR_tilde
        self.mesh = mesh
        self.PAWdata = PAWdata
        self.PAWNucdata = PAWNucdata
        self.Rgrid = Rgrid
        self.cell = mol
        self.pcell = pmol
        self.gcell = gmol
        self.ctr_coeff = ctr_coeff
        self.BeckeGrid = BeckeGrid
        self.alpha0 = alpha0
        
        logger.info(self, "alpha0      : %10.2f", alpha0)
        logger.info(self, "Rb          : %10.2f", numpy.max(Rs))
        logger.info(self, "Nelection   : %10d", nelec)
        logger.info(self, "Ngrid points: %10d", numpy.prod(self.cell.mesh))
        logger.info(self, "delta-a     : %10.2f", (self.cell.vol/numpy.prod(self.cell.mesh))**(1./3.))

    def get_nuc(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        if self.gdfNuc:
            dfbuilder = _RSNucBuilder(cell, kpts).build()
            nuc = dfbuilder.get_nuc()
        else:
            nuc = PAWutils.getnuc_PAW(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                      self.PAWNucdata, self.mg_ni, self.Periodic, self.with_multigrid)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_paw1(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        nuc = PAWutils.getNucPAWSmoothPW(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                 self.PAWNucdata, self.Periodic)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_paw2(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        nuc = PAWutils.getNucPAWSharpLocal(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                 self.PAWNucdata, self.Periodic)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_paw3(self, kpts=None):
        '''Get the periodic nuc-el AO matrix, with G=0 removed.
        '''
        # TODO: kpt not implemented
        cell = self.cell
        kpts, is_single_kpt = _check_kpts(self, kpts)
        
        nuc = PAWutils.getNucPAWSmoothLocal(cell, self.mesh, self.aoOnR_tilde, self.PAWdata,
                                 self.PAWNucdata, self.Periodic)
        if is_single_kpt:
            nuc = nuc[0]
        return nuc
    
    def get_nuc_rsdf(self, kpts=None):
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
    
    def getJ1(self, dm, hermi=1, kpts=None, kpts_band=None,
              with_j=True, with_k=True, omega=None, exxdiv=None):
        # TODO: this might be problematic
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                        omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)

        # recreate occMo from DM if not available
        cell = self.cell
        if isinstance(dm, list):
            dm = numpy.asarray(dm)
        nk = kpts.shape[0] # what is kpt default for no kpts (kpt=None)?
        nao = cell.nao
        # if with_k:
        if with_j or with_k:
            dm = tag_dm(self, dm, cell, kpts, nk, nao)
        J1 = PAWutils.getjSmoothPW2(self.cell, dm, self.mg_ni, self.mesh, self.PAWdata, Periodic=self.Periodic)
        
        return J1
    
    def getJ2(self, dm, hermi=1, kpts=None, kpts_band=None,
              with_j=True, with_k=True, omega=None, exxdiv=None):
        # TODO: this might be problematic
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                        omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)

        # recreate occMo from DM if not available
        cell = self.cell
        if isinstance(dm, list):
            dm = numpy.asarray(dm)
        nk = kpts.shape[0] # what is kpt default for no kpts (kpt=None)?
        nao = cell.nao
        # if with_k:
        if with_j or with_k:
            dm = tag_dm(self, dm, cell, kpts, nk, nao)
        J2 = PAWutils.getjSharpLocal(self.cell, dm, self.aoOnR_tilde, self.mesh,
                                   self.PAWdata, self.Periodic)
        
        return J2
    
    def getJ3(self, dm, hermi=1, kpts=None, kpts_band=None,
              with_j=True, with_k=True, omega=None, exxdiv=None):
        # TODO: this might be problematic
        if omega is not None:  # J/K for RSH functionals
            with self.range_coulomb(omega) as rsh_df:
                return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
                                        omega=None, exxdiv=exxdiv)

        kpts, is_single_kpt = _check_kpts(self, kpts)

        # recreate occMo from DM if not available
        cell = self.cell
        if isinstance(dm, list):
            dm = numpy.asarray(dm)
        nk = kpts.shape[0] # what is kpt default for no kpts (kpt=None)?
        nao = cell.nao
        # if with_k:
        if with_j or with_k:
            dm = tag_dm(self, dm, cell, kpts, nk, nao)
        J3 = PAWutils.getjSmoothLocal(self.cell, dm, self.aoOnR_tilde, self.mesh,
                                   self.PAWdata, self.Periodic)
        
        return J3
    
    @classmethod
    def from_mf(cls, mf, cell=None, gaussgrid=None, **kwargs):
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
            occri = cls(cell, gaussgrid, Periodic=False, **kwargs)
        occri.method = method

        # cell will be modified for molecular case
        

        return occri
