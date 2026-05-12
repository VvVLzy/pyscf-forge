import numpy
from pyscf import lib
from pyscf.pbc.dft.numint import NumInt, KNumInt, nr_rks, nr_uks
from pyscf.pbc.dft.gen_grid import BeckeGrids
from pyscf.dft.gen_grid import Grids
from pyscf.lib import logger
from .PAWutils import buildPmolAtom, makeWignerSeitz, get_vxc_and_j_smooth
# Use the multigrid_pair version as requested
from pyscf.pbc.dft.multigrid.multigrid_pair import MultiGridNumInt as MultiGridNumInt2
from pyscf.pbc.dft.multigrid import MultiGridNumInt

from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

def smoothCell(mol, alpha0):
    smoothMol = mol.copy()
    env2Mod = smoothMol._env.copy()
    for shell in smoothMol._bas:
        nprim = shell[2]
        nconc = shell[3]

        # locate sharp alphas
        alphaStart = shell[-3]
        alphas = smoothMol._env[alphaStart:alphaStart+nprim]
        sharpAlphaId = numpy.where(alphas > alpha0)[0]

        # set sharp contraction coeff to 0
        coeffStart = [shell[-2] + nprim* i for i in range(nconc)]
        for c in coeffStart:
            env2Mod[sharpAlphaId+c] = 0
        # print(env2Mod)
    smoothMol._env = env2Mod
    return smoothMol

def smoothCell2(mol, alpha0):
    smoothMol = mol.copy()
    new_bas = smoothMol._bas.copy()
    
    # Identify the start of basis data in env to keep coordinates and constants
    # Basis pointers (exponents and coefficients) are in columns 5 and 6
    bas_pointers = numpy.concatenate([mol._bas[:, 5], mol._bas[:, 6]])
    split_ptr = int(numpy.min(bas_pointers)) # all exp and conc coeff start from this index
    
    new_env = list(mol._env[:split_ptr])
    
    for i in range(len(new_bas)):
        nprim = new_bas[i, 2]
        nconc = new_bas[i, 3]
        ptr_exp = new_bas[i, 5]
        ptr_coeff = new_bas[i, 6]
        
        exps = mol._env[ptr_exp : ptr_exp + nprim]
        coeffs = mol._env[ptr_coeff : ptr_coeff + nprim * nconc].reshape(nconc, nprim)
        
        mask = exps <= alpha0
        new_nprim = numpy.count_nonzero(mask)
        
        if new_nprim > 0:
            filtered_exps = exps[mask]
            filtered_coeffs = coeffs[:, mask]
        else:
            # Keep one dummy primitive with zero coefficient to maintain nao consistency
            # and avoid nprim=0 which might cause issues in some PySCF routines.
            new_nprim = 1
            filtered_exps = numpy.array([alpha0])
            filtered_coeffs = numpy.zeros((nconc, 1))
            
        # Update env and pointers
        new_bas[i, 5] = len(new_env)
        new_env.extend(filtered_exps)
        new_bas[i, 6] = len(new_env)
        new_env.extend(filtered_coeffs.ravel())
        new_bas[i, 2] = new_nprim
        
    smoothMol._env = numpy.array(new_env)
    smoothMol._bas = new_bas
    return smoothMol

class PAWNumInt(NumInt):
    def __init__(self, mydf, mf, uniform=True, with_multigrid=0, use_merged_multigrid=False):
        self.mf = mf
        self.cell = mydf.cell
        self.smoothCell = smoothCell2(self.cell, mydf.alpha0)
        self.pcell = mydf.pcell
        self.max_memory = mydf.max_memory

        self.with_multigrid = with_multigrid
        self.use_merged_multigrid = use_merged_multigrid
        
        # Prefer reusing mg_ni from the density fitting object (mydf)
        if getattr(mydf, 'mg_ni', None) is not None:
            self.mg_ni = mydf.mg_ni
            self.smoothCell = getattr(mydf, 'smoothCell', self.smoothCell)
            self.with_multigrid = mydf.with_multigrid
            self.use_merged_multigrid = mydf.use_merged_multigrid
        elif with_multigrid == 1:
            # Initialize Multigrid (pair version) for the smooth component
            self.mg_ni = MultiGridNumInt(self.smoothCell)
            self.mg_ni.mesh = mydf.grids.mesh
            self.mg_ni.xc_with_j = False
            self.mg_ni.build()
        elif with_multigrid == 2:
            # Initialize Multigrid (pair version) for the smooth component
            # Ensure smoothCell has enough precision for multigrid_pair (version 2)
            # which lacks the internal EXPDROP safeguard present in version 1.
            self.smoothCell.precision = min(self.cell.precision, 1e-10)
            self.mg_ni = MultiGridNumInt2(self.smoothCell)
            self.mg_ni.mesh = mydf.grids.mesh
            self.mg_ni.xc_with_j = False
            self.mg_ni.ntasks = 5
            self.mg_ni.build()
        else:
            self.mg_ni = None

        if uniform:
            self.uniform_grid = mydf.grids 
            self.pcellAtoms, self.smoothPcellAtoms, self.atomicGrids = self.prepareAtomicData(mydf.alpha0, getattr(mydf, 'augRadius', None))
        else:
            self.uniform_grid = BeckeGrids(self.pcell).build()
            self.pcellAtoms, self.smoothPcellAtoms, self.atomicGrids = [], [], []
        self.localIdx = mydf.PAWdata[0]
        self.F_PmuArr = mydf.PAWdata[1]
        self.Ftilde_PmuArr = mydf.PAWdata[2]
        super().__init__()

    def prepareAtomicData(self, alpha0, augRadius):
        pcellAtoms = []
        smoothPcellAtoms = []
        grids = []

        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = buildPmolAtom(self.cell, self.pcell, atomI, self.pcell.cart)
            smoothpcellAtom = smoothCell2(pcellAtom, alpha0)

            # filter the grid
            atomGrid = BeckeGrids(pcellAtom)
            atomGrid.build()
            
            if augRadius is not None:
                radius = augRadius
                if isinstance(augRadius, (list, numpy.ndarray)):
                    radius = augRadius[atomI]
                
                atomGridDist = makeWignerSeitz(atomGrid.coords, pcellAtom, Periodic=True)[1]
                allIdx = numpy.where(atomGridDist[0, :] < radius)[0]
                atomGrid.coords = atomGrid.coords[allIdx]
                atomGrid.weights = atomGrid.weights[allIdx]
                
                # IMPORTANT: rebuild non0tab after filtering to ensure consistency
                # and performance in block_loop
                atomGrid.non0tab = atomGrid.make_mask(pcellAtom, atomGrid.coords)
            
            logger.info(self.mf, 'Atom %d atomic grid size: %d', atomI, atomGrid.coords.shape[0])

            pcellAtoms.append(pcellAtom)
            smoothPcellAtoms.append(smoothpcellAtom)
            grids.append(atomGrid)

        return pcellAtoms, smoothPcellAtoms, grids

    def nr_rks_profiled(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
                        kpts=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpts is None:
            kpts = numpy.zeros((1,3))
        kpts = kpts.reshape(-1,3)

        xctype = self._xc_type(xc_code)
        if xctype == 'LDA':
            ao_deriv = 0
        elif xctype == 'GGA':
            ao_deriv = 1
        elif xctype == 'MGGA':
            ao_deriv = 1
        elif xctype == 'HF':
            ao_deriv = 0
        
        make_rho, nset, nao = self._gen_rho_evaluator(cell, dms, hermi, False)
        
        nelec = numpy.zeros(nset)
        excsum = numpy.zeros(nset)
        shls_slice = (0, cell.nbas)
        ao_loc = cell.ao_loc
        deriv = 1
        vmat = [0]*nset
        v_hermi = 1
        
        # Sub-timers (Wall time)
        t_ao = 0.0
        t_rho = 0.0
        t_xc = 0.0
        t_vmat = 0.0
        
        loop = self.block_loop(cell, grids, nao, ao_deriv, kpts, kpts_band, max_memory)
        
        while True:
            t0 = logger.perf_counter()
            try:
                ao_k1, ao_k2, mask, weight, coords = next(loop)
            except StopIteration:
                break
            t_ao += logger.perf_counter() - t0
            
            for i in range(nset):
                t0 = logger.perf_counter()
                rho = make_rho(i, ao_k2, mask, xctype).real
                t_rho += logger.perf_counter() - t0
                
                t0 = logger.perf_counter()
                exc, vxc = self.eval_xc_eff(xc_code, rho, deriv, xctype=xctype, spin=0)[:2]
                t_xc += logger.perf_counter() - t0
                
                if xctype == 'LDA':
                    den = rho*weight
                else:
                    den = rho[0]*weight
                nelec[i] += den.sum()
                excsum[i] += den.dot(exc)
                
                t0 = logger.perf_counter()
                wv = weight * vxc
                vmat[i] += self._vxc_mat(cell, ao_k1, wv, mask, xctype,
                                       shls_slice, ao_loc, v_hermi)
                t_vmat += logger.perf_counter() - t0

        vmat = numpy.stack(vmat)
        vmat = vmat + vmat.conj().swapaxes(-2,-1)
        if nset == 1:
            nelec = nelec[0]
            excsum = excsum[0]
            vmat = vmat[0]
            
        logger.info(self.mf, 'Uniform Grid Wall-Time Profiling (nset=%d):', nset)
        logger.info(self.mf, '  - AO Eval:        %10.4f s', t_ao)
        logger.info(self.mf, '  - Density Build:  %10.4f s', t_rho)
        logger.info(self.mf, '  - XC Evaluation:  %10.4f s', t_xc)
        logger.info(self.mf, '  - Vxc Matrix:     %10.4f s', t_vmat)
        
        return nelec, excsum, vmat

    def nr_rks_uniform_smooth(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        # Switch between Multigrid and Profiled standard integration
        if self.with_multigrid:
            if self.use_merged_multigrid:
                dm_multigrid = dms
                if dm_multigrid.ndim == 2:
                    dm_multigrid = dm_multigrid[numpy.newaxis, ...]
                if dm_multigrid.ndim == 3: # (nkpts, nao, nao)
                    dm_multigrid = dm_multigrid[numpy.newaxis, ...]

                # Call combined Hartree and XC multigrid pass
                nelec1, exc1, ecoul1, vxc1, vj1 = get_vxc_and_j_smooth(
                    self.cell, dm_multigrid, self.mg_ni, self.mg_ni.mesh, self.mf.with_df.PAWdata, 
                    xc_code, Periodic=self.mf.with_df.Periodic, hermi=hermi, kpt=kpt
                )
                
                # Cache vj1 for NewPAW.get_jk
                self.mf.with_df._cached_vj1 = vj1
                self.mf.with_df._cached_vj1_dm = dms
                
                vxc1 = lib.tag_array(vxc1, ecoul=ecoul1, exc=exc1, vj=vj1, vk=None)
                
                return nelec1, exc1, vxc1
            else:
                # vj1 will be computed separately in get_jk
                dm_multigrid = dms
                if dm_multigrid.ndim == 2:
                    dm_multigrid = dm_multigrid[numpy.newaxis, ...]
                
                # multigrid_pair.nr_rks returns (nelec, excsum, vmat) 
                # This matches standard PySCF order.
                nelec1, exc1, vxc1 = self.mg_ni.nr_rks(self.smoothCell, self.uniform_grid, xc_code, dm_multigrid, 
                                                    relativity=relativity, hermi=hermi, 
                                                    kpts=kpt, kpts_band=kpts_band,
                                                    max_memory=max_memory, verbose=verbose)
                
                if isinstance(vxc1, numpy.ndarray) and vxc1.ndim == 3 and vxc1.shape[0] == 1:
                    vxc1 = vxc1[0]
                
                return nelec1, exc1, vxc1

        # Profiled standard path
        if self._xc_type(xc_code) != 'HF':
             return self.nr_rks_profiled(self.smoothCell, self.uniform_grid, xc_code, dms, 
                                         relativity, hermi, kpt, kpts_band, max_memory, verbose)

        # Fallback for HF (still want to count electrons)
        nelec1, exc1, vxc1 = self.nr_rks_profiled(self.smoothCell, self.uniform_grid, 'lda', dms, 
                                                  relativity, hermi, kpt, kpts_band, max_memory, verbose)
        return nelec1, 0, numpy.zeros_like(vxc1)
    
    def nr_rks_atomic_sharp(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        nelec2 = 0
        exc2 = 0
        vxc2 = numpy.zeros((self.cell.nao, self.cell.nao))
        if self.smoothPcellAtoms == []:
            return nelec2, exc2, vxc2

        f = self.F_PmuArr
        l = self.localIdx
        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = self.pcellAtoms[atomI]
            grids = self.atomicGrids[atomI]
            if grids.coords.size == 0:
                continue

            dm_loc = dms[l[atomI]][:, l[atomI]]
            dm = smart_einsum('mn,Pm,Qn->PQ', dm_loc, f[atomI], f[atomI])

            if self._xc_type(xc_code) == 'HF':
                temp = xc_code
                xc_code = 'lda' # still want to check electron density
                nelec2a, exc2a, vxc2a = nr_rks(
                    self, pcellAtom, grids, xc_code, dm, spin, relativity,
                    hermi, kpt, kpts_band, max_memory, verbose
                )
                exc2a = 0
                vxc2a = numpy.zeros_like(vxc2a)
                xc_code = temp
            else:
                nelec2a, exc2a, vxc2a = nr_rks(
                    self, pcellAtom, grids, xc_code, dm, spin, relativity,
                    hermi, kpt, kpts_band, max_memory, verbose
                )
            # import pdb; pdb.set_trace()
            nelec2 += nelec2a
            exc2 += exc2a
            numpy.add.at(
                vxc2, numpy.ix_(l[atomI], l[atomI]),
                smart_einsum('PQ,Pm,Qn->mn', vxc2a, f[atomI], f[atomI])
            )
            # vxc2 += smart_einsum('PQ,Pm,Qn->mn', vxc2a, f[atomI], f[atomI])

        return nelec2, exc2, vxc2

    def nr_rks_atomic_smooth(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        nelec3 = 0
        exc3 = 0
        vxc3 = numpy.zeros((self.cell.nao, self.cell.nao))
        if self.smoothPcellAtoms == []:
            return nelec3, exc3, vxc3

        f = self.Ftilde_PmuArr
        l = self.localIdx
        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = self.smoothPcellAtoms[atomI]
            grids = self.atomicGrids[atomI]
            if grids.coords.size == 0:
                continue

            dm_loc = dms[l[atomI]][:, l[atomI]]
            dm = smart_einsum('mn,Pm,Qn->PQ', dm_loc, f[atomI], f[atomI])

            if self._xc_type(xc_code) == 'HF':
                temp = xc_code
                xc_code = 'lda' # still want to check electron density
                nelec3a, exc3a, vxc3a = nr_rks(
                    self, pcellAtom, grids, xc_code, dm, spin, relativity,
                    hermi, kpt, kpts_band, max_memory, verbose
                )
                exc3a = 0
                vxc3a = numpy.zeros_like(vxc3a)
                xc_code = temp
            else:
                nelec3a, exc3a, vxc3a = nr_rks(
                    self, pcellAtom, grids, xc_code, dm, spin, relativity,
                    hermi, kpt, kpts_band, max_memory, verbose
                )

            nelec3 += nelec3a
            exc3 += exc3a
            numpy.add.at(
                vxc3, numpy.ix_(l[atomI], l[atomI]),
                smart_einsum('PQ,Pm,Qn->mn', vxc3a, f[atomI], f[atomI])
            )

        return nelec3, exc3, vxc3

    def nr_uks_uniform_smooth(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        if self.with_multigrid:
            nk = getattr(self.mf.with_df, 'kpts', numpy.zeros((1,3))).reshape(-1, 3).shape[0]
            nao = self.cell.nao
            dm_multigrid = dms.reshape(-1, nk, nao, nao)

            if self.use_merged_multigrid:
                # Call combined Hartree and XC multigrid pass
                nelec1, exc1, ecoul1, vxc1, vj1 = get_vxc_and_j_smooth(
                    self.cell, dm_multigrid, self.mg_ni, self.mg_ni.mesh, self.mf.with_df.PAWdata, 
                    xc_code, Periodic=self.mf.with_df.Periodic, hermi=hermi, kpt=kpt
                )
                
                # Cache vj1 for NewPAW.get_jk
                # get_vxc_and_j_smooth now returns vj1 as (2, nao, nao) for UKS
                
                # As requested: store total DM for cache key
                dm_total = dms[0] + dms[1]
                dm_total_tagged = dm_total.reshape(-1, nk, nao, nao)
                self.mf.with_df._cached_vj1_dm = dm_total_tagged
                self.mf.with_df._cached_vj1 = vj1[0] + vj1[1]
                
                vxc1 = lib.tag_array(vxc1, ecoul=ecoul1, exc=exc1, vj=vj1, vk=None)
                
                return nelec1, exc1, vxc1
            else:
                nelec1, exc1, vxc1 = self.mg_ni.nr_uks(self.smoothCell, self.uniform_grid, xc_code, dm_multigrid, 
                                                    relativity=relativity, hermi=hermi, 
                                                    kpts=kpt, kpts_band=kpts_band,
                                                    max_memory=max_memory, verbose=verbose)
                
                if isinstance(vxc1, numpy.ndarray) and vxc1.ndim == 4 and vxc1.shape[1] == 1:
                    vxc1 = vxc1[:, 0]
                
                self.mf.with_df._cached_vj1_dm = dm_multigrid
                
                return nelec1, exc1, vxc1

        if self._xc_type(xc_code) != 'HF':
             return nr_uks(self, self.smoothCell, self.uniform_grid, xc_code, dms, 
                           spin=1, relativity=relativity, hermi=hermi, kpts=kpt, kpts_band=kpts_band, max_memory=max_memory, verbose=verbose)

        nelec1, exc1, vxc1 = nr_uks(self, self.smoothCell, self.uniform_grid, 'lda', dms, 
                                    spin=1, relativity=relativity, hermi=hermi, kpts=kpt, kpts_band=kpts_band, max_memory=max_memory, verbose=verbose)
        return nelec1, 0, numpy.zeros_like(vxc1)

    def nr_uks_atomic_sharp(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        nelec2 = numpy.zeros(2)
        exc2 = 0
        vxc2 = numpy.zeros((2, self.cell.nao, self.cell.nao))
        if self.smoothPcellAtoms == []:
            return nelec2, exc2, vxc2

        f = self.F_PmuArr
        l = self.localIdx
        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = self.pcellAtoms[atomI]
            grids = self.atomicGrids[atomI]
            if grids.coords.size == 0:
                continue

            dm_loc = dms[:, l[atomI]][:, :, l[atomI]]
            dm = smart_einsum('smn,Pm,Qn->sPQ', dm_loc, f[atomI], f[atomI])

            if self._xc_type(xc_code) == 'HF':
                temp = xc_code
                xc_code = 'lda' # still want to check electron density
                nelec2a, exc2a, vxc2a = nr_uks(
                    self, pcellAtom, grids, xc_code, dm, spin=1, relativity=relativity,
                    hermi=hermi, kpts=kpt, kpts_band=kpts_band, max_memory=max_memory, verbose=verbose
                )
                exc2a = 0
                vxc2a = numpy.zeros_like(vxc2a)
                xc_code = temp
            else:
                nelec2a, exc2a, vxc2a = nr_uks(
                    self, pcellAtom, grids, xc_code, dm, spin=1, relativity=relativity,
                    hermi=hermi, kpts=kpt, kpts_band=kpts_band, max_memory=max_memory, verbose=verbose
                )

            nelec2 += nelec2a
            exc2 += exc2a
            for s in range(2):
                numpy.add.at(
                    vxc2[s], numpy.ix_(l[atomI], l[atomI]),
                    smart_einsum('PQ,Pm,Qn->mn', vxc2a[s], f[atomI], f[atomI])
                )

        return nelec2, exc2, vxc2

    def nr_uks_atomic_smooth(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        nelec3 = numpy.zeros(2)
        exc3 = 0
        vxc3 = numpy.zeros((2, self.cell.nao, self.cell.nao))
        if self.smoothPcellAtoms == []:
            return nelec3, exc3, vxc3

        f = self.Ftilde_PmuArr
        l = self.localIdx
        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = self.smoothPcellAtoms[atomI]
            grids = self.atomicGrids[atomI]
            if grids.coords.size == 0:
                continue

            dm_loc = dms[:, l[atomI]][:, :, l[atomI]]
            dm = smart_einsum('smn,Pm,Qn->sPQ', dm_loc, f[atomI], f[atomI])

            if self._xc_type(xc_code) == 'HF':
                temp = xc_code
                xc_code = 'lda' # still want to check electron density
                nelec3a, exc3a, vxc3a = nr_uks(
                    self, pcellAtom, grids, xc_code, dm, spin=1, relativity=relativity,
                    hermi=hermi, kpts=kpt, kpts_band=kpts_band, max_memory=max_memory, verbose=verbose
                )
                exc3a = 0
                vxc3a = numpy.zeros_like(vxc3a)
                xc_code = temp
            else:
                nelec3a, exc3a, vxc3a = nr_uks(
                    self, pcellAtom, grids, xc_code, dm, spin=1, relativity=relativity,
                    hermi=hermi, kpts=kpt, kpts_band=kpts_band, max_memory=max_memory, verbose=verbose
                )

            nelec3 += nelec3a
            exc3 += exc3a
            for s in range(2):
                numpy.add.at(
                    vxc3[s], numpy.ix_(l[atomI], l[atomI]),
                    smart_einsum('PQ,Pm,Qn->mn', vxc3a[s], f[atomI], f[atomI])
                )

        return nelec3, exc3, vxc3
    
    def get_rho(self, cell, dm, grids, kpts=None, max_memory=2000):
        if kpts is not None and not isinstance(kpts, (numpy.ndarray, list)):
            max_memory = kpts
            kpts = None

        if not getattr(cell, 'a', None):
            cell = self.cell

        return super().get_rho(cell, dm, grids, kpts, max_memory)

    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if not isinstance(kpt, (numpy.ndarray, list)):
            # Molecular call: nr_rks(mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose)
            verbose = kpts_band
            max_memory = kpt
            kpt = numpy.zeros(3)
            kpts_band = None

        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = self.view(KNumInt)
            nao = dms.shape[-1]
            dms = dms.reshape(-1,nao,nao)
            return ni.nr_rks(cell, grids, xc_code, dms, relativity,
                             hermi, kpt.reshape(1,3), kpts_band, max_memory, verbose)
        spin = 0
        
        # uniform grid
        cpu0 = (logger.process_clock(), logger.perf_counter())
        nelec1, exc1, vxc1 = self.nr_rks_uniform_smooth(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )
        logger.timer(self.mf.with_df, 'vxc uniform', *cpu0)
        # atomic grid
        cpu0 = (logger.process_clock(), logger.perf_counter())
        nelec2, exc2, vxc2 = self.nr_rks_atomic_sharp(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )
        nelec3, exc3, vxc3 = self.nr_rks_atomic_smooth(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )
        logger.timer(self.mf.with_df, 'vxc atomic', *cpu0)
        logger.info(self.mf, 'smooth nelec with uniform grids = %s', nelec1)
        logger.info(self.mf, 'all nelec with atomic grids = %s', nelec2)
        logger.info(self.mf, 'smooth nelec with atomic grids = %s', nelec3)
        logger.info(self.mf, 'sharp nelec with atomic grids = %s', nelec2 - nelec3)
        nelec = nelec1 + nelec2 - nelec3
        exc = exc1 + exc2 - exc3
        vxc = vxc1 + vxc2 - vxc3
        return nelec, exc, vxc

    @lib.with_doc(nr_uks.__doc__)
    def nr_uks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpt=None, kpts_band=None, max_memory=2000, verbose=None):
        if kpt is None:
            kpt = numpy.zeros((1,3))
        if not isinstance(kpt, (numpy.ndarray, list)) or numpy.asarray(kpt).ndim == 0:
            verbose = kpts_band
            max_memory = kpt
            kpt = numpy.zeros((1,3))
            kpts_band = None

        kpt = numpy.asarray(kpt).reshape(-1, 3)

        if kpts_band is not None:
            ni = self.view(KNumInt)
            nao = dms.shape[-1]
            dms = dms.reshape(-1,2,nao,nao)
            return ni.nr_uks(cell, grids, xc_code, dms, relativity,
                             hermi, kpt, kpts_band, max_memory, verbose)
        spin = 1
        
        cpu0 = (logger.process_clock(), logger.perf_counter())
        nelec1, exc1, vxc1 = self.nr_uks_uniform_smooth(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )
        logger.timer(self.mf.with_df, 'vxc uniform', *cpu0)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        nelec2, exc2, vxc2 = self.nr_uks_atomic_sharp(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )
        nelec3, exc3, vxc3 = self.nr_uks_atomic_smooth(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )
        logger.timer(self.mf.with_df, 'vxc atomic', *cpu0)
        logger.info(self.mf, 'smooth nelec with uniform grids = %s', nelec1)
        logger.info(self.mf, 'all nelec with atomic grids = %s', nelec2)
        logger.info(self.mf, 'smooth nelec with atomic grids = %s', nelec3)
        logger.info(self.mf, 'sharp nelec with atomic grids = %s', nelec2 - nelec3)
        nelec = nelec1 + nelec2 - nelec3
        exc = exc1 + exc2 - exc3
        vxc = vxc1 + vxc2 - vxc3
        return nelec, exc, vxc
