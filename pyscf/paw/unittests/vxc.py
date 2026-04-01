import numpy
from pyscf import lib
from pyscf.dft import libxc
import sys
from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

# def get_vxc(ni, xc_code, grids_smooth, grids_hard, rho_smooth, rho_hard, rho_smooth_hard):
#     '''
#     Evaluate the GAPW/PAW exchange-correlation energy and potential.

#     Args:
#         ni : numint object
#             used to evaluate the XC functional
#         xc_code : str
#             Exchange-correlation functional code
#         grids_smooth : Uniform grid object
#             The grid for the global smooth density
#         grids_hard : dict or list of Grid objects
#             The local atomic grids (e.g. Becke grids) for the sharp and smooth local densities
#         rho_smooth : ndarray
#             The global smooth density evaluated on grids_smooth
#         rho_hard : list of ndarray
#             The all-electron sharp density evaluated on local grids_hard for each atom
#         rho_smooth_hard : list of ndarray
#             The pseudo smooth density evaluated on local grids_hard for each atom

#     Returns:
#         exc_total : float
#             Total exchange-correlation energy
#         vxc : tuple
#             Exchange-correlation potentials corresponding to the input grids
#     '''
#     # 1. Smooth global contribution
#     exc_s, vxc_s, fxc_s, kxc_s = ni.eval_xc(xc_code, rho_smooth, deriv=1)
#     weight_s = grids_smooth.weights
#     # E_xc = \int exc * rho * dr
#     if exc_s.ndim == 0:  # scalar handle
#         exc_s_tot = 0.0
#     else:
#         # Note: eval_xc returns energy density per particle `exc`
#         # Energy = \int exc * rho * dr
#         if rho_smooth.ndim == 1:
#             exc_s_tot = np.dot(exc_s * rho_smooth, weight_s)
#         else: # spin-polarized or generalized gradient
#             exc_s_tot = np.dot(exc_s * rho_smooth[0], weight_s)

#     exc_total = exc_s_tot
#     vxc_hard_list = []
#     vxc_smooth_hard_list = []

#     # 2. Local atomic contributions
#     # \sum_a (E_xc[rho_hard^a] - E_xc[rho_smooth_hard^a])
#     for a in range(len(grids_hard)):
#         grid_a = grids_hard[a]
#         weights_a = grid_a.weights
        
#         rho_h_a = rho_hard[a]
#         rho_sh_a = rho_smooth_hard[a]

#         # Sharp local contribution
#         exc_h, vxc_h, _, _ = ni.eval_xc(xc_code, rho_h_a, deriv=1)
#         if rho_h_a.ndim == 1:
#             exc_h_tot = np.dot(exc_h * rho_h_a, weights_a)
#         else:
#             exc_h_tot = np.dot(exc_h * rho_h_a[0], weights_a)
        
#         # Smooth local contribution
#         exc_sh, vxc_sh, _, _ = ni.eval_xc(xc_code, rho_sh_a, deriv=1)
#         if rho_sh_a.ndim == 1:
#             exc_sh_tot = np.dot(exc_sh * rho_sh_a, weights_a)
#         else:
#             exc_sh_tot = np.dot(exc_sh * rho_sh_a[0], weights_a)

#         exc_total += (exc_h_tot - exc_sh_tot)
        
#         vxc_hard_list.append(vxc_h)
#         vxc_smooth_hard_list.append(vxc_sh)

#     return exc_total, (vxc_s, vxc_hard_list, vxc_smooth_hard_list)


def smoothCell(mol, alpha0):
    smoothMol = mol.copy()
    env2Mod = smoothMol._env.copy()
    for shell in smoothMol._bas:
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

from pyscf.pbc.dft.numint import NumInt, KNumInt, nr_rks
from pyscf.pbc.dft.gen_grid import BeckeGrids
from PAWutilsNumpy import buildPmolAtom

class PAWNumInt(NumInt):
    def __init__(self, mydf):
        self.cell = mydf.cell
        self.smoothCell = smoothCell(self.cell, mydf.alpha0)
        self.pcell = mydf.pcell
        # self.smoothPcell = smoothCell(self.pcell, mydf.alpha0)
        self.uniform_grid = mydf.grids # this should be a uniform grid

        # atomic data
        self.pcellAtoms, self.smoothPcellAtoms, self.atomicGrids = self.prepareAtomicData(mydf.alpha0)
        self.F_PmuArr = mydf.PAWdata[1]
        self.Ftilde_PmuArr = mydf.PAWdata[2]
        super().__init__()

    def prepareAtomicData(self, alpha0):
        pcellAtoms = []
        smoothPcellAtoms = []
        grids = []

        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = buildPmolAtom(self.cell, self.pcell, atomI, self.pcell.cart)
            smoothpcellAtom = smoothCell(pcellAtom, alpha0)
            grid = BeckeGrids(pcellAtom)
            # grid.level = 5 # doesn't affect much

            pcellAtoms.append(pcellAtom)
            smoothPcellAtoms.append(smoothpcellAtom)
            grids.append(grid)

        return pcellAtoms, smoothPcellAtoms, grids

    def nr_rks_uniform_smooth(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        cell = self.smoothCell
        grids = self.uniform_grid
        return nr_rks(self, cell, grids, xc_code, dms, spin, relativity,
                      hermi, kpt, kpts_band, max_memory, verbose)
    
    def nr_rks_atomic_sharp(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        nelec2 = 0
        exc2 = 0
        vxc2 = numpy.zeros((self.cell.nao, self.cell.nao))

        f = self.F_PmuArr
        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = self.pcellAtoms[atomI]
            grids = self.atomicGrids[atomI]
            dm = smart_einsum('mn,Pm,Qn->PQ', dms, f[atomI], f[atomI])

            nelec2a, exc2a, vxc2a = nr_rks(
                self, pcellAtom, grids, xc_code, dm, spin, relativity,
                hermi, kpt, kpts_band, max_memory, verbose
            )

            nelec2 += nelec2a
            exc2 += exc2a
            vxc2 += smart_einsum('PQ,Pm,Qn->mn', vxc2a, f[atomI], f[atomI])

        return nelec2, exc2, vxc2

    def nr_rks_atomic_smooth(self, xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose):
        nelec3 = 0
        exc3 = 0
        vxc3 = numpy.zeros((self.cell.nao, self.cell.nao))

        f = self.Ftilde_PmuArr
        for atomI in range(self.cell._atm.shape[0]):
            pcellAtom = self.smoothPcellAtoms[atomI]
            grids = self.atomicGrids[atomI]
            dm = smart_einsum('mn,Pm,Qn->PQ', dms, f[atomI], f[atomI])

            nelec3a, exc3a, vxc3a = nr_rks(
                self, pcellAtom, grids, xc_code, dm, spin, relativity,
                hermi, kpt, kpts_band, max_memory, verbose
            )

            nelec3 += nelec3a
            exc3 += exc3a
            vxc3 += smart_einsum('PQ,Pm,Qn->mn', vxc3a, f[atomI], f[atomI])

        return nelec3, exc3, vxc3
    
    @lib.with_doc(nr_rks.__doc__)
    def nr_rks(self, cell, grids, xc_code, dms, relativity=0, hermi=1,
               kpt=numpy.zeros(3), kpts_band=None, max_memory=2000, verbose=None):
        if kpts_band is not None:
            # To compute Vxc on kpts_band, convert the NumInt object to KNumInt object.
            ni = self.view(KNumInt)
            nao = dms.shape[-1]
            dms = dms.reshape(-1,nao,nao)
            return ni.nr_rks(cell, grids, xc_code, dms, relativity,
                             hermi, kpt.reshape(1,3), kpts_band, max_memory, verbose)
        spin = 0
        
        # uniform grid
        nelec1, exc1, vxc1 = self.nr_rks_uniform_smooth(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )

        # atomic grid
        nelec2, exc2, vxc2 = self.nr_rks_atomic_sharp(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )
        nelec3, exc3, vxc3 = self.nr_rks_atomic_smooth(
            xc_code, dms, spin, relativity, hermi, kpt, kpts_band, max_memory, verbose
        )

        nelec = nelec1 + nelec2 - nelec3
        exc = exc1 + exc2 - exc3
        vxc = vxc1 + vxc2 - vxc3
        return nelec, exc, vxc