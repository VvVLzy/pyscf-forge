import numpy
from pyscf import lib
from pyscf.pbc.dft.numint import NumInt, KNumInt, nr_rks
from pyscf.pbc.dft.gen_grid import BeckeGrids
from .PAWutils import buildPmolAtom

from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

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