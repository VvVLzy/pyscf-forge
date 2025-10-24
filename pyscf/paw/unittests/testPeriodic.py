from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf import scf, gto

from pyscf.paw import PAW
import pyscf

import numpy

import unittest

def setUpModule():
    global cell, init_guess, xc, auxbasis

    L = 3.  # box size
    # x = L/2 # atom in the center
    x = 0

    atom =  f"""
    He      {x} {x} {x}
    """
    basis = "ccpvtz"
    verbose = 3
    a = numpy.eye(3) * L
    ke_cutoff = 200

    cell = pgto.M(
        atom        = atom,
        basis       = basis,
        verbose     = verbose,
        a           = a,
        ke_cutoff   = ke_cutoff,
        precision   = 1e-5
    )

    # TODO: the default minao guess usually give mo that is not (cell.nao, cell.nao)
    # and thus will give an error
    init_guess = '1e'
    xc = ''
    auxbasis = pyscf.df.aug_etb(cell, beta=1.3)

def tearDownModule():
    global cell, init_guess, xc, auxbasis
    del cell, init_guess, xc, auxbasis

class testPAW(unittest.TestCase):
    def test_periodic_dft(self):
        """Test RHF at gamma point against FFTDF reference"""
        # Reference GDF calculation
        mf_per_rks = pscf.RKS(cell)
        mf_per_rks.init_guess = init_guess
        mf_per_rks.xc = xc
        mf_per_rks = mf_per_rks.density_fit(auxbasis=auxbasis)
        mf_per_rks.kernel()
        e_ref = mf_per_rks.e_tot/cell.natm

        # # Reference FFTDF calculation
        # mf_per_rks_fftdf = pscf.RKS(cell)
        # mf_per_rks_fftdf.init_guess = init_guess
        # mf_per_rks_fftdf.xc = xc
        # mf_per_rks_fftdf.kernel()
        # e_ref = mf_per_rks_fftdf.e_tot/cell.natm

        # PAW calculation
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8).build()
        mf_per_rks_paw.with_df = mydf
        
        mf_per_rks_paw.kernel()
        e_occri = mf_per_rks_paw.e_tot/cell.natm

        # # Check convergence and energy agreement
        # self.assertTrue(mf_per_rks_paw.converged)
        # self.assertAlmostEqual(e_ref, e_occri, places=3)

        # Done: compare original GAPW scf and pyscf periodic to see if error is similar
        # the current GAPW is more accurate than GAPW before (Yes!)



if __name__ == "__main__":
    print("Running PAW test suite...")
    unittest.main()