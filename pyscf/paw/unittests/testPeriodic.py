from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf import scf, gto

from pyscf.paw import PAW
from pyscf.paw import NewPAW
import pyscf

import numpy

import unittest

def setUpModule():
    global cell, cell_fftdf, init_guess, xc, auxbasis

    L = 3.  # box size
    x = L/2 # atom in the center
    # x = 0

    atom =  f"""
    He      {x} {x} {x}
    """
    basis = "ccpvdz"
    # basis = "sto6g"
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

    cell_fftdf = pgto.M(
        atom        = atom,
        basis       = basis,
        verbose     = verbose,
        a           = a,
        ke_cutoff   = 3000,
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
        # mf_per_rks = pscf.RKS(cell)
        # mf_per_rks.init_guess = init_guess
        # mf_per_rks.xc = xc
        # mf_per_rks = mf_per_rks.density_fit()
        # mf_per_rks.kernel()
        # e_ref = mf_per_rks.e_tot/cell.natm

        # # Reference FFTDF calculation
        mf_per_rks_fftdf = pscf.RKS(cell_fftdf)
        mf_per_rks_fftdf.init_guess = init_guess
        mf_per_rks_fftdf.xc = xc
        mf_per_rks_fftdf.kernel()
        e_ref = mf_per_rks_fftdf.e_tot/cell_fftdf.natm

        # PAW calculation
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        # mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8).build()
        mydf = PAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=10, augRadius=1.5).build()
        mf_per_rks_paw.with_df = mydf
        mf_per_rks_paw.kernel()
        e_paw = mf_per_rks_paw.e_tot/cell.natm

        # # Check convergence and energy agreement
        self.assertTrue(mf_per_rks_paw.converged)
        print(f'E(PAW)-E(FFTDF) = {numpy.abs(e_paw-e_ref)}')
        # self.assertAlmostEqual(e_ref, e_paw, places=5)

    def test_periodic_dft_paw_nuc(self):
        """Test RHF at gamma point against FFTDF reference"""
        # Reference GDF calculation
        # mf_per_rks = pscf.RKS(cell)
        # mf_per_rks.init_guess = init_guess
        # mf_per_rks.xc = xc
        # mf_per_rks = mf_per_rks.density_fit()
        # mf_per_rks.kernel()
        # e_ref = mf_per_rks.e_tot/cell.natm

        # # Reference FFTDF calculation
        mf_per_rks_fftdf = pscf.RKS(cell_fftdf)
        mf_per_rks_fftdf.init_guess = init_guess
        mf_per_rks_fftdf.xc = xc
        mf_per_rks_fftdf.kernel()
        e_ref = mf_per_rks_fftdf.e_tot/cell_fftdf.natm

        # PAW calculation
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        # mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8).build()
        mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=20, augRadius=5).build()
        mf_per_rks_paw.with_df = mydf
        mf_per_rks_paw.kernel()
        e_paw = mf_per_rks_paw.e_tot/cell.natm

        # # Check convergence and energy agreement
        self.assertTrue(mf_per_rks_paw.converged)
        print(f'E(PAW)-E(FFTDF) = {numpy.abs(e_paw-e_ref)}')
        # self.assertAlmostEqual(e_ref, e_paw, places=5)

        # Done: compare original GAPW scf and pyscf periodic to see if error is similar
        # the current GAPW is more accurate than GAPW before (Yes!)

    def testPAWNuc(self):
        # PAW calculation
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        # mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8).build()
        mydf = NewPAW.from_mf(mf_per_rks_paw, PWAccuracy=1e-8, alpha0=12.04, augRadius=2).build()
        
        mf_per_rks_paw1 = pscf.RKS(cell)
        mf_per_rks_paw1.init_guess = init_guess
        mf_per_rks_paw1.xc = xc
        mydf1 = PAW.from_mf(mf_per_rks_paw1, PWAccuracy=1e-8).build()

        nuc_paw = mydf.get_nuc()
        nuc1_paw = mydf.get_nuc1()
        nuc2_paw = mydf.get_nuc2()
        nuc3_paw = mydf.get_nuc3()
        nuc_rsdf = mydf1.get_nuc()

        print(f'Difference in smooth part: {numpy.max(numpy.abs(nuc1_paw+nuc3_paw))}')
        print(f'Difference in sharp part: {numpy.max(numpy.abs(nuc2_paw-nuc_rsdf))}')

        import pdb; pdb.set_trace()

if __name__ == "__main__":
    print("Running PAW test suite...")
    unittest.main()