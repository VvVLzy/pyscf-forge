from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf import scf, gto

# from paw import PAW
from pyscf.paw import PAW
# from paw import PAW

import numpy

import unittest

def setUpModule():
    global cell, mol, init_guess

    # shared parameters of mol and periodic
    atom =  """
    O        0.000000    0.000000    0.117790
    H        0.000000   -0.755453   -0.471161
    H        0.000000    0.755453   -0.471161
    """
    basis = "cc-pvtz"
    verbose = 4

    # periodic box
    a = numpy.eye(3) * 20
    ke_cutoff = 50

    # Setup basic cell for gamma point tests
    cell = pgto.M(
        atom        = atom,
        basis       = basis,
        verbose     = verbose,
        a           = a,
        ke_cutoff   = ke_cutoff,
    )
    
    # Set up corresponding mol object

    mol = gto.M(
        atom        = atom,
        basis       = basis,
        verbose     = verbose,
    )

    # TODO: the default minao guess usually give mo that is not (cell.nao, cell.nao)
    # and thus will give an error
    init_guess = '1e'

def tearDownModule():
    global cell, mol, init_guess
    del cell, mol, init_guess

class testPAW(unittest.TestCase):
    def test_molecular_rhf(self):
        """Test RHF at gamma point against FFTDF reference"""
        # Reference molecular calculation
        mf_mol_rhf = scf.RHF(mol)
        mf_mol_rhf.init_guess = init_guess
        mf_mol_rhf.kernel()
        e_ref = mf_mol_rhf.e_tot/cell.natm

        # OCCRI calculation
        mf_mol_rhf_paw = scf.RHF(mol).density_fit()
        mf_mol_rhf_paw.init_guess = init_guess
        mydf = PAW.from_mf(mf_mol_rhf_paw, cell).build()
        mf_mol_rhf_paw.with_df = mydf
        
        mf_mol_rhf_paw.kernel()
        e_occri = mf_mol_rhf_paw.e_tot/cell.natm

        # Check convergence and energy agreement
        self.assertTrue(mf_mol_rhf_paw.converged)
        self.assertAlmostEqual(e_ref, e_occri, places=3)

    def test_molecular_dft(self):
        """Test RHF at gamma point against FFTDF reference"""
        # Reference molecular calculation
        mf_mol_rhf = scf.RKS(mol)
        mf_mol_rhf.init_guess = init_guess
        mf_mol_rhf.kernel()
        e_ref = mf_mol_rhf.e_tot/cell.natm

        # OCCRI calculation
        mf_mol_rhf_paw = scf.RKS(mol).density_fit()
        mf_mol_rhf_paw.init_guess = init_guess
        mydf = PAW.from_mf(mf_mol_rhf_paw, cell).build()
        mf_mol_rhf_paw.with_df = mydf
        
        mf_mol_rhf_paw.kernel()
        e_occri = mf_mol_rhf_paw.e_tot/cell.natm

        # Check convergence and energy agreement
        self.assertTrue(mf_mol_rhf_paw.converged)
        self.assertAlmostEqual(e_ref, e_occri, places=3)


if __name__ == "__main__":
    print("Running PAW test suite...")
    unittest.main()