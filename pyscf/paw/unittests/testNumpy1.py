import sys
from pathlib import Path
import numpy

# Get the path of the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)

# Add the parent directory to sys.path
sys.path.append(parent_dir)


from pyscf import gto, dft
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import NewPAW as PAW
from pyscf.pbc import tools
import pyscf

import time

def testMolecule():
    SPACING = 3.0    # Spacing between Ne atoms in Bohr
    BASIS = 'cc-pvtz'

    # molecule
    def make_neon_chain(n):
        """Creates a simple chain of Neon atoms along the X axis."""
        atoms = []
        for i in range(n):
            atoms.append(['Ne', [i * SPACING, 0, 0]])
        return atoms

    geo = make_neon_chain(50)

    mol = gto.M(atom=geo, basis=BASIS, unit='Bohr', verbose=0)
    mol.max_memory = 15000

    # paw init
    cell = pgto.M(atom=geo, basis=BASIS, a=numpy.eye(3)*100, unit='Bohr', verbose=0, ke_cutoff=50)
    cell.max_memory = 15000
    # mesh = pyscf.pbc.tools.cutoff_to_mesh(cell.lattice_vectors(), cell.ke_cutoff)
    # Rgrid = cell.get_uniform_grids(mesh=mesh, wrap_around=False)
    mf_paw = dft.RKS(mol).density_fit()
    mf_paw.xc = ''
    mydf = PAW.from_mf(mf_paw, cell).build()
    mf_paw.with_df = mydf
    from paw_helper import makeAugmentationSphere1, makeAugmentationSphere
    from paw_helper import makeWignerSeitz

    L = numpy.array([6])
    alpha0 = mydf.alpha0
    Periodic = False
    Rgrid = mydf.Rgrid

    # import pdb; pdb.set_trace()
    # start = time.time()
    # data = makeWignerSeitz(Rgrid, cell, Periodic=Periodic)
    # result = makeAugmentationSphere(data, cell, L, alpha0, Rb=None, epsilon=1.e-5)
    # print(f'old takes {time.time() - start: .2f} s')
    start = time.time()
    result1 = makeAugmentationSphere1(
        Rgrid, cell, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=Periodic
    )
    print(f'new takes {time.time() - start: .2f} s')

    # assert(numpy.isclose(result[0][10], result1[0][10]).all())

def testPeriodic():
    L = 5.443702372939453  # box size

    # specify He2
    atom    = '''
    Si 4.082777 4.082777 1.360926
    Si -0.000000 2.721851 2.721851
    Si 4.082777 1.360926 4.082777
    Si 0.000000 0.000000 0.000000
    Si 1.360926 4.082777 4.082777
    Si 2.721851 2.721851 0.000000
    Si 1.360926 1.360926 1.360926
    Si 2.721851 0.000000 2.721851
    '''
    rscales = numpy.linspace(0.7, 1.5, 15)
    # rscales = [1.5]
    print(rscales)

    basis = "ccpvdz"
    verbose = 4
    a = numpy.eye(3) * L
    ke_cutoff = 200
    precision = 1e-5

    cell = pgto.M(
        atom        = atom,
        basis       = basis,
        verbose     = verbose,
        a           = a,
        ke_cutoff   = ke_cutoff,
        precision   = precision
    )

    ncopy = [1, 1, 1]
    cell = tools.super_cell(cell, ncopy)
    # mesh = pyscf.pbc.tools.cutoff_to_mesh(cell.lattice_vectors(), cell.ke_cutoff)
    # Rgrid = cell.get_uniform_grids(mesh=mesh, wrap_around=False)
    mf_paw = pscf.RKS(cell)
    mf_paw.xc = ''
    mydf = PAW.from_mf(mf_paw).build()
    mf_paw.with_df = mydf
    from paw_helper import makeAugmentationSphere1, makeAugmentationSphere
    from paw_helper import makeWignerSeitz

    L = numpy.array([6])
    alpha0 = mydf.alpha0
    Periodic = True
    Rgrid = mydf.Rgrid

    import pdb; pdb.set_trace()
    start = time.time()
    data = makeWignerSeitz(Rgrid, cell, Periodic=Periodic)
    result = makeAugmentationSphere(data, cell, L, alpha0, Rb=None, epsilon=1.e-5)
    print(f'old takes {time.time() - start: .2f} s')
    start = time.time()
    result1 = makeAugmentationSphere1(
        Rgrid, cell, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=Periodic
    )
    print(f'new takes {time.time() - start: .2f} s')
    for i in range(cell._atm.shape[0]):
        assert(numpy.isclose(result[0][i], result1[0][i]).all())


def main():
    # testMolecule()
    testPeriodic()

if __name__ == '__main__':
    main()

