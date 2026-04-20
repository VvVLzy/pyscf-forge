from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc.gto.eval_gto import _estimate_rcut
import pyscf
import numpy

from pyscf.paw import PAWutils

# specify He2
L = 3.  # box size
a_He = numpy.eye(3) * L
r1      = 0.    # location of first carbon
r0      = 0.7    # N-N bond length
atom_He   = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
He = (a_He, atom_He)
# rscales = numpy.linspace(0.9, 1.1, 5)
# print(rscales)

a, atom = He

# basis = "ccpvtz"
basis = {'He': gto.basis.parse('''
He    S
    234.0000000              0.0025870        
     35.1600000              0.0195330        
      7.9890000              0.0909980        
      2.2120000              0.2720500        
He    S
      0.6669000              1.0000000        
He    S
      0.1089000              1.0000000        
He    P
      3.0440000              1.0000000        
He    P
      0.7580000              1.0000000        
He    D
      1.9650000              1.0000000
''')}
verbose = 4
ke_cutoff = 200
precision = 1e-8

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision
)

mol = cell
alpha0_wf = 12.5

pmol, ctr_coeff = mol.decontract_basis()
pmol._basis = PAWutils.modifyMolBasis(pmol._basis)

mesh = pyscf.pbc.tools.cutoff_to_mesh(pmol.lattice_vectors(), pmol.ke_cutoff)
Rgrid = pmol.get_uniform_grids(mesh=mesh, wrap_around=False)

# aoOnR, aoOnR_tilde = PAWutils.partitionAOs(mol, pmol, Rgrid, ctr_coeff, alpha0_wf)
rcut = _estimate_rcut(pmol, 0)
aoOnR_prim1 = pmol.pbc_eval_gto('GTOval', Rgrid, rcut=numpy.array([30]))
# aoOnR_prim2 = pmol.pbc_eval_gto('GTOval', Rgrid, rcut=numpy.array([30]))
# aoOnR_prim3 = pmol.pbc_eval_gto('GTOval', Rgrid, rcut=numpy.array([40]))
aoOnR_prim = pmol.pbc_eval_gto('GTOval', Rgrid)
import pdb; pdb.set_trace()