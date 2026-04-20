import numpy

import pyscf
from pyscf.pbc import gto as pgto
from pyscf import gto

import sys
from pathlib import Path

# 1. Get the absolute path to the parent directory
parent_dir = str(Path(__file__).resolve().parent.parent)

# 2. Add the parent directory to sys.path
sys.path.append(parent_dir)

# 3. Now you can import your module!
import PAWutilsNumpy as PAWutils



# specify He2
L = 3.  # box size
a_He = numpy.eye(3) * L
r1      = 0.    # location of first carbon
r0      = 1.    # N-N bond length
atom_He   = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
He = (a_He, atom_He)

a_C = numpy.array([[ 2.18050236,  0.,          1.25891346],
                 [ 0.72683412,  2.05579703,  1.25891346],
                 [-0.,          0.,          2.51782692]])

atom_C = '''
C 2.543919 1.798822 4.406197
C 0.363417 0.256975 0.629457
'''
C = (a_C, atom_C)

zeta = 'dz'
basis = "ccpv"+zeta
# ccpvdz
# basis = {'He': gto.basis.parse('''
# He    S
#      38.3600000              0.0238090        
#       5.7700000              0.1548910        
#       1.2400000              0.4699870        
# He    S
#       0.2976000              1.0000000        
# He    P
#       1.2750000              1.0000000 
# ''')}
# basis = {'C': gto.basis.parse('''
# C    S
#    6665.0000000              0.0006920             -0.0001460        
#    1000.0000000              0.0053290             -0.0011540        
#     228.0000000              0.0270770             -0.0057250        
#      64.7100000              0.1017180             -0.0233120        
#      21.0600000              0.2747400             -0.0639550        
#       7.4950000              0.4485640             -0.1499810        
#       2.7970000              0.2850740             -0.1272620        
#       0.5215000              0.0152040              0.5445290        
# C    S
#       0.1596000              1.0000000        
# C    P
#       9.4390000              0.0381090        
#       2.0020000              1.2094800        
#       0.5456000              0.5085570        
# C    P
#       0.1517000              1.0000000        
# C    D
#       0.5500000              1.0000000  
# ''')}



a, atom = He

verbose = 4
ke_cutoff = 200
precision = 1e-8

mol = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision
)

# uncontract basis
pmol, ctr_coeff = mol.decontract_basis()
pmol._basis = PAWutils.modifyMolBasis(pmol._basis)
mesh = pyscf.pbc.tools.cutoff_to_mesh(pmol.lattice_vectors(), pmol.ke_cutoff)
Rgrid = pmol.get_uniform_grids(mesh=mesh, wrap_around=False)

pgInvNP = PAWutils.obtainLocalFnsNewer(
    pmol, mol, ctr_coeff, Rgrid, 12.5, r=None, epsilon=1e-8, Rb=1.2, Periodic=False, rtol=1e-8
)
print("S-block pgInvNP (PySCF) [:4][:,:4]:")
print(pgInvNP[5:10][:,5:10])
print("P-block pgInvNP (PySCF) [4:7, 4:7]:")
print(pgInvNP[13:25][:, 13:25])
print("d-block pgInvNP (PySCF) [4:7, 4:7]:")
print(pgInvNP[25:35][:, 25:35])
print("d-block pgInvNP (PySCF) [4:7, 4:7]:")
print(pgInvNP[35:42][:, 35:42])

# pgInvNP_list = PAWutils.obtainLocalFnsPySCFMatchCP2K(
#     pmol, mol, ctr_coeff, 1.2, 1e-8, 1e-8, rtol=1e-8
# )
# pgInvNP = pgInvNP_list[0]

# print("Submatrix pgInvNP (PySCF) [:4][:,:4]:")
# print(pgInvNP[5:9][:,5:9])
# # print("P-block pgInvNP (PySCF) [4:7, 4:7]:")
# # print(pgInvNP[9:21][:, 9:21])
# print("d-block pgInvNP (PySCF) [4:7, 4:7]:")
# print(pgInvNP[21:][:, 21:])

# pgInvNP_analytical_list = PAWutils.obtainLocalFnsAnalytical(
#     pmol, mol, ctr_coeff, 1.2, 1e-8, 1e-8, rtol=1e-8
# )
# pgInvNP_analytical = pgInvNP_analytical_list[0]

# print("\nSubmatrix pgInvNP (Analytical) [:4][:,:4]:")
# print(pgInvNP[5:9][:,5:9])
# # print("P-block pgInvNP (PySCF) [4:7, 4:7]:")
# # print(pgInvNP[9:21][:, 9:21])
# print("d-block pgInvNP (PySCF) [4:7, 4:7]:")
# print(pgInvNP[21:][:, 21:])