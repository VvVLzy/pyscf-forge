import numpy
import pyscf

from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.pbc.df.rsdf_builder import _RSNucBuilder


def addSharpGTO2Atom(mol):
    mol = mol.copy()
    # 2. Define the new s-type Gaussian primitive to add
    # Here, we'll add a single s-type primitive with an exponent and a coefficient.
    new_s_gaussian = [0, [1e12, 1.0]]  # Angular momentum = 0, exponent = 1e12, coefficient = 1.0

    # 3. Create a new basis set dictionary
    custom_basis = {}

    # Iterate through each atom in the molecule and modify its basis
    for atm_symbol, basis_def in mol._basis.items():
        # Make a copy of the current basis definition for the atom
        modified_basis_def = [new_s_gaussian]
        
        # Add the new s-type Gaussian to the list of basis functions
        modified_basis_def.extend(list(basis_def))
        
        # Update the custom basis dictionary
        custom_basis[atm_symbol] = modified_basis_def

    # 4. Modify the molecule's basis set with the new custom basis
    mol.basis = custom_basis

    # 5. Rebuild the molecule object to apply the new basis set
    mol.build()

    return mol

def getAuxbasis(pmol):
    auxbas = {}
    for atom, basOnA in pmol._basis.items():
        shellAtom = []
        for i in range(len(basOnA)):
            shell1 = basOnA[i]
            l1, exp1 = shell1[0], shell1[1][0]
            assert(shell1[1][1] == 1) # should be true for primitive mol
            for j in range(i, len(basOnA)):
                shell2 = basOnA[j]
                l2, exp2 = shell2[0], shell2[1][0]
                assert(shell2[1][1] == 1)
                lmax = l1+l2+1
                lmin = numpy.abs(l1-l2)
                shell = [[l, [exp1+exp2, 1.]] for l in range(lmin, lmax)]
                shellAtom.extend(shell)
        auxbas[atom] = shellAtom

    return auxbas

L = 3  # box size

# specify He2
r1      = 0.    # location of first carbon
r0      = 1.5    # N-N bond length
# atom    = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
atom    = f'He 0 0 0'

# zeta = 'dz'
# basis = "unc-ccpv"+zeta
basis = {'He': gto.basis.parse('''
He    S
     38.3600000              1.0000000        
He    S
      5.7700000              1.0000000        
He    S
      1.2400000              1.0000000        
He    S
      0.2976000              1.0000000        
He    P
      1.2750000              1.0000000 
''')}
verbose = 3
a = numpy.eye(3) * L
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

cellAtom = addSharpGTO2Atom(cell) # add a sharp GTO to represent nuclear charge

mydf = pyscf.pbc.df.RSDF(cellAtom)
mydf.auxbasis = getAuxbasis(cellAtom) # get complete DF basis for single atom
mydf.build()
vpqrs = mydf.get_eri(compact=False).reshape((cellAtom.nao, cellAtom.nao, cellAtom.nao, cellAtom.nao))
nuc_paw = vpqrs[0,0,1:,1:]

dfbuilder = _RSNucBuilder(cell, kpts=numpy.zeros((1,3))).build()
nuc_rsdf = -dfbuilder.get_nuc()[0]/cellAtom._atm[0, 0] # divide by nuclear charge to get the integral (00|RS)

print(numpy.max(numpy.abs(nuc_rsdf-nuc_paw))) # error quite big

import pdb; pdb.set_trace()