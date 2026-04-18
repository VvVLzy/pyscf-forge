from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import pyscf

import numpy
from cp2kparser import load_cp2k_molden

# specify He2
L = 3.  # box size
a_He = numpy.eye(3) * L
r1      = 0.    # location of first carbon
r0      = 1.0    # N-N bond length
atom_He   = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
He = (a_He, atom_He)

a, atom = He

basis = {'He': gto.basis.parse('''
He    S
      1.2400000              1.0       
He    S
      0.0576000              1.0000000        
''')}
verbose = 4
ke_cutoff = 400
precision = 1e-8

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision
)

init_guess = '1e'
xc = ''
alpha0 = 12.5

def main():
    ## tz cp2k converged
    n_basis = cell.nao
    n_mo = cell.nelec[0]
    mo_coeff, mo_energy, mo_occ = load_cp2k_molden('He2-MO.molden', n_basis, n_mo)
    mo_coeff_cp2k = numpy.zeros((n_basis,n_basis))
    mo_coeff_cp2k[:, :n_mo] = mo_coeff

    mo_occ_cp2k = numpy.zeros(n_basis) 
    mo_occ_cp2k[:n_mo] = numpy.array(mo_occ)
    dm_cp2k = pscf.hf.make_rdm1(mo_coeff_cp2k, mo_occ_cp2k)

    mesh = pyscf.pbc.tools.cutoff_to_mesh(cell.lattice_vectors(), cell.ke_cutoff)
    Rgrid = cell.get_uniform_grids(mesh=mesh, wrap_around=False)
    dv = cell.vol / Rgrid.shape[0]

    aoOnR_tilde = cell.pbc_eval_gto('GTOval', Rgrid)
    smoothrho = numpy.einsum('rm,rn,mn->r', aoOnR_tilde, aoOnR_tilde, dm_cp2k)
    nelec_smooth2 = smoothrho.sum() * dv
    print(nelec_smooth2)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()