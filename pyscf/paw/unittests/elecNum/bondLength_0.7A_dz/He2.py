from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf import lib
from my_eval_gto import my_pbc_eval_gto, eval_s_gto
import pyscf

import numpy
from cp2kparser import load_cp2k_molden

from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt, smoothCell

# specify He2
L = 3.  # box size
a_He = numpy.eye(3) * L
r1      = 0.    # location of first carbon
r0      = 1.0    # N-N bond length
atom_He   = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
# atom_He   = f'He {r1} {r1} {r1}'
He = (a_He, atom_He)
# rscales = numpy.linspace(0.9, 1.1, 5)
# print(rscales)

a, atom = He

basis = "ccpvdz"
basis = {'He': gto.basis.parse('''
# He    S
#      38.3600000              0.0238090        
# He    S
#     #   5.7700000              0.1548910        
#       2.7700000              1.0       
He    S
    #   1.2400000              0.4699870        
      1.2400000              1.0       
He    S
    #   0.1976000              1.0000000        
      0.0576000              1.0000000        
# He    P
#     #   1.2750000              1.0000000
#       0.1050000              1.0000000
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

# tz cp2k converged
n_basis = cell.nao
n_mo = cell.nelec[0]
mo_coeff, mo_energy, mo_occ = load_cp2k_molden('He2-MO.molden', n_basis, n_mo)
mo_coeff_cp2k = numpy.zeros((n_basis,n_basis))
mo_coeff_cp2k[:, :n_mo] = mo_coeff

mo_occ_cp2k = numpy.zeros(n_basis) 
mo_occ_cp2k[:n_mo] = numpy.array([2]*n_mo)
dm_cp2k = pscf.hf.make_rdm1(mo_coeff_cp2k, mo_occ_cp2k)
lib.tag_array(dm_cp2k, mo_coeff=mo_coeff_cp2k, mo_occ=mo_occ_cp2k)

def paw():
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 50
    mydf = PAW.from_mf(mf_per_rks_paw, alpha0=alpha0, augRadius=1.7, gdfNuc=False).build()
    mf_per_rks_paw.with_df = mydf
    pawnumint = PAWNumInt(mydf, mf_per_rks_paw)
    mf_per_rks_paw._numint = pawnumint

    # hand calc e
    # dm = mf_per_rks_paw.make_rdm1()
    # J = mf_per_rks_paw.get_j(dm=dm_cp2k)
    # hcore = mf_per_rks_paw.get_hcore()
    # e = mf_per_rks_paw.energy_tot(dm_cp2k, hcore, J)
    # print(e)
    return mf_per_rks_paw

def gdf():
    from pyscf.pbc import df
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 50
    gdf_obj = df.GDF(cell)
    mf_per_rks_paw.with_df = gdf_obj
    mf_per_rks_paw.verbose = 1

    return mf_per_rks_paw

def fftdf():
    from pyscf.pbc import df
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 50
    mf_per_rks_paw.verbose = 1

    # J = mf_per_rks_paw.get_j(dm=dm_cp2k)
    # hcore = mf_per_rks_paw.get_hcore()
    # import pdb; pdb.set_trace()
    # e = mf_per_rks_paw.energy_tot(dm=dm_cp2k, h1e=hcore, vhf=J)
    # print(e)

    return mf_per_rks_paw


def main():
    mf_fftdf = fftdf()
    mf_gdf = gdf()
    mf_paw = paw()
    import pdb; pdb.set_trace()
    e_fftdf = mf_fftdf.energy_tot(dm=dm_cp2k)
    e_gdf = mf_gdf.energy_tot(dm=dm_cp2k)
    e_paw = mf_paw.energy_tot(dm=dm_cp2k)

    # smoothcell = smoothCell(cell, alpha0)
    # mesh = pyscf.pbc.tools.cutoff_to_mesh(smoothcell.lattice_vectors(), smoothcell.ke_cutoff)
    # Rgrid = smoothcell.get_uniform_grids(mesh=mesh, wrap_around=False)
    # # import pdb; pdb.set_trace()
    # smoothAOOnR = my_pbc_eval_gto(smoothcell, 'GTOval', Rgrid, nimgs=5)
    # dv = smoothcell.vol / Rgrid.shape[0]

    # smoothrho = numpy.einsum('rm,rn,mn->r', smoothAOOnR, smoothAOOnR, dm_cp2k)
    # nelec_smooth1 = smoothrho.sum() * dv
    # print(nelec_smooth1)

    # aoOnR_tilde = smoothcell.pbc_eval_gto('GTOval', Rgrid)
    # smoothrho = numpy.einsum('rm,rn,mn->r', aoOnR_tilde, aoOnR_tilde, dm_cp2k)
    # nelec_smooth2 = smoothrho.sum() * dv
    # print(nelec_smooth2)

    # print(cell.pbc_intor('int1e_ovlp'))

    print(f'e_paw: {e_paw}')
    print(f'e_fftdf: {e_fftdf}')
    print(f'e_gdf: {e_gdf}')

    # mf_gdf = gdf()
    # dm = mf_gdf.make_rdm1()

    # j1 = mf_paw.get_j(dm=dm)
    # j2 = mf_gdf.get_j(dm=dm)
    # n1 = mf_paw.get_hcore()
    # n2 = mf_gdf.get_hcore()
    # print(numpy.max(numpy.abs(j1-j2)))
    # print(numpy.max(numpy.abs(n1-n2)))
    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()