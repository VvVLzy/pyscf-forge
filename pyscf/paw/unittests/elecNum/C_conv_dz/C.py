from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf import lib
import numpy
from cp2kparser import load_cp2k_molden

import pyscf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt, smoothCell

# specify He2
L = 3.56074511  # box size
a = numpy.eye(3) * L

# specify He2
atom    = '''
C 0.000000 0.000000 1.780373
C 0.890186 0.890186 2.670559
C 0.000000 1.780373 0.000000
C 0.890186 2.670559 0.890186
C 1.780373 0.000000 0.000000
C 2.670559 0.890186 0.890186
C 1.780373 1.780373 1.780373
C 2.670559 2.670559 2.670559
'''

# basis = "ccpvtz"
basis = {'C': gto.basis.parse('''
C    S
   6665.0000000              0.0006920             -0.0001460        
   1000.0000000              0.0053290             -0.0011540        
    228.0000000              0.0270770             -0.0057250        
     64.7100000              0.1017180             -0.0233120        
     21.0600000              0.2747400             -0.0639550        
      7.4950000              0.4485640             -0.1499810        
      2.7970000              0.2850740             -0.1272620        
      0.5215000              0.0152040              0.5445290        
C    S
      0.1596000              1.0000000        
C    P
      9.4390000              0.0381090        
      2.0020000              0.2094800        
      0.5456000              0.5085570        
C    P
      0.1517000              1.0000000        
C    D
      0.5500000              1.0000000
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

init_guess = '1e'
xc = ''
alpha0 = 12.5

## tz cp2k converged
n_basis = cell.nao
n_mo = cell.nelec[0]
mo_coeff, mo_energy, mo_occ = load_cp2k_molden('C-MO.molden', n_basis, n_mo)
mo_coeff_cp2k = numpy.zeros((n_basis,n_basis))
mo_coeff_cp2k[:, :n_mo] = mo_coeff

mo_occ_cp2k = numpy.zeros(n_basis) 
mo_occ_cp2k[:n_mo] = numpy.array([2]*n_mo)
dm_cp2k = pscf.hf.make_rdm1(mo_coeff_cp2k, mo_occ_cp2k)
lib.tag_array(dm_cp2k, mo_coeff=mo_coeff_cp2k, mo_occ=mo_occ_cp2k)
import pdb; pdb.set_trace()

def paw():
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 50
    mydf = PAW.from_mf(mf_per_rks_paw, alpha0=alpha0, augRadius=1.2, gdfNuc=False).build()
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
    # mf_gdf = gdf()
    mf_paw = paw()
    ovlp = cell.pbc_intor('int1e_ovlp')
    import pdb; pdb.set_trace()
    e_fftdf = mf_fftdf.energy_tot(dm=dm_cp2k)
    # e_gdf = mf_gdf.energy_tot(dm=dm_cp2k)
    e_paw = mf_paw.energy_tot(dm=dm_cp2k)

    # smoothcell = smoothCell(cell, alpha0)
    # mesh = pyscf.pbc.tools.cutoff_to_mesh(smoothcell.lattice_vectors(), smoothcell.ke_cutoff)
    # Rgrid = smoothcell.get_uniform_grids(mesh=mesh, wrap_around=False)
    # # import pdb; pdb.set_trace()
    # smoothAOOnR = smoothcell.pbc_eval_gto('GTOval', Rgrid)
    # dv = smoothcell.vol / Rgrid.shape[0]

    # smoothrho = numpy.einsum('rm,rn,mn->r', smoothAOOnR, smoothAOOnR, dm_cp2k)
    # nelec_smooth1 = smoothrho.sum() * dv
    # print(nelec_smooth1)

    # aoOnR_tilde = smoothcell.pbc_eval_gto('GTOval', Rgrid)
    # smoothrho = numpy.einsum('rm,rn,mn->r', aoOnR_tilde, aoOnR_tilde, dm_cp2k)
    # nelec_smooth2 = smoothrho.sum() * dv
    # print(nelec_smooth2)

    print(f'e_paw: {e_paw}')
    print(f'e_fftdf: {e_fftdf}')
    # print(f'e_gdf: {e_gdf}')


if __name__ == '__main__':
    main()