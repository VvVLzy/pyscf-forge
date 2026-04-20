from pyscf import gto
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf import lib
import numpy
from cp2kparser import load_cp2k_molden

import pyscf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt, smoothCell

a = numpy.array([[ 2.18050236,  0.,          1.25891346],
                 [ 0.72683412,  2.05579703,  1.25891346],
                 [-0.,          0.,          2.51782692]])

atom = '''
C 2.543919 1.798822 4.406197
C 0.363417 0.256975 0.629457
'''

# basis = "ccpvtz"
# basis_hand = {'C': gto.basis.parse('''
# C    S
#    8236.0000000              0.0005310             -0.0001130        
#    1235.0000000              0.0041080             -0.0008780        
#     280.8000000              0.0210870             -0.0045400        
#      79.2700000              0.0818530             -0.0181330        
#      25.5900000              0.2348170             -0.0557600        
#       8.9970000              0.4344010             -0.1268950        
#       3.3190000              0.3461290             -0.1703520        
#       0.3643000             -0.0089830              0.5986840        
# C    S
#       0.9059000              1.0000000        
# C    S
#       0.1285000              1.0000000        
# C    P
#      18.7100000              0.0140310        
#       4.1330000              0.0868660        
#       1.2000000              0.2902160        
# C    P
#       0.3827000              1.0000000        
# C    P
#       0.1209000              1.0000000        
# C    D
#       1.0970000              1.0000000        
# C    D
#       0.3180000              1.0000000        
# C    F
#       0.7610000              1.0000000
# ''')}
basis = "ccpvtz"
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
xc = 'pbe'
alpha0 = 12.5

# tz cp2k converged
# n_basis = cell.nao
# n_mo = cell.nelec[0]
# mo_coeff, mo_energy, mo_occ = load_cp2k_molden('C-MO.molden', n_basis, n_mo)
# mo_coeff_cp2k = numpy.zeros((n_basis,n_basis))
# mo_coeff_cp2k[:, :n_mo] = mo_coeff

# mo_occ_cp2k = numpy.zeros(n_basis) 
# mo_occ_cp2k[:n_mo] = numpy.array([2]*n_mo)
# dm_cp2k = pscf.hf.make_rdm1(mo_coeff_cp2k, mo_occ_cp2k)
# lib.tag_array(dm_cp2k, mo_coeff=mo_coeff_cp2k, mo_occ=mo_occ_cp2k)

def paw(cell):
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
    mf_paw = paw(cell)
    e = mf_paw.kernel()

if __name__ == '__main__':
    main()