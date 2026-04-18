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
L = 3.  # box size
a_He = numpy.eye(3) * L
r1      = 0.    # location of first carbon
r0      = 1.0    # N-N bond length
atom_He   = f'He {r1} {r1} {r1}; He {r1+r0} {r1} {r1}'
He = (a_He, atom_He)
# rscales = numpy.linspace(0.9, 1.1, 5)
# print(rscales)

a, atom = He

# basis = "ccpvtz"
basis = {'He': gto.basis.parse('''
He    S
    # 234.0000000              0.0025870        
     35.1600000              0.0195330
      7.9890000              0.0909980
      2.2120000              0.2720500
# He    S
#       0.6669000              1.0000000        
He    S
      0.1089000              1.0000000        
# He    P
#       3.0440000              1.0000000        
# He    P
#       0.7580000              1.0000000        
# He    D
#       1.9650000              1.0000000
''')}
verbose = 4
ke_cutoff = 1500
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
mo_coeff, mo_energy, mo_occ = load_cp2k_molden('He2-MO.molden', n_basis, n_mo)
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
    mf_gdf = gdf()
    mf_paw = paw()
    ovlp = cell.pbc_intor('int1e_ovlp')
    print(ovlp)
    import pdb; pdb.set_trace()
    e_fftdf = mf_fftdf.energy_tot(dm=dm_cp2k)
    e_gdf = mf_gdf.energy_tot(dm=dm_cp2k)
    e_paw = mf_paw.energy_tot(dm=dm_cp2k)
    e_paw1 = mf_paw.energy_tot(dm=dm_cp2k, h1e=mf_fftdf.get_hcore())

    ks_cp2k = numpy.array([
        [0.7504973867031, -0.5846200447774, -0.0901759397758, -0.5208937836981],
        [-0.5846200447774, -0.1688378486388, -0.5208937750218, -0.1826734477934],
        [-0.0901759397758, -0.5208937750218, 0.7504974192164, -0.5846200344691],
        [-0.5208937836981, -0.1826734477934, -0.5846200344691, -0.1688378425689]
    ])
    e_cp2k = numpy.einsum('mn,mn', ks_cp2k, dm_cp2k)
    print(f'e_paw: {e_paw}')
    print(f'e_paw (fftdf core): {e_paw1}')
    print(f'e_cp2k: {e_cp2k}')
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
