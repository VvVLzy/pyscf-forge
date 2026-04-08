from pyscf.pbc import dft
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import numpy

from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

# ## conventional
# # lattice
# L = 3.56074511  # box size
# a = numpy.eye(3) * L

# # specify He2
# atom    = '''
# C 0.000000 0.000000 1.780373
# C 0.890186 0.890186 2.670559
# C 0.000000 1.780373 0.000000
# C 0.890186 2.670559 0.890186
# C 1.780373 0.000000 0.000000
# C 2.670559 0.890186 0.890186
# C 1.780373 1.780373 1.780373
# C 2.670559 2.670559 2.670559
# '''

# primitive
a = numpy.array([[ 2.18050236,  0.,          1.25891346],
                 [ 0.72683412,  2.05579703,  1.25891346],
                 [-0.,          0.,          2.51782692]])

atom = '''
C 2.543919 1.798822 4.406197
C 0.363417 0.256975 0.629457
'''
rscales = numpy.linspace(0.9, 1.1, 5)
print(rscales)

basis = "ccpvdz"
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

coords_bohr = cell.atom_coords() 
lattice_bohr = cell.lattice_vectors()

init_guess = '1e'
xc = ''

def paw():
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 50
    mydf = PAW.from_mf(mf_per_rks_paw, alpha0=10, augRadius=1.7, gdfNuc=False).build()
    # import pdb; pdb.set_trace()
    mf_per_rks_paw.with_df = mydf
    pawnumint = PAWNumInt(mydf, mf_per_rks_paw)
    mf_per_rks_paw._numint = pawnumint
    mf_per_rks_paw.kernel()
    return mf_per_rks_paw

def gdf():
    from pyscf.pbc import df
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 50
    gdf_obj = df.GDF(cell)
    mf_per_rks_paw.with_df = gdf_obj
    mf_per_rks_paw.kernel()

    return mf_per_rks_paw

def main():
    mf_paw = paw()
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