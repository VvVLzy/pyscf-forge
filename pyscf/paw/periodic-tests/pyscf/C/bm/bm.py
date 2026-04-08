from pyscf.pbc import dft
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import numpy

from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

L = 3.56074511  # box size

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
rscales = numpy.linspace(0.9, 1.1, 5)
print(rscales)

basis = "ccpvdz"
verbose = 4
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

coords_bohr = cell.atom_coords() 
lattice_bohr = cell.lattice_vectors()

init_guess = '1e'
xc = ''

volumes = []
es = []
for x in numpy.linspace(0.9, 1.1, 5):
    # x=1
    # scale cell and coord
    scaled_coords = coords_bohr * x
    scaled_lattice = lattice_bohr * x

    cell.set_geom_(scaled_coords, unit='Bohr')
    cell.a = scaled_lattice
    cell.build()


    # paw calculation
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 50
    mydf = PAW.from_mf(mf_per_rks_paw, alpha0=10, augRadius=1.7, gdfNuc=False).build()
    mf_per_rks_paw.with_df = mydf
    pawnumint = PAWNumInt(mydf)
    mf_per_rks_paw._numint = pawnumint
    mf_per_rks_paw.kernel()
    # import pdb; pdb.set_trace()

    # from pyscf.pbc import df
    # mf_per_rks_paw = pscf.RKS(cell)
    # mf_per_rks_paw.init_guess = init_guess
    # mf_per_rks_paw.xc = xc
    # mf_per_rks_paw.max_cycle = 50
    # gdf_obj = df.GDF(cell)
    # mf_per_rks_paw.with_df = gdf_obj

    # convergence trick
    # mf_per_rks_paw.damp = 0.6
    # mf_per_rks_paw.diis_start_cycle = 12
    # mf_per_rks_paw.kernel()


    volumes.append(cell.vol)
    es.append(mf_per_rks_paw.e_tot)

print(volumes)
print(es)
    
# from ase.eos import EquationOfState
# from ase.units import kJ


# eos = EquationOfState(volumes, np.array(es)*Ha)
# v0, e0, B = eos.fit()
# print(B / kJ * 1.0e24, 'GPa')
# eos.plot('Diamond-eos.png')