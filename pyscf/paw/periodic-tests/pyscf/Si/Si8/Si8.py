import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

L = 5.443702372939453  # box size

# specify He2
atom    = '''
Si 4.082777 4.082777 1.360926
Si -0.000000 2.721851 2.721851
Si 4.082777 1.360926 4.082777
Si 0.000000 0.000000 0.000000
Si 1.360926 4.082777 4.082777
Si 2.721851 2.721851 0.000000
Si 1.360926 1.360926 1.360926
Si 2.721851 0.000000 2.721851
'''
rscales = numpy.linspace(0.7, 1.5, 15)
# rscales = [1.5]
print(rscales)

basis = "ccpvdz"
verbose = 4
a = numpy.eye(3) * L
ke_cutoff = 200
precision = 1e-5

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision
)

cell_fftdf = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = 3,
    a           = a,
    ke_cutoff   = 2000,
    precision   = precision
)

# TODO: the default minao guess usually give mo that is not (cell.nao, cell.nao)
# and thus will give an error
init_guess = '1e'
xc = 'pbe'

def single_point_paw_manual(alpha0=None, Rb=None):
    mf_per_rks_paw = pscf.RKS(cell)
    mf_per_rks_paw.init_guess = init_guess
    mf_per_rks_paw.xc = xc
    mf_per_rks_paw.max_cycle = 2
    mydf = PAW.from_mf(mf_per_rks_paw, alpha0=alpha0, augRadius=Rb).build()
    mf_per_rks_paw.with_df = mydf
    pawnumint = PAWNumInt(mydf, mf_per_rks_paw, with_multigrid=2)
    mf_per_rks_paw._numint = pawnumint
    mf_per_rks_paw.kernel()

def main():
    single_point_paw_manual(alpha0=10, Rb=1.5)

if __name__ == '__main__':
    main()
