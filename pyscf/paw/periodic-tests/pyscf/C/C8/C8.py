import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
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
rscales = numpy.linspace(0.7, 1.5, 15)
# rscales = [1.5]
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
    mf_per_rks_paw.max_cycle = 50
    mydf = PAW.from_mf(mf_per_rks_paw, alpha0=alpha0, augRadius=Rb).build()
    mf_per_rks_paw.with_df = mydf
    pawnumint = PAWNumInt(mydf)
    mf_per_rks_paw._numint = pawnumint
    mf_per_rks_paw.kernel()

def main():
    single_point_paw_manual(alpha0=10, Rb=1.5)

if __name__ == '__main__':
    main()
