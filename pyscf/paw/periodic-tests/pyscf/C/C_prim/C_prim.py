import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

a = numpy.array([[ 2.18050236,  0.,          1.25891346],
                 [ 0.72683412,  2.05579703,  1.25891346],
                 [-0.,          0.,          2.51782692]])

atom = '''
C 2.543919 1.798822 4.406197
C 0.363417 0.256975 0.629457
'''

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
    pawnumint = PAWNumInt(mydf, mf_per_rks_paw)
    mf_per_rks_paw._numint = pawnumint
    mf_per_rks_paw.kernel()

def main():
    single_point_paw_manual(alpha0=10, Rb=1.5)

if __name__ == '__main__':
    main()
