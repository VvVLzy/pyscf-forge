import numpy
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf

L = numpy.array([0, 0, 0, 0, 1, 1, 1], dtype=numpy.int32)
M = numpy.array([0, 0, 0, 0, 1, -1, 0], dtype=numpy.int32)
alpha = numpy.array([38.36, 5.77, 1.24, 0.2976, 1.275, 1.275, 1.275])



def npy():
    import scipy
    def gammaBar(n, l, a):
        """Vectorized Gamma calculation"""
        L_val = n + 2 + l
        LL = (L_val + 1) / 2
        # Use scipy.special.gamma instead of jsp.special.gamma
        return scipy.special.gamma(LL) / (2 * a**(LL))
    l1 = L[:, None]
    l2 = L[None, :]

    alpha1 = alpha[:, None]
    alpha2 = alpha[None, :]
    gamma0 = gammaBar(0, l1+l2, alpha1+alpha2)
    
    return gamma0

def jax():
    import jax
    jax.config.update("jax_enable_x64",True)
    jax.config.update('jax_platform_name', 'cpu')
    import jax.scipy as jsp
    from jax import jit, vmap
    
    @jit
    def gammaBar(n, l, a):
        L = n + 2 + l
        LL = (L+1)/2
        return jsp.special.gamma(LL) / (2*a**(LL))

    @jit
    def getgamma0(l1, m1, l2, m2, alpha1, alpha2, alpha3):
        gamma0 = gammaBar(0, l1+l2, alpha1+alpha2)
        return gamma0
    
    getgamma = vmap(vmap(getgamma0, (None, None, 0, 0, None, 0, None)),
                    (0, 0, None, None, 0, None, None))
    gamma0 = getgamma(L, M, L, M, alpha, alpha, 10.0)

    return gamma0


print(jax()-npy())