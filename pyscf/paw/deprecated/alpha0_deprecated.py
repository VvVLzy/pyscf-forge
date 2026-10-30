'''Deprecated (2026-10-30): PAW's first alpha0 rule, the sharpest exponent whose probe Coulomb integral
the mesh reproduces to PWAccuracy. Superseded by paw_alpha0.choose_alpha0; kept for reference and geminit.'''
import numpy
import scipy

from pyscf.pbc import gto as pgto
import pyscf
from pyscf.lib import logger

from ..paw_helper import getAlphaAtomsL, getCoulombFF, modifyMolBasis, smart_einsum


def getIntegralDiff(dmol, L, Rgrid, mesh, FF, Ng, f, Periodic=False, diag=True, gridIdx=None):
    '''
    Calculate the difference between analytical and numerical 2c2e integral
    given a mol object
    '''
    aoOnR = dmol.pbc_eval_gto('GTOval', Rgrid)
    L = aoOnR.shape[1]

    # calculate numerical integral
    density_L = aoOnR.T
    if isinstance(FF, OpenBoundaryCoulomb):
        # the molecule's own box: the open-boundary solve (alpha0_lowmem=False)
        potential_L = FF.potential(density_L.reshape(L, *mesh)).reshape(L, Ng)
    else:
        potential_L = numpy.fft.ifftn(density_L.reshape(L, *mesh), axes=(-3, -2, -1))
        potential_L = numpy.fft.fftn(potential_L * FF, axes=(-3,-2,-1))
        potential_L = potential_L.real.reshape(L, Ng)

    if gridIdx is not None:
        potential_L = potential_L[gridIdx]
    

    if diag:
        VLL_numeric = smart_einsum('Gr, rG->G', potential_L, aoOnR) * f
    else:
        VLL_numeric = smart_einsum('Gr, rL->GL', potential_L, aoOnR) * f

    # calculate analytical integral
    if not Periodic:
        integral = dmol.intor('int2c2e')
        VLL_analytical = numpy.diag(integral) if diag else integral
    else:
        dfbuilder = pyscf.pbc.df.rsdf_builder._RSGDFBuilder(dmol, dmol).build()
        # not sure if this is the right way to do periodic integral
        j2c = dfbuilder.get_2c2e(numpy.zeros((1, 3)))[0]
        VLL_analytical = numpy.diag(j2c) if diag else j2c

    return numpy.abs(VLL_numeric - VLL_analytical)

def getBoundaryAlphaFromBasis(alpha, pmol, Rgrid, mesh, FF, Ng, f, tol=1e-3, alpha0_cutoff=(1,100), Periodic=False):
    '''
    Return the biggest exponent in the basis set that can be accurately represented
    given Rgrid up to tolerance, also return the next biggest exponent
    '''
    # sort all exponents in ascending order
    alphaSorted = numpy.unique(alpha)
    # the range 1 to 10 should cover all practical alpha's in reality
    alphaSorted = alphaSorted[(alpha0_cutoff[0] < alphaSorted) & (alphaSorted < alpha0_cutoff[1])]

    # initialize dummy atom at the center of unit cell
    center_coords = numpy.sum(pmol.a, axis=0) * 0.5
    atom = f'He {center_coords[0]} {center_coords[1]} {center_coords[2]}'

    # s-type function for each exponent
    bas = {'He': [ [0, [a*2, 1.]] for a in alphaSorted]}
    dmol = pgto.M(atom=atom, basis=bas, a=pmol.a, unit=pmol.unit)

    VLL_diff = getIntegralDiff(dmol, len(alphaSorted), Rgrid, mesh, FF, Ng, f, Periodic=Periodic)

    # determine which exponents are too sharp to be represented by PWs
    sharpIdx = numpy.where(VLL_diff > tol)[0]
    if len(sharpIdx) > 0:
        minSharpIdx = sharpIdx[0]
        assert(len(sharpIdx) == len(alphaSorted)-minSharpIdx) # this should be true ideally
        # assert(sharpIdx[0]+1 == sharpIdx[1])
        maxSoftIdx = minSharpIdx-1 if minSharpIdx !=0 else None
    else:
        minSharpIdx = None
        maxSoftIdx = len(VLL_diff) - 1

    # determine the value of the min sharp exponent and the max soft exponent
    maxSoftAlpha = alpha0_cutoff[0]*2 if maxSoftIdx is None else alphaSorted[maxSoftIdx]*2
    minSharpAlpha = alpha0_cutoff[1]*2 if minSharpIdx is None else alphaSorted[minSharpIdx]*2

    return maxSoftAlpha, minSharpAlpha

def fineGrainAlpha0(maxSoftAlpha, minSharpAlpha, pmol, Rgrid, mesh, FF, Ng, f, tol=1e-3, nsteps=20, Periodic=False):
    '''
    Fine grain the exponent given the boundaries exponents
    We want to use the result of this as our exponent for compensating charge
    '''
    # fine grain the compensating charge exponent
    alphas = numpy.linspace(maxSoftAlpha, minSharpAlpha, nsteps)

    # initialize dummy atom at the center of unit cell
    center_coords = numpy.sum(pmol.a, axis=0) * 0.5
    atom = f'He {center_coords[0]} {center_coords[1]} {center_coords[2]}'

    # s-type function for each exponent
    bas = {'He': [ [0, [a, 1.]] for a in alphas]}
    dmol = pgto.M(atom=atom, basis=bas, a=pmol.a, unit=pmol.unit)
    VLL_diff_fine = getIntegralDiff(dmol, nsteps, Rgrid, mesh, FF, Ng, f, Periodic)
    # pick the biggest one for compensating charge exponent
    softIdx = numpy.where(VLL_diff_fine < tol)[0]
    maxSoftIdx = softIdx[-1] if len(softIdx) > 0 else 0
    alpha0 = alphas[maxSoftIdx]
    alpha0_wf = alpha0 / 2

    return alpha0, alpha0_wf + 1e-5 # just to make sure the edge case works

def getAlpha0(pmol, Rgrid, mesh, Periodic=False, tol=1e-3, coulomb_kernel=None):
    '''
    Get alpha0 (compensating charge), and the threshold (alpha0_wf) below which
    the GTOs are treated as diffuse (on PWs). With coulomb_kernel (an
    OpenBoundaryCoulomb, molecular boxes) the probe potentials are open-boundary;
    without it the cell's form factor (getCoulombFF) is used.
    '''
    alpha, atoms, _, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    if coulomb_kernel is not None and not Periodic:
        FF = coulomb_kernel
    else:
        FF = getCoulombFF(mesh, pmol, Periodic)
    Ng = numpy.prod(mesh)
    f = (pmol.vol/Ng)

    maxSoftAlpha, minSharpAlpha = getBoundaryAlphaFromBasis(alpha, pmol, Rgrid, mesh, FF, Ng, f,
                                                            tol=tol, Periodic=Periodic)
    alpha0, alpha0_wf = fineGrainAlpha0(maxSoftAlpha, minSharpAlpha, pmol, Rgrid, mesh, FF, Ng, f,
                                        tol=tol, Periodic=Periodic)

    # print(f'Recommended alpha0 for the given PW cutoff {alpha0}')
    logger.info(pmol, "Recommended alpha0 for the given PW cutoff: %s", alpha0)
    return alpha0, alpha0_wf

def calculate_alpha0_dict(pmol, Rgrid, mesh, alpha0_input, PWAccuracy, Periodic, alpha0_lowmem, mol=None, neighbor_list=None, coulomb_kernel=None):
    """
    Calculates and constructs a dictionary of alpha0 values for each element.
    If alpha0_input is None, it computes the reference alpha0.
    If it's a number, it's used globally.
    If it's a dict, it uses the provided values and falls back to the reference.
    """
    if alpha0_lowmem:
        # Mini-cell implementation: Use a small box with same resolution (ke_cutoff)
        basis = []
        for v in pmol._basis.values():
            basis.extend(v)
        test_cell = pgto.M(
            atom='He 5 5 5',
            basis={'He': basis},
            a=numpy.eye(3) * 10.0,
            unit='B',
            ke_cutoff=pmol.ke_cutoff,
            verbose=0
        )
        t_pmol, _ = test_cell.decontract_basis()
        t_pmol._basis = modifyMolBasis(t_pmol._basis)
        t_mesh = t_pmol.mesh
        t_Rgrid = t_pmol.get_uniform_grids(mesh=t_mesh)
        alpha0_ref, alpha0_wf_ref = getAlpha0(t_pmol, t_Rgrid, t_mesh, Periodic=Periodic, tol=PWAccuracy)
    else:
        # The molecule's own box, so the molecule's open-boundary solve (the mini-cell
        # above is a 10 Bohr cube the sphere-truncated kernel always fits).
        alpha0_ref, alpha0_wf_ref = getAlpha0(pmol, Rgrid, mesh, Periodic=Periodic, tol=PWAccuracy,
                                              coulomb_kernel=coulomb_kernel)

    elements = list(pmol._basis.keys())
    alpha0_dict = {}

    log_target = mol if mol is not None else pmol

    if alpha0_input is None:
        alpha0_dict = {el: alpha0_ref for el in elements}
    elif isinstance(alpha0_input, dict):
        coords = mol.atom_coords() if mol is not None else None
        if Periodic and mol is not None:
            L = mol.lattice_vectors()
            L_inv = numpy.linalg.inv(L)
            
        for el in elements:
            if el in alpha0_input:
                val = alpha0_input[el]
                if val is None and neighbor_list is not None and mol is not None:
                    # Calculate min distance
                    type_indices = [i for i in range(mol.natm) if mol.atom_symbol(i) == el]
                    min_dist = float('inf')
                    for i in type_indices:
                        for j in neighbor_list[i]:
                            if i == j or mol.atom_symbol(j) not in alpha0_input: continue
                            diff = coords[i] - coords[j]
                            if Periodic:
                                frac_diff = numpy.dot(diff, L_inv)
                                frac_diff -= numpy.round(frac_diff)
                                diff = numpy.dot(frac_diff, L)
                            dist = numpy.linalg.norm(diff)
                            if dist < min_dist:
                                min_dist = dist
                    
                    if min_dist != float('inf'):
                        print(f"Smallest nearest neighbor distance for element {el}: {min_dist:.4f} Bohr")
                        val = solve_alpha(min_dist / 2.0)
                        print(f"Calculated alpha0 for {el} based on Rb={min_dist/2.0:.4f}: {val:.4f}")
                    else:
                        val = alpha0_ref # Fallback if no neighbors found
            else:
                val = alpha0_ref
                
            alpha0_dict[el] = val
            if val > alpha0_ref:
                logger.warn(log_target, 'Provided alpha0 (%.3f) for element %s is greater than the recommended alpha0_ref (%.3f). The provided PW cutoff may not be accurate enough.', val, el, alpha0_ref)
    else: # single number
        alpha0_dict = {el: alpha0_input for el in elements}
        if alpha0_input > alpha0_ref:
            logger.warn(log_target, 'Provided alpha0 (%.3f) is greater than the recommended alpha0_ref (%.3f). The provided PW cutoff may not be accurate enough.', alpha0_input, alpha0_ref)

    return alpha0_dict


def solve_alpha(r_c, l=1, epsilon=1e-4):
    """
    Solves for alpha in the equation:
    sqrt((2*(2*alpha)**(l+1.5))/gamma(l+1.5)) * r_c**l * exp(-alpha * r_c**2) = epsilon
    """
    
    # 1. Define the substituted variables to keep the code clean
    p = (l + 1.5) / 2.0
    
    # Calculate the constant C
    numerator_C = 2**(l + 2.5)
    denominator_C = scipy.special.gamma(l + 1.5)
    C = (r_c**l) * numpy.sqrt(numerator_C / denominator_C)
    
    # 2. Calculate z (the argument for the Lambert W function)
    z = -(r_c**2 / p) * (epsilon / C)**(1.0 / p)
    
    # 3. Apply the Lambert W function
    # We use the k=-1 branch because we expect a small epsilon and a large positive alpha.
    # The argument z will be small and negative (between -1/e and 0).
    W_val = scipy.special.lambertw(z, k=-1)
    
    # 4. Solve for alpha
    alpha = -(p / r_c**2) * W_val
    
    # The Lambert W function returns a complex number type in SciPy.
    # The physical solution we want is purely real, so we return the real part.
    return alpha.real
