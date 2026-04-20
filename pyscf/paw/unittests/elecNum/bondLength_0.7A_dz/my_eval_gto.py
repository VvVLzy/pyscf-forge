import numpy as np
from pyscf import gto
from pyscf.paw.PAWutils import basisnorm

def my_pbc_eval_gto(cell, eval_name, coords, kpts=None, nimgs=None):
    """
    Evaluate GTOs with periodic boundary conditions by explicitly summing over 
    real-space lattice translations using the non-periodic Mole.eval_gto.
    
    This function is intended for debugging discrepancies in pyscf's pbc_eval_gto.
    
    Args:
        cell : pyscf.pbc.gto.Cell
            The periodic cell object.
        eval_name : str
            Type of evaluation, e.g., 'GTOval', 'GTOval_ip', 'GTOval_sph'.
        coords : (N, 3) array
            Coordinates of the grids where the GTOs are evaluated.
        kpts : (3,) array, (nkpts, 3) array, or None
            k-point(s) for the PBC evaluation. Defaults to Gamma point if None.
        nimgs : int or (3,) sequence of ints, optional
            Number of periodic images to include in each direction (sum from -n to n).
            If None, defaults to cell.nimgs.
            
    Returns:
        ao_kpts : (nkpts, ..., N, nao) or (..., N, nao) array
            The periodic AO values. Matches the output format of cell.pbc_eval_gto.
    """
    if nimgs is None:
        # Use nimgs calculated by PySCF based on precision if not provided
        nimgs = cell.nimgs
    
    if isinstance(nimgs, (int, np.integer)):
        nx = ny = nz = nimgs
    else:
        nx, ny, nz = nimgs
        
    a = cell.lattice_vectors()
    
    # Create a non-periodic Mole object copy for standard evaluation
    mol = cell.copy()
    
    # Handle k-points
    if kpts is None:
        kpts_lst = np.zeros((1, 3))
        is_single_kpt = True
    else:
        kpts_lst = np.reshape(kpts, (-1, 3))
        # Match pyscf convention: return single array if kpts is None or (3,)
        is_single_kpt = (np.ndim(kpts) == 1 and len(kpts) == 3) or (kpts is None)
        
    nkpts = len(kpts_lst)
    nao = mol.nao_nr()
    ngrids = coords.shape[0]
    
    # Determine the component count (e.g., 3 for 'GTOval_ip')
    # We evaluate a single point to probe the return shape of mol.eval_gto
    sample = mol.eval_gto(eval_name, coords[:1])
    if sample.ndim == 2:
        # Shape: (ngrids, nao)
        comp = 1
        out_shape = (nkpts, ngrids, nao)
    else:
        # Shape: (comp, ngrids, nao)
        comp = sample.shape[0]
        out_shape = (nkpts, comp, ngrids, nao)
        
    # Accumulate results in complex array
    out = np.zeros(out_shape, dtype=np.complex128)
    
    # Explicit lattice sum
    for ix in range(-nx, nx + 1):
        for iy in range(-ny, ny + 1):
            for iz in range(-nz, nz + 1):
                T = ix * a[0] + iy * a[1] + iz * a[2]
                
                # Shifted coordinates evaluation:
                # phi_periodic(r) = sum_T exp(i k.T) * phi(r - T)
                # phi(r - T - R_atom) = phi((r - T) - R_atom)
                ao_T = mol.eval_gto(eval_name, coords - T)
                
                for ik, kpt in enumerate(kpts_lst):
                    phase = np.exp(1j * np.dot(kpt, T))
                    if comp == 1:
                        out[ik] += phase * ao_T
                    else:
                        for c in range(comp):
                            out[ik, c] += phase * ao_T[c]
                            
    # Final formatting and real-conversion for Gamma point
    results = []
    for ik, kpt in enumerate(kpts_lst):
        v = out[ik]
        if np.allclose(kpt, 0):
            # Strip near-zero imaginary part at Gamma point
            v = v.real
        results.append(v)
        
    if is_single_kpt:
        return results[0]
    else:
        return results

def eval_s_gto(Rgrid, alpha, a, nimg=[1,1,1]):
    """
    Evaluate an unnormalized s-type GTO and its periodic images on a grid.
    
    Args:
        Rgrid : (N, 3) array
            Grid points.
        alpha : float
            Exponent of the s-type GTO.
        a : (3, 3) array
            Lattice vectors.
        nimg : list of 3 ints
            Number of periodic images in each direction.
            
    Returns:
        out : (N,) array
            Summed GTO values.
    """
    nx, ny, nz = nimg
    out = np.zeros(Rgrid.shape[0])
    for ix in range(-nx, nx + 1):
        for iy in range(-ny, ny + 1):
            for iz in range(-nz, nz + 1):
                T = ix * a[0] + iy * a[1] + iz * a[2]
                r_diff = Rgrid - T
                r2 = np.sum(r_diff**2, axis=1)
                out += basisnorm(alpha,0) / np.sqrt(4*np.pi) * np.exp(-alpha * r2)
    return out
