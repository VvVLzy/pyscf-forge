import numpy as np
from pyscf import gto

def my_pbc_eval_gto(cell, eval_name, coords, kpts=None, nimgs=None, rcut=None):
    """
    Evaluate GTOs with periodic boundary conditions by explicitly summing over 
    real-space lattice translations using the non-periodic Mole.eval_gto.
    
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
            Number of periodic images to include in each direction.
        rcut : float, optional
            If provided, only images T where dist(grid, atom+T) < rcut will be 
            included for that specific grid point.
            
    Returns:
        ao_kpts : (nkpts, ..., N, nao) or (..., N, nao) array
            The periodic AO values.
    """
    if nimgs is None:
        nimgs = cell.nimgs
    
    if isinstance(nimgs, (int, np.integer)):
        nx = ny = nz = nimgs
    else:
        nx, ny, nz = nimgs
        
    a = cell.lattice_vectors()
    atom_coords = cell.atom_coords()
    
    mol = cell.copy()
    
    if kpts is None:
        kpts_lst = np.zeros((1, 3))
        is_single_kpt = True
    else:
        kpts_lst = np.reshape(kpts, (-1, 3))
        is_single_kpt = (np.ndim(kpts) == 1 and len(kpts) == 3) or (kpts is None)
        
    nkpts = len(kpts_lst)
    nao = mol.nao_nr()
    ngrids = coords.shape[0]
    
    # Determine comp from a sample evaluation
    sample = mol.eval_gto(eval_name, coords[:1])
    if sample.ndim == 2:
        comp = 1
        out_shape = (nkpts, ngrids, nao)
    else:
        comp = sample.shape[0]
        out_shape = (nkpts, comp, ngrids, nao)
        
    out = np.zeros(out_shape, dtype=np.complex128)
    
    Ts = []
    for ix in range(-nx, nx + 1):
        for iy in range(-ny, ny + 1):
            for iz in range(-nz, nz + 1):
                Ts.append(ix * a[0] + iy * a[1] + iz * a[2])
    
    # Correctly identify AO ranges per atom
    shl_ao_loc = mol.ao_loc_nr()
    atom_ao_loc = [shl_ao_loc[np.where(mol._bas[:, 0] == iat)[0][0]] for iat in range(mol.natm)]
    atom_ao_loc.append(nao)

    for iat in range(mol.natm):
        ao_start, ao_end = atom_ao_loc[iat], atom_ao_loc[iat+1]
        r_iat = atom_coords[iat]
        
        # Determine shells for this atom
        shl_ids = np.where(mol._bas[:, 0] == iat)[0]
        if len(shl_ids) == 0: continue
        shl_slice = (shl_ids[0], shl_ids[-1] + 1)
        
        for T in Ts:
            pos_iat_T = r_iat + T
            if rcut is not None:
                dists = np.linalg.norm(coords - pos_iat_T, axis=1)
                mask = dists < rcut
                if not np.any(mask): continue
                
                ao_T_full = mol.eval_gto(eval_name, coords[mask] - T, shls_slice=shl_slice)
                
                for ik, kpt in enumerate(kpts_lst):
                    phase = np.exp(1j * np.dot(kpt, T))
                    if comp == 1:
                        out[ik, mask, ao_start:ao_end] += phase * ao_T_full
                    else:
                        for c in range(comp):
                            out[ik, c, mask, ao_start:ao_end] += phase * ao_T_full[c]
            else:
                ao_T_full = mol.eval_gto(eval_name, coords - T, shls_slice=shl_slice)
                for ik, kpt in enumerate(kpts_lst):
                    phase = np.exp(1j * np.dot(kpt, T))
                    if comp == 1:
                        out[ik, :, ao_start:ao_end] += phase * ao_T_full
                    else:
                        for c in range(comp):
                            out[ik, c, :, ao_start:ao_end] += phase * ao_T_full[c]
                            
    results = []
    for ik, kpt in enumerate(kpts_lst):
        v = out[ik]
        if np.allclose(kpt, 0):
            v = v.real
        results.append(v)
        
    return results[0] if is_single_kpt else results
