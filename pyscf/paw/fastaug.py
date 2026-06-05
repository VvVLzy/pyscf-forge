import numpy
from scipy.spatial import cKDTree
from .paw_helper import get_gaussian_radius

def makeAugmentationSphere_Fast(Rgrid, mol, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=False):
    '''
    O(N log Ng) version of makeAugmentationSphere using a cKDTree
    for fast spatial queries, replacing the O(N*Ng) distance broadcasting.
    '''
    # Build the KDTree - O(Ng log Ng)
    # boxsize for periodic boundary conditions in cKDTree
    lattice = mol.lattice_vectors()
    boxsize = lattice.diagonal() if Periodic else None
    
    tree = cKDTree(Rgrid, boxsize=boxsize)
    
    gridIdx = []
    Rs = []
    L_max = L.max()
    
    for atomI in range(mol._atm.shape[0]):
        maxR = Rb if Rb is not None else get_gaussian_radius(alpha0, L_max, epsilon)
        atom_pos = mol.atom_coord(atomI)
        
        # Query points within radius - O(log Ng) per atom
        allIdx = tree.query_ball_point(atom_pos, maxR)
        
        gridIdx.append(numpy.array(allIdx, dtype=int))
        Rs.append(maxR)

    return gridIdx, [None]*len(gridIdx), [None]*len(gridIdx), Rs

def makeAugmentationSphere_BoundingBox(Rgrid, mol, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=False):
    '''
    Bounding box version of makeAugmentationSphere.
    Filters Rgrid points using a Cartesian bounding box for each atom,
    then performs a final distance check.
    Very slow!
    '''
    gridIdx = []
    Rs = []
    L_max = L.max()
    
    for atomI in range(mol._atm.shape[0]):
        maxR = Rb if Rb is not None else get_gaussian_radius(alpha0, L_max, epsilon)
        atom_pos = mol.atom_coord(atomI)
        
        # 1. Cartesian Bounding Box Filter (O(Ng), but with low constant factor)
        # Using numpy vectorized operations is very fast
        mask_box = (numpy.abs(Rgrid[:, 0] - atom_pos[0]) <= maxR) & \
                   (numpy.abs(Rgrid[:, 1] - atom_pos[1]) <= maxR) & \
                   (numpy.abs(Rgrid[:, 2] - atom_pos[2]) <= maxR)
        
        # Candidate indices
        candidate_indices = numpy.where(mask_box)[0]
        
        # 2. Distance Filter
        if len(candidate_indices) > 0:
            sub_coords = Rgrid[candidate_indices]
            diff = sub_coords - atom_pos
            
            # Periodic handling (minimal image)
            if Periodic:
                lattice = mol.lattice_vectors()
                inv_lattice = numpy.linalg.inv(lattice)
                frac_diff = numpy.dot(diff, inv_lattice.T)
                frac_diff -= numpy.round(frac_diff)
                diff = numpy.dot(frac_diff, lattice.T)
            
            dist_sq = numpy.sum(diff**2, axis=-1)
            valid_mask = dist_sq < maxR**2
            
            gridIdx.append(candidate_indices[valid_mask])
        else:
            gridIdx.append(numpy.array([], dtype=int))
            
        Rs.append(maxR)

    return gridIdx, [None]*len(gridIdx), [None]*len(gridIdx), Rs
