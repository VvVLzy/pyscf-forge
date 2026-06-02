import itertools
import numpy
import scipy

import jax
jax.config.update("jax_enable_x64",True)
jax.config.update('jax_platform_name', 'cpu')

import pyscf
from pyscf import gto, lib
from pyscf.pbc import gto as pgto
from pyscf import __config__
from pyscf.lib import logger


from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

def get_Gv(nmesh,reciprocal_vecs):
    rx = numpy.fft.fftfreq(nmesh[0], 1./nmesh[0])
    ry = numpy.fft.fftfreq(nmesh[1], 1./nmesh[1])
    rz = numpy.fft.fftfreq(nmesh[2], 1./nmesh[2])
    return numpy.dot(pyscf.lib.cartesian_prod((rx,ry,rz)), reciprocal_vecs).astype(numpy.float64)

def getFormFactor_Truncated(nmesh,cell):
    L = cell.lattice_vectors().max()/2.
    G2 = get_Gv(nmesh,cell.reciprocal_vectors())**2
    G2 = numpy.sum( G2, axis = 1)

    idx = G2 < 1.e-10
    G2[idx] = 1.e-8    
    G = G2**0.5
    FF = 2 * (numpy.sin(G * L /2.)/G)**2
    FF[idx] = L**2/2.
    return FF * 4 *numpy.pi

def getFormFactor(nmesh,cell):
    G2 = get_Gv(nmesh,cell.reciprocal_vectors())**2
    G2 = numpy.sum( G2, axis = 1)
    FF = numpy.zeros((G2.shape[0]), numpy.float64)
    idx = numpy.greater( G2, 0.)
    FF[idx] = 4. * numpy.pi /G2[idx]
    return FF

def prepareMolForPAW(mol, auto_box=False):
    if auto_box:
        box_lengths = estimate_box_size(mol)
        print(box_lengths)
        logger.info(mol, "Estimated rectangular box size (A): %s", box_lengths)
        if mol.unit[0].capitalize() == 'A':
            mol.a = numpy.diag(box_lengths)
        else:
            mol.a = numpy.diag(box_lengths) / pyscf.data.nist.BOHR

    ##move the atoms so they are in the center of the cell
    atomPos = numpy.asarray([mol._atom[i][1] for i in range(len(mol._atom))])
    center = numpy.sum(atomPos, axis=0)/atomPos.shape[0]    


    displacement = mol.lattice_vectors().diagonal()/2. - center

    atomPos = [(atom[0], [(atom[1][0]+displacement[0])*pyscf.data.nist.BOHR, 
                          (atom[1][1]+displacement[1])*pyscf.data.nist.BOHR, 
                          (atom[1][2]+displacement[2])*pyscf.data.nist.BOHR]) for atom in mol._atom]

    if getattr(mol, 'a', None) is not None:
        if mol.unit[0].capitalize()=='A':
            k=mol.a
        else:
            k=pyscf.data.nist.BOHR*mol.a
    else:
        raise ValueError("mol.a must be defined or auto_box must be True")

    return pgto.M(atom = atomPos, basis = mol.basis, a = k, ke_cutoff = mol.ke_cutoff, unit='A', max_memory=mol.max_memory)

def addSharpGTO2Atom(mol):
    mol = mol.copy()
    # 2. Define the new s-type Gaussian primitive to add
    # Here, we'll add a single s-type primitive with an exponent and a coefficient.
    new_s_gaussian = [0, [1e12, 1.0]]  # Angular momentum = 0, exponent = 1e12, coefficient = 1.0

    # 3. Create a new basis set dictionary
    custom_basis = {}

    # Iterate through each atom in the molecule and modify its basis
    for atm_symbol, basis_def in mol._basis.items():
        # Make a copy of the current basis definition for the atom
        modified_basis_def = [new_s_gaussian]
        
        # Add the new s-type Gaussian to the list of basis functions
        modified_basis_def.extend(list(basis_def))
        
        # Update the custom basis dictionary
        custom_basis[atm_symbol] = modified_basis_def

    # 4. Modify the molecule's basis set with the new custom basis
    mol.basis = custom_basis

    # 5. Rebuild the molecule object to apply the new basis set
    mol.build()

    return mol


def makeWignerSeitz(Rgrid, mol, Periodic=False):
    '''
    low memory version
    '''
    def makeWignerSeitz4Grid(grid):
        grid = numpy.array(grid) # (N, 3)
        atomPos = numpy.asarray([mol._atom[i][1] for i in range(len(mol._atom))]) # [A, 3]
        # Optimized: Calculate distances one atom at a time to avoid (A, N, 3) intermediate
        atomGridDistance = numpy.empty((len(atomPos), len(grid)))
        for i, pos in enumerate(atomPos):
            atomGridDistance[i] = numpy.sqrt(numpy.sum((grid - pos)**2, axis=-1))
        return atomGridDistance

    if Periodic:
        nx = (-1, 0, 1)
        ny = (-1, 0, 1)
        nz = (-1, 0, 1)
        prod = list(itertools.product(nx, ny, nz))
        L = mol.lattice_vectors()

        atomGridDistance = None
        for coeff in prod:
            shift = numpy.dot(coeff, L)
            Rgrid_shifted = Rgrid + shift
            dist = makeWignerSeitz4Grid(Rgrid_shifted)
            if atomGridDistance is None:
                atomGridDistance = dist
            else:
                # Optimized: Keep running minimum to avoid storing 27 full distance arrays
                numpy.minimum(atomGridDistance, dist, out=atomGridDistance)
    else:
        atomGridDistance = makeWignerSeitz4Grid(Rgrid)

    closestAtomToGrid = numpy.argmin(atomGridDistance, axis=0)

    return closestAtomToGrid, atomGridDistance, Rgrid

def getAtomsL(bas, env, cart=False):
    l, atmId, nexp = bas[:,1], bas[:,0], bas[:,3]

    if not cart :
        L     = numpy.concatenate([(2*l[i]+1)*[l[i]]*nexp[i] for i in range(len(l))])
        M     = numpy.concatenate([[1,-1,0]*nexp[i] if l[i] == 1 else list(range(-l[i],l[i]+1))*nexp[i] for i in range(len(l)) ])
        atoms = numpy.concatenate([(2*l[i]+1)*[atmId[i]]*nexp[i] for i in range(len(l))])
    else :
        L     = numpy.concatenate([(l[i]+1)*(l[i]+2)//2*[l[i]]*nexp[i] for i in range(len(l))])
        M     = numpy.concatenate([[1,2,3]*nexp[i] if l[i] == 1 else list(range(tillL(l[i]-1),tillL(l[i]-1)+(l[i]+1)*(l[i]+2)//2))*nexp[i] for i in range(len(l)) ])
        atoms = numpy.concatenate([(l[i]+1)*(l[i]+2)//2*[atmId[i]]*nexp[i] for i in range(len(l))])
    return atoms, L, M

def getAlphaAtomsL(bas, env, cart=False):
    # should only use for primitive basis!!
    assert(len(numpy.unique(bas[:, 2])) == 1)
    assert(numpy.unique(bas[:, 2])[0] == 1)
    l, atmId = bas[:,1], bas[:,0]
    # alpha    = numpy.concatenate([env[bas[i,5]:bas[i,6]] for i in range(bas.shape[0])])
    alpha    = numpy.concatenate([env[bas[i,5]:bas[i,5]+1] for i in range(bas.shape[0])])

    if not cart :
        alpha = numpy.concatenate([(2*li+1)*[a] for li,a in zip(l, alpha)])
        L     = numpy.concatenate([(2*li+1)*[li] for li,a in zip(l, alpha)])
        M     = numpy.concatenate([[1,-1,0] if li == 1 else list(range(-li,li+1)) for li,a in zip(l, alpha) ])
        atoms = numpy.concatenate([(2*li+1)*[a] for li,a in zip(l, atmId)])
    else :
        alpha = numpy.concatenate([(li+1)*(li+2)//2*[a] for li,a in zip(l, alpha)])
        L     = numpy.concatenate([(li+1)*(li+2)//2*[li] for li,a in zip(l, alpha)])
        M     = numpy.concatenate([[1,2,3] if li == 1 else list(range(tillL(li-1),tillL(li-1)+(li+1)*(li+2)//2)) for li,a in zip(l, alpha) ])
        atoms = numpy.concatenate([(li+1)*(li+2)//2*[a] for li,a in zip(l, atmId)])

    return alpha, atoms, L.astype(numpy.int64), M.astype(numpy.int64)



def buildPmolAtom(mol, pmol, atomI, cart):
    # if mol.nao != pmol.nao and isinstance(pmol.basis, str): # mol uses a contracted basis
    #         pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = 'unc-'+pmol.basis, a = pmol.lattice_vectors(), unit='B', cart = cart) 
    # elif mol.nao != pmol.nao or isinstance(pmol.basis, dict):
    #     pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = pmol._basis, a = pmol.lattice_vectors(), unit='B', cart = cart)
    if mol.nao != pmol.nao:
        pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = pmol._basis, a = pmol.lattice_vectors(), unit='B', cart = cart)
    else:
        assert(mol.nao == pmol.nao)
        pmolAtom = pgto.M(atom = [pmol._atom[atomI]], basis = pmol.basis, a = pmol.lattice_vectors(), unit='B',  cart = cart)

    return pmolAtom

def intor_cross(intor, mol1, mol2, shls_slice, comp=None):
    nbas1 = len(mol1._bas)
    nbas2 = len(mol2._bas)
    atmc, basc, envc = gto.mole.conc_env(mol1._atm, mol1._bas, mol1._env,
                                mol2._atm, mol2._bas, mol2._env)


    if (intor.endswith('_sph') or intor.startswith('cint') or
        intor.endswith('_spinor') or intor.endswith('_cart')):
        return gto.moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart == mol2.cart:
        intor = mol1._add_suffix(intor)
        return gto.moleintor.getints(intor, atmc, basc, envc, shls_slice, comp, 0)
    elif mol1.cart:
        mat = gto.moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return gto.numpy.dot(mat, mol2.cart2sph_coeff())
    else:
        mat = gto.moleintor.getints(intor+'_cart', atmc, basc, envc, shls_slice, comp, 0)
        return gto.numpy.dot(mol1.cart2sph_coeff().T, mat)

tillL = lambda l : int( (l+1) * (l+2) * (l+3)//6 )


###
# Modified functions for contracted basis
####

def partitionAOs(mol, pmol, Rgrid, ctr_coeff, alpha0):
    aoOnR = mol.pbc_eval_gto('GTOval', Rgrid)

    aoOnR_prim = pmol.pbc_eval_gto('GTOval', Rgrid)
    alpha, atoms, L, M = getAlphaAtomsL(pmol._bas, pmol._env)

    assert(aoOnR_prim.shape[1] == len(alpha))

    def reconstructAOFromPrim():
        count_prim = 0
        count_contracted = 0
        # aoOnR_from_prim = numpy.zeros_like(aoOnR)
        aoOnR_tilde = numpy.zeros_like(aoOnR)
        for coeff in ctr_coeff:
            nprim = coeff.shape[0]
            ncontracted = coeff.shape[1]
            
            alphaa = alpha[count_prim: count_prim+nprim]
            diffuseIdx = numpy.where(alphaa < alpha0)[0]

            aoOnR_p = aoOnR_prim[:, count_prim: count_prim+nprim]

            # assertions
            LL = numpy.unique(L[count_prim: count_prim+nprim])
            aa = numpy.unique(atoms[count_prim: count_prim+nprim])
            assert(len(LL) == 1)
            assert(len(aa) == 1)
            ###

            # aoOnR_from_prim[:, count_contracted: count_contracted+ncontracted] = aoOnR_p @ coeff
            aoOnR_tilde[:, count_contracted: count_contracted+ncontracted] = aoOnR_p[:, diffuseIdx] @ coeff[diffuseIdx, :]

            count_prim += nprim
            count_contracted += ncontracted

        assert(count_prim == aoOnR_prim.shape[1])
        assert(count_contracted == aoOnR.shape[1])

        return aoOnR, aoOnR_tilde
    
    # test differences
    # print(numpy.max(numpy.abs(aoOnR_from_prim - aoOnR)))


    return reconstructAOFromPrim()

def get_gaussian_radius(alpha, l, eps):
    # Find r where N * r^l * exp(-alpha * r^2) = eps
    # Using a 1D search for simplicity and accuracy
    r = numpy.linspace(0, 50, 5000) 
    # Normalization factor for spherical GTO
    from scipy.special import factorial2
    norm = (2*alpha)**(l/2.+0.75) * numpy.sqrt(2**l / factorial2(2*l+1) / numpy.sqrt(numpy.pi))
    val = norm * (r**l) * numpy.exp(-alpha * (r**2))
    idx = numpy.where(val > eps)[0]
    if len(idx) == 0: return 0.0
    return r[idx[-1]]

def estimate_box_size(mol, epsilon=1e-8):
    """
    Estimates the required box size to contain a molecule's basis functions
    up to a threshold epsilon. Returns the box lengths in Angstrom.
    """
    coords = mol.atom_coords() # Returns coords in Bohr
    natm = mol.natm
    atom_radii = numpy.zeros(natm)
    
    # Find max radius for each atom
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        # Smallest exponent is the most diffuse
        alpha_min = numpy.min(mol.bas_exp(ib))
        r = get_gaussian_radius(alpha_min, l, epsilon)
        atom_radii[ia] = max(atom_radii[ia], r)
    
    # Calculate bounds in Bohr
    lower = numpy.min(coords - atom_radii[:, None], axis=0)
    upper = numpy.max(coords + atom_radii[:, None], axis=0)
    
    # Add a small buffer (e.g., 1.0 Bohr)
    box_lengths_bohr = (upper - lower) + 1.0
    
    # Convert to Angstrom for consistency with your script units
    from pyscf.data.nist import BOHR
    return box_lengths_bohr * BOHR

def makeAugmentationSphere(WignerSeitzData, mol, L, alpha0, Rb=None, epsilon=1.e-5):
    ##this is useful for finding points that are close to atom
    ##while taking into account the periodic images
    '''
    use highest L to find the radius of sphere, include all points within the radius, return both overlapped and not overlapped
    '''
    grid2Atom, atomGridDist, Rgrid = WignerSeitzData
    
    logger.debug(mol, 'L max is :%d', L.max())

    gridIdx, masked_gridIdx, masks, Rs = [], [], [], []
    for atomI in range(mol._atm.shape[0]):
        if Rb is None:
            maxR = get_gaussian_radius(alpha0, L.max(), epsilon)
        else:
            maxR = Rb
        
        # print(f'The augmentation radius is: {maxR}')
        allIdx = numpy.where(atomGridDist[atomI, :] < maxR)[0]
        mask = grid2Atom[allIdx] == atomI
        masked_idx = allIdx[mask] # ensure the grids are in the aug sphere

        gridIdx.append(allIdx)
        masked_gridIdx.append(masked_idx)
        masks.append(mask)
        Rs.append(maxR)

    logger.info(mol, 'The max augmentation radius is: %s Bohr', max(Rs))
    
    return gridIdx, masked_gridIdx, masks, Rs

def get_grid_indices_within_radius(Rgrid, atom_pos, radius, cell_lattice=None):
    """
    Finds indices of points in Rgrid that are within `radius` of `atom_pos`.
    
    Args:
        Rgrid: (N, 3) array of grid coordinates.
        atom_pos: (3,) array of the atom's coordinate.
        radius: float, the maximum distance.
        cell_lattice: (3, 3) array of lattice vectors (optional, for Periodic=True).
                      If None, assumes open boundary conditions.
                      
    Returns:
        1D numpy array of integer indices.
    """
    # 1. Calculate displacement vectors from the atom to all grid points
    # Shape: (N, 3)
    diff = Rgrid - atom_pos
    
    # 2. Apply Minimum Image Convention (PBC wrapping) if lattice is provided
    if cell_lattice is not None:
        # Get fractional coordinates by multiplying by inverse lattice
        L_inv = numpy.linalg.inv(cell_lattice)
        frac_diff = numpy.dot(diff, L_inv)
        
        # Wrap fractional coordinates to [-0.5, 0.5] to find the shortest path
        frac_diff -= numpy.round(frac_diff)
        
        # Convert back to Cartesian coordinates
        diff = numpy.dot(frac_diff, cell_lattice)
        
    # 3. Filter out a small box containing the sphere
    box_mask = (numpy.abs(diff[:, 0]) <= radius) & \
               (numpy.abs(diff[:, 1]) <= radius) & \
               (numpy.abs(diff[:, 2]) <= radius)
    box_indices = numpy.where(box_mask)[0]
    diff_box = diff[box_indices]
    
    # 4. Calculate squared distances within that box
    dist_sq = numpy.sum(diff_box**2, axis=-1)
    
    # 5. Find indices where distance is less than radius
    # We compare squared distances to avoid the expensive numpy.sqrt
    radius_sq = radius**2
    valid_in_box = numpy.where(dist_sq < radius_sq)[0]
    
    valid_indices = box_indices[valid_in_box]
    
    return valid_indices

def makeAugmentationSphere1(Rgrid, mol, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=False):
    '''
    Memory efficient version of makeAugmentationSphere that computes distances
    on-the-fly using get_grid_indices_within_radius instead of requiring a
    pre-computed Wigner-Seitz distance matrix.
    '''
    logger.debug(mol, 'L max is :%d', L.max())

    gridIdx, masked_gridIdx, masks, Rs = [], [], [], []
    
    cell_lattice = mol.lattice_vectors() if Periodic else None
    
    for atomI in range(mol._atm.shape[0]):
        if Rb is None:
            maxR = get_gaussian_radius(alpha0, L.max(), epsilon)
        else:
            maxR = Rb
            
        atom_pos = mol.atom_coord(atomI)
        
        # Get indices directly without storing N_atoms x N_grid matrix
        allIdx = get_grid_indices_within_radius(Rgrid, atom_pos, maxR, cell_lattice=cell_lattice)
        
        # Placeholders for Wigner-Seitz dependent outputs to maintain uniform signature
        mask = None
        masked_idx = None
        
        gridIdx.append(allIdx)
        masked_gridIdx.append(masked_idx)
        masks.append(mask)
        Rs.append(maxR)

    logger.info(mol, 'The max augmentation radius is: %s Bohr', max(Rs))
    
    return gridIdx, masked_gridIdx, masks, Rs

def makeAugmentationSphere_Fast(Rgrid, mol, L, alpha0, Rb=None, epsilon=1.e-5, Periodic=False):
    '''
    O(N log Ng) version of makeAugmentationSphere using a cKDTree
    for fast spatial queries, replacing the O(N*Ng) distance broadcasting.
    '''
    # Build the KDTree - O(Ng log Ng)
    # boxsize for periodic boundary conditions in cKDTree
    lattice = mol.lattice_vectors()
    boxsize = lattice.diagonal() if Periodic else None
    
    tree = scipy.spatial.cKDTree(Rgrid, boxsize=boxsize)
    
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

def getPrimIdxFromAOIdx(mol, pmol):
    ao2prim = []
    count = 0
    for bas in mol._bas:
        l = bas[1]
        nprim = bas[2]
        nao = bas[3]

        primIdx = numpy.arange((2*l+1)*nprim) + count
        ao2prim.extend([primIdx for _ in range((2*l+1)*nao)])

        count = primIdx[-1] + 1

    assert(len(ao2prim) == mol.nao)
    assert(ao2prim[-1][-1] == pmol.nao-1)

    return ao2prim

def getContMatFromID(idx, primIdx, ctr_coeff, labels, mol):
    '''
    return N_prim x N_ao matrix
    '''
    Nao = len(idx)
    NPrim = len(primIdx)
    C_mua = numpy.zeros((NPrim, Nao))

    # determine which shell a given AO is in
    binId = numpy.digitize(idx, labels)

    # determine the relative index with in the shell of a given AO
    labels_buff = numpy.append(labels, 0)
    aoIdInBin = idx - labels_buff[binId-1]

    count = 0 # adding AOs one by one
    NPrimStart = 0
    binLast = binId[0]
    for bI, aobI in zip(binId, aoIdInBin):
        binNew = binId[count]
        NPrimStart = NPrimStart if binNew == binLast else NPrimStart + nPrim
        binLast = binNew

        submat = ctr_coeff[bI][:, aobI]
        nPrim = submat.shape[0]
        C_mua[NPrimStart: NPrimStart+nPrim, count] = submat

        count += 1
        
    assert(count == Nao)

    return C_mua

def labelShellWithIdx(ctr_coeff, mol):
    labels = numpy.zeros((len(ctr_coeff),), dtype=int)

    count = 0
    for i, c in enumerate(ctr_coeff):
        count += c.shape[1]
        labels[i] = count

    assert(count == mol.nao)

    return labels

def getIntegralDiff(dmol, L, Rgrid, mesh, FF, Ng, f, Periodic=False, diag=True, gridIdx=None):
    '''
    Calculate the difference between analytical and numerical 2c2e integral
    given a mol object
    '''
    aoOnR = dmol.pbc_eval_gto('GTOval', Rgrid)

    # calculate numerical integral
    density_L = aoOnR.T
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

def getAlpha0(pmol, Rgrid, mesh, Periodic=False, tol=1e-3):
    '''
    Get alpha0 (compensating charge), and the threshold (alpha0_wf) below which
    the GTOs are treated as diffuse (on PWs)
    '''
    alpha, atoms, _, _ = getAlphaAtomsL(pmol._bas, pmol._env)
    FF = getFormFactor(mesh, pmol).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, pmol).reshape(mesh)
    Ng = numpy.prod(mesh)
    f = (pmol.vol/Ng)

    maxSoftAlpha, minSharpAlpha = getBoundaryAlphaFromBasis(alpha, pmol, Rgrid, mesh, FF, Ng, f,
                                                            tol=tol, Periodic=Periodic)
    alpha0, alpha0_wf = fineGrainAlpha0(maxSoftAlpha, minSharpAlpha, pmol, Rgrid, mesh, FF, Ng, f,
                                        tol=tol, Periodic=Periodic)

    # print(f'Recommended alpha0 for the given PW cutoff {alpha0}')
    logger.info(pmol, "Recommended alpha0 for the given PW cutoff: %s", alpha0)
    return alpha0, alpha0_wf


def make_natural_orbitals(cell, kpts, dms):
    """
    Construct natural orbitals from density matrix.

    This is a pure mathematical operation that performs eigenvalue decomposition
    of density matrices to obtain natural orbitals and their occupations.

    Parameters:
    -----------
    cell : Cell
        PySCF cell object
    kpts : ndarray
        K-point coordinates
    dms : ndarray
        Density matrices with shape (nset, nk, nao, nao)

    Returns:
    --------
    ndarray
        Tagged density matrix array with mo_coeff and mo_occ attributes
    """
    nk = kpts.shape[0]
    nao = cell.nao
    nset = dms.shape[0]

    # Compute k-point dependent overlap matrices
    sk = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    if abs(dms.imag).max() < 1.0e-6:
        sk = [s.real.astype(numpy.float64) for s in sk]

    mo_coeff = numpy.zeros_like(dms)
    mo_occ = numpy.zeros((nset, nk, nao), numpy.float64)

    for i, dm in enumerate(dms):
        for k, s in enumerate(sk):
            # Diagonalize the DM in AO basis: S^{1/2} * DM * S^{1/2}
            A = lib.reduce(numpy.dot, (s, dm[k], s))
            w, v = scipy.linalg.eigh(A, b=s)

            # Sort eigenvalues/eigenvectors in descending order
            mo_occ[i][k] = numpy.flip(w)
            mo_coeff[i][k] = numpy.flip(v, axis=1)

    return lib.tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)

def modifyMolBasis(bas):
    '''
    uncontract mol.basis in the case of a dictionary
    '''
    result = {}
    for atom, basis in bas.items():
        result[atom] = list()
        for shell in basis:
            if len(shell) > 2:
                # this is a contracted shell
                l = shell[0]
                for pgto in shell[1:]:
                    result[atom].append([l, [pgto[0], 1.]])
            else:
                result[atom].append(shell)
    
    return result

def separateNuclearElectron(pmol, VPQRSArr, M_PQLarr, V_PQLarr, V_LMarr):
    VPQRSArrElec, VPQRSArrNuc = [], []
    M_PQLarrElec, M_PQLarrNuc = [], []
    V_PQLarrElec = []
    ZNucArr = []
    for atomI in range(pmol._atm.shape[0]):
    
        VPQRSArrElec.append(VPQRSArr[atomI][1:, 1:, 1:, 1:])
        VPQRSArrNuc.append(VPQRSArr[atomI][0, 0, 1:, 1:])

        M_PQLarrElec.append(M_PQLarr[atomI][1:, 1:, :])
        M_PQLarrNuc.append(M_PQLarr[atomI][0, 0, :])

        V_PQLarrElec.append(V_PQLarr[atomI][1:, 1:, :])

        ZNucArr.append(-pmol._atm[atomI][0])

    V_LMarrElec = V_LMarr # no new compensating charge is added

    result = (VPQRSArrElec, M_PQLarrElec, V_PQLarrElec, V_LMarrElec,
              VPQRSArrNuc, M_PQLarrNuc, ZNucArr
    )

    return result

def getAuxbasis(pmol):
    auxbas = {}
    for atom, basOnA in pmol._basis.items():
        shellAtom = []
        for i in range(len(basOnA)):
            shell1 = basOnA[i]
            l1, exp1 = shell1[0], shell1[1][0]
            assert(shell1[1][1] == 1) # should be true for primitive mol
            for j in range(i, len(basOnA)):
                shell2 = basOnA[j]
                l2, exp2 = shell2[0], shell2[1][0]
                assert(shell2[1][1] == 1)
                lmax = l1+l2+1
                lmin = numpy.abs(l1-l2)
                shell = [[l, [exp1+exp2, 1.]] for l in range(lmin, lmax)]
                shellAtom.extend(shell)
        auxbas[atom] = shellAtom

    return auxbas

def get_atom_radii(mol, epsilon=1e-8):
    """
    For each atom, determine the radius within which its most diffuse 
    basis function decays below epsilon, accounting for contraction coefficients.
    """
    natm = mol.natm
    radii = numpy.zeros(natm)
    for ib in range(mol.nbas):
        ia = mol.bas_atom(ib)
        l = mol.bas_angular(ib)
        exps = mol.bas_exp(ib)
        # bas_ctr_coeff returns (n_prim, n_ctr) matrix
        coeffs = mol.bas_ctr_coeff(ib)
        
        # We care about the tail, dominated by the smallest exponent
        idx_min = numpy.argmin(exps)
        alpha_min = exps[idx_min]
        
        # Max coefficient across all contracted functions for this shell's diffuse primitive
        c_max = numpy.max(numpy.abs(coeffs[idx_min, :]))
        
        # Effective threshold adjusted by the coefficient: |c| * N * G(r) < epsilon
        # So we find where N * G(r) < epsilon / |c|
        eps_eff = epsilon / max(c_max, 1e-15)
        
        r = get_gaussian_radius(alpha_min, l, eps_eff)
        radii[ia] = max(radii[ia], r)
    return radii

def get_neighbor_list(mol, epsilon=1e-8, Periodic=False):
    """
    Generates a neighbor list where atoms i and j are neighbors if
    distance(i, j) < radius_i + radius_j.
    """
    coords = mol.atom_coords() # Coords in Bohr
    natm = mol.natm
    radii = get_atom_radii(mol, epsilon)
    
    neighbor_list = []
    
    if Periodic:
        L = mol.lattice_vectors()
        L_inv = numpy.linalg.inv(L)
    
    for i in range(natm):
        # Displacement from atom i to all other atoms
        diff = coords - coords[i]
        
        if Periodic:
            # Minimum Image Convention
            frac_diff = numpy.dot(diff, L_inv)
            frac_diff -= numpy.round(frac_diff)
            diff = numpy.dot(frac_diff, L)
            
        dist = numpy.sqrt(numpy.sum(diff**2, axis=-1))
        
        # Two atoms are neighbors if their spheres of influence overlap
        thresholds = radii[i] + radii
        neighbors = numpy.where(dist < thresholds)[0]
        neighbor_list.append(neighbors)
        
    return neighbor_list