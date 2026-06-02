import numpy
import time
from pyscf.pbc.dft.multigrid import multigrid_pair
from pyscf.pbc.df.ft_ao import ft_ao
from pyscf.pbc import gto as pgto
from .paw_jk import getFormFactor, getFormFactor_Truncated
from .paw_1c import addSharpGTO2Atom, modifyMolBasis, getAlphaAtomsL, getMPQLarray, getVLLVPQLarray, getmpql
from .paw_Integrals import intor_cross

from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')

def build_g_G_periodic(gmolCart, gmolSph, Gv, gmax):
    g_G_cart = ft_ao(gmolCart, Gv)
    g_G_sph = ft_ao(gmolSph, Gv)
    
    g_G_list = []
    for atomI in range(gmolCart._atm.shape[0]):
        cart_slice = gmolCart.aoslice_by_atom()[atomI]
        cart_AOs = g_G_cart[:, cart_slice[2]:cart_slice[3]]
        
        if gmax < 2:
            g_G_list.append(cart_AOs[:, [0, 1, 4, 6]])
        else:
            sph_slice = gmolSph.aoslice_by_atom()[atomI]
            sph_AOs = g_G_sph[:, sph_slice[2]:sph_slice[3]]
            
            s00 = cart_AOs[:, 0:1]
            l2 = cart_AOs[:, [4, 7, 9]]
            
            maskSph = numpy.zeros(sph_AOs.shape[-1], dtype=bool)
            maskSph[[0, 6, 8]] = True
            sph = sph_AOs[:, ~maskSph]
            
            g_G_list.append(numpy.hstack([s00, l2, sph]))
            
    return numpy.hstack(g_G_list)

def build_g_R_block_periodic(pmol, gmolCart, gmolSph, coords, ni, mask, gmax):
    ao_cart = ni.eval_ao(gmolCart, coords, non0tab=mask)
    ao_sph = ni.eval_ao(gmolSph, coords, non0tab=mask)
    
    g_R_list = []
    for atomI in range(pmol._atm.shape[0]):
        cart_slice = gmolCart.aoslice_by_atom()[atomI]
        cart_AOs = ao_cart[:, cart_slice[2]:cart_slice[3]]
        
        if gmax < 2:
            g_R_list.append(cart_AOs[:, [0, 1, 4, 6]])
        else:
            sph_slice = gmolSph.aoslice_by_atom()[atomI]
            sph_AOs = ao_sph[:, sph_slice[2]:sph_slice[3]]
            
            s00 = cart_AOs[:, 0:1]
            l2 = cart_AOs[:, [4, 7, 9]]
            
            maskSph = numpy.zeros(sph_AOs.shape[-1], dtype=bool)
            maskSph[[0, 6, 8]] = True
            sph = sph_AOs[:, ~maskSph]
            
            g_R_list.append(numpy.hstack([s00, l2, sph]))
            
    return numpy.hstack(g_R_list)

def compensatingChargeSph_onthefly(pmol, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmol):
    from . import ClebschGordan
    CG = ClebschGordan.RealCG
    M_PQLarray, V_PQLarray, V_LLarray = [], [], []

    atomBass = {}
    atomData = {}
    
    def getCalcIdx(atomInfo, atomBas):
        for i, info in enumerate(atomBas):
            if (atomInfo==info).all():
                return i
        return None

    for atomI in range(pmol._atm.shape[0]):
        shellsA, shellsB = numpy.where(pmol._bas[:,0] == atomI)[0], numpy.where(gmol._bas[:,0] == atomI)[0]

        idx_bas = numpy.where(pmol._bas[:,0] == atomI)[0]
        atomZ = pmol._atm[atomI, 0]
        atomL = pmol._bas[idx_bas][:, 1]
        atomExp = pmol._env[pmol._bas[idx_bas][:, -3]]
        atomInfo = numpy.concatenate((atomL, atomExp))
        
        atomBas = atomBass.get(atomZ, [])
        i = getCalcIdx(atomInfo, atomBas)
        
        if i is not None:
            mpql, vpql, vll = atomData[atomZ][i]
            M_PQLarray.append(mpql)
            V_PQLarray.append(vpql)
            V_LLarray.append(vll)
        else:
            idx  = (atoms==atomI)
            idxg = (atomsG==atomI)
            
            mpql = getmpql(L, M, alpha, LG, MG, alphaG, idx, idxg)

            VPQL = intor_cross('int3c2e', pmol, gmol,  
                                shls_slice=(shellsA[0], shellsA[-1]+1, shellsA[0], shellsA[-1]+1, pmol.nbas + shellsB[0], pmol.nbas + shellsB[-1]+1))

            VLL = gmol.intor('int2c2e', shls_slice=(shellsB[0], shellsB[-1]+1,shellsB[0], shellsB[-1]+1))

            M_PQLarray.append(mpql)
            V_PQLarray.append(VPQL)
            V_LLarray.append(VLL)

            atomBas.append(atomInfo)
            atomBass[atomZ] = atomBas
            data_list = atomData.get(atomZ, [])
            data_list.append((mpql, VPQL, VLL))
            atomData[atomZ] = data_list

    return M_PQLarray, V_PQLarray, V_LLarray, None, None, gmol

def mergeCompensatingCharge_onthefly(pmol, mol, alpha0, Rgrid, epsilon, Rb=None, Periodic = False):
    pmolNuc = addSharpGTO2Atom(pmol)
    alpha, atoms, L, M = getAlphaAtomsL(pmolNuc._bas, pmolNuc._env)
    
    gmax = int(L.max()*2)
    gbasCart, gbasSph = {}, {}
    for atomI in range(pmol._atm.shape[0]):
        elem = pmol._atom[atomI][0]
        gbasSph[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]
        if gmax < 2:
            gbasCart[elem] = [ [0, [alpha0, 1.]], [2, [alpha0, 1.]]]
        else:
            gbasCart[elem] = [ [l, [alpha0, 1.]] for l in range(gmax+1)]

    gmolCart = pgto.M(atom=pmol.atom, basis=gbasCart, a=pmol.a, unit=pmol.unit, cart=True)
    gmolSph = pgto.M(atom=pmol.atom, basis=gbasSph, a=pmol.a, unit=pmol.unit, cart=False)
    alphaG, atomsG, LG, MG = getAlphaAtomsL(gmolSph._bas, gmolSph._env, cart=gmolSph.cart)

    if Periodic:
        M_PQLarray = getMPQLarray(pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmax)
        V_LLarray, V_PQLarray = getVLLVPQLarray(mol, pmolNuc, gmolCart, gmax)
        gmol = gmolCart
    else:
        M_PQLarray, V_PQLarray, V_LLarray, _, _, gmol = compensatingChargeSph_onthefly(
            pmolNuc, atoms, L, M, alpha, atomsG, LG, MG, alphaG, gmolSph
        )

    # Return structure matching what getjSmoothPW2Approach1 needs, dropping massive arrays
    return M_PQLarray, V_PQLarray, V_LLarray, None, None, gmol, gmolCart, gmolSph, gmax, None

def getjSmoothPW2Approach3(cell, dm, mg_ni, mesh, PAWdata, pmol, Periodic=False):
    from pyscf.pbc.dft import gen_grid, numint
    
    # We expect PAWdata to be 13 elements: localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx_none, gOnR_none, _gmol, _gmolCart, _gmolSph, _gmax
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr = PAWdata[:7]
    gmol, gmolCart, gmolSph, gmax = PAWdata[9:13]

    nset = dm.shape[0]
    Ng = numpy.prod(mesh)
    dv = cell.vol / Ng
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    FF_flat = FF.flatten()

    # Pass 1: AO -> Grid (Smooth Density)
    rhoG_eval = multigrid_pair._eval_rhoG(mg_ni, dm, hermi=1, kpts=numpy.zeros((1,3)), deriv=0)
    
    # Custom Mock Grid object using standard cell.get_uniform_grids Rgrid coordinates
    # to perfectly align with the expected domain.
    Rgrid = cell.get_uniform_grids(mesh=mesh, wrap_around=False)
    class MockGrids:
        def __init__(self, coords):
            self.coords = coords
            self.weights = numpy.ones(coords.shape[0])
            self.cutoff = 1e-15
            self.non0tab = None
            self.mesh = mesh
            self.mol = gmol
        def build(self, *args, **kwargs):
            pass
    mock_grids = MockGrids(Rgrid)
    
    ni = numint.NumInt()

    J_all = []
    for i in range(nset):
        dm_val = dm[i, 0, :, :]
        
        Q_total_list = []
        for atomI in range(cell._atm.shape[0]):
            submat = dm_val[localIdx[atomI]][:, localIdx[atomI]]
            DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
            DPQtilde = smart_einsum('Pm,Qn,mn->PQ', Ftilde_Pmu[atomI], Ftilde_Pmu[atomI], submat)
            zg = smart_einsum('PQ,PQL->L', DPQ-DPQtilde, M_PQLarr[atomI])
            Q_total_list.append(zg)
        
        Q_total = numpy.concatenate(Q_total_list)
        
        # Real-space block-by-block construction
        rhoR_comp = numpy.zeros(Ng)
        p1 = 0
        for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(gmol, mock_grids, nao=gmol.nao):
            p0, p1 = p1, p1 + coords.shape[0]
            if Periodic and gmolCart is not None and gmolSph is not None:
                ao = build_g_R_block_periodic(pmol, gmolCart, gmolSph, coords, ni, mask, gmax)
            else:
                ao = ao_k1
            rhoR_comp[p0:p1] += numpy.dot(ao, Q_total)
            
        rhoG_comp = numpy.fft.fftn(rhoR_comp.reshape(mesh)).flatten() * dv
        
        wv_freq = (rhoG_eval[i, 0, :] + rhoG_comp) * FF_flat
        
        # Pass 2: G-space -> Matrix (Smooth Potential Integration)
        Ji = multigrid_pair._get_j_pass2(mg_ni, wv_freq[numpy.newaxis, numpy.newaxis, :], kpts=numpy.zeros((1,3)), hermi=1)
        while Ji.ndim > 2:
            Ji = Ji[0]
            
        potential = numpy.fft.ifftn(wv_freq.reshape(mesh)).real.flatten()
        zg2_total = numpy.zeros_like(Q_total)
        
        p1 = 0
        for ao_k1, ao_k2, mask, weight, coords in ni.block_loop(gmol, mock_grids, nao=gmol.nao):
            p0, p1 = p1, p1 + coords.shape[0]
            if Periodic and gmolCart is not None and gmolSph is not None:
                ao = build_g_R_block_periodic(pmol, gmolCart, gmolSph, coords, ni, mask, gmax)
            else:
                ao = ao_k1
            
            v_block = potential[p0:p1]
            zg2_total += numpy.dot(ao.T, v_block) * dv
        
        count = 0
        for atomI in range(cell._atm.shape[0]):
            L_size = M_PQLarr[atomI].shape[-1]
            zg2 = zg2_total[count : count+L_size]
            count += L_size
            
            GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
            numpy.add.at(
                Ji, numpy.ix_(localIdx[atomI], localIdx[atomI]),
                 smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])                -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
            )
        J_all.append(Ji)
    
    J = numpy.stack(J_all)
    if nset == 1:
        J = J[0]
    return J


def get_vxc_and_j_smooth_Approach3(cell, dm, mg_ni, mesh, PAWdata, pmol, xc_code, Periodic=False, hermi=1, kpt=None):
    from pyscf.pbc.dft import gen_grid, numint
    from pyscf import __config__
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr = PAWdata[:7]
    gmol, gmolCart, gmolSph, gmax = PAWdata[9:13]

    GGA_METHOD = getattr(__config__, 'pbc_dft_multigrid_gga_method', 'FFT')

    if kpt is None:
        kpt = numpy.zeros((1,3))

    nset = dm.shape[0]
    is_uks = (nset == 2)
    
    xctype = mg_ni._xc_type(xc_code)
    if xctype in (None, 'LDA', 'HF'):
        deriv = 0
    elif xctype == 'GGA':
        deriv = 1
    else:
        raise NotImplementedError(f'xctype {xctype} not implemented in PAW multigrid')
        
    # Pass 1: AO -> Grid (Smooth Density)
    rhoG = multigrid_pair._eval_rhoG(mg_ni, dm, hermi=hermi, kpts=kpt, deriv=deriv)
    
    Ng = numpy.prod(mesh)
    dv = cell.vol / Ng
    weight = dv
    
    # XC Evaluation on pseudo-density only
    rhoR = numpy.fft.ifftn(rhoG.reshape(nset, -1, *mesh), axes=(2,3,4)).real.reshape(nset, -1, Ng) * (1./weight)
    
    if xctype == 'HF': # No XC part
        if is_uks:
            nelec = numpy.array([numpy.sum(rhoR[0,0]) * weight, numpy.sum(rhoR[1,0]) * weight])
        else:
            nelec = numpy.sum(rhoR[0,0]) * weight
        excsum = 0
        vxc_G = numpy.zeros_like(rhoG)
    else:
        spin = 1 if is_uks else 0
        if not is_uks:
            rho_eval = rhoR[0]
        else:
            rho_eval = rhoR
        
        exc, vxcR = mg_ni.eval_xc_eff(xc_code, rho_eval, deriv=1, xctype=xctype, spin=spin)[:2]
        
        if is_uks:
            nelec = numpy.array([numpy.sum(rhoR[0,0]) * weight, numpy.sum(rhoR[1,0]) * weight])
            excsum = numpy.dot(rhoR[0,0] + rhoR[1,0], exc) * weight
            vxc_G = numpy.fft.fftn(vxcR.reshape(nset, -1, *mesh), axes=(2,3,4)).reshape(nset, -1, Ng)
        else:
            nelec = numpy.sum(rhoR[0,0]) * weight
            excsum = numpy.dot(rhoR[0,0], exc) * weight
            vxc_G = numpy.fft.fftn(vxcR.reshape(nset, -1, *mesh), axes=(2,3,4)).reshape(nset, -1, Ng)
        
        if xctype == 'GGA' and GGA_METHOD.upper() == 'FFT':
            from pyscf.pbc.dft.multigrid import _backend_c as backend
            Gv = cell.get_Gv(mesh)
            if is_uks:
                vxc_G_new = numpy.zeros_like(vxc_G[:, :1, :])
                vxc_G_new[0] = backend.get_gga_vrho_gs(vxc_G[0, 0:1], vxc_G[0, 1:4], Gv, weight, Ng, 1.)
                vxc_G_new[1] = backend.get_gga_vrho_gs(vxc_G[1, 0:1], vxc_G[1, 1:4], Gv, weight, Ng, 1.)
                vxc_G = vxc_G_new
            else:
                vxc_G = backend.get_gga_vrho_gs(vxc_G[0, 0:1], vxc_G[0, 1:4], Gv, weight, Ng, 1.)[numpy.newaxis, ...]
        else:
            vxc_G *= weight

    # Hartree Evaluation on pseudo-density + compensating charges
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    
    Rgrid = cell.get_uniform_grids(mesh=mesh, wrap_around=False)
    class MockGrids:
        def __init__(self, coords):
            self.coords = coords
            self.weights = numpy.ones(coords.shape[0])
            self.cutoff = 1e-15
            self.non0tab = None
            self.mesh = mesh
            self.mol = gmol
        def build(self, *args, **kwargs):
            pass
    mock_grids = MockGrids(Rgrid)
    ni = numint.NumInt()

    J_all = []
    ecoul_total = 0.
    for i in range(nset):
        dm_val = dm[i, 0, :, :]
        rhoR_Ji = rhoR[i, 0].copy()

        Q_total_list = []
        for atomI in range(cell._atm.shape[0]):
            submat = dm_val[localIdx[atomI]][:, localIdx[atomI]]
            DPQ = F_Pmu[atomI] @ submat @ F_Pmu[atomI].T
            DPQtilde = Ftilde_Pmu[atomI] @ submat @ Ftilde_Pmu[atomI].T
            diff_DPQ = (DPQ - DPQtilde).ravel()
            zg = diff_DPQ @ M_PQLarr[atomI].reshape(-1, M_PQLarr[atomI].shape[-1])
            Q_total_list.append(zg)
            
        Q_total = numpy.concatenate(Q_total_list)
        
        # Real-space block-by-block construction
        p1 = 0
        for ao_k1, ao_k2, mask, weight_grid, coords in ni.block_loop(gmol, mock_grids, nao=gmol.nao):
            p0, p1 = p1, p1 + coords.shape[0]
            if Periodic and gmolCart is not None and gmolSph is not None:
                ao = build_g_R_block_periodic(pmol, gmolCart, gmolSph, coords, ni, mask, gmax)
            else:
                ao = ao_k1
            rhoR_Ji[p0:p1] += numpy.dot(ao, Q_total)

        rhoG_Ji = numpy.fft.fftn(rhoR_Ji.reshape(mesh)).flatten()
        vG_Ji = rhoG_Ji * FF.flatten()
        
        potential_Ji = numpy.fft.ifftn(vG_Ji.reshape(mesh)).real.flatten()
        ecoul_total += 0.5 * numpy.dot(rhoR_Ji, potential_Ji) * dv
        
        vj_mat_i = multigrid_pair._get_j_pass2(mg_ni, vG_Ji * weight, kpts=kpt, hermi=hermi)
        while vj_mat_i.ndim > 2: vj_mat_i = vj_mat_i[0]
        
        zg2_total = numpy.zeros_like(Q_total)
        p1 = 0
        for ao_k1, ao_k2, mask, weight_grid, coords in ni.block_loop(gmol, mock_grids, nao=gmol.nao):
            p0, p1 = p1, p1 + coords.shape[0]
            if Periodic and gmolCart is not None and gmolSph is not None:
                ao = build_g_R_block_periodic(pmol, gmolCart, gmolSph, coords, ni, mask, gmax)
            else:
                ao = ao_k1
            
            v_block = potential_Ji[p0:p1]
            zg2_total += numpy.dot(ao.T, v_block) * dv

        count = 0
        for atomI in range(cell._atm.shape[0]):
            L_size = M_PQLarr[atomI].shape[-1]
            zg2 = zg2_total[count : count+L_size]
            count += L_size
            
            GL = (M_PQLarr[atomI].reshape(-1, M_PQLarr[atomI].shape[-1]) @ zg2).reshape(M_PQLarr[atomI].shape[:-1])
            
            vj_mat_i[numpy.ix_(localIdx[atomI], localIdx[atomI])] +=                 F_Pmu[atomI].T @ GL @ F_Pmu[atomI] -                 Ftilde_Pmu[atomI].T @ GL @ Ftilde_Pmu[atomI]
                
        J_all.append(vj_mat_i)

    vj_mat = numpy.stack(J_all)
    if not is_uks:
        vj_mat = vj_mat[0]

    # Pass 2: Grid -> Matrix (Vxc)
    if xctype == 'GGA' and GGA_METHOD.upper() != 'FFT':
        vxc_mat = multigrid_pair._get_gga_pass2(mg_ni, vxc_G, kpts=kpt, hermi=hermi)
    else:
        if vxc_G.ndim == 3 and vxc_G.shape[1] == 1:
            vxc_G_input = vxc_G[:, 0, :]
        else:
            vxc_G_input = vxc_G
        vxc_mat = multigrid_pair._get_j_pass2(mg_ni, vxc_G_input, kpts=kpt, hermi=hermi)

    if not is_uks:
        while vxc_mat.ndim > 2: vxc_mat = vxc_mat[0]
    else:
        if vxc_mat.shape[1] == 1: 
            vxc_mat = vxc_mat[:, 0]

    return nelec, excsum, ecoul_total, vxc_mat, vj_mat
