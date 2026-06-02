import numpy

import jax
jax.config.update("jax_enable_x64",True)
jax.config.update('jax_platform_name', 'cpu')

from pyscf.pbc.dft.multigrid import multigrid_pair
from pyscf.pbc.dft.multigrid import _backend_c as backend
from pyscf import __config__
import pyscf

from .paw_helper import *

from functools import partial
smart_einsum = partial(numpy.einsum, optimize='optimal')


def getj_PAW(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    J1 = getjSmoothPW(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=Periodic)
    J2 = getjSharpLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=Periodic)
    J3 = getjSmoothLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=Periodic)

    return J1+J2+J3

def getjSmoothPW(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    # TODO: generalize to multiple k-points
    # use list of numpy arrays instead of jax arrays
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    nset = dm.shape[0]
    J_all = []
    for i in range(nset):
        dm_val = dm[i, 0, :, :]

        Ng = numpy.prod(mesh)
        dv = (cell.vol/Ng)
        FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
        density = smart_einsum('ra,ab,rb->r', aoOnR_tilde.conj(), dm_val, aoOnR_tilde).real

        for atomI in range(cell._atm.shape[0]):
            submat = dm_val[localIdx[atomI]][:, localIdx[atomI]]
            DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
            DPQtilde = smart_einsum('Pm,Qn,mn->PQ', Ftilde_Pmu[atomI], Ftilde_Pmu[atomI], submat)
            zg = smart_einsum('PQ,PQL->L', DPQ-DPQtilde, M_PQLarr[atomI])
            numpy.add.at(density, gridIdx[atomI], smart_einsum('L,rL->r', zg, gOnR[atomI]))
        potential = numpy.fft.ifftn( FF * numpy.fft.fftn(density.reshape(mesh))).real.flatten()

        Ji = smart_einsum('ra,r,rb->ab', aoOnR_tilde.conj(), potential, aoOnR_tilde)*dv
        assert(Ji.shape[0] == cell.nao)
        assert(Ji.shape[1] == cell.nao)

        for atomI in range(cell._atm.shape[0]):
            # el+comp-compOnA
            zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*dv
            GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
            numpy.add.at(
                Ji, numpy.ix_(localIdx[atomI], localIdx[atomI]),
                 smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
                -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
            )
        J_all.append(Ji)
    
    J = numpy.stack(J_all)
    if nset == 1:
        J = J[0]
    return J

def get_vxc_and_j_smooth(cell, dm, mg_ni, mesh, PAWdata, xc_code, Periodic=False, hermi=1, kpt=None):
    GGA_METHOD = getattr(__config__, 'pbc_dft_multigrid_gga_method', 'FFT')

    if kpt is None:
        kpt = numpy.zeros((1,3))
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

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
    # eval_rhoG processes the nset dimension natively
    # start = time.time()
    rhoG = multigrid_pair._eval_rhoG(mg_ni, dm, hermi=hermi, kpts=kpt, deriv=deriv)
    # print(f'Eval density on grid takes {time.time() - start : .1f} seconds')
    
    Ng = numpy.prod(mesh)
    dv = cell.vol / Ng
    weight = dv
    
    # start = time.time()
    # XC Evaluation on pseudo-density only
    rhoR = numpy.fft.ifftn(rhoG.reshape(nset, -1, *mesh), axes=(2,3,4)).real.reshape(nset, -1, Ng) * (1./weight)
    # print(f'First FFT from G to R takes {time.time() - start: .2f} seconds')
    
    if xctype == 'HF': # No XC part
        if is_uks:
            nelec = numpy.array([numpy.sum(rhoR[0,0]) * weight, numpy.sum(rhoR[1,0]) * weight])
        else:
            nelec = numpy.sum(rhoR[0,0]) * weight
        excsum = 0
        vxc_G = numpy.zeros_like(rhoG)
    else:
        # Evaluate XC on density
        spin = 1 if is_uks else 0
        # If UKS, rhoR shape is (2, deriv, Ng). We reshape to match PySCF expected: (2, deriv, Ng) 
        # or just pass it as is.
        if not is_uks:
            rho_eval = rhoR[0]
        else:
            rho_eval = rhoR
        
        exc, vxcR = mg_ni.eval_xc_eff(xc_code, rho_eval, deriv=1, xctype=xctype, spin=spin)[:2]
        
        if is_uks:
            nelec = numpy.array([numpy.sum(rhoR[0,0]) * weight, numpy.sum(rhoR[1,0]) * weight])
            # vxcR for UKS is shape (2, deriv, Ng). Or list of length 2? 
            # eval_xc returns vxc as (2, deriv, Ng) if spin=1.
            excsum = numpy.dot(rhoR[0,0] + rhoR[1,0], exc) * weight
            vxc_G = numpy.fft.fftn(vxcR.reshape(nset, -1, *mesh), axes=(2,3,4)).reshape(nset, -1, Ng)
        else:
            nelec = numpy.sum(rhoR[0,0]) * weight
            excsum = numpy.dot(rhoR[0,0], exc) * weight
            vxc_G = numpy.fft.fftn(vxcR.reshape(nset, -1, *mesh), axes=(2,3,4)).reshape(nset, -1, Ng)
        
        if xctype == 'GGA' and GGA_METHOD.upper() == 'FFT':
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
    
    J_all = []
    ecoul_total = 0.
    for i in range(nset):
        dm_val = dm[i, 0, :, :]
        rhoR_Ji = rhoR[i, 0].copy()

        # start = time.time()
        for atomI in range(cell._atm.shape[0]):
            submat = dm_val[localIdx[atomI]][:, localIdx[atomI]]
            # Use @ for faster matrix multiplication
            DPQ = F_Pmu[atomI] @ submat @ F_Pmu[atomI].T
            DPQtilde = Ftilde_Pmu[atomI] @ submat @ Ftilde_Pmu[atomI].T
            
            # zg = (DPQ - DPQtilde) : M_PQL
            diff_DPQ = (DPQ - DPQtilde).ravel()
            zg = diff_DPQ @ M_PQLarr[atomI].reshape(-1, M_PQLarr[atomI].shape[-1])
            
            # rhoR_J += zg @ gOnR^T
            rhoR_Ji[gridIdx[atomI]] += gOnR[atomI] @ zg
        # print(f'Density augmentation takes {time.time() - start: .2f} seconds')

        # start = time.time()
        rhoG_Ji = numpy.fft.fftn(rhoR_Ji.reshape(mesh)).flatten()
        vG_Ji = rhoG_Ji * FF.flatten()
        
        # Use real-space integration for ecoul to ensure consistent scaling
        potential_Ji = numpy.fft.ifftn(vG_Ji.reshape(mesh)).real.flatten()
        # print(f'Poisson solve takes {time.time() - start: .2f} seconds')
        ecoul_total += 0.5 * numpy.dot(rhoR_Ji, potential_Ji) * dv
        
        # Pass 2: Grid -> Matrix
        # start = time.time()
        vj_mat_i = multigrid_pair._get_j_pass2(mg_ni, vG_Ji * weight, kpts=kpt, hermi=hermi)
        while vj_mat_i.ndim > 2: vj_mat_i = vj_mat_i[0]
        # print(f'Pass 2 integration takes {time.time() - start: .2f} seconds')
        
        # Local Hartree Corrections
        # start = time.time()
        for atomI in range(cell._atm.shape[0]):
            # zg2 = \int v(r) g_L(r) dr
            zg2 = potential_Ji[gridIdx[atomI]] @ gOnR[atomI] * dv
            
            # GL = zg2 : M_PQL
            # GL_RS = \sum_L zg2_L * M_RSL
            GL = (M_PQLarr[atomI].reshape(-1, M_PQLarr[atomI].shape[-1]) @ zg2).reshape(M_PQLarr[atomI].shape[:-1])
            
            # vj_mat += F @ GL @ F^T
            vj_mat_i[numpy.ix_(localIdx[atomI], localIdx[atomI])] += \
                F_Pmu[atomI].T @ GL @ F_Pmu[atomI] - \
                Ftilde_Pmu[atomI].T @ GL @ Ftilde_Pmu[atomI]
        # print(f'Local corrections takes {time.time() - start: .2f} seconds')
        J_all.append(vj_mat_i)

    vj_mat = numpy.stack(J_all)
    if not is_uks:
        vj_mat = vj_mat[0]

    # Pass 2: Grid -> Matrix (Vxc)
    # start = time.time()
    if xctype == 'GGA' and GGA_METHOD.upper() != 'FFT':
        # multigrid_pair._get_gga_pass2 processes the nset dimension
        vxc_mat = multigrid_pair._get_gga_pass2(mg_ni, vxc_G, kpts=kpt, hermi=hermi)
    else:
        # _get_j_pass2 natively expects (nset, nkpts, Ng) but since it's just a dot product, it works fine
        if vxc_G.ndim == 3 and vxc_G.shape[1] == 1:
            vxc_G_input = vxc_G[:, 0, :]
        else:
            vxc_G_input = vxc_G
        vxc_mat = multigrid_pair._get_j_pass2(mg_ni, vxc_G_input, kpts=kpt, hermi=hermi)
    # print(f'Eval matrix from potential takes: {time.time() - start: .2f} seconds')
    # Squeeze J and Vxc appropriately
    if not is_uks:
        while vxc_mat.ndim > 2: vxc_mat = vxc_mat[0]
    else:
        # For UKS, vxc_mat should end up as (2, nao, nao)
        if vxc_mat.shape[1] == 1: # if (2, 1, nao, nao)
            vxc_mat = vxc_mat[:, 0]

    return nelec, excsum, ecoul_total, vxc_mat, vj_mat

def getjSmoothPW2(cell, dm, mg_ni, mesh, PAWdata, Periodic=False):
    # import time
    # t_start_total = time.time()
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    nset = dm.shape[0]
    Ng = numpy.prod(mesh)
    dv = cell.vol / Ng
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)

    # Pass 1: AO -> Grid (Smooth Density)
    # t0 = time.time()
    rhoG = multigrid_pair._eval_rhoG(mg_ni, dm, hermi=1, kpts=numpy.zeros((1,3)), deriv=0)
    # print(f"[getjSmoothPW2] _eval_rhoG takes: {time.time() - t0:.4f} seconds")
    
    J_all = []
    for i in range(nset):
        dm_val = dm[i, 0, :, :]
        
        # t0 = time.time()
        rhoR = numpy.fft.ifftn(rhoG[i].reshape(mesh)).real.flatten() * (1./dv)
        # print(f"[getjSmoothPW2] Set {i}: ifftn for rhoR takes: {time.time() - t0:.4f} seconds")
        
        # t0 = time.time()
        for atomI in range(cell._atm.shape[0]):
            submat = dm_val[localIdx[atomI]][:, localIdx[atomI]]
            DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
            DPQtilde = smart_einsum('Pm,Qn,mn->PQ', Ftilde_Pmu[atomI], Ftilde_Pmu[atomI], submat)
            zg = smart_einsum('PQ,PQL->L', DPQ-DPQtilde, M_PQLarr[atomI])
            numpy.add.at(rhoR, gridIdx[atomI], smart_einsum('L,rL->r', zg, gOnR[atomI]))
        # print(f"[getjSmoothPW2] Set {i}: Atom loop for density augmentation takes: {time.time() - t0:.4f} seconds")

        # Poisson in G-space
        # t0 = time.time()
        rhoG_total = numpy.fft.fftn(rhoR.reshape(mesh))
        vG = rhoG_total * FF
        # print(f"[getjSmoothPW2] Set {i}: fftn for rhoG_total takes: {time.time() - t0:.4f} seconds")
        
        # Pass 2: Grid -> Matrix (Smooth Potential Integration)
        # t0 = time.time()
        wv_freq = vG * dv
        Ji = multigrid_pair._get_j_pass2(mg_ni, wv_freq[numpy.newaxis, numpy.newaxis, :], kpts=numpy.zeros((1,3)), hermi=1)
        while Ji.ndim > 2:
            Ji = Ji[0]
        # print(f"[getjSmoothPW2] Set {i}: _get_j_pass2 takes: {time.time() - t0:.4f} seconds")
        
        # Local Matrix Corrections (zg2 terms)
        # t0 = time.time()
        potential = numpy.fft.ifftn(vG).real.flatten()
        # print(f"[getjSmoothPW2] Set {i}: ifftn for potential takes: {time.time() - t0:.4f} seconds")
        
        # t0 = time.time()
        for atomI in range(cell._atm.shape[0]):
            # el+comp-compOnA
            zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*dv
            GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
            numpy.add.at(
                Ji, numpy.ix_(localIdx[atomI], localIdx[atomI]),
                 smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
                -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
            )
        # print(f"[getjSmoothPW2] Set {i}: Atom loop for local corrections takes: {time.time() - t0:.4f} seconds")
        
        J_all.append(Ji)
    
    J = numpy.stack(J_all)
    if nset == 1:
        J = J[0]
    # print(f"[getjSmoothPW2] Total time: {time.time() - t_start_total:.4f} seconds")
    return J

def getjSharpLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    # TODO: generalize to multiple k-points
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    nset = dm.shape[0]
    J_all = []
    for i in range(nset):
        dm_val = dm[i, 0, :, :]
        Ji = numpy.zeros((cell.nao, cell.nao))
        for atomI in range(cell._atm.shape[0]):
            submat = dm_val[localIdx[atomI]][:,localIdx[atomI]]
            DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
            numpy.add.at(
                Ji, numpy.ix_(localIdx[atomI], localIdx[atomI]),
                smart_einsum('PQ,PQRS,Rm,Sn->mn', DPQ, VPQRSarray[atomI], F_Pmu[atomI], F_Pmu[atomI])
            )
        J_all.append(Ji)

    J = numpy.stack(J_all)
    if nset == 1:
        J = J[0]
    return J

def getjSmoothLocal(cell, dm, aoOnR_tilde, mesh, PAWdata, Periodic=False):
    # TODO: generalize to multiple k-points
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWdata

    nset = dm.shape[0]
    J_all = []
    for i in range(nset):
        dm_val = dm[i, 0, :, :]
        Ji = numpy.zeros((cell.nao, cell.nao))
        for atomI in range(cell._atm.shape[0]):
            submat = dm_val[localIdx[atomI]][:,localIdx[atomI]]
            DPQ = smart_einsum('Pm,Qn,mn->PQ', F_Pmu[atomI], F_Pmu[atomI], submat)
            DPQtilde = smart_einsum('Pm,Qn,mn->PQ', Ftilde_Pmu[atomI], Ftilde_Pmu[atomI], submat)
            zg = smart_einsum('PQ,PQL->L', DPQ-DPQtilde, M_PQLarr[atomI])

            # smooth RHS
            # smoothel-smoothel
            A = -smart_einsum('PQ,PQRS->RS', DPQtilde, VPQRSarray[atomI])
            # smoothel-smoothcomp
            A += smart_einsum('PQ,PQL,RSL->RS', DPQtilde, V_PQLarr[atomI], M_PQLarr[atomI])
            # comp-smoothel
            A -= smart_einsum('L,RSL->RS', zg, V_PQLarr[atomI])
            # comp-smoothcomp
            A += smart_einsum('L,LM,RSM->RS', zg, V_LMarr[atomI], M_PQLarr[atomI])

            # sharp RHS
            # smoothel-sharpcomp
            B = -smart_einsum('PQ,PQL,RSL->RS', DPQtilde, V_PQLarr[atomI], M_PQLarr[atomI])
            # comp-sharpcomp
            B -= smart_einsum('L,LM,RSM->RS', zg, V_LMarr[atomI], M_PQLarr[atomI])

            numpy.add.at(
                Ji, numpy.ix_(localIdx[atomI], localIdx[atomI]),
                smart_einsum('RS,Rm,Sn->mn', A, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])+\
                smart_einsum('RS,Rm,Sn->mn', B, F_Pmu[atomI], F_Pmu[atomI])
            )
        J_all.append(Ji)

    J = numpy.stack(J_all)
    if nset == 1:
        J = J[0]
    return J

# PAW nuclear
def getnuc_PAW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, mg_ni, Periodic=False, with_multigrid=2):
    # cell, self.mesh, self.aoOnR_tilde, self.PAWdata, self.PAWNucdata, self.Periodic
    # nuc1 = getNucPAWSmoothPW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic=Periodic)
                            #  cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic
    if with_multigrid > 0:
        nuc1 = getNucPAWSmoothPW2(cell, mesh, mg_ni, PAWElecdata, PAWNucdata, Periodic)
    else:
        assert(with_multigrid == 0)
        nuc1 = getNucPAWSmoothPW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic)
    nuc2 = getNucPAWSharpLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic=Periodic)
    nuc3 = getNucPAWSmoothLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic=Periodic)

    return nuc1+nuc2+nuc3


def getNucPAWSmoothPW(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic):
    # TODO: generalize to multiple k-points
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    Ng = numpy.prod(mesh)
    f = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    density = numpy.zeros((Ng,))

    for atomI in range(cell._atm.shape[0]):
        update = smart_einsum('g,rg->r', M_PQLarrNuc[atomI], gOnR[atomI])*ZNucArr[atomI]
        numpy.add.at(density, gridIdx[atomI], update)
    potential = numpy.fft.ifftn( FF * numpy.fft.fftn(density.reshape(mesh))).real.flatten()

    nuc = smart_einsum('ra,r,rb->ab', aoOnR_tilde, potential, aoOnR_tilde)*f
    assert(nuc.shape[0] == cell.nao)
    assert(nuc.shape[1] == cell.nao)

    for atomI in range(cell._atm.shape[0]):
        # compnuc-compOnA
        zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*f
        GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
             smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
            -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
        )
    
    return numpy.array([nuc])

def getNucPAWSmoothPW2(cell, mesh, mg_ni, PAWElecdata, PAWNucdata, Periodic):
    # TODO: generalize to multiple k-points
    
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    Ng = numpy.prod(mesh)
    dv = (cell.vol/Ng)
    FF = getFormFactor(mesh, cell).reshape(mesh) if Periodic else getFormFactor_Truncated(mesh, cell).reshape(mesh)
    rhoR = numpy.zeros((Ng,))

    for atomI in range(cell._atm.shape[0]):
        update = smart_einsum('g,rg->r', M_PQLarrNuc[atomI], gOnR[atomI])*ZNucArr[atomI]
        numpy.add.at(rhoR, gridIdx[atomI], update)

    rhoG_total = numpy.fft.fftn(rhoR.reshape(mesh))
    vG = rhoG_total * FF
    
    # Pass 2: Grid -> Matrix (Smooth Potential Integration)
    # Note: _get_j_pass2 expects vG * weight
    wv_freq = vG * dv
    nuc = multigrid_pair._get_j_pass2(mg_ni, wv_freq, kpts=numpy.zeros((1,3)), hermi=1)
    while nuc.ndim > 2:
        nuc = nuc[0]

    potential = numpy.fft.ifftn(vG).real.flatten()
    for atomI in range(cell._atm.shape[0]):
        # compnuc-compOnA
        zg2 = smart_einsum('r,rL->L', potential[gridIdx[atomI]], gOnR[atomI])*dv
        GL  = smart_einsum('L,RSL->RS', zg2, M_PQLarr[atomI])
        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
             smart_einsum('RS,Rm,Sn->mn', GL, F_Pmu[atomI], F_Pmu[atomI])\
            -smart_einsum('RS,Rm,Sn->mn', GL, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])
        )
    
    return numpy.array([nuc])

def getNucPAWSharpLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic):
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    nao = cell.nao
    nuc = numpy.zeros((nao, nao))

    for atomI in range(cell._atm.shape[0]):
        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
            smart_einsum('RS,Rm,Sn->mn', VPQRSArrNuc[atomI]*ZNucArr[atomI], F_Pmu[atomI], F_Pmu[atomI])
        )
    
    return numpy.array([nuc])

def getNucPAWSmoothLocal(cell, mesh, aoOnR_tilde, PAWElecdata, PAWNucdata, Periodic):
    localIdx, F_Pmu, Ftilde_Pmu, VPQRSarray, M_PQLarr, V_PQLarr, V_LMarr, gridIdx, gOnR = PAWElecdata
    VPQRSArrNuc, M_PQLarrNuc, ZNucArr = PAWNucdata

    nao = cell.nao
    nuc = numpy.zeros((nao, nao))

    for atomI in range(cell._atm.shape[0]):
        # smooth RHS
        # compnuc-smoothel
        A = -smart_einsum('g,PQg->PQ', M_PQLarrNuc[atomI], V_PQLarr[atomI])*ZNucArr[atomI]
        # compnuc-smoothcomp
        A += smart_einsum('g, gf, PQf->PQ', M_PQLarrNuc[atomI], V_LMarr[atomI], M_PQLarr[atomI])*ZNucArr[atomI]

        # sharp RHS
        # compnuc-sharpcomp
        B = -smart_einsum('g,gf, PQf->PQ', M_PQLarrNuc[atomI], V_LMarr[atomI], M_PQLarr[atomI])*ZNucArr[atomI]

        numpy.add.at(
            nuc, numpy.ix_(localIdx[atomI], localIdx[atomI]),
            smart_einsum('RS,Rm,Sn->mn', A, Ftilde_Pmu[atomI], Ftilde_Pmu[atomI])+\
            smart_einsum('RS,Rm,Sn->mn', B, F_Pmu[atomI], F_Pmu[atomI])
        )

    return numpy.array([nuc])