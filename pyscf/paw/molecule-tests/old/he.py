from pyscf import gto, scf, dft, sgx
from pyscf.pbc import gto as pgto
import numpy as np
import jax, pyscf
from pyscf.paw import NewPAW as PAW


jax.config.update("jax_enable_x64", True)

L=30
#alpha0=2.48
alpha0=10
he=[['He',[L/2,L/2,L/2]]]
ne=[['Ne',[L/2,L/2,L/2]]]
he2=[['He',[L/2-3,L/2,L/2]],
['He',[L/2+3,L/2,L/2]]]
ne2=[['Ne',[L/2-3,L/2,L/2]],
['Ne',[L/2+3,L/2,L/2]]]
n2=[
    ['N', [L/2+1.037, L/2, L/2]],
    ['N', [L/2-1.037, L/2, L/2]]
]
methane=[
    ['C', [L/2,           L/2,           L/2          ]],
    ['H', [L/2 + 1.18588, L/2 + 1.18588, L/2 + 1.18588]],
    ['H', [L/2 - 1.18588, L/2 - 1.18588, L/2 + 1.18588]],
    ['H', [L/2 - 1.18588, L/2 + 1.18588, L/2 - 1.18588]],
    ['H', [L/2 + 1.18588, L/2 - 1.18588, L/2 - 1.18588]]
]
ethylene=[
    ['C', [L/2 + 1.265,L/2,        L/2]],
    ['C', [L/2 - 1.265,L/2,        L/2]],
    ['H', [L/2 + 2.329,L/2 + 1.756,L/2]],
    ['H', [L/2 + 2.329,L/2 - 1.756,L/2]],
    ['H', [L/2 - 2.329,L/2 + 1.756,L/2]],
    ['H', [L/2 - 2.329,L/2 - 1.756,L/2]]
]
benzene=[
    # Carbon ring
    ['C', [L/2 + 2.62672, L/2,           L/2]],
    ['C', [L/2 + 1.31336, L/2 + 2.27481, L/2]],
    ['C', [L/2 - 1.31336, L/2 + 2.27481, L/2]],
    ['C', [L/2 - 2.62672, L/2,           L/2]],
    ['C', [L/2 - 1.31336, L/2 - 2.27481, L/2]],
    ['C', [L/2 + 1.31336, L/2 - 2.27481, L/2]],
    
    # Hydrogen ring
    ['H', [L/2 + 4.68652, L/2,           L/2]],
    ['H', [L/2 + 2.34326, L/2 + 4.05865, L/2]],
    ['H', [L/2 - 2.34326, L/2 + 4.05865, L/2]],
    ['H', [L/2 - 4.68652, L/2,           L/2]],
    ['H', [L/2 - 2.34326, L/2 - 4.05865, L/2]],
    ['H', [L/2 + 2.34326, L/2 - 4.05865, L/2]]
]

cyclohexane = [
    ['C', [L/2 - 2.600830, L/2 + 0.245097, L/2 - 0.788583]],
    ['C', [L/2 - 1.235503, L/2 + 2.343449, L/2 + 0.669719]],
    ['C', [L/2 + 1.607023, L/2 + 2.393338, L/2 + 0.068975]],
    ['C', [L/2 + 2.655821, L/2 - 0.231869, L/2 - 0.607925]],
    ['C', [L/2 + 1.190905, L/2 - 2.362724, L/2 + 0.701844]],
    ['C', [L/2 - 1.618928, L/2 - 2.386157, L/2 - 0.044975]],
    ['H', [L/2 - 2.090415, L/2 + 4.196515, L/2 + 0.227145]],
    ['H', [L/2 - 1.510458, L/2 + 2.052431, L/2 + 2.723284]],
    ['H', [L/2 + 1.975897, L/2 + 3.703863, L/2 - 1.516316]],
    ['H', [L/2 + 2.638624, L/2 + 3.173039, L/2 + 1.710202]],
    ['H', [L/2 + 2.572106, L/2 - 0.506069, L/2 - 2.680009]],
    ['H', [L/2 + 4.678584, L/2 - 0.338639, L/2 - 0.096376]],
    ['H', [L/2 + 2.059801, L/2 - 4.206530, L/2 + 0.246609]],
    ['H', [L/2 + 1.357390, L/2 - 2.140115, L/2 + 2.775441]],
    ['H', [L/2 - 1.930922, L/2 - 3.706887, L/2 - 1.633479]],
    ['H', [L/2 - 2.743504, L/2 - 3.136000, L/2 + 1.549386]],
    ['H', [L/2 - 4.658552, L/2 + 0.359048, L/2 - 0.451833]],
    ['H', [L/2 - 2.336079, L/2 + 0.539706, L/2 - 2.842715]]
]

propene = [
    ['C', [L/2 + 0.00000, L/2 + 0.00000, L/2 + 0.00000]],
    ['C', [L/2 + 2.52000, L/2 + 0.00000, L/2 + 0.00000]],
    ['C', [L/2 - 1.43000, L/2 + 2.38000, L/2 + 0.00000]],
    ['H', [L/2 - 0.95000, L/2 - 1.65000, L/2 + 0.00000]],
    ['H', [L/2 + 3.00000, L/2 - 1.65000, L/2 + 0.00000]],
    ['H', [L/2 + 3.00000, L/2 + 1.65000, L/2 + 0.00000]],
    ['H', [L/2 - 0.80000, L/2 + 4.20000, L/2 + 0.00000]],
    ['H', [L/2 - 2.30000, L/2 + 2.20000, L/2 + 1.60000]],
    ['H', [L/2 - 2.30000, L/2 + 2.20000, L/2 - 1.60000]]
]

butadiene = [
    ['C', [L/2 - 2.52000, L/2 - 1.34000, L/2 + 0.00000]],
    ['C', [L/2 - 0.00000, L/2 - 0.00000, L/2 + 0.00000]],
    ['C', [L/2 + 2.52000, L/2 - 0.00000, L/2 + 0.00000]],
    ['C', [L/2 + 5.04000, L/2 + 1.34000, L/2 + 0.00000]],
    ['H', [L/2 - 4.20000, L/2 - 0.60000, L/2 + 0.00000]],
    ['H', [L/2 - 2.80000, L/2 - 3.30000, L/2 + 0.00000]],
    ['H', [L/2 - 0.30000, L/2 + 2.00000, L/2 + 0.00000]],
    ['H', [L/2 + 2.80000, L/2 - 2.00000, L/2 + 0.00000]],
    ['H', [L/2 + 5.34000, L/2 + 3.30000, L/2 + 0.00000]],
    ['H', [L/2 + 6.72000, L/2 + 0.60000, L/2 + 0.00000]]
]

cyclobutane = [
    ['C', [L/2 + 1.45000, L/2 + 1.45000, L/2 + 0.00000]],
    ['C', [L/2 - 1.45000, L/2 + 1.45000, L/2 + 0.00000]],
    ['C', [L/2 - 1.45000, L/2 - 1.45000, L/2 + 0.00000]],
    ['C', [L/2 + 1.45000, L/2 - 1.45000, L/2 + 0.00000]],
    ['H', [L/2 + 2.50000, L/2 + 2.50000, L/2 + 1.00000]],
    ['H', [L/2 + 2.50000, L/2 + 2.50000, L/2 - 1.00000]],
    ['H', [L/2 - 2.50000, L/2 + 2.50000, L/2 + 1.00000]],
    ['H', [L/2 - 2.50000, L/2 + 2.50000, L/2 - 1.00000]],
    ['H', [L/2 - 2.50000, L/2 - 2.50000, L/2 + 1.00000]],
    ['H', [L/2 - 2.50000, L/2 - 2.50000, L/2 - 1.00000]],
    ['H', [L/2 + 2.50000, L/2 - 2.50000, L/2 + 1.00000]],
    ['H', [L/2 + 2.50000, L/2 - 2.50000, L/2 - 1.00000]]
]

ddict2={'H':1/2,'He':1/2,'C':1/2,'N':1/2,'Ne':1/2}
ddict3={'H':1/3,'He':1/3,'C':1/3,'N':1/3,'Ne':1/3}
ddict4={'H':1/4,'He':1/4,'C':1/4,'N':1/4,'Ne':1/4}
ddict5={'H':1/5,'He':1/5,'C':1/5,'N':1/5,'Ne':1/5}
ddict6={'H':1/5,'He':1/6,'C':1/6,'N':1/6,'Ne':1/6}
ddict7={'H':1/5,'He':1/7,'C':1/7,'N':1/7,'Ne':1/7}
ddict8={'H':1/5,'He':1/8,'C':1/8,'N':1/8,'Ne':1/8}

#for ddict in [ddict2,ddict3,ddict4,ddict5,ddict6,ddict7,ddict8]:
#    for atom in [he,ne,he2,ne2,n2,methane,ethylene]:
for ddict in [ddict8]:
    for atom in [butadiene]:
        verbose = 1

        cell = pgto.M(
            atom=atom,
            basis='cc-pvdz',
            a=np.array([[L,0,0],[0,L,0],[0,0,L]]),
            verbose=verbose,
            ke_cutoff=50,
            unit='B')

        mol = gto.M(atom=cell.atom, basis=cell.basis, verbose=verbose,unit='B')

        # paw calculation
        #mf=scf.RHF(mol)
        mf=dft.RKS(mol)
        mf.xc = ''
        mf.kernel()

        #mf_mol_paw = scf.RHF(mol).density_fit()
        mf_mol_paw = dft.RKS(mol).density_fit()
        mf_mol_paw.xc = ''
        #mydf = PAW.from_mf(mf_mol_paw, cell,alpha0=alpha0,#augRadius=50,
        mydf = PAW.from_mf(mf_mol_paw, cell,#alpha0=alpha0,#augRadius=50,
                            #PAWorbitalCutOff=0).build()
                            #PAWorbitalCutOff=1e-10).build()
                            ).build()
        mf_mol_paw.with_df = mydf
        mf_mol_paw.kernel()
        print("Original SCF: ",mf.e_tot)
        print("PAW SCF: ",mf_mol_paw.e_tot)
        print("Difference (PAW - Original): ",mf_mol_paw.e_tot-mf.e_tot,flush=True)
        exit()

        # Conventional PySCF Density Fitting
        mf_df = dft.RKS(mol).density_fit()
        mf_df.xc = ''
        mf_df.kernel()
        print("Conventional DF SCF: ", mf_df.e_tot)
        print("Difference (DF - Original): ", mf_df.e_tot - mf.e_tot, flush=True)

    
        print("Smooth JK Differences")
        # Calculate analytic K matrix and E_K for smooth portions
        smooth_mol = gausslet.smooth_mol
        from Gausslets.addGTO import get_ao_radii
        radii = get_ao_radii(smooth_mol)
        nPAWidx = np.where(radii > 0)[0]
        
        DM = mf.make_rdm1()
        
        # Zero out non-smooth AO contributions in DM to isolate smooth exchange
        DM_smooth_full = np.zeros_like(DM)
        DM_smooth_full[np.ix_(nPAWidx, nPAWidx)] = DM[np.ix_(nPAWidx, nPAWidx)]
        
        # Use PySCF's efficient J and K build
        J_anal_full, K_anal_full = pyscf.scf.hf.get_jk(smooth_mol, DM_smooth_full, with_j=True, with_k=True)
        EJ = np.einsum('ij,ij', J_anal_full, DM_smooth_full)
        EK = 0.5 * np.einsum('ij,ij', K_anal_full, DM_smooth_full)

        J = gausslet.compute_coulomb(DM)
        EJ_isdf = np.einsum('ij,ji->', DM, J)
        K = gausslet.compute_exchange(DM)
        EK_isdf = 0.5 * np.einsum('ij,ji->', DM, K)
        print(f"Coulomb Difference: {abs(EJ_isdf - EJ):.4e}")
        print(f"Exchange Difference: {abs(EK_isdf - EK):.4e}")
