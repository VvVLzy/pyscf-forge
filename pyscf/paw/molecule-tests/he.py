from pyscf import gto, scf, dft, sgx
from pyscf.pbc import gto as pgto
import numpy as np
import jax, pyscf
from pyscf.paw import PAW
from Gausslets import GaussletGrid


jax.config.update("jax_enable_x64", True)

L=40
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

ddict={'H':1/4,'He':1/4,'C':1/7,'N':1/7,'Ne':1/7}

#for atom in [he,ne,he2,ne2,n2,methane,ethylene]:
#for atom in [he,n2]:
#for atom in [methane]:
#for atom in [he,ne]:
#for atom in [ne2]:
for atom in [n2]:

    verbose = 1

    cell = pgto.M(
        atom=atom,
        basis='unc-cc-pvdz',
        a=np.array([[L,0,0],[0,L,0],[0,0,L]]),
        verbose=verbose,
        ke_cutoff=100,
        unit='B')

    mol = gto.M(atom=cell.atom, basis=cell.basis, verbose=verbose,unit='B')

    # paw calculation
    mf=dft.RKS(mol)
    mf.xc = 'pbe'
    mf.kernel()


    gausslet=GaussletGrid.GaussletGrid(mf,L,[ddict[i] for i in mol.elements],uniform=6,spread=0.7,isdf=True,gto=False,analcoeffs=False,maxgto=alpha0)
    ao=gausslet.getaocoeffs()
    v2e=gausslet.formFullIntegrals()
    analteis=mol.intor('int2e')
    myslice=gausslet.gtodata['gtoid']
    analteis=analteis[np.ix_(myslice,myslice,myslice,myslice)]
    numteis=np.einsum('ix,jx,xy,ky,ly->ijkl',ao,ao,v2e,ao,ao,optimize='optimal')
    print('max int error: ',np.max(np.abs(numteis-analteis)))

    mf_mol_paw = dft.RKS(mol).density_fit()
    mf_mol_paw.xc = 'pbe'
    mydf = PAW.from_mf(mf_mol_paw, cell,gausslet,alpha0=alpha0,#augRadius=1.5,
                        #PAWorbitalCutOff=0).build()
                        PAWorbitalCutOff=0).build()
    mf_mol_paw.with_df = mydf
    mf_mol_paw.kernel()
    print("Original SCF: ",mf.e_tot)
    print("PAW SCF: ",mf_mol_paw.e_tot)
    print("Difference: ",mf_mol_paw.e_tot-mf.e_tot,flush=True)
    print("")
