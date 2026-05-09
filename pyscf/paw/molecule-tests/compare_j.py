import numpy as np
from pyscf import gto, scf, lib, dft
from pyscf.pbc import gto as pgto
import jax
from pyscf.paw import NewPAW as PAW

jax.config.update("jax_enable_x64", True)

# Path setup
# Assuming script runs from /home/lebox/Github/pyscf-forge/pyscf/paw
geom_file = '/home/lebox/Github/pyscf-forge/pyscf/paw/molecule-tests/SLAC/dz/geoms/HighSpin-slab.xyz'
chk_file = '/home/lebox/Github/pyscf-forge/pyscf/paw/molecule-tests/SLAC/dz/DFT/chkfiles/High_spin_clean_pbe-d3bj.chk'

# Parameters
basis = 'def2-svpd'
spin = 4
auxbasis = 'def2-universal-jkfit'
ke_cutoff = 50
alpha0 = 5

# 1. Build Molecule
mol = gto.Mole()
mol.atom = geom_file
mol.basis = basis
mol.spin = spin
mol.verbose = 4
mol.build()

# 2. Build Cell for PAW
cell = pgto.M(
    atom = geom_file,
    basis = basis,
    spin = spin,
    a = np.eye(3) * 30.0,
    verbose = 4,
    ke_cutoff = ke_cutoff,
)

# 3. Load Density Matrix from chkfile
print(f"\nLoading density matrix from {chk_file}...")
# UKS DM has shape (2, nao, nao)
mo_coeff = scf.chkfile.load(chk_file, 'scf/mo_coeff')
mo_occ = scf.chkfile.load(chk_file, 'scf/mo_occ')
mf_dummy = dft.KS(mol)
dm = mf_dummy.make_rdm1(mo_coeff, mo_occ)

# # 4. Exact J Calculation
# print("\n--- Calculating Exact J (Analytic) ---")
# vj_exact = mf_dummy.get_j(dm=dm)

# 5. Regular DF J Calculation
print("\n--- Calculating Regular DF J ---")
mf_df = mf_dummy.density_fit()
mf_df.with_df.auxbasis = auxbasis
vj_df = mf_df.get_j(dm=dm)
# 6. PAW J Calculation
print("\n--- Calculating PAW J ---")
# Use UKS density fit as base
mf_paw = dft.KS(mol).density_fit()
mydf = PAW.from_mf(mf_paw, cell,
                    alpha0=alpha0, augRadius=None, with_multigrid=2, alpha0_lowmem=True).build()
mf_paw.with_df = mydf
# Use the PAW get_j logic via our recently implemented UKS support
vj_paw = mf_paw.get_j(dm=dm)


# 7. Comparison
def print_diff(label, vj_comp, vj_ref):
    diff = vj_comp - vj_ref
    # UKS J is (2, nao, nao)
    norm_total = np.linalg.norm(diff)
    norm_alpha = np.linalg.norm(diff[0])
    norm_beta  = np.linalg.norm(diff[1])
    print(f"\nDifferences for {label}:")
    print(f"  Total Norm: {norm_total:.2e}")
    print(f"  Alpha Norm: {norm_alpha:.2e}")
    print(f"  Beta Norm:  {norm_beta:.2e}")

# Compare PAW and Regular DF
print_diff("PAW vs Regular DF", vj_paw, vj_df)

# 8. Energy Comparison
print("\n" + "="*50)
print("  TOTAL ENERGY COMPARISON (No SCF)")
print("="*50)

xc = 'pbe-d3bj'
mf_dummy.xc = xc
mf_df.xc = xc
mf_paw.xc = xc

def get_energy_breakdown(mf, dm_in, label):
    # PySCF energy components
    enuc = mf.energy_nuc()
    # energy_elec returns (e_tot_elec, e_coul)
    # where e_tot_elec = e1 + e_coul + e_xc
    e_elec, ecoul = mf.energy_elec(dm=dm_in)
    h1 = mf.get_hcore()
    e1 = np.einsum('sij,ij->', dm_in, h1).real
    exc = e_elec - e1 - ecoul
    etot = e_elec + enuc
    
    print(f"\n{label}:")
    print(f"  One-electron: {e1:18.10f} Ha")
    print(f"  Coulomb:      {ecoul:18.10f} Ha")
    print(f"  XC:           {exc:18.10f} Ha")
    print(f"  Nuclear:      {enuc:18.10f} Ha")
    print(f"  Total:        {etot:18.10f} Ha")
    return {'e1': e1, 'ecoul': ecoul, 'exc': exc, 'etot': etot}

res_df = get_energy_breakdown(mf_df, dm, "Regular DF")
res_paw = get_energy_breakdown(mf_paw, dm, "PAW")

print("\n" + "="*50)
print("  DIFFERENCES (PAW - DF)")
print("="*50)
print(f"  One-electron: {res_paw['e1'] - res_df['e1']:12.4e} Ha")
print(f"  Coulomb:      {res_paw['ecoul'] - res_df['ecoul']:12.4e} Ha")
print(f"  XC:           {res_paw['exc'] - res_df['exc']:12.4e} Ha")
print(f"  Total:        {res_paw['etot'] - res_df['etot']:12.4e} Ha")
print("="*50)

