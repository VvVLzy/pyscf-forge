import os
import csv
import time
from pyscf import gto, scf
from pyscf.pbc import gto as pgto
import numpy as np
from pyscf.paw import NewPAW as PAW

verbose = 3
ke_cutoff = 100
init_guess = '1e'
box = np.array([[20, 0, 0], [0, 20, 0], [0, 0, 20]])

geom_dir = os.path.join(os.path.dirname(__file__), 'finalGeom')
xyz_files = sorted(f for f in os.listdir(geom_dir) if f.endswith('.xyz'))

results = []

for xyz_file in xyz_files:
    name = xyz_file[:-4]
    xyz_path = os.path.join(geom_dir, xyz_file)
    print(f'=== {name} ===')

    cell = pgto.M(
        atom=xyz_path,
        basis='cc-pvtz',
        a=box,
        verbose=verbose,
        ke_cutoff=ke_cutoff,
        max_memory=15000)

    mol = gto.M(atom=cell.atom, basis=cell.basis, verbose=verbose, max_memory=15000)

    # exact calculation
    mf_mol_exact = scf.RKS(mol)
    mf_mol_exact.xc = 'pbe'
    mf_mol_exact.init_guess = init_guess
    mf_mol_exact.kernel()

    # paw calculation
    start = time.time()
    mf_mol_paw = scf.RKS(mol).density_fit()
    mf_mol_paw.xc = 'pbe'
    mf_mol_paw.init_guess = init_guess
    mydf = PAW.from_mf(mf_mol_paw, cell,
                        alpha0={'H': None}, augRadius=None, with_multigrid=2,
                        alpha0_lowmem=True, auto_box=True).build()
    mf_mol_paw.with_df = mydf
    print(f"alpha0      : {str(mydf.alpha0)}")
    print(f"Rb          : {np.max(mydf.augRadius):10.2f}")
    print(f"Nelection   : {np.sum(mydf.cell.nelec)}")
    print(f"Ngrid points: {np.prod(mydf.cell.mesh):10d}")
    delta_a = (mydf.cell.vol / np.prod(mydf.cell.mesh)) ** (1. / 3.)
    print(f"delta-a     : {delta_a:10.2f}")
    mf_mol_paw.kernel()
    paw_time = time.time() - start
    print(f'{name} paw takes {paw_time:.2f} seconds')

    e_diff = mf_mol_exact.e_tot - mf_mol_paw.e_tot
    print(f'{name} energy difference: {e_diff}')

    results.append({
        'molecule': name,
        'e_exact': mf_mol_exact.e_tot,
        'e_paw': mf_mol_paw.e_tot,
        'e_diff': e_diff,
        'alpha0': mydf.alpha0,
        'delta_a': delta_a,
        'paw_time_s': paw_time,
    })

csv_path = os.path.join(os.path.dirname(__file__), 'smallMol_results1.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['molecule', 'e_exact', 'e_paw', 'e_diff', 'alpha0', 'delta_a', 'paw_time_s'])
    writer.writeheader()
    writer.writerows(results)

print(f'\nWrote results for {len(results)} molecules to {csv_path}')
