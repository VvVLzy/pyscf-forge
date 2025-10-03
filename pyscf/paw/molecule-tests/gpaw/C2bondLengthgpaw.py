# gpaw_c2_pbe.py
import numpy as np
from ase import Atoms
from gpaw import GPAW, PW, FermiDirac
from ase.units import Ha

# ke cutoff
ke_cut = 50 # pyscf cutoff in hartree
factor = 4  # this is the factor to convert pyscf cutoff to gpaw cutoff

rs, etot_gpaw, converged_gpaw = [], [], []
for rscale in np.linspace(0.7, 1.5, 15):
    # geometry: two C atoms separated by r (Angstrom)
    r1 = 10.
    r = 1.33*rscale  # bond length in Angstrom (you can change to 1.5)
    pos1 = np.array([r1, r1, r1])
    pos2 = np.array([r / np.sqrt(3), r / np.sqrt(3), r / np.sqrt(3)]) + pos1  # diagonal placement so distance = r
    atoms = Atoms('C2', positions=[pos1, pos2])

    # cell and periodicity
    a = np.eye(3) * 20.0  # 20 Å cubic cell (large vacuum)
    atoms.set_cell(a)
    atoms.set_pbc([True, True, True])  # PW mode expects periodic cell; large cell -> isolated molecule


    ecut_eV_optionB = ke_cut*Ha/factor

    # pick one:
    ecut = ecut_eV_optionB

    # GPAW calculator: PBE + PW
    calc = GPAW(mode=PW(ecut),           # plane-wave cutoff in eV
                xc='PBE',                # PBE functional
                kpts=(1, 1, 1),          # Gamma point (isolated molecule in big cell)
                txt=f'out/gpaw_C2_pbe_{r: .2f}.txt')   # output log

    atoms.set_calculator(calc)

    # run calculation
    energy = atoms.get_total_energy()

    # save data
    rs.append(r)
    etot_gpaw.append(energy/Ha)
    converged_gpaw.append(calc.scf.converged)

# write to csv
import csv

path = f'../data/C2bondLengthgpaw_{ke_cut/factor: .2f}.csv'
print(f'Calculation done. Saving data to {path}...')
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['r', "etot_gpaw", 'conv_gpaw'])
    for a, b, c in zip(rs, etot_gpaw, converged_gpaw):
        writer.writerow([a, b, c])