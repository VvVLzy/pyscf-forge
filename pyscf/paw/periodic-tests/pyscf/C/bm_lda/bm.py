import csv
from functools import reduce

from pyscf import scf
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf as pscf
import numpy

from pyscf.paw import NewPAW as PAW
from pyscf.paw.vxc import PAWNumInt

L = 3.56074511  # box size

# specify He2
atom    = '''
C 0.000000 0.000000 1.780373
C 0.890186 0.890186 2.670559
C 0.000000 1.780373 0.000000
C 0.890186 2.670559 0.890186
C 1.780373 0.000000 0.000000
C 2.670559 0.890186 0.890186
C 1.780373 1.780373 1.780373
C 2.670559 2.670559 2.670559
'''
rscales = numpy.linspace(0.9, 1.1, 5)
print(rscales)

zeta = 'qz'
basis = "ccpv"+zeta

verbose = 4
a = numpy.eye(3) * L
ke_cutoff = 200
precision = 1e-8

cell = pgto.M(
    atom        = atom,
    basis       = basis,
    verbose     = verbose,
    a           = a,
    ke_cutoff   = ke_cutoff,
    precision   = precision
)
cell.exp_to_discard = 0.2

coords_bohr = cell.atom_coords() 
lattice_bohr = cell.lattice_vectors()

init_guess = '1e'
xc = 'lda,pw'

alpha0=10
Rb=1.5


def eig_lindep(h, s):
    d, t = numpy.linalg.eigh(s)
    idx_good = d > 1e-7
    idx_bad = ~idx_good
    
    # Orthogonalize good subspace
    x_good = t[:,idx_good] / numpy.sqrt(d[idx_good])
    xhx_good = reduce(numpy.dot, (x_good.T, h, x_good))
    e_good, c_good = numpy.linalg.eigh(xhx_good)
    mo_coeff_good = numpy.dot(x_good, c_good)
    
    # Bad subspace: just use normalized t vectors
    mo_coeff_bad = t[:,idx_bad] 
    e_bad = numpy.ones(numpy.sum(idx_bad)) * 1e9
    
    mo_energy = numpy.concatenate([e_good, e_bad])
    mo_coeff = numpy.concatenate([mo_coeff_good, mo_coeff_bad], axis=1)
    
    # Sort them by energy
    idx_sort = numpy.argsort(mo_energy)
    return mo_energy[idx_sort], mo_coeff[:,idx_sort]

def calc_pes():
    results = []
    for x in numpy.linspace(0.9, 1.1, 5):
        # x=1
        # scale cell and coord
        scaled_coords = coords_bohr * x
        scaled_lattice = lattice_bohr * x

        cell.set_geom_(scaled_coords, unit='Bohr')
        cell.a = scaled_lattice
        cell.build()

        # paw calculation
        mf_per_rks_paw = pscf.RKS(cell)
        mf_per_rks_paw.init_guess = init_guess
        mf_per_rks_paw.xc = xc
        mf_per_rks_paw.max_cycle = 50
        mf_per_rks_paw.eig = eig_lindep
        mydf = PAW.from_mf(mf_per_rks_paw, alpha0=alpha0, augRadius=Rb, gdfNuc=False).build()
        mf_per_rks_paw.with_df = mydf
        pawnumint = PAWNumInt(mydf, mf_per_rks_paw)
        mf_per_rks_paw._numint = pawnumint
        mf_per_rks_paw.kernel()

        # from pyscf.pbc import df
        # mf_per_rks_gdf = pscf.RKS(cell)
        # mf_per_rks_gdf.init_guess = init_guess
        # mf_per_rks_gdf.xc = xc
        # mf_per_rks_gdf.max_cycle = 50
        # mf_per_rks_gdf.eig = eig_lindep
        # gdf_obj = df.GDF(cell)
        # mf_per_rks_gdf.with_df = gdf_obj
        # mf_per_rks_gdf.kernel()

        results.append({
                "volume": cell.vol,
                "etot_paw": mf_per_rks_paw.e_tot,
                # "etot_gdf": mf_per_rks_gdf.e_tot,
                "etot_gdf": 0,
                "conv_paw": mf_per_rks_paw.converged,
                # "conv_gdf": mf_per_rks_gdf.converged
                "conv_gdf": False
            })
    
    prefix = './'
    path = prefix + f"data/CBulkModulus_{ke_cutoff:.2f}_a_{alpha0}_r_{Rb}_{zeta}_{precision:.0e}_{xc}_nuc.csv"
    print(f"Calculation done. Saving data to {path}...")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "volume", "etot_paw", "etot_gdf", "conv_paw", "conv_gdf"
        ])
        writer.writeheader()
        writer.writerows(results)
    return path

def read_data(path):
    volumes = []
    etot_paw = []
    etot_gdf = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            volumes.append(float(row["volume"]))
            etot_paw.append(float(row["etot_paw"]))
            etot_gdf.append(float(row["etot_gdf"]))
    return numpy.array(volumes), numpy.array(etot_paw), numpy.array(etot_gdf)

def analyze_bm(path):
    from ase.eos import EquationOfState
    from ase.units import Hartree, Bohr, GPa
    
    volumes, etot_paw, etot_gdf = read_data(path)
    
    # Convert to ASE units (eV and Angstrom)
    volumes_ang = volumes * Bohr**3
    energies_paw_ev = etot_paw * Hartree
    energies_gdf_ev = etot_gdf * Hartree
    
    print(f"\nAnalysis for {path}:")
    
    for label, es in [("PAW", energies_paw_ev), ("GDF", energies_gdf_ev)]:
        eos = EquationOfState(volumes_ang, es)
        v0, e0, B = eos.fit()
        a0 = v0**(1/3)
        print(f"{label:3s}: B = {B/GPa:10.4f} GPa, v0 = {v0/Bohr**3:10.4f} Bohr^3, a0 = {a0:10.6f} Angstrom, e0 = {e0/Hartree:10.8f} Hartree")

def main():
    import os
    prefix = './'
    path = prefix + f"data/CBulkModulus_{ke_cutoff:.2f}_a_{alpha0}_r_{Rb}_{zeta}_{precision:.0e}_{xc}_nuc.csv"
    
    if not os.path.exists(path):
        path = calc_pes()
    
    analyze_bm(path)


if __name__ == '__main__':
    main()
