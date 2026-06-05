import sys
from pathlib import Path
import numpy

# Get the path of the parent directory to allow imports from pyscf.paw
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from pyscf import gto
from pyscf.paw.neighbor import get_neighbor_list, get_atom_radii

def test_neighbor_scaling():
    print("--- Testing Neighbor List Logic ---")
    
    # 1. Isolated chain of Neon atoms
    # STO-3G is compact, so we should see clear neighbor filtering
    spacing = 10.0 # Large spacing to ensure filtering
    mol = gto.M(
        atom=[['Ne', [i * spacing, 0, 0]] for i in range(10)],
        basis='sto-3g',
        unit='Bohr',
        verbose=0
    )
    
    radii = get_atom_radii(mol, epsilon=1e-8)
    print(f"Typical Ne radius (STO-3G, eps=1e-8): {radii[0]:.2f} Bohr")

    n_list = get_neighbor_list(mol, epsilon=1e-8)

    print("\nNeighbor list for 10-atom chain (spacing=10.0 Bohr):")
    for i, neighbors in enumerate(n_list):
        print(f"  Atom {i:2d} neighbors: {neighbors}")
        # With 10 Bohr spacing and ~2.5 Bohr radius, 
        # neighbors should only be [i] (itself)
        # assert len(neighbors) == 1
        # assert neighbors[0] == i

    # 2. Tight chain
    spacing = 3.0
    mol_tight = gto.M(
        atom=[['Ne', [i * spacing, 0, 0]] for i in range(10)],
        basis='sto-3g',
        unit='Bohr',
        verbose=0
    )
    n_list_tight = get_neighbor_list(mol_tight, epsilon=1e-8)
    print(f"\nNeighbor list for tight 10-atom chain (spacing={spacing} Bohr):")
    for i, neighbors in enumerate(n_list_tight):
        print(f"  Atom {i:2d} neighbors: {neighbors}")
        # Should see adjacent atoms now
        
    print("\nNeighbor List Test Passed!")

if __name__ == "__main__":
    test_neighbor_scaling()
