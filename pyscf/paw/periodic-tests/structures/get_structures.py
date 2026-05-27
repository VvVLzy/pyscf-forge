from mp_api.client import MPRester
from pymatgen.io.xyz import XYZ
from pymatgen.core import Molecule

# Replace with your actual API key from the Materials Project dashboard
API_KEY = "ld5HdyuSrVhvlNZKJIPy2iM58UCKuX2V"

structure_name = ['Si',
                  'NaCl',
                  'BaTiO3',
                  'FeS2',
                  'SnO2',
                  'C']
structure_id = ['mp-149',
                'mp-22862',
                'mp-2998',
                'mp-226',
                'mp-856',
                'mp-66']

for i, id in enumerate(structure_id):
    with MPRester(API_KEY) as mpr:
        # 1. Get the structure object
        structure = mpr.get_structure_by_material_id(
            id,
            conventional_unit_cell=True
        ) 
        
        # 2. Convert to a 'Molecule' object to write to XYZ 
        # (XYZ doesn't store lattice vectors by default)
        mol = Molecule(structure.species, structure.cart_coords)
        
        # 3. Save to file
        mol.to(fmt="xyz", filename=f"{structure_name[i]}.xyz")

        print(f'Pyscf lattice matrix for {structure_name[i]}')
        lattice_matrix = structure.lattice.matrix
        print(lattice_matrix)
        print(f'Cp2k lattice matrix for {structure_name[i]}')
        print('&CELL')
        print(f'  A\t{lattice_matrix[0][0]}\t{lattice_matrix[0][1]}\t{lattice_matrix[0][2]}')
        print(f'  B\t{lattice_matrix[1][0]}\t{lattice_matrix[1][1]}\t{lattice_matrix[1][2]}')
        print(f'  C\t{lattice_matrix[2][0]}\t{lattice_matrix[2][1]}\t{lattice_matrix[2][2]}')
        print('  PERIODIC XYZ')
        print('&END CELL')
        print('')
