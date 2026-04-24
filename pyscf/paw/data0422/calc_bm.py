import pandas as pd
import numpy as np
from ase.eos import EquationOfState
from ase.units import Hartree, Bohr, kJ
import json
import os

# Conversion factor for Bulk Modulus: eV/Angstrom^3 to GPa
# 1 eV/Angstrom^3 = 160.21766208 GPa
EV_ANG3_TO_GPA = 160.21766208

def parse_csv(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    data_blocks = {}
    current_label = None
    
    for line in lines:
        clean = line.strip().strip(',')
        if clean in ['DZ', 'TZ', 'QZ']:
            current_label = clean
            data_blocks[current_label] = []
            continue
        
        if current_label and clean and not clean.startswith('volume'):
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    v = float(parts[0])
                    e_paw = float(parts[1])
                    e_gdf = float(parts[2])
                    e_cp2k = float(parts[5])
                    data_blocks[current_label].append({
                        'Volume': v, # Bohr^3
                        'PAW': e_paw, # Hartree
                        'GDF': e_gdf,
                        'CP2K': e_cp2k
                    })
                except ValueError:
                    continue
                    
    return {k: pd.DataFrame(v) for k, v in data_blocks.items()}

def calculate_properties(volumes_bohr3, energies_hartree, label):
    # Convert to ASE standard units: eV and Angstrom^3
    volumes_ang3 = volumes_bohr3 * (Bohr**3)
    energies_ev = energies_hartree * Hartree
    
    try:
        eos = EquationOfState(volumes_ang3, energies_ev, eos='birchmurnaghan')
        v0, e0, B_ev_ang3 = eos.fit()
        
        # a0 = v0^(1/3) for the cubic cell (8 atoms)
        a0 = v0**(1/3)
        B_gpa = B_ev_ang3 * EV_ANG3_TO_GPA
        
        return {
            'a0': round(a0, 6),
            'B_gpa': round(B_gpa, 2),
            'v0': round(v0, 4)
        }
    except Exception as e:
        return None

file_path = 'data0422/TheorySeminarData - BM_C.csv'
dfs = parse_csv(file_path)

results = {}

print(f"{'Basis':<5} | {'Method':<6} | {'a0 (Ang)':<10} | {'B (GPa)':<8}")
print("-" * 40)

for basis, df in dfs.items():
    results[basis] = {}
    for method in ['PAW', 'CP2K', 'GDF']:
        # Skip if all energies are zero (e.g. QZ GDF)
        if (df[method] == 0).all():
            continue
            
        prop = calculate_properties(df['Volume'].values, df[method].values, f"{basis} {method}")
        if prop:
            results[basis][method] = prop
            print(f"{basis:<5} | {method:<6} | {prop['a0']:<10} | {prop['B_gpa']:<8}")

# Save results to JSON
output_path = 'data0422/bm_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {output_path}")
