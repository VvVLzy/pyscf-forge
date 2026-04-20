import numpy as np
from pyscf import gto, scf

def load_cp2k_molden(filepath, n_basis, n_mo):
    """
    Reads a sparse CP2K molden file and returns dense PySCF-compatible arrays.
    Missing AO coefficients are automatically left as 0.0.
    """
    # Initialize a matrix of pure zeros
    mo_coeffs = np.zeros((n_basis, n_mo))
    mo_energies = []
    mo_occs = []

    # Pre-parse [GTO] to build basis mapping from Molden to PySCF
    spherical = True
    in_gto = False
    mapping = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        if '[6D' in line.upper() or '[6D10F]' in line.upper():
            spherical = False
        elif '[5D' in line.upper() or '[5D7F]' in line.upper():
            spherical = True

    for line in lines:
        line_strip = line.strip()
        if not line_strip: continue
        if '[GTO]' in line_strip:
            in_gto = True
            continue
        if in_gto and line_strip.startswith('[') and not '[GTO]' in line_strip:
            break
        if in_gto:
            parts = line_strip.split()
            if len(parts) >= 2 and parts[0].isalpha() and parts[0].isascii():
                shell_type = parts[0].lower()
                if shell_type == 's':
                    mapping.append(len(mapping))
                elif shell_type == 'p':
                    start = len(mapping)
                    mapping.extend([start, start+1, start+2])
                elif shell_type == 'sp':
                    start = len(mapping)
                    mapping.extend([start, start+1, start+2, start+3])
                elif shell_type == 'd':
                    start = len(mapping)
                    if spherical:
                        # Molden: z2, xz, yz, x2-y2, xy
                        # PySCF: xy, yz, z2, xz, x2-y2
                        mapping.extend([start+2, start+3, start+1, start+4, start+0])
                    else:
                        # Molden: xx, yy, zz, xy, xz, yz
                        # PySCF: xx, xy, xz, yy, yz, zz
                        mapping.extend([start+0, start+3, start+5, start+1, start+2, start+4])
                elif shell_type == 'f':
                    start = len(mapping)
                    if spherical:
                        # Molden: z3, xz2, yz2, x(x2-3y2), y(3x2-y2), xyz, z(x2-y2)
                        # PySCF: y(3x2-y2), xyz, yz2, z3, xz2, z(x2-y2), x(x2-3y2)
                        mapping.extend([start+3, start+4, start+2, start+6, start+0, start+1, start+5])
                    else:
                        mapping.extend(range(start, start+10))
                elif shell_type == 'g':
                    start = len(mapping)
                    mapping.extend(range(start, start+(9 if spherical else 15)))

    # Fallback identity mapping if GTO block is incomplete or missing
    if len(mapping) < n_basis:
        mapping.extend(range(len(mapping), n_basis))
    
    current_mo = -1
    in_mo = False
    for line in lines:
        if '[MO]' in line:
            in_mo = True
            continue
        if not in_mo:
            continue
        
        # Stop parsing if we hit another section
        if line.startswith('[') and not '[MO]' in line:
            break
            
        if 'Ene=' in line:
            current_mo += 1
            mo_energies.append(float(line.split('=')[1]))
        elif 'Occup=' in line:
            mo_occs.append(float(line.split('=')[1]))
        elif 'Spin=' in line:
            continue
        elif line.strip():
            parts = line.split()
            if len(parts) == 2:
                ao_idx = int(parts[0]) - 1 # 0-indexed molden AO
                if ao_idx < len(mapping):
                    ao_idx = mapping[ao_idx] # map to PySCF
                coeff = float(parts[1].replace('E', 'e'))
                mo_coeffs[ao_idx, current_mo] = coeff
                    
    return mo_coeffs, np.array(mo_energies), np.array(mo_occs)