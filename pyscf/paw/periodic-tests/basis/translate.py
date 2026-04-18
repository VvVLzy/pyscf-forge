import re

def convert_basis(input_file, output_file, basis_name):
    with open(input_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]

    # Using a dictionary to group sets per element key
    element_blocks = {}
    current_element = None
    current_set = None

    for line in lines:
        if line.startswith('#'):
            continue
            
        # Detect Element Header (e.g., "Na S")
        if re.match(r'^[A-Za-z]+\s+[SPDFG]', line, re.IGNORECASE):
            parts = line.split()
            current_element = parts[0]
            orb_type = {'S': 0, 'P': 1, 'D': 2, 'F': 3, 'G': 4}[parts[1].upper()]
            
            # Initialize element entry in dictionary if it doesn't exist
            if current_element not in element_blocks:
                element_blocks[current_element] = []
            
            current_set = {'l': orb_type, 'data': []}
            element_blocks[current_element].append(current_set)
        
        elif current_set is not None:
            # Clean data: Replace Fortran D/d with 'e'
            row = re.sub(r'([0-9\.\-])[Dd]([+-]?[0-9]+)', r'\1e\2', line)
            try:
                current_set['data'].append([float(x) for x in row.split()])
            except ValueError:
                continue

    # Write output file
    with open(output_file, 'w') as f:
        for element, sets in element_blocks.items():
            f.write(f"{element}  {basis_name} {basis_name}-q2\n")
            f.write(f"  {len(sets)}\n")
            for s in sets:
                nexp = len(s['data'])
                # nshells is the count of coefficient columns (total columns - 1 for exponent)
                nshells = len(s['data'][0]) - 1
                l = s['l']
                # CP2K format: n(placeholder) lmin lmax nexp nshell
                f.write(f"  {l+1} {l} {l} {nexp} {nshells}\n")
                for row in s['data']:
                    f.write("    " + "  ".join(f"{x:>16.8f}" for x in row) + "\n")

# Usage:
# convert_basis('ccpvdz.dat', 'ccpvdz_cp2k.dat', 'CC-PVDZ')
convert_basis('ccpvtz.dat', 'ccpvtz_cp2k.dat', 'CC-PVTZ')