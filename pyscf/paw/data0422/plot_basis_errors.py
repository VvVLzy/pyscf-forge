import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the data
json_path = 'data0422/paw_basis_errors.json'
with open(json_path, 'r') as f:
    data = json.load(f)

basis_sets = list(data.keys())
all_molecules = []
for b in basis_sets:
    for m in data[b].keys():
        if m not in all_molecules:
            all_molecules.append(m)

# Sort molecules by complexity (rough estimate based on appearance in JSON)
molecules = ["He", "Ne", "He2", "Ne2", "N2", "Methane", "Ethylene", "Propene", "Butadiene", "Cyclobutane", "Benzene"]
molecules = [m for m in molecules if m in all_molecules]

# Plotting setup
fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(molecules))
width = 0.25 

# Plot bars for each basis set
# Set zorder=2 so they are behind the grid lines
for i, basis in enumerate(basis_sets):
    errors = []
    for mol in molecules:
        if mol in data[basis]:
            errors.append(data[basis][mol]['err_scf'])
        else:
            errors.append(0)
    
    offset = (i - (len(basis_sets)-1)/2) * width
    ax.bar(x + offset, errors, width, label=basis, zorder=2)

# Configure axes
ax.set_yscale('log')
ax.set_ylabel('Energy Error (Hartree)', fontsize=20, fontweight='bold')
ax.set_title('PAW Energy Error For Molecules', fontsize=22, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(molecules, rotation=45, ha='right', fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelsize=14)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Configure grid: only major ticks, dashed, in front of bars (zorder=3)
ax.grid(True, which='major', axis='y', linestyle='--', color='black', alpha=0.3, zorder=3)
ax.grid(False, which='minor', axis='y')

# Legend and spines
ax.legend(fontsize=14, frameon=True, loc='upper left')

plt.tight_layout()
output_dir = 'data0422/img'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'paw_basis_errors.png')
plt.savefig(output_path, dpi=300)
print(f"Bar chart saved to {output_path}")
