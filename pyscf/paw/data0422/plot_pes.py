import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

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
            # Split line and take relevant columns (volume, etot_paw, etot_cp2k)
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    v = float(parts[0])
                    e_paw = float(parts[1])
                    e_gdf = float(parts[2])
                    e_cp2k = float(parts[5])
                    data_blocks[current_label].append([v, e_paw, e_cp2k, e_gdf])
                except ValueError:
                    continue
                    
    return {k: pd.DataFrame(v, columns=['Volume', 'PAW', 'CP2K', 'GDF']) for k, v in data_blocks.items()}

file_path = 'data0422/TheorySeminarData - BM_C.csv'
dfs = parse_csv(file_path)

fig, ax = plt.subplots(figsize=(12, 8))

colors = {'DZ': '#1f77b4', 'TZ': '#2ca02c', 'QZ': '#d62728'}
markers = {'PAW': 'o', 'CP2K': 'x', 'GDF': 's'}

for label, df in dfs.items():
    c = colors.get(label, 'black')
    # Sort by volume for smooth lines
    df = df.sort_values('Volume')
    
    ax.plot(df['Volume'], df['PAW'], marker=markers['PAW'], linestyle='-', color=c, label=f'PAW {label}', linewidth=2)
    ax.plot(df['Volume'], df['CP2K'], marker=markers['CP2K'], linestyle='--', color=c, label=f'CP2K {label}', linewidth=1.5, alpha=0.7)
    
    # Only plot GDF if values are not zero/missing (e.g. QZ is 0 in csv)
    if not (df['GDF'] == 0).all():
        ax.plot(df['Volume'], df['GDF'], marker=markers['GDF'], linestyle=':', color=c, label=f'GDF {label}', linewidth=1.5, alpha=0.6)

ax.set_xlabel('Volume (Bohr$^3$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Energy (Hartree)', fontsize=14, fontweight='bold')
ax.set_title('Potential Energy Surface For Carbon (Diamond)', fontsize=16, fontweight='bold')

ax.tick_params(axis='both', which='major', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(fontsize=12, loc='best')

# Format y-axis to show decimals clearly
ax.ticklabel_format(useOffset=False, style='plain', axis='y')

plt.tight_layout()
output_dir = 'data0422/img'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'bm_c_pes.png')
plt.savefig(output_path, dpi=300)
print(f"PES plot saved to {output_path}")
