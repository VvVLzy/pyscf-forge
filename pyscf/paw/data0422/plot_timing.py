import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import ScalarFormatter

# Load the data
file_path = "data0422/TheorySeminarData - Timing.csv"
df = pd.read_csv(file_path, skiprows=3)
df.columns = [c.strip() for c in df.columns]
df = df.dropna(subset=['Nao', 'cp2k 1st scf', 'pyscf 1st scf', 'vxc'])

def fit_power_law(x, y):
    p, log_C = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(log_C), p

# Perform fits
c_cp2k, p_cp2k = fit_power_law(df['Nao'], df['cp2k 1st scf'])
c_pyscf, p_pyscf = fit_power_law(df['Nao'], df['pyscf 1st scf'])
c_vxc, p_vxc = fit_power_law(df['Nao'], df['vxc'])

# Improve figure aesthetics
fig, ax = plt.subplots(figsize=(10, 8))

# Define smooth range for fit lines
x_fit = np.linspace(min(df['Nao']), max(df['Nao']), 100)

# Plot Data Points
ax.scatter(df['Nao'], df['cp2k 1st scf'], color='#1f77b4', marker='o', s=100, zorder=5)
ax.scatter(df['Nao'], df['pyscf 1st scf'], color='#ff7f0e', marker='s', s=100, zorder=5)
ax.scatter(df['Nao'], df['vxc'], color='#2ca02c', marker='d', s=100, zorder=5)

# Plot Fitted Lines
ax.loglog(x_fit, c_cp2k * x_fit**p_cp2k, color='#1f77b4', label='CP2K', linewidth=2.5, zorder=4)
ax.loglog(x_fit, c_pyscf * x_fit**p_pyscf, color='#ff7f0e', label='PySCF', linewidth=2.5, zorder=4)
ax.loglog(x_fit, c_vxc * x_fit**p_vxc, color='#2ca02c', linestyle='--', label='PySCF Vxc', linewidth=2.0, alpha=0.8, zorder=4)

# Configure axes
ax.set_xlabel('Number of Atomic Orbitals ($N_{ao}$)', fontsize=16, fontweight='bold')
ax.set_ylabel('Time per SCF Cycle (seconds)', fontsize=16, fontweight='bold')
ax.set_title('Computational Timing Scaling', fontsize=18, fontweight='bold')

# Set natural limits
ax.set_xlim(15, 200)
ax.set_ylim(0.1, 20)

# Formatting ticks and grid
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xticks([20, 40, 60, 80, 100, 150])
ax.tick_params(axis='both', which='major', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Create a nice text box for the equations with BIGGER font
fit_text = (
    r"$\bf{Scaling:}$" + "\n"
    rf"CP2K: $T \propto N_{{ao}}^{{{p_cp2k:.2f}}}$" + "\n"
    rf"PySCF: $T \propto N_{{ao}}^{{{p_pyscf:.2f}}}$" + "\n"
    rf"Vxc: $T \propto N_{{ao}}^{{{p_vxc:.2f}}}$"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax.text(0.05, 0.95, fit_text, transform=ax.transAxes, fontsize=16, # Increased font size
        verticalalignment='top', bbox=props)

ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(fontsize=14, loc='lower right')

plt.tight_layout()
output_dir = 'data0422/img'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'timing_scaling.png')
plt.savefig(output_path, dpi=300)
print(f"Improved timing plot with fitted lines saved to {output_path}")
