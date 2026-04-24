import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data, skipping the metadata rows
file_path = 'data0422/TheorySeminarData - compCharge.csv'
# The header is on the 4th line (index 3)
df = pd.read_csv(file_path, skiprows=3)

# Filter out any empty rows if they exist
df = df.dropna(subset=['Bond Length (Å)', 'before', 'after', 'cp2k'])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(df['Bond Length (Å)'], df['before'], 'o-', label='Before Correction', linewidth=2)
ax.plot(df['Bond Length (Å)'], df['after'], 's-', label='After Correction', linewidth=2)
ax.plot(df['Bond Length (Å)'], df['cp2k'], '^-', label='CP2K', linewidth=2)

# Configure axes
ax.set_yscale('log')
ax.set_xlabel('Bond Length (Å)', fontsize=14, fontweight='bold')
ax.set_ylabel('Energy Error (Hartree)', fontsize=14, fontweight='bold')
ax.set_title('Effect of Periodic Correction to Compensating Charge on PAW Accuracy', fontsize=16, fontweight='bold')
ax.tick_params(axis='y', labelsize=14)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

ax.grid(True, which='major', linestyle='--', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)
ax.legend(fontsize=12)

plt.tight_layout()
output_dir = 'data0422/img'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'comp_charge_errors.png')
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
