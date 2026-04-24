import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Radial grid
r = np.linspace(-3.0, 3.0, 1000)

# Basis set parameters from basis.png
# Full AO primitives (all in green box)
# Only the last primitive (2.212) is treated as the "smooth" part here
alpha_full = np.array([234.0, 35.16, 7.989, 2.212 * 0.4])
coeff_full = np.array([0.0025870, 0.0195330, 0.0909980, 0.2720500])

# Smooth AO primitives (now choosing only the most diffuse one: 2.212)
alpha_smooth = np.array([2.212 * 0.4])
coeff_smooth = np.array([0.2720500])

def calc_psi(r, alpha, coeff):
    psi = np.zeros_like(r)
    for a, c in zip(alpha, coeff):
        # Using a simple Gaussian form for visualization
        # We include the normalization factor for s-orbitals: N = (2*alpha/pi)^(3/4)
        norm = (2 * a / np.pi)**(0.75)
        psi += c * norm * np.exp(-a * r**2)
    return psi

psi_full = calc_psi(r, alpha_full, coeff_full)
psi_smooth = calc_psi(r, alpha_smooth, coeff_smooth)

# Increase amplitude for visibility (manual scaling)
scale_factor = 3.0
psi_full *= scale_factor
psi_smooth *= scale_factor

# Calculate the difference: full AO minus smooth AO
psi_diff = psi_full - psi_smooth

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8))

# Draw the Helium atom as a blue sphere (smaller radius)
radius = 1.2
circle = patches.Circle((0, 0), radius=radius, color='#5c7fcf', alpha=0.9, zorder=1)
ax.add_patch(circle)

# Add a subtle radial gradient to the sphere
for i in range(20, 0, -1):
    c = patches.Circle((0, 0), radius=radius * i / 20, color='white', alpha=0.02, zorder=2)
    ax.add_patch(c)

# Plot the orbitals
ax.plot(r, psi_full, color='orange', linewidth=4, label=r'$|\psi\rangle$ (Full AO)', zorder=4)
ax.plot(r, psi_smooth, color='red', linewidth=4, linestyle='--', label=r'$|\tilde{\psi}\rangle$ (Smooth AO)', zorder=5)
ax.plot(r, psi_diff, color='#2e7d32', linewidth=3, alpha=0.8, label=r'$|\psi\rangle - |\tilde{\psi}\rangle$', zorder=3)

# Set plot limits
ax.set_xlim(-3, 3)
max_val = max(np.max(psi_full), np.max(psi_smooth), np.max(psi_diff))
ax.set_ylim(-1.3, max_val * 1.2)
ax.set_aspect('equal')
ax.axis('off')

# Annotations
# Position labels strategically for maximum clarity
# Full AO (orange) - Top right
r_label_full = 1.2
y_full = psi_full[np.argmin(np.abs(r - r_label_full))]
ax.text(r_label_full + 0.2, y_full + 0.4, r'$|\psi^a\rangle$', color='orange', fontsize=22, fontweight='bold')

# Smooth AO (red dashed) - Far right
r_label_tilde = 2.4
y_tilde = psi_smooth[np.argmin(np.abs(r - r_label_tilde))]
ax.text(r_label_tilde + 0.1, y_tilde + 0.1, r'$|\tilde{\psi}\rangle$', color='red', fontsize=22, fontweight='bold')

# Difference (green) - Top left
r_label_diff = -0.5
y_diff = psi_diff[np.argmin(np.abs(r - r_label_diff))]
ax.text(r_label_diff - 2.3, y_diff + 0.2, r'$|\psi^a\rangle - |\tilde{\psi}^a\rangle$', color='#2e7d32', fontsize=20, fontweight='bold')

# U_a (atom region) - Bottom center
ax.text(0.3, -0.6, r'$U_a$', fontsize=24, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig('data0422/paw_wf/he_ao_plot.png', dpi=300, transparent=True)
print("Plot saved to data0422/paw_wf/he_ao_plot.png")
