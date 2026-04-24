import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Radial grid for a single atom
r_single = np.linspace(-3.0, 3.0, 1000)

# Parameters (consistent with previous plot)
# Exponent 2.212 scaled by 0.4
alpha_full = np.array([234.0, 35.16, 7.989, 2.212 * 0.4])
coeff_full = np.array([0.0025870, 0.0195330, 0.0909980, 0.2720500])
alpha_smooth = np.array([2.212 * 0.4])
coeff_smooth = np.array([0.2720500])

def calc_psi(r, alpha, coeff):
    psi = np.zeros_like(r)
    for a, c in zip(alpha, coeff):
        norm = (2 * a / np.pi)**(0.75)
        psi += c * norm * np.exp(-a * r**2)
    return psi

# Calculate base orbitals and scale them
scale_factor = 3.0
psi_full_base = calc_psi(r_single, alpha_full, coeff_full) * scale_factor
psi_smooth_base = calc_psi(r_single, alpha_smooth, coeff_smooth) * scale_factor
psi_diff_base = psi_full_base - psi_smooth_base

# Setup multi-atom plot
fig, ax = plt.subplots(figsize=(16, 6))
spacing = 3.8
centers = [i * spacing for i in range(4)]

for x0 in centers:
    # Draw sphere
    radius = 1.2
    circle = patches.Circle((x0, 0), radius=radius, color='#5c7fcf', alpha=0.9, zorder=1)
    ax.add_patch(circle)
    for i in range(15, 0, -1):
        c = patches.Circle((x0, 0), radius=radius * i / 15, color='white', alpha=0.02, zorder=2)
        ax.add_patch(c)
    
    # Plot orbitals shifted to center, but truncate tails to avoid overlap clutter
    mask_ao = np.abs(r_single) < 1.9
    mask_diff = np.abs(r_single) < 1.2  # Difference is only within the sphere
    
    r_plot_ao = r_single[mask_ao] + x0
    r_plot_diff = r_single[mask_diff] + x0
    
    ax.plot(r_plot_ao, psi_full_base[mask_ao], color='orange', linewidth=3, zorder=4)
    ax.plot(r_plot_ao, psi_smooth_base[mask_ao], color='red', linewidth=3, linestyle='--', zorder=5)
    ax.plot(r_plot_diff, psi_diff_base[mask_diff], color='#2e7d32', linewidth=2, alpha=0.7, zorder=3)
    
    # Label U_a for each
    ax.text(x0 + 0.3, -0.6, r'$U_a$', fontsize=18, fontweight='bold')

ax.set_xlim(centers[0] - 3, centers[-1] + 3)
ax.set_ylim(-1.3, np.max(psi_full_base) * 1.3)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('data0422/paw_wf/he_array_plot.png', dpi=300, transparent=True)
print("Array plot saved to data0422/paw_wf/he_array_plot.png")
