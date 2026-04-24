import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Radial grid for a single atom
R_SINGLE = np.linspace(-3.0, 3.0, 1000)

# Shared Parameters
# Exponent 2.212 scaled by 0.4 for visibility
ALPHA_FULL = np.array([234.0, 35.16, 7.989, 2.212 * 0.4])
COEFF_FULL = np.array([0.0025870, 0.0195330, 0.0909980, 0.2720500])
ALPHA_SMOOTH = np.array([2.212 * 0.4])
COEFF_SMOOTH = np.array([0.2720500])
SCALE_FACTOR = 3.0
ATOM_RADIUS = 1.2

def calc_psi(r, alpha, coeff):
    psi = np.zeros_like(r)
    for a, c in zip(alpha, coeff):
        # Gaussian normalization factor for s-orbitals
        norm = (2 * a / np.pi)**(0.75)
        psi += c * norm * np.exp(-a * r**2)
    return psi * SCALE_FACTOR

# Base orbitals
PSI_FULL_BASE = calc_psi(R_SINGLE, ALPHA_FULL, COEFF_FULL)
PSI_SMOOTH_BASE = calc_psi(R_SINGLE, ALPHA_SMOOTH, COEFF_SMOOTH)
PSI_DIFF_BASE = PSI_FULL_BASE - PSI_SMOOTH_BASE

def draw_he_atom(ax, x0, y0, radius):
    """Draws a Helium atom as a blue sphere with a radial gradient."""
    circle = patches.Circle((x0, y0), radius=radius, color='#5c7fcf', alpha=0.9, zorder=1)
    ax.add_patch(circle)
    # Radial gradient effect
    for i in range(20, 0, -1):
        c = patches.Circle((x0, y0), radius=radius * i / 20, color='white', alpha=0.02, zorder=2)
        ax.add_patch(c)

def plot_single_he(output_path='data0422/paw_wf/he_ao_plot.png'):
    """Generates the single Helium atom plot with all 3 orbital representations."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    draw_he_atom(ax, 0, 0, ATOM_RADIUS)
    
    # Truncate tails significantly
    truncation_radius = 2.2
    mask = np.abs(R_SINGLE) < truncation_radius
    r_plot = R_SINGLE[mask]
    
    # Plot orbitals
    ax.plot(r_plot, PSI_FULL_BASE[mask], color='orange', linewidth=4, label=r'$|\psi\rangle$', zorder=4)
    ax.plot(r_plot, PSI_SMOOTH_BASE[mask], color='red', linewidth=4, linestyle='--', label=r'$|\tilde{\psi}\rangle$', zorder=5)
    ax.plot(r_plot, PSI_DIFF_BASE[mask], color='#2e7d32', linewidth=3, alpha=0.8, label=r'$|\psi\rangle - |\tilde{\psi}\rangle$', zorder=3)
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.3, max(np.max(PSI_FULL_BASE), np.max(PSI_SMOOTH_BASE)) * 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"Single atom plot saved to {output_path}")

def plot_he_array(output_path='data0422/paw_wf/he_array_plot.png'):
    """Generates a horizontal array of 4 Helium atoms."""
    fig, ax = plt.subplots(figsize=(16, 6))
    spacing = 3.8
    centers = [i * spacing for i in range(4)]
    truncation_radius = 1.9

    for x0 in centers:
        draw_he_atom(ax, x0, 0, ATOM_RADIUS)
        
        # Plot orbitals shifted to center, but truncate tails to avoid overlap clutter
        mask = np.abs(R_SINGLE) < truncation_radius
        r_plot = R_SINGLE[mask] + x0
        
        ax.plot(r_plot, PSI_FULL_BASE[mask], color='orange', linewidth=3, zorder=4)
        ax.plot(r_plot, PSI_SMOOTH_BASE[mask], color='red', linewidth=3, linestyle='--', zorder=5)
        ax.plot(r_plot, PSI_DIFF_BASE[mask], color='#2e7d32', linewidth=2, alpha=0.7, zorder=3)
        
    ax.set_xlim(centers[0] - 3, centers[-1] + 3)
    ax.set_ylim(-1.3, np.max(PSI_FULL_BASE) * 1.3)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"Array plot saved to {output_path}")

if __name__ == "__main__":
    # Create target directory if it doesn't exist
    os.makedirs('data0422/paw_wf', exist_ok=True)
    
    # Toggle plots here
    plot_single_he()
    plot_he_array()
