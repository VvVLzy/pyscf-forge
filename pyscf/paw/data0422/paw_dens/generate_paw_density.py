import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy.special import erf

# Radial grid
R = np.linspace(-5.0, 5.0, 2000)

# Shared Parameters
ALPHA_FULL = np.array([234.0, 35.16, 7.989 * 0.4, 2.212 * 0.4])
COEFF_FULL = np.array([0.0025870, 0.0195330, 0.0909980, 0.2720500])
ALPHA_SMOOTH = np.array([7.989 * 0.4, 2.212 * 0.4])
COEFF_SMOOTH = np.array([0.0909980, 0.2720500])
ATOM_RADIUS = 0.8
BOND_LENGTH = 2.4

def calc_psi_at(r, x0, alpha, coeff):
    psi = np.zeros_like(r)
    r_rel = r - x0
    for a, c in zip(alpha, coeff):
        norm = (2 * a / np.pi)**(0.75)
        psi += c * norm * np.exp(-a * r_rel**2)
    return psi

def draw_he_atom(ax, x0, y0, radius, alpha=0.8):
    circle = patches.Circle((x0, y0), radius=radius, color='#5c7fcf', alpha=alpha, zorder=1)
    ax.add_patch(circle)
    for i in range(15, 0, -1):
        c = patches.Circle((x0, y0), radius=radius * i / 15, color='white', alpha=0.02 * (alpha/0.8), zorder=2)
        ax.add_patch(c)

def get_shared_data():
    """Utility to calculate shared densities across functions."""
    n_atoms = 7
    spacing = 2.4
    centers = [(i - (n_atoms-1)/2) * spacing for i in range(n_atoms)]
    R_chain = np.linspace(centers[0] - 5, centers[-1] + 5, 4000)
    x_center = centers[3]
    
    local_diffs = []
    for x0 in centers:
        psi_f = calc_psi_at(R_chain, x0, ALPHA_FULL, COEFF_FULL)
        psi_s = calc_psi_at(R_chain, x0, ALPHA_SMOOTH, COEFF_SMOOTH)
        local_diffs.append((psi_f**2 - psi_s**2) * 5.0)
        
    mask_q = np.abs(R_chain - x_center) < 1.4
    q_elec = np.trapz(local_diffs[3][mask_q], R_chain[mask_q])
    
    return R_chain, centers, x_center, local_diffs, q_elec

def plot_he_chain(output_path='data0422/paw_dens/he_chain_density.png'):
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    psi_tot_smooth = np.zeros_like(R_chain)
    for x0 in centers:
        psi_s = calc_psi_at(R_chain, x0, ALPHA_SMOOTH, COEFF_SMOOTH)
        psi_tot_smooth += psi_s
    n_smooth_tot = (psi_tot_smooth**2) * 5.0
    
    fig, ax = plt.subplots(figsize=(20, 6))
    for i, x0 in enumerate(centers):
        is_focus = (i in [2, 3, 4])
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8 if is_focus else 0.15)
        mask_local = np.abs(R_chain - x0) < 1.4
        ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, alpha=0.8 if is_focus else 0.2, zorder=3)
    
    focus_limit_left = centers[2] - 1.2
    focus_limit_right = centers[4] + 1.2
    mask_left, mask_focus, mask_right = R_chain < focus_limit_left, (R_chain >= focus_limit_left) & (R_chain <= focus_limit_right), R_chain > focus_limit_right
    
    ax.plot(R_chain[mask_left], n_smooth_tot[mask_left], color='red', linewidth=4, linestyle='--', alpha=0.15, zorder=4)
    ax.plot(R_chain[mask_focus], n_smooth_tot[mask_focus], color='red', linewidth=4, linestyle='--', alpha=0.9, zorder=5)
    ax.plot(R_chain[mask_right], n_smooth_tot[mask_right], color='red', linewidth=4, linestyle='--', alpha=0.15, zorder=4)
    
    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    ax.set_ylim(-1.5, max(np.max(n_smooth_tot), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()

def plot_he_chain_with_potential(output_path='data0422/paw_dens/he_chain_potential.png'):
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    r_dist = np.abs(R_chain - x_center)
    v_elec = np.zeros_like(R_chain)
    mask_nz = r_dist > 1e-5
    alpha_elec, V_scale = 50.0, 1.2 * q_elec
    v_elec[mask_nz] = (V_scale / r_dist[mask_nz]) * erf(np.sqrt(alpha_elec) * r_dist[mask_nz])
    v_elec[~mask_nz] = V_scale * 2 * np.sqrt(alpha_elec / np.pi)
    
    fig, ax = plt.subplots(figsize=(20, 10))
    for i, x0 in enumerate(centers):
        is_focus = (i in [2, 3, 4])
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8 if is_focus else 0.15)
        if is_focus:
            mask_local = np.abs(R_chain - x0) < 1.4
            ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, zorder=3)
    
    ax.plot(R_chain, v_elec, color='black', linewidth=3, zorder=6, alpha=0.8)
    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    ax.set_ylim(-1.5, max(np.max(v_elec), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()

def plot_he_chain_compensate(output_path='data0422/paw_dens/he_chain_compensate.png'):
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    alpha_comp, V_scale = 5.0, 1.2 * q_elec
    r_dist = np.abs(R_chain - x_center)
    v_elec, v_comp = np.zeros_like(R_chain), np.zeros_like(R_chain)
    mask_nz = r_dist > 1e-5
    v_elec[mask_nz] = (V_scale / r_dist[mask_nz]) * erf(np.sqrt(50.0) * r_dist[mask_nz])
    v_elec[~mask_nz] = V_scale * 2 * np.sqrt(50.0 / np.pi)
    v_comp[mask_nz] = (-V_scale / r_dist[mask_nz]) * erf(np.sqrt(alpha_comp) * r_dist[mask_nz])
    v_comp[~mask_nz] = -V_scale * 2 * np.sqrt(alpha_comp / np.pi)
    
    n_comp = (-q_elec / np.sqrt(np.pi / alpha_comp)) * np.exp(-alpha_comp * r_dist**2)

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, x0 in enumerate(centers):
        is_focus = (i in [2, 3, 4])
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8 if is_focus else 0.15)
        if is_focus:
            mask_local = np.abs(R_chain - x0) < 1.4
            ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, zorder=3)
    
    ax.plot(R_chain, v_elec, color='black', linewidth=3, zorder=6, alpha=0.8)
    ax.plot(R_chain, v_comp, color='black', linewidth=3, zorder=6, alpha=0.8)
    
    mask_c = np.abs(R_chain - x_center) < 1.4
    ax.plot(R_chain[mask_c], n_comp[mask_c], color='#9c27b0', linewidth=4, zorder=5)

    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    label_y_low = np.min(v_comp) - 0.5
    ax.set_ylim(label_y_low - 1.0, max(np.max(v_elec), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()

def plot_he_chain_net_potential(output_path='data0422/paw_dens/he_chain_net.png'):
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    V_scale = 1.2 * q_elec
    r_dist = np.abs(R_chain - x_center)
    v_elec, v_comp = np.zeros_like(R_chain), np.zeros_like(R_chain)
    mask_nz = r_dist > 1e-5
    v_elec[mask_nz] = (V_scale / r_dist[mask_nz]) * erf(np.sqrt(50.0) * r_dist[mask_nz])
    v_elec[~mask_nz] = V_scale * 2 * np.sqrt(50.0 / np.pi)
    v_comp[mask_nz] = (-V_scale / r_dist[mask_nz]) * erf(np.sqrt(5.0) * r_dist[mask_nz])
    v_comp[~mask_nz] = -V_scale * 2 * np.sqrt(5.0 / np.pi)
    v_net = v_elec + v_comp
    
    n_comp = (-q_elec / np.sqrt(np.pi / 5.0)) * np.exp(-5.0 * r_dist**2)

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, x0 in enumerate(centers):
        is_focus = (i in [2, 3, 4])
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8 if is_focus else 0.15)
        if is_focus:
            mask_local = np.abs(R_chain - x0) < 1.4
            ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, zorder=3)
    
    ax.plot(R_chain, v_net, color='black', linewidth=4, zorder=7)
    mask_c = np.abs(R_chain - x_center) < 1.4
    ax.plot(R_chain[mask_c], n_comp[mask_c], color='#9c27b0', linewidth=4, zorder=5)
    
    label_y_low = np.min(v_comp) - 0.5
    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    ax.set_ylim(label_y_low - 1.0, max(np.max(v_net), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()

def plot_he_chain_net_multi(output_path='data0422/paw_dens/he_chain_net_multi.png'):
    """Generates a plot with compensating charges and net potentials for atoms a1, a2, and a3."""
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    alpha_comp = 5.0
    alpha_elec = 50.0
    V_scale = 1.2 * q_elec
    from scipy.special import erf

    v_net_total = np.zeros_like(R_chain)
    n_comp_total = np.zeros_like(R_chain)
    
    # Calculate for atoms at centers[2], centers[3], centers[4]
    focus_indices = [2, 3, 4]
    for idx in focus_indices:
        x0 = centers[idx]
        r_dist = np.abs(R_chain - x0)
        
        # Local Potentials
        v_e = np.zeros_like(R_chain)
        v_c = np.zeros_like(R_chain)
        mask_nz = r_dist > 1e-5
        
        v_e[mask_nz] = (V_scale / r_dist[mask_nz]) * erf(np.sqrt(alpha_elec) * r_dist[mask_nz])
        v_e[~mask_nz] = V_scale * 2 * np.sqrt(alpha_elec / np.pi)
        
        v_c[mask_nz] = (-V_scale / r_dist[mask_nz]) * erf(np.sqrt(alpha_comp) * r_dist[mask_nz])
        v_c[~mask_nz] = -V_scale * 2 * np.sqrt(alpha_comp / np.pi)
        
        v_net_total += (v_e + v_c)
        
        # Local compensating charge
        amp_comp = -q_elec / np.sqrt(np.pi / alpha_comp)
        n_comp_total += amp_comp * np.exp(-alpha_comp * (R_chain - x0)**2)

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, x0 in enumerate(centers):
        is_focus = (i in focus_indices)
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8 if is_focus else 0.15)
        mask_local = np.abs(R_chain - x0) < 1.4
        ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, alpha=0.8 if is_focus else 0.2, zorder=3)
    
    # Plot Total Net Potential (Black solid)
    ax.plot(R_chain, v_net_total, color='black', linewidth=4, zorder=7)
    
    # Plot Total Compensating Charges (Purple, upside down, truncated locally for each)
    for idx in focus_indices:
        x0 = centers[idx]
        r_dist = np.abs(R_chain - x0)
        mask_c = r_dist < 1.4
        # Extract the local part of the total n_comp for this atom
        # (Since they overlap slightly, plotting individual Gaussian peaks is clearer)
        amp_comp = -q_elec / np.sqrt(np.pi / alpha_comp)
        n_local = amp_comp * np.exp(-alpha_comp * (R_chain - x0)**2)
        ax.plot(R_chain[mask_c], n_local[mask_c], color='#9c27b0', linewidth=4, zorder=5)

    label_y_low = -1.5 # Consistent with other plots
    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    # Ensure ylim covers the net potential peaks and the charges
    ax.set_ylim(-2.5, max(np.max(v_net_total), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"Multi-atom net potential plot saved to {output_path}")

def plot_he_chain_net_multi_tail(output_path='data0422/paw_dens/he_chain_net_multi_tail.png'):
    """Generates a plot with compensating charges and net potentials with negative tails outside spheres."""
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    alpha_comp = 5.0
    alpha_elec = 50.0
    V_scale = 1.2 * q_elec
    from scipy.special import erf

    v_net_total = np.zeros_like(R_chain)
    
    focus_indices = [2, 3, 4]
    for idx in focus_indices:
        x0 = centers[idx]
        r_dist = np.abs(R_chain - x0)
        
        v_e = np.zeros_like(R_chain)
        v_c = np.zeros_like(R_chain)
        mask_nz = r_dist > 1e-5
        
        v_e[mask_nz] = (V_scale / r_dist[mask_nz]) * erf(np.sqrt(alpha_elec) * r_dist[mask_nz])
        v_e[~mask_nz] = V_scale * 2 * np.sqrt(alpha_elec / np.pi)
        
        # Revert to matched scaling (1.2) for the high-quality local peaks
        v_c[mask_nz] = (-V_scale / r_dist[mask_nz]) * erf(np.sqrt(alpha_comp) * r_dist[mask_nz])
        v_c[~mask_nz] = -V_scale * 2 * np.sqrt(alpha_comp / np.pi)
        
        v_net_total += (v_e + v_c)

    # Apply a stretching transformation:
    # v_new = (v / v_max) * (v_max + shift) - shift
    # This keeps the peak at v_max and sets the tails at -shift
    shift = 0.4
    v_max_orig = np.max(v_net_total)
    v_net_total = (v_net_total / v_max_orig) * (v_max_orig + shift) - shift

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, x0 in enumerate(centers):
        is_focus = (i in focus_indices)
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8 if is_focus else 0.15)
        mask_local = np.abs(R_chain - x0) < 1.4
        ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, alpha=0.8 if is_focus else 0.2, zorder=3)
    
    # Plot Total Net Potential with negative tails (Black solid)
    ax.plot(R_chain, v_net_total, color='black', linewidth=4, zorder=7)
    
    for idx in focus_indices:
        x0 = centers[idx]
        mask_c = np.abs(R_chain - x0) < 1.4
        amp_comp = -q_elec / np.sqrt(np.pi / alpha_comp)
        n_local = amp_comp * np.exp(-alpha_comp * (R_chain - x0)**2)
        ax.plot(R_chain[mask_c], n_local[mask_c], color='#9c27b0', linewidth=4, zorder=5)

    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    ax.set_ylim(-3.0, max(np.max(v_net_total), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"Multi-atom net potential with tails plot saved to {output_path}")

def plot_he_chain_net_tail(output_path='data0422/paw_dens/he_chain_net_tail.png'):
    """Generates a single-atom net potential plot with stretched negative tails."""
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    V_scale = 1.2 * q_elec
    r_dist = np.abs(R_chain - x_center)
    v_elec, v_comp = np.zeros_like(R_chain), np.zeros_like(R_chain)
    mask_nz = r_dist > 1e-5
    v_elec[mask_nz] = (V_scale / r_dist[mask_nz]) * erf(np.sqrt(50.0) * r_dist[mask_nz])
    v_elec[~mask_nz] = V_scale * 2 * np.sqrt(50.0 / np.pi)
    v_comp[mask_nz] = (-V_scale / r_dist[mask_nz]) * erf(np.sqrt(5.0) * r_dist[mask_nz])
    v_comp[~mask_nz] = -V_scale * 2 * np.sqrt(5.0 / np.pi)
    v_net = v_elec + v_comp
    
    # Apply stretching transformation
    shift = 0.4
    v_max_orig = np.max(v_net)
    v_net = (v_net / v_max_orig) * (v_max_orig + shift) - shift
    
    n_comp = (-q_elec / np.sqrt(np.pi / 5.0)) * np.exp(-5.0 * r_dist**2)

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, x0 in enumerate(centers):
        # All atoms fully opaque
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8)
        # Green electronic correction on ALL atoms
        mask_local = np.abs(R_chain - x0) < 1.4
        ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, zorder=3)
    
    ax.plot(R_chain, v_net, color='black', linewidth=4, zorder=7)
    mask_c = np.abs(R_chain - x_center) < 1.4
    ax.plot(R_chain[mask_c], n_comp[mask_c], color='#9c27b0', linewidth=4, zorder=5)
    
    label_y_low = -2.0
    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    ax.set_ylim(label_y_low - 1.0, max(np.max(v_net), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"Single-atom net potential with tails plot saved to {output_path}")

def plot_he_chain_net_all_opaque(output_path='data0422/paw_dens/he_chain_net_all.png'):
    """Generates a plot with all 7 atoms opaque and green charges on all of them."""
    R_chain, centers, x_center, local_diffs, q_elec = get_shared_data()
    
    V_scale = 1.2 * q_elec
    r_dist = np.abs(R_chain - x_center)
    v_elec, v_comp = np.zeros_like(R_chain), np.zeros_like(R_chain)
    mask_nz = r_dist > 1e-5
    v_elec[mask_nz] = (V_scale / r_dist[mask_nz]) * erf(np.sqrt(50.0) * r_dist[mask_nz])
    v_elec[~mask_nz] = V_scale * 2 * np.sqrt(50.0 / np.pi)
    v_comp[mask_nz] = (-V_scale / r_dist[mask_nz]) * erf(np.sqrt(5.0) * r_dist[mask_nz])
    v_comp[~mask_nz] = -V_scale * 2 * np.sqrt(5.0 / np.pi)
    v_net = v_elec + v_comp
    
    n_comp = (-q_elec / np.sqrt(np.pi / 5.0)) * np.exp(-5.0 * r_dist**2)

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, x0 in enumerate(centers):
        # All atoms fully opaque
        draw_he_atom(ax, x0, 0, ATOM_RADIUS, alpha=0.8)
        # Green electronic correction on ALL atoms
        mask_local = np.abs(R_chain - x0) < 1.4
        ax.plot(R_chain[mask_local], local_diffs[i][mask_local], color='#2e7d32', linewidth=3, zorder=3)
    
    # Net potential and compensating charge on center atom
    ax.plot(R_chain, v_net, color='black', linewidth=4, zorder=7)
    mask_c = np.abs(R_chain - x_center) < 1.4
    ax.plot(R_chain[mask_c], n_comp[mask_c], color='#9c27b0', linewidth=4, zorder=5)
    
    label_y_low = -2.0
    ax.set_xlim(centers[0] - 4, centers[-1] + 4)
    ax.set_ylim(label_y_low - 1.0, max(np.max(v_net), np.max(local_diffs[0])) * 1.7)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, transparent=True)
    plt.close()
    print(f"All-opaque net potential plot saved to {output_path}")

if __name__ == "__main__":
    # plot_he_chain()
    # plot_he_chain_with_potential()
    # plot_he_chain_compensate()
    # plot_he_chain_net_potential()
    # plot_he_chain_net_multi()
    plot_he_chain_net_multi_tail()
    plot_he_chain_net_tail()
    plot_he_chain_net_all_opaque()
