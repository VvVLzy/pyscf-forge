import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

from matplotlib.colors import LogNorm

# Orbital counts per atom for N (Nitrogen) with different basis sets
BASIS_STRUCTURES = {
    28:  {'name': 'dz', 'groups': [3, 6, 5], 'labels': ['s', 'p', 'd']},
    60:  {'name': 'tz', 'groups': [4, 9, 10, 7], 'labels': ['s', 'p', 'd', 'f']},
    110: {'name': 'qz', 'groups': [5, 12, 15, 14, 9], 'labels': ['s', 'p', 'd', 'f', 'g']},
}

def plot_j_error(npz_path):
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    # Filter for j_diff keys
    keys = sorted([k for k in data.keys() if k.startswith('j_diff_')], 
                  key=lambda x: float(x.split('_')[-1]))
    
    if not keys:
        print("No j_diff arrays found in the file.")
        return

    # Find global maximum to unify the color scale
    global_max = 0
    for key in keys:
        global_max = max(global_max, np.max(np.abs(data[key])))
    
    print(f"Global maximum absolute error: {global_max:.2e}")
    # Fixed scale for all plots regardless of input
    global_vmin = 1e-10
    global_vmax = 1e-1
    norm = LogNorm(vmin=global_vmin, vmax=global_vmax)

    num_plots = len(keys)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    axes = axes.flatten()

    for i, key in enumerate(keys):
        j_diff = data[key]
        alpha0 = key.split('_')[-1]
        
        # Find the local largest entry for reporting
        abs_j_diff = np.abs(j_diff)
        max_val = np.max(abs_j_diff)
        max_idx = np.unravel_index(np.argmax(abs_j_diff), j_diff.shape)
        
        print(f"Alpha0 {alpha0}: Max absolute error = {max_val:.2e} at index {max_idx}")
        
        # Plot absolute value of the error in log scale using unified norm
        im = axes[i].imshow(abs_j_diff, cmap='viridis', interpolation='nearest', 
                            norm=norm)
        axes[i].set_title(f"|J Error| (alpha0={alpha0})\nMax: {max_val:.2e} at {max_idx}", pad=30)
        plt.colorbar(im, ax=axes[i])

        # Add orbital boundaries if the structure is known
        N = j_diff.shape[0]
        if N in BASIS_STRUCTURES:
            struct = BASIS_STRUCTURES[N]
            # Mirror groups for two atoms
            full_groups = struct['groups'] + struct['groups']
            full_labels = struct['labels'] + struct['labels']
            
            cum_groups = np.cumsum(full_groups)
            boundaries = cum_groups[:-1]
            ticks = cum_groups - np.array(full_groups) / 2 - 0.5
            
            for b in boundaries:
                axes[i].axvline(b - 0.5, color='white', linestyle='--', alpha=0.5, lw=0.8)
                axes[i].axhline(b - 0.5, color='white', linestyle='--', alpha=0.5, lw=0.8)
            
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(full_labels)
            axes[i].set_yticks(ticks)
            axes[i].set_yticklabels(full_labels)
            
            # Label atoms using axes coordinates (0 to 1) to avoid overlap
            axes[i].text(0.25, 1.02, 'Atom 1', transform=axes[i].transAxes, 
                         ha='center', va='bottom', color='black', weight='bold')
            axes[i].text(0.75, 1.02, 'Atom 2', transform=axes[i].transAxes, 
                         ha='center', va='bottom', color='black', weight='bold')
            axes[i].text(-0.15, 0.75, 'Atom 1', transform=axes[i].transAxes, 
                         ha='right', va='center', rotation=90, color='black', weight='bold')
            axes[i].text(-0.15, 0.25, 'Atom 2', transform=axes[i].transAxes, 
                         ha='right', va='center', rotation=90, color='black', weight='bold')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    output_plot = npz_path.replace('.npz', '_heatmap.png')
    print(f"Saving heatmap plot to {output_plot}...")
    plt.savefig(output_plot)

def main():
    parser = argparse.ArgumentParser(description="Plot heatmaps of J matrix errors from .npz files.")
    parser.add_argument('npz_file', help="Path to the .npz file containing J difference matrices.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_file):
        print(f"Error: File {args.npz_file} not found.")
        return
        
    plot_j_error(args.npz_file)

if __name__ == '__main__':
    main()
