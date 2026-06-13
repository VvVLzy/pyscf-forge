import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

# Orbital counts per atom for N (Nitrogen) with different basis sets
BASIS_STRUCTURES = {
    28:  {'name': 'dz', 'groups': [3, 6, 5], 'labels': ['s', 'p', 'd']},
    60:  {'name': 'tz', 'groups': [4, 9, 10, 7], 'labels': ['s', 'p', 'd', 'f']},
    110: {'name': 'qz', 'groups': [5, 12, 15, 14, 9], 'labels': ['s', 'p', 'd', 'f', 'g']},
}

def plot_dm(npz_path):
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    
    if 'dm' not in data:
        print("Error: 'dm' array not found in the file.")
        return

    dm = data['dm']
    N = dm.shape[0]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Use symmetric scale for signed DM
    max_val = np.max(np.abs(dm))
    im = ax.imshow(dm, cmap='RdBu_r', interpolation='nearest', 
                   vmin=-max_val, vmax=max_val)
    
    ax.set_title(f"Density Matrix\nMax Absolute Value: {max_val:.2e}", pad=30)
    plt.colorbar(im, ax=ax)

    # Add orbital boundaries if the structure is known
    if N in BASIS_STRUCTURES:
        struct = BASIS_STRUCTURES[N]
        full_groups = struct['groups'] + struct['groups']
        full_labels = struct['labels'] + struct['labels']
        
        cum_groups = np.cumsum(full_groups)
        boundaries = cum_groups[:-1]
        ticks = cum_groups - np.array(full_groups) / 2 - 0.5
        
        for b in boundaries:
            ax.axvline(b - 0.5, color='gray', linestyle='--', alpha=0.5, lw=0.8)
            ax.axhline(b - 0.5, color='gray', linestyle='--', alpha=0.5, lw=0.8)
        
        ax.set_xticks(ticks)
        ax.set_xticklabels(full_labels)
        ax.set_yticks(ticks)
        ax.set_yticklabels(full_labels)
        
        # Label atoms using axes coordinates
        ax.text(0.25, 1.02, 'Atom 1', transform=ax.transAxes, 
                 ha='center', va='bottom', color='black', weight='bold')
        ax.text(0.75, 1.02, 'Atom 2', transform=ax.transAxes, 
                 ha='center', va='bottom', color='black', weight='bold')
        ax.text(-0.12, 0.75, 'Atom 1', transform=ax.transAxes, 
                 ha='right', va='center', rotation=90, color='black', weight='bold')
        ax.text(-0.12, 0.25, 'Atom 2', transform=ax.transAxes, 
                 ha='right', va='center', rotation=90, color='black', weight='bold')

    plt.tight_layout()
    output_plot = npz_path.replace('.npz', '_dm_heatmap.png')
    print(f"Saving DM heatmap to {output_plot}...")
    plt.savefig(output_plot)

def main():
    parser = argparse.ArgumentParser(description="Plot heatmap of the density matrix from .npz files.")
    parser.add_argument('npz_file', help="Path to the .npz file containing the 'dm' array.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npz_file):
        print(f"Error: File {args.npz_file} not found.")
        return
        
    plot_dm(args.npz_file)

if __name__ == '__main__':
    main()
