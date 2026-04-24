import matplotlib.pyplot as plt
import pandas as pd
import os

def render_table(data, title, output_path, col_widths):
    fig, ax = plt.subplots(figsize=(10, 4.5)) # Slightly taller for footnote
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=data.values, 
                     colLabels=data.columns, 
                     cellLoc='center', 
                     loc='center',
                     colWidths=col_widths)
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2.5)
    
    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#404040') # Dark gray header
        elif row > 0:
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2') # Zebra striping
            
            # Bold the first column (Basis Set)
            if col == 0:
                cell.set_text_props(weight='bold')
            
            # Bold the GPAW reference row
            if row == len(data):
                cell.set_text_props(weight='bold')

    # Title closer to the table (reduced pad)
    plt.title(title, fontsize=18, fontweight='bold', pad=10)
    
    # Add footnote
    plt.figtext(0.5, 0.05, "* modified basis set", ha="center", fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Table saved to {output_path}")

# Data 1: Lattice Constant (with asterisk)
lattice_data = pd.DataFrame([
    ["cc-pVDZ", "3.667", "3.667", "3.661"],
    ["cc-pVTZ", "3.648", "3.651", "3.623"],
    ["cc-pVQZ*", "3.646", "3.647", "—"],
    ["PW (GPAW)", "3.650", "—", "—"]
], columns=["Basis Set", "PAW", "CP2K", "GDF"])

# Data 2: Bulk Modulus (with asterisk)
bm_data = pd.DataFrame([
    ["cc-pVDZ", "368", "366", "371"],
    ["cc-pVTZ", "376", "372", "402"],
    ["cc-pVQZ*", "385", "385", "—"],
    ["PW (GPAW)", "375", "—", "—"]
], columns=["Basis Set", "PAW", "CP2K", "GDF"])

output_dir = 'data0422/img'
os.makedirs(output_dir, exist_ok=True)

render_table(lattice_data, "Equilibrium Lattice Constant $a_0$ (Å)", 
             os.path.join(output_dir, "lattice_table.png"), [0.25, 0.2, 0.2, 0.2])

render_table(bm_data, "Bulk Modulus $B$ (GPa)", 
             os.path.join(output_dir, "bm_table.png"), [0.25, 0.2, 0.2, 0.2])
