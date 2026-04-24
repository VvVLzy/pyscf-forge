import matplotlib.pyplot as plt
import pandas as pd
import os

def render_latex_table(data, title, output_path):
    # Set plot params to look like LaTeX (Serif fonts)
    plt.rcParams.update({
        "text.usetex": False, # Tex not assumed to be installed, use mathtext
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    })
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=data.values, 
                     colLabels=data.columns, 
                     cellLoc='center', 
                     loc='center',
                     edges='horizontal') # LaTeX tables rarely have vertical lines
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.1, 2.5)
    
    # Apply "booktabs" style lines manually
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0) # Default no lines
        
        # Row 0 is header
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_edgecolor('black')
            cell.visible_edges = 'TB' # Top and Bottom of header
            if row == 0: cell.set_linewidth(2) # Thicker top rule
        
        # Last row
        elif row == len(data):
            cell.set_edgecolor('black')
            cell.visible_edges = 'B' # Bottom rule
            cell.set_linewidth(2)
        
        # First column bold
        if col == 0:
            cell.set_text_props(weight='bold')

    plt.title(title, fontsize=18, family='serif', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()
    print(f"LaTeX-style table saved to {output_path}")

# Data definitions
lattice_data = pd.DataFrame([
    ["cc-pVDZ", "3.667", "3.667", "3.661"],
    ["cc-pVTZ", "3.648", "3.651", "3.623"],
    ["cc-pVQZ", "3.646", "3.647", "—"],
    ["PW (GPAW)", "3.650", "—", "—"]
], columns=["Basis Set", "PAW", "CP2K", "GDF"])

bm_data = pd.DataFrame([
    ["cc-pVDZ", "368", "366", "371"],
    ["cc-pVTZ", "376", "372", "402"],
    ["cc-pVQZ", "385", "385", "—"],
    ["PW (GPAW)", "375", "—", "—"]
], columns=["Basis Set", "PAW", "CP2K", "GDF"])

output_dir = 'data0422/img'
os.makedirs(output_dir, exist_ok=True)

render_latex_table(lattice_data, "Equilibrium Lattice Constant $a_0$ (Å)", 
                   os.path.join(output_dir, "lattice_table_latex.png"))

render_latex_table(bm_data, "Bulk Modulus $B$ (GPa)", 
                   os.path.join(output_dir, "bm_table_latex.png"))
