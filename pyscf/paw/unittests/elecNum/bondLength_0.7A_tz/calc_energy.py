import numpy as np

# Energy Verification Script for He2.py (GAPW)
# This script extracts matrices from He2.out and calculates the Total Energy
# to verify it matches the reported CP2K Total Energy.

# --- 1. Load Matrices from CP2K Output (He2.out) ---

# Overlap Matrix (S)
S = np.array([
    [1.0000000000000, 0.2960762097805, 0.0119738258937, 0.2538185316064],
    [0.2960762097805, 2.4569420322423, 0.2538185316064, 2.4309333599904],
    [0.0119738258937, 0.2538185316064, 1.0000000000000, 0.2960762097805],
    [0.2538185316064, 2.4309333599904, 0.2960762097805, 2.4569420322423]
])

# Core Hamiltonian Matrix (Hcore)
Hcore = np.array([
    [ 3.4704680931774, -0.0829622353472, -0.0726799231157, -0.0924968054827],
    [-0.0829622353472,  0.0098260697704, -0.0924968054827, -0.0060469604974],
    [-0.0726799231157, -0.0924968054827,  3.4704680931774, -0.0829622353472],
    [-0.0924968054827, -0.0060469604974, -0.0829622353472,  0.0098260697704]
])

# Kohn-Sham Matrix (F)
F = np.array([
    [ 0.7504973867031, -0.5846200447774, -0.0901759397758, -0.5208937836981],
    [-0.5846200447774, -0.1688378486388, -0.5208937750218, -0.1826734477934],
    [-0.0901759397758, -0.5208937750218,  0.7504974192164, -0.5846200344691],
    [-0.5208937836981, -0.1826734477934, -0.5846200344691, -0.1688378425689]
])

# Molecular Orbital Coefficients (C) - Occupied only
C = np.array([
    [ 0.271821,  0.367941],
    [ 0.265953,  3.202035],
    [ 0.271821, -0.367941],
    [ 0.265953, -3.202035]
])

# --- 2. Density Matrix Calculation ---
P = 2 * np.dot(C, C.T)

# --- 3. Reported Components from He2.out (Absolute Ground Truth) ---
E_nuc_overlap = 0.00000000005017
E_nuc_self    = -11.28379167095513
E_hcore_rep   = 3.56334671528124
E_hartree_rep = 5.10254525820603
E_xc_rep      = 0.00000000000000
E_local_rep   = -0.05279651205868
E_total_rep   = -2.67069620947638

# --- 4. Matrix-Based Energy Calculation ---

# Formula for Total Energy in GAPW (No XC):
# E_tot = E_overlap + E_self + E_hcore + E_hartree + E_local

# We can calculate E_hcore directly from the matrix:
E_hcore_calc = np.trace(np.dot(P, Hcore))

# The Electronic Energy (Tr(PF) part) is:
# E_elec_part = Tr(P * F)
E_PF = np.trace(np.dot(P, F))

# In a pure Hartree system: E_elec = 0.5 * Tr(P * (F + Hcore))
# This represents E_hcore + 0.5 * Tr(P * J)
# In GAPW, 0.5 * Tr(P * (F - Hcore)) corresponds to the electronic part of Hartree energy.
E_hartree_elec_calc = 0.5 * np.trace(np.dot(P, F - Hcore))

# Verification of Total Energy:
# We sum the calculated core energy and the reported other terms:
E_total_calc = E_nuc_overlap + E_nuc_self + E_hcore_calc + E_hartree_rep + E_local_rep

print("--- GAPW Total Energy Verification ---")
print(f"Reported Total Energy:          {E_total_rep:18.14f}")
print(f"Calculated Total Energy:        {E_total_calc:18.14f}")
print(f"Difference:                     {E_total_calc - E_total_rep:18.14f}")

print("\n--- Matrix vs Reported Component Check ---")
print(f"Calculated Tr(PHcore):          {E_hcore_calc:18.14f}")
print(f"Reported Core Hamiltonian:      {E_hcore_rep:18.14f}")
print(f"Difference in Core:             {E_hcore_calc - E_hcore_rep:18.14f}")

print("\n--- Hartree Term Analysis ---")
print(f"Reported Hartree Energy:        {E_hartree_rep:18.14f}")
print(f"Calculated 0.5 * Tr(P(F-Hcore)): {E_hartree_elec_calc:18.14f}")
print(f"Definition Offset:              {E_hartree_rep - E_hartree_elec_calc:18.14f}")
print("(This offset is the self-interaction of compensation charges and core charges included in E_hartree)")

print("\n--- Final Verification Formula ---")
print("E_total = E_overlap + E_self + Tr(PHcore) + E_hartree_rep + E_local")
print(f"E_total = {E_nuc_overlap:.4f} + ({E_nuc_self:.4f}) + {E_hcore_calc:.4f} + {E_hartree_rep:.4f} + ({E_local_rep:.4f})")
print(f"E_total = {E_nuc_overlap + E_nuc_self + E_hcore_calc + E_hartree_rep + E_local_rep:.14f}")
