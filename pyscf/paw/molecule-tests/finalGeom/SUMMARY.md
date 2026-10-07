# Optimized geometries -- MP2(FC)/cc-pvtz

Cartesian coordinates, **units = Angstrom**. Relaxed from `../initGeom/` with analytic MP2 gradients (geomeTRIC, default convergence thresholds).

| molecule | formula | motif | atoms | AOs | E(MP2)/Eh | steps | \|g\|max | RMS shift/A | wall/s |
|---|---|---|---|---|---|---|---|---|---|
| `acetone.xyz` | C3H6O | Ketone (C=O) | 10 | 204 | -192.774850 | 6 | 2.1e-05 | 0.0233 | 169 |
| `acetylene.xyz` | C2H2 | Alkyne (C#C) | 4 | 88 | -77.159196 | 3 | 1.0e-05 | 0.0042 | 4 |
| `benzene.xyz` | C6H6 | Aromatic | 12 | 264 | -231.729480 | 3 | 4.6e-06 | 0.0021 | 281 |
| `butane.xyz` | C4H10 | Alkane chain (screening probe) | 14 | 260 | -158.073920 | 5 | 2.9e-05 | 0.0127 | 368 |
| `cyclobutadiene.xyz` | C4H4 | Antiaromatic | 8 | 176 | -154.328522 | 5 | 1.0e-04 | 0.0450 | 107 |
| `dimethyl_ether.xyz` | C2H6O | Ether (C-O-C) | 9 | 174 | -154.724567 | 4 | 6.5e-05 | 0.0106 | 86 |
| `dimethyl_peroxide.xyz` | C2H6O2 | Peroxide (O-O) | 10 | 204 | -229.753530 | 7 | 3.1e-05 | 0.0318 | 193 |
| `ethane.xyz` | C2H6 | Alkane (C-C) | 8 | 144 | -79.629908 | 4 | 8.4e-06 | 0.0032 | 36 |
| `ethylene.xyz` | C2H4 | Alkene (C=C) | 6 | 116 | -78.399306 | 5 | 1.0e-06 | 0.0021 | 18 |
| `fluoromethane.xyz` | CH3F | Alkyl fluoride (C-F) | 5 | 102 | -139.528578 | 4 | 2.0e-06 | 0.0032 | 10 |
| `formaldehyde.xyz` | CH2O | Aldehyde (-CHO) | 4 | 88 | -114.307198 | 6 | 1.6e-05 | 0.0069 | 8 |
| `formamide.xyz` | CH3NO | Amide (-CONH2) | 6 | 132 | -169.605449 | 7 | 4.0e-05 | 0.0090 | 45 |
| `formic_acid.xyz` | CH2O2 | Carboxylic acid (-COOH) | 5 | 118 | -189.471134 | 6 | 1.0e-04 | 0.0095 | 27 |
| `hydrogen_cyanide.xyz` | HCN | Nitrile (C#N) | 3 | 74 | -93.253589 | 4 | 4.2e-06 | 0.0042 | 3 |
| `methanimine.xyz` | CH3N | Imine (C=N) | 5 | 102 | -94.438795 | 6 | 1.1e-05 | 0.0106 | 14 |
| `methanol.xyz` | CH4O | Alcohol (-OH) | 6 | 116 | -115.517475 | 5 | 4.1e-05 | 0.0169 | 19 |
| `methyl_chloride.xyz` | CH3Cl | Alkyl chloride (C-Cl) | 5 | 106 | -499.520239 | 3 | 2.8e-04 | 0.0037 | 10 |
| `methyl_formate.xyz` | C2H4O2 | Ester (-COO-) | 8 | 176 | -228.676395 | 7 | 7.3e-05 | 0.0132 | 109 |
| `methylamine.xyz` | CH5N | Amine (-NH2) | 7 | 130 | -95.657699 | 5 | 1.3e-04 | 0.0397 | 31 |
| `neon.xyz` | Ne | Noble gas (single atom) | 1 | 30 | -128.796184 | n/a | 0 (by symmetry) | n/a | 0 |
| `nitromethane.xyz` | CH3NO2 | Nitro (-NO2) | 7 | 162 | -244.621097 | 6 | 2.4e-05 | 0.0138 | 99 |
| `oxirane.xyz` | C2H4O | Epoxide | 7 | 146 | -153.501120 | 5 | 8.7e-05 | 0.0381 | 51 |
| `vinyl_fluoride.xyz` | C2H3F | Vinyl halide (=C-F) | 6 | 132 | -177.529174 | 7 | 4.9e-05 | 0.0344 | 46 |

A single atom has no internal degrees of freedom, so it is a single point rather than an optimization; its force is zero by translational invariance.
