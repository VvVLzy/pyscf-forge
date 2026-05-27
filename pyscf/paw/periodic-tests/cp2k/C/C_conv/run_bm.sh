#!/bin/bash

# Go to the directory where the script is located
cd "$(dirname "$0")"

# Parameters for filename (matching C.inp)
CUTOFF="400"
ALPHA0="10"
RADIUS="1.5"
ZETA="qz"
XC="lda" 

CSV_FILE="data/CCP2KBulkModulus_${CUTOFF}_a_${ALPHA0}_r_${RADIUS}_${ZETA}_${XC}.csv"

mkdir -p data
echo "volume,etot_cp2k,conv_cp2k" > "$CSV_FILE"

SCALES=("0.9" "0.95" "1.0" "1.05" "1.1")

# Constants
ANG2BOHR=1.8897261246
ANG3_TO_BOHR3=$(echo "$ANG2BOHR * $ANG2BOHR * $ANG2BOHR" | bc -l)

for x in "${SCALES[@]}"; do
    DIR="scale_$x"
    if [ -d "$DIR" ]; then
        echo "Processing $DIR..."
        cd "$DIR"
        
        # Run CP2K
        echo "  Running CP2K..."
        /home/lebox/sw/cp2k/install/bin/launch cp2k -o C.out C.inp
        
        # Extract volume (in Angstrom^3) and convert to Bohr^3
        VOL_ANG=$(grep "CELL| Volume \[angstrom^3\]:" C.out | tail -n 1 | awk '{print $NF}')
        if [ -z "$VOL_ANG" ]; then
            VOL_BOHR="NaN"
        else
            VOL_BOHR=$(echo "$VOL_ANG * $ANG3_TO_BOHR3" | bc -l)
        fi
        
        # Extract energy (in Hartree)
        ENERGY=$(grep "ENERGY| Total FORCE_EVAL ( QS ) energy" C.out | tail -n 1 | awk '{print $NF}')
        if [ -z "$ENERGY" ]; then
            ENERGY="NaN"
        fi
        
        # Check convergence
        if grep -q "SCF run converged" C.out; then
            CONV="True"
        else
            CONV="False"
        fi
        
        cd ..
        echo "$VOL_BOHR,$ENERGY,$CONV" >> "$CSV_FILE"
        echo "  Done. Volume: $VOL_BOHR Bohr^3, Energy: $ENERGY Hartree"
    else
        echo "Directory $DIR not found!"
    fi
done

echo ""
echo "All runs completed. Results saved to $CSV_FILE"
