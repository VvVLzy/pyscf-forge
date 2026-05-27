#!/bin/bash

# Base parameters
L_BASE="3.56074511"
SCALES=("0.9" "0.95" "1.0" "1.05" "1.1")

# Go to the directory where the script is located
cd "$(dirname "$0")"

for x in "${SCALES[@]}"; do
    DIR="scale_$x"
    rm -rf "$DIR"
    mkdir -p "$DIR"
    
    # Calculate scaled L
    L_SCALED=$(echo "scale=10; $L_BASE * $x" | bc -l)
    
    # Copy static dependencies
    cp POTENTIAL CCPVxZ "$DIR/"
    
    # Create scaled C.xyz
    head -n 2 C.xyz > "$DIR/C.xyz"
    tail -n +3 C.xyz | awk -v x="$x" '{printf "%s %12.8f %12.8f %12.8f\n", $1, $2*x, $3*x, $4*x}' >> "$DIR/C.xyz"
    
    # Create scaled C.inp
    # We replace the ABC line with scaled values
    sed "s/ABC .*/ABC $L_SCALED $L_SCALED $L_SCALED/" C.inp > "$DIR/C.inp"
    
    echo "Created directory $DIR with scaled L = $L_SCALED"
done
