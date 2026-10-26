#!/bin/bash
# Collect the converged total energies of one parameter set into a csv
# usage: ./analyze.sh <alpha0> <Rb>

set_dir=a_${1}_r_${2}

bondlengths="0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"

plot_file=$set_dir/He2bondLength_cp2k.csv

echo "r,etot_cp2k,conv_cp2k" > $plot_file
for ii in $bondlengths ; do
    out=$set_dir/bondLength_${ii}A/He2.out
    total_energy=$(grep -e '^[ \t]*Total energy:' $out | tail -n 1 | awk '{print $3}')
    if grep -q "SCF run converged" $out ; then conv=True ; else conv=False ; fi
    printf "%s,%s,%s\n" $ii $total_energy $conv >> $plot_file
done
