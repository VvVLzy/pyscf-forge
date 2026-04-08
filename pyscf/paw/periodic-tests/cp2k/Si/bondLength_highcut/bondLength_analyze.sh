#!/bin/bash
 
cutoffs="0.7 0.75714286 0.81428571 0.87142857 0.92857143 0.98571429 1.04285714\
 1.1 1.15714286 1.21428571 1.27142857 1.32857143\
 1.38571429 1.44285714 1.5"
 
output_file=He2.out
cutoff=1600.00
prec=1e-8
plot_file="He2bondLength_${cutoff}_a_10_r_1.5_dz_${prec}_cp2k.csv"
 
echo "# Bond Length vs total energy" > $plot_file
echo "# Date: $(date)" >> $plot_file
echo "# PWD: $PWD" >> $plot_file
echo "# CUTOFF = $cutoff" >> $plot_file
echo -n "# BondLength (A),Total Energy (Ha)" >> $plot_file
grid_header=true
for ii in $cutoffs ; do
    work_dir=bondLength_${ii}A
    total_energy=$(grep -e '^[ \t]*Total energy' $work_dir/$output_file | awk '{print $3}')
    ngrids=$(grep -e '^[ \t]*QS| Number of grid levels:' $work_dir/$output_file | \
             awk '{print $6}')
    if $grid_header ; then
        for ((igrid=1; igrid <= ngrids; igrid++)) ; do
            printf ",NG on grid %d" $igrid >> $plot_file
        done
        printf "\n" >> $plot_file
        grid_header=false
    fi
    printf "%10.8f,%15.10f" $ii $total_energy >> $plot_file
    for ((igrid=1; igrid <= ngrids; igrid++)) ; do
        grid=$(grep -e '^[ \t]*count for grid' $work_dir/$output_file | \
               awk -v igrid=$igrid '(NR == igrid){print $5}')
        printf ",%6d" $grid >> $plot_file
    done
    printf "\n" >> $plot_file
done
