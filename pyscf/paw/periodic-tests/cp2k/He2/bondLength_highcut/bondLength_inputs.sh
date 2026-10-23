#!/bin/bash
 
cutoffs="0.7 0.75714286 0.81428571 0.87142857 0.92857143 0.98571429 1.04285714\
 1.1 1.15714286 1.21428571 1.27142857 1.32857143\
 1.38571429 1.44285714 1.5"
 
# basis_file=BASIS_SET
# potential_file=GTH_POTENTIALS
template_file=template.inp
input_file=He2.inp
 
# rel_cutoff=60
 
for ii in $cutoffs ; do
    work_dir=bondLength_${ii}A
    if [ ! -d $work_dir ] ; then
        mkdir $work_dir
    else
        rm -r $work_dir/*
    fi
    # sed -e "s/LT_rel_cutoff/${rel_cutoff}/g" \
    sed -e "s/LT_bondLength/${ii}/g" \
        $template_file > $work_dir/$input_file
    cp ../energy.sh $work_dir
    cp ../CCPVxZ $work_dir
done
