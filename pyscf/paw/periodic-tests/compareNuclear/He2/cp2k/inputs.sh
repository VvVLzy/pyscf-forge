#!/bin/bash
# Generate one CP2K input per He2 bond length from template.inp
# usage: ./inputs.sh <alpha0> <Rb>      e.g. ./inputs.sh 12.5 1.7  ->  a_12.5_r_1.7/

alpha0=$1
rb=$2
set_dir=a_${alpha0}_r_${rb}

bondlengths="0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"

template_file=template.inp
input_file=He2.inp

for ii in $bondlengths ; do
    work_dir=$set_dir/bondLength_${ii}A
    mkdir -p $work_dir
    sed -e "s/LT_bondLength/${ii}/g" \
        -e "s/LT_alpha0/${alpha0}/g" \
        -e "s/LT_rb/${rb}/g" \
        $template_file > $work_dir/$input_file
    cp CCPVxZ POTENTIAL $work_dir
done
