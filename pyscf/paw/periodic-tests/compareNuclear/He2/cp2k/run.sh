#!/bin/bash
# Run every bond length of one parameter set locally
# usage: ./run.sh <alpha0> <Rb>
# (cp2k -o appends to an existing output, so old outputs are removed first)

set_dir=a_${1}_r_${2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

bondlengths="0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"

cd $set_dir
for ii in $bondlengths ; do
    work_dir=bondLength_${ii}A
    echo $work_dir
    cd $work_dir
    rm -f He2.out
    /home/lebox/sw/cp2k/install/bin/launch cp2k -o He2.out He2.inp
    cd ..
done
