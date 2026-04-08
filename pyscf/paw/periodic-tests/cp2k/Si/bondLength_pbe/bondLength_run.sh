#!/bin/bash


cutoffs="0.7 0.75714286 0.81428571 0.87142857 0.92857143 0.98571429 1.04285714\
 1.1 1.15714286 1.21428571 1.27142857 1.32857143\
 1.38571429 1.44285714 1.5"
name=He2


for ii in $cutoffs ; do
    work_dir=bondLength_${ii}A
    echo $work_dir
    cd $work_dir
	export OMP_NUM_THREADS=1; /home/lebox/sw/cp2k/install/bin/launch cp2k -o $name.out $name.inp
	wait
    cd ..
    echo "Done!"
done
