#!/bin/bash
# Run the He2 bond scan with CP2K and PySCF (gdfNuc=False), then compare
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qchem
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8

sets="12.5,1.7"
for s in $sets ; do
    a=${s%,*} ; rb=${s#*,}
    echo "== CP2K alpha0=$a Rb=$rb"
    (cd cp2k && ./inputs.sh $a $rb && ./run.sh $a $rb > a_${a}_r_${rb}/run.log 2>&1 && ./analyze.sh $a $rb)
    for g in False ; do
        echo "== PySCF alpha0=$a Rb=$rb gdfNuc=$g"
        (cd pyscf && python He2bondLength.py scan --alpha0 $a --Rb $rb --gdfNuc $g > scan_a_${a}_r_${rb}_gdfNuc_${g}.out 2>&1)
    done
    python compare.py $a $rb
done
