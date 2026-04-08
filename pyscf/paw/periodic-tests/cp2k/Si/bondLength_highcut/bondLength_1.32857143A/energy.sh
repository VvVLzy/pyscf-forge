#!/bin/bash
#SBATCH --job-name=He2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=3:00:00
#SBATCH --account=befl-delta-cpu
#SBATCH --partition=cpu


no_proc_per_calc=$SLURM_NTASKS_PER_NODE
if [ "$no_proc_per_calc" -gt 1 ]; then
    cp2k_bin=cp2k.psmp
    echo "Using parallel run with $no_proc_per_calc processes"
else
    cp2k_bin=cp2k.ssmp
    echo "Using single run"
fi

name=$1
input_file=$name.inp
output_file=$name.out

if [ -f $output_file ] && [ "$2" = "T" ]; then
    rm $output_file
fi

source /u/zluo4/packages/cp2k/tools/toolchain/install/setup
srun --ntasks=$no_proc_per_calc $cp2k_bin -o $output_file $input_file &
wait
