#!/bin/bash
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=calculate_morgan_fing

source ~/.bashrc
conda activate $4

start=`date +%s`

python -u morgan_fp.py -sfp $1 -fn $2 -tp $3

end=`date +%s`
runtime=$((end-start))
echo $runtime
