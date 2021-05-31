#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --job-name=phase_3

file_path=`sed -n '1p' $3/$4/logs.txt`
protein=`sed -n '2p' $3/$4/logs.txt`
grid_file=`sed -n '3p' $3/$4/logs.txt`
glide_input_file=`sed -n '8p' $3/$4/logs.txt`

morgan_directory=`sed -n '4p' $3/$4/logs.txt`
smile_directory=`sed -n '5p' $3/$4/logs.txt`

python jobid_writer.py -pt $protein -fp $file_path -n_it $1 -jid phase_3 -jn phase_3.txt

njobs=$(($2/3))
python scripts_1/input_glide.py -pt $protein -fp $file_path -gf $grid_file -n_it $1 -g_in $glide_input_file
cd $file_path/$protein/iteration_$1/docked
for f in *.in;do $SCHRODINGER/glide -HOST slurm-compute -NJOBS $njobs -OVERWRITE -JOBNAME phase_3_${f%.*} $f;done
