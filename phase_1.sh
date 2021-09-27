#!/bin/bash
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60       #change this to match the number of processors you want to use
#SBATCH --mem=0
#SBATCH --job-name=phase_1

source ~/.bashrc
conda activate $6

start=`date +%s`

file_path=`sed -n '1p' $3/$4/logs.txt`
protein=`sed -n '2p' $3/$4/logs.txt`
n_mol=`sed -n '8p' $3/$4/logs.txt`

pr_it=$(($1-1)) 

t_cpu=$2

mol_to_dock=$5

if [ $1 == 1 ]
then 
	to_d=$((n_mol+n_mol+mol_to_dock))
else
	to_d=$mol_to_dock
fi

echo $to_d
echo $t_cpu

python jobid_writer.py -pt $protein -fp $file_path -n_it $1 -jid $SLURM_JOB_NAME -jn $SLURM_JOB_NAME.txt

morgan_directory=`sed -n '4p' $3/$4/logs.txt`
smile_directory=`sed -n '5p' $3/$4/logs.txt`
sdf_directory=`sed -n '6p' $3/$4/logs.txt`

if [ $1 == 1 ];then pred_directory=$morgan_directory;else pred_directory=$file_path/$protein/iteration_$pr_it/morgan_1024_predictions;fi

python scripts_1/molecular_file_count_updated.py -pt $protein -it $1 -cdd $pred_directory -t_pos $t_cpu -t_samp $to_d
python scripts_1/sampling.py -pt $protein -fp $file_path -it $1 -dd $pred_directory -t_pos $t_cpu -tr_sz $mol_to_dock -vl_sz $n_mol
python scripts_1/sanity_check.py -pt $protein -fp $file_path -it $1
python scripts_1/extracting_morgan.py -pt $protein -fp $file_path -it $1 -md $morgan_directory -t_pos $t_cpu
python scripts_1/extracting_smiles.py -pt $protein -fp $file_path -it $1 -smd $smile_directory -t_pos $t_cpu

end=`date +%s`
runtime=$((end-start))
echo $runtime
