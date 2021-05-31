#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0               # memory per node
#SBATCH --job-name=phase_5

env=${6}

source ~/.bashrc
conda activate $env

file_path=`sed -n '1p' $2/$3/logs.txt`
protein=`sed -n '2p' $2/$3/logs.txt`    # name of project folder

morgan_directory=`sed -n '4p' $2/$3/logs.txt`

num_molec=`sed -n '8p' $2/$3/logs.txt`

gpu_part=$5

python jobid_writer.py -pt $protein -fp $file_path -n_it $1 -jid $SLURM_JOB_NAME -jn $SLURM_JOB_NAME.txt

echo "Starting Evaluation"
python -u scripts_2/hyperparameter_result_evaluation.py -n_it $1 -d_path $file_path/$protein -mdd $morgan_directory -n_mol $num_molec -ct $4
echo "Creating simple_job_predictions"
python scripts_2/simple_job_predictions.py -pt $protein -fp $file_path -n_it $1 -mdd $morgan_directory -gp $gpu_part -tf_e $env

cd $file_path/$protein/iteration_$1/simple_job_predictions/
echo "running simple_jobs"
for f in *;do sbatch $f;done
