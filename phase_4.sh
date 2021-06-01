#!/bin/bash
#SBATCH --cpus-per-task=3
#SBATCH --ntasks=1
#SBATCH --mem=0               # memory per node
#SBATCH --job-name=phase_4

env=${11}
time=${10}

source ~/.bashrc
conda activate $env

file_path=`sed -n '1p' $3/$4/logs.txt`
protein=`sed -n '2p' $3/$4/logs.txt`

morgan_directory=`sed -n '4p' $3/$4/logs.txt`
smile_directory=`sed -n '5p' $3/$4/logs.txt`
nhp=`sed -n '7p' $3/$4/logs.txt`    # number of hyperparameters
sof=`sed -n '6p' $3/$4/logs.txt`    # The docking software used

rec=$9

num_molec=`sed -n '8p' $3/$4/logs.txt`

echo "writing jobs"
python jobid_writer.py -pt $protein -fp $file_path -n_it $1 -jid $SLURM_JOB_NAME -jn $SLURM_JOB_NAME.txt

t_pos=$2    # total number of processers available
echo "Extracting labels"

if [ $sof = 'Glide' ]; then
   kw='r_i_docking_score'
elif [ $sof = 'FRED' ]; then
   kw='FRED Chemgauss4 score'
fi   

python scripts_2/extract_labels.py -n_it $1 -pt $protein -fp $file_path -t_pos $t_pos -score "$kw"

if [ $? != 0 ]; then
  echo "Extract_labels failed... terminating"
  exit
fi

part_gpu=$5

if [ $6 = $1 ]; then
   last='True'
else
   last='False'
fi

echo "Creating simple jobs"
python scripts_2/simple_job_models.py -n_it $1 -mdd $morgan_directory -time $time -file_path $file_path/$protein -nhp $nhp -titr $6 -n_mol $num_molec -pfm $7 -plm $8 -ct $rec -gp $part_gpu -tf_e $env -isl $last

cd $file_path/$protein/iteration_$1
rm model_no.txt
cd simple_job

echo "Running simple jobs"
#Executes all the files that were created in the simple_jobs directory
for f in *;do sbatch $f;done
