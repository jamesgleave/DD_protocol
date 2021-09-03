#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0               # memory per node
#SBATCH --job-name=progressive_evaluation

eval_location=$1
project_name=$2
name_gpu_partition=$3
percent_first_mols=$4
percent_last_mols=$5
recall_value=$6
max_size=$7
min_size=$8
n_steps=$9
time=${10}
env=${11}

source ~/.bashrc

morgan_path=$(sed -n '4p' $eval_location'/'$project_name'/'logs.txt)
n_mol_vt=$(sed -n '8p' $eval_location'/'$project_name'/'logs.txt)

# Calculate the step size
let step_size=($max_size-$min_size)/$(expr $n_steps - 1)

evaluation(){
  # Run the evaluation at each sample size
  echo ">> Running iteration $i with sample size $size"
  sbatch phase_4_evaluator.sh 1 3 $eval_location $project_name $name_gpu_partition 2 $percent_first_mols $percent_last_mols $recall_value $time $env $size

  # Wait for phase 4 to complete
  python3 -u scripts_2/progressive_evaluator.py --sample_size $size --project_name $project_name --project_path $eval_location --mode wait_phase_4

  # Activate the conda env and run the hyperparameter result eval script
  conda activate $env
  python -u scripts_2/hyperparameter_result_evaluation.py -n_it 1 -d_path $eval_location/$project_name -mdd $morgan_path -n_mol $n_mol_vt

  # Wait for phase 4 to complete
  python3 scripts_2/progressive_evaluator.py --sample_size $size --project_name $project_name --project_path $eval_location --mode finished_iteration
}


# Run eval
# Loop through each value in the search
for ((i = 0 ; i < $n_steps ; i++)); do
   # Calculate sample size
   size=$(($min_size + $i*$step_size))
   echo $size
   evaluation
done

