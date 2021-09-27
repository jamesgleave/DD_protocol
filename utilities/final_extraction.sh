#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=60     #change this to match the number of processors you want to use
#SBATCH --job-name=extract

source ~/.bashrc
conda activate $5

start=`date +%s`

if [ $4 = 'all_mol' ]; then
   echo "Extracting all SMILES"
   python final_extraction.py -smile_dir $1 -prediction_dir $2 -processors $3
else
   python final_extraction.py -smile_dir $1 -prediction_dir $2 -processors $3 -mols_to_dock $4
fi

end=`date +%s`
runtime=$((end-start))
echo $runtime
