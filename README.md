# The Deep Docking protocol

Deep docking (DD) is a deep learning-based tool developed to accelerate docking-based virtual screening. Using a docking program of choice, the method allows to virtually screem extensive chemical libraries 50 times faster than conventional docking without losing valuable drug candidates. For further details into the processes behind DD, please refer to our [paper](https://doi.org/10.1021/acscentsci.0c00229). This repository provides all the scripts required to run different stages of DD, and also slurm programs to automatically perform the protocol on a computing cluster using either Glide SP or FRED docking. The [protocol](https://www.nature.com/articles/s41596-021-00659-2) can be trivially adapted to any other docking program.

If you use DD in your research, please cite:

Gentile, F. et al. *Deep Docking: A Deep Learning Platform for Augmentation of Structure Based Drug Discovery.* ACS Cent. Sci. 6, 939–949 (2020)  
Gentile, F. et al. *Artificial intelligence–enabled virtual screening of ultra-large chemical libraries with deep docking.* Nat. Protoc. 17, 672–697 (2022)



## Requirements
The only installation step required for DD is to create a conda environment and install the following packages within it:
* rdkit
* tensorflow >= 1.14.0 (1.15 GPU version recommended. If you are using cuda11, please use [nvidia-tensorflow](https://developer.nvidia.com/blog/accelerating-tensorflow-on-a100-gpus/))
* pandas
* numpy
* keras
* matplotlib
* scikit-learn

The majority of the DD steps can be run on regular CPU cores; for model training and inference however, the use of GPUs is recommended. To run the automated version (see below, B section), a computing cluster with slurm job scheduler is required.


## Test data
A complete example of DD iteration for testing the protocol is available from https://doi.org/10.20383/102.0489. A DD-prepared version of the ZINC20 library (as available in March 2021) that can readily be screened with the protocol is available at https://files.docking.org/zinc20-ML/


## Help
For help with the options of a specific script, type

```
python script.py -h
```


## A. Run Deep Docking manually
Here we present in details how to use each individual script constituting DD. If you have access to an HPC cluster, you may want to consider running the automated protocol using Slurm job scheduler (see Run automated Deep Docking on HPC clusters section below).

### i. Preparing a database
In order to be prepared for DD virtual screening, the chemical library must be in SMILES format. DD requires Morgan fingerprints of radius 2 and size 1024 bits for each molecule, represented in a compressed form as a list of the indexes of bits that are set to 1. 
It is recommended to split the library of SMILES into a number of evenly populated files to facilitate other steps such as random sampling and inference, and place these files into a new folder. This reorganization can be achieved for example with the `split` bash command. For example, consider a `smiles.smi` file with a billion compounds, to obtain 1000 files of 1 million molecules each you can run:

```bash
split -d -l 1000000 smiles.smi smile_all_ --additional-suffix=.smi
```

Ideally the number of split files should be equal to the number of CPUs used for random sampling. After this step, stereoisomers, tautomers and protomers should be calculated for each molecule and stored in txt files (e.g. smiles_all_1.txt).

Once the SMILES have been prepared, activate the conda environment with rdkit and calculate Morgan fingerprints using the `morgan_fp.py` script (located in the `utilities` folder):

```bash
python morgan_fp.py --smile_folder_path path_smiles_library/smiles_library --folder_name path_morgan_library/morgan_library --tot_process num_cpus
```
which will create all the fingerprints and place them in `path_morgan_library/morgan_library`. The *--tot_process* argument controls how many files will be processed in parallel using multiprocessing.

### ii. Phase 1. Random sampling of molecules
In phase 1 molecules are randomly sampled from the database to generate or augment the training set. During the first iteration, this phase also samples molecules for the validation and test sets.

Create a project folder. Run the following scripts (from *scripts_1* folder) sequentially with the activated conda environment:

```bash
python molecular_file_count_updated.py --project_name project --n_iteration current_iteration --data_directory left_mol_directory --tot_process num_cpus --tot_sampling  molecules_to_dock
python sampling.py --project_name  project --file_path path_project --n_iteration current_iteration --data_director left_mol_directory --tot_process num_cpus --train_size train_size --val_size val_size
python sanity_check.py  --project_name project --file_path path_project --n_iteration current_iteration
python extracting_morgan.py --project_name project --file_path path_project --n_iteration current_iteration --morgan_directory path_morgan_library/morgan_library --tot_process num_cpus
python extracting_smiles.py --project_name project --file_path path_project --n_iteration current_iteration --smile_directory path_smiles_library/smiles_library --tot_process num_cpus
```

* `molecular_file_count_updated.py` determines the number of molecules to be sampled from each file of the database. The sample sizes (per million) are stored in `Mol_ct_file_updated_project_name.csv` file created in the `left_mol_directory` directory.

* `sampling.py` randomly samples the specified number of molecules for the training, validation and testing sets (the validation and testing sets are generated only in the first iteration). 

* `sanity_check.py` removes overlaps between the sets.

* `extracting_morgan.py` and `extracting_smiles.py` extract Morgan fingerprints and SMILES for the molecules that were randomly sampled, and saves them in `morgan` and `smiles` folders inside the directory of the current iteration.

**IMPORTANT:** For `molecular_file_count_updated.py` AND `sampling.py` the option `left_mol_directory` must be the directory from where molecules are sampled; for iteration 1, `left_mol_directory` is the directory where the Morgan fingerprints of the library are stored; BUT for successive iterations this must be the path to `morgan_1024_predictions` folder of the previous iteration. This will ensure that sampling is done progressively on better scoring subsets of the database over the course of DD.

### iii. Phase 2 and phase 3. Ligand preparation and docking
After phase 1 is completed, molecules grouped in the *smiles* folder can be prepared and docked to the target. Use your favourite tools for this step. It is important that docking results are saved as SDF files in a *docked* folder in the current iteration directory, keeping the same name convention of the files in the *smile* folder (the name of the originating set (training, validation, testing) must always be present in the name of the respective SDF file with docking results).

### iv. Phase 4. Neural network training
In phase 4, deep neural network models are trained with the docking scores from the previous phase. Run the following scripts from *scripts_2*, after activating the environment:

```bash
python extract_labels.py --project_name project --file_path path_to_project --iteration_no current_iteration --tot_process num_cpus -score score_keyword
python simple_job_models_manual.py --iteration_no current_iteration --morgan_directory path_morgan_library/morgan_library --file_path path_project/project --number_of_hyp num_models_to_train --total_iterations number_total_iterations --is_last is_this_last_iteration (False/True)? --number_mol num_molecules_test_valid_set --percent_first_mols percent_first_molecules --percent_last_mols percent_last_mols --recall recall_value 
```

* `extract_labels.py` extracts docking scores for model training. It should generate three comma-seperated files, `training_labels.txt`, `validation_labels.txt` and `testing_labels.txt` inside the current iteration folder.

* `simple_job_models_manual.py` creates bash scripts to run model training using the `progressive_docking.py` script. These scripts are generated inside the `simple_job` folder in the current iteration. Note that if `--recall` is not specified, the recall value will be set to 0.9.

The bash scripts generated by `simple_job_models.py` in the iteration directory, *simple_job* foldder, should be then run on GPUs to train DD models. The resulting models will be saved in the `all_models` folder in the current iteration.

### v. Phase 5. Selection of best model and prediction of virtual hits
In phase 5 the models are evalauted with a grid search, and the model with the highest precision is selected for predicting scores of all the molecules in the database. This step will create a `morgan_1024_predictions` folder in the iteration directory, which will contain all the molecules that are predicted as virtual hits. 
To run phase 5, use the following scripts from *scripts_2* with the conda environment activated:

```bash
python hyperparameter_result_evaluation.py --n_iteration current_iteration --data_path path_project/project --morgan_directory path_morgan_library/morgan_library --number_mol num_molecules --recall recall_value
python simple_job_predictions_manual.py --project_name project --file_path path_project --n_iteration current_iteration --morgan_directory path_morgan_library/morgan_library

```

* `hyperparameter_result_evaluation.py` evaluates the models generated in phase 4 and select the best (most precise) one.

* `simple_job_predictions.py` creates bash scripts to run the predictions over the full database using the `Prediction_morgan_1024.py` script. These scripts will be saved in the `simple_job_predictions` folder of the current iteration, and they should be run on GPU nodes in order to predict virtual hits from the full database. Prospective hits will then be saved in `morgan_1024_predictions` folder of the current iteration, together with their virutal hit likeness.

### vi. Final docking phase
After the last iteration of DD is complete, SMILES of all or a ranked subset of the predicted virtual hits can be obtained for the final docking. Ranking is based on the probabilities of being virtual hits. With the environment activated, use the following script (available in `utilities`):

```bash
python final_extraction.py -smile_dir path_smiles_library/smiles_library -prediction_dir path_last_iteration/morgan_1024_predictions -processors n_cpus -mols_to_dock num_molecules_to_dock
```

This will return a list of SMILES of all the predicted virtual hits of the last iteration or the top `num_molecules_to_dock` molecules ranked by their virtual hit likeness. If *mols_to_dock* is not specified, all the prospective hits will be extracted. Virtual hit likeness will also be returned in a separated file. These molecules can be then docked into the target of interest.


## B. Run automated Deep Docking on HPC clusters
As part of this repository, we provide a series of scripts to run the process on a Slurm cluster using the preparation and docking tools that we regularly employ in virtual screening campaigns. The workflow can be trivially adapted to any other set of tools by modifying the scripts of phase 2, 3 and 4. Additionally, the user will need to either modify the headers of the slurm scripts or pass the #SBATCH values from command line in order to satisfy the requirements of the cluster that is being used. 

### i. Automated library preparation
In our DD automated version, SMILES preparation is performed using OpenEye tools (flipper, https://docs.eyesopen.com/applications/omega/flipper.html and tautomers, https://docs.eyesopen.com/applications/quacpac/tautomers/tautomers.html, both require license). Use the `compute_states.sh` script provided in `utilities` to submit a preparation job for each original SMILES file:

```bash
for i in $(ls smiles/smile_all_*.smi); do sbatch compute_states.sh $i library_prepared; done
```

The SMILES will be prepared and saved into the `library_prepared` folder. Note that this program will enumerate all the unspecified chiral centers of the molecules and assign unique names to each isomer, and then calculate the dominant tautomer at pH 7.4. The next step is the calculation of Morgan fingerprints; run:

```bash
sbatch --cpus-per-task n_cpus_per_node compute_morgan_fp.sh library_prepared library_prepared_fp n_cpus_per_node conda_environment
```
to calculate the Morgan fingerprints and save them in `library_prepared_fp`.

### ii. Project file preparation
Create a `logs.txt` file with parameters that will be read by the slurm scripts, and save it in the project folder. An example file is provided in `utilities` (remove the comments delimited by #).

### iii. Automated phase 1
Run:

```bash
sbatch --cpus-per-task n_cpus_per_node phase_1.sh current_iteration n_cpus_per_node path_project project training_sample_size conda_env
```

### iv. Automated phase 2
The provided automated version of DD works with Glide docking or FRED docking; the choice of the program (listed in `logs.txt`) influences both how the SMILES are translated to 3D structures and how they are successively docked. For creating conformations for Glide docking, run:

```bash
sbatch phase_2_glide.sh current_iteration n_cpus_per_node path_project project name_cpu_partition
```

OR for creating conformations for FRED docking, run:

```bash
sbatch phase_2_fred.sh current_iteration n_cpus_per_node path_project project name_cpu_partition
```

### v. Automated phase 3
For Glide docking, run:

```bash
sbatch phase_3_glide.sh current_iteration n_cpus_total path_project project
```

For FRED docking, run:

```bash
sbatch phase_3_fred.sh current_iteration n_cpus_per_node path_project project name_cpu_partition
```

### iv. Automated phase 4
Run:

```bash
sbatch phase_4.sh current_iteration 3 path_project project name_gpu_partition tot_number_iterations percent_first_mols percent_last_mols_value recall_value 00-15:00 conda_env
```
00-15:00 is the maximal training time (days-hours:mins) after which Slurm will cancel the job. Usually each model does not require more than 12 hours to complete the training.

### v. Automated phase 5
Run:

```bash
sbatch phase_5.sh current_iteration path_project project recall_value name_gpu_partition conda_env
```

### vi. Automated final phase
Run:

```bash
sbatch --cpus-per-task n_cpus_per_node final_extraction.sh path_smiles_library/smiles_library path_last_iteration/morgan_1024_predictions n_cpus_per_node 'all_mol' conda_env
```

*all_mol* will extract all the prospective virtual hits. If less molecules need to be selected for docking (for example, if the final number of predicted hits is too high to be docked with the available resources), change the argument accordingly to extract a smaller subset of molecules ranked by their virtual hit likeness. The resulting SMILES can then be prepared and docked into the target.
