import argparse
import glob
import time
import os
import shutil

def wait_phase_4(project, project_path):
    # Get the full path
    full_path = os.path.join(project_path, project)

    # simple job models path
    simple_path = os.path.join(full_path, "iteration_1/simple_job")

    created = False
    initiated = False
    finished = False

    # First check if the folder is created
    while not created:
        time.sleep(10)
        created = True
        if os.path.isdir(simple_path) == False:
            created = False
    else:
        print("simple_job created. Starting to check slurm files...") 

    # Then check if all training jobs ahve been submitted (otherwise it will not have some of the out files to check and may terminate it earlier
    while not initiated:
        time.sleep(10)
        initiated = True
        if len(glob.glob(f"{simple_path}/*out")) != len(glob.glob(f"{simple_path}/*sh")):
            initiated = False
    else:
        print("All jobs started. Starting checking completion...")

    # Finally check if all the jobs are completed or have been cancelled due to the time limit
    while not finished:
        # Grab each slurm file
        time.sleep(10)
        finished = True
        for slurm_out_file in glob.glob(f"{simple_path}/*out"):
            # Read the file
            with open(slurm_out_file, "r") as f:
                # Read the lines and check if complete
                ln = f.readlines()[-1]
                if ("complete" not in ln) and ("DUE TO TIME LIMIT" not in ln):
                    finished = False
    else:
        print("All jobs finished")

def add_train_num_mol(project, project_path, training_size):
    # Get the full path
    full_path = os.path.join(project_path, project)
    full_path = os.path.join(full_path, "iteration_1/simple_job")

    # Add train_num_mol to each slurm script
    for file in glob.glob(full_path + "/*.sh"):
        fp = open(file, "r")
        lines = fp.readlines()
        lines[-2] = lines[-2][:-1] + f" --train_num_mol {training_size}\n"

        fp = open(file, "w")
        fp.writelines(lines)
        fp.close()


def finish_iteration(project, project_path, training_size):
    # Get the full path
    full_path = os.path.join(project_path, project)

    # Best model stats file
    bms = os.path.join(full_path, "iteration_1/best_model_stats.txt")

    with open(bms, "r") as fp:
        bms_contents = fp.readlines()

    # Copy and remove old files
    os.mkdir(f"{full_path}/evaluation_{training_size}")
    shutil.copytree(full_path + "/iteration_1/simple_job", f"{full_path}/evaluation_{training_size}/simple_job")
    shutil.copytree(full_path + "/iteration_1/all_models", f"{full_path}/evaluation_{training_size}/all_models")
    shutil.copytree(full_path + "/iteration_1/best_models", f"{full_path}/evaluation_{training_size}/best_models")
    shutil.copyfile(full_path + "/iteration_1/hyperparameter_morgan_with_freq_v3.csv", f"{full_path}/evaluation_{training_size}/hyperparameter_morgan_with_freq_v3.csv")
    shutil.copyfile(full_path + "/iteration_1/hyperparameter_morgan_with_freq_v3.txt", f"{full_path}/evaluation_{training_size}/hyperparameter_morgan_with_freq_v3.txt")
    shutil.copyfile(full_path + "/iteration_1/best_model_stats.txt", f"{full_path}/evaluation_{training_size}/best_model_stats.txt")
    shutil.rmtree(full_path + "/iteration_1/simple_job", ignore_errors=True)
    shutil.rmtree(full_path + "/iteration_1/all_models", ignore_errors=True)
    shutil.rmtree(full_path + "/iteration_1/best_models", ignore_errors=True)
    os.remove(full_path + "/iteration_1/hyperparameter_morgan_with_freq_v3.csv")
    os.remove(full_path + "/iteration_1/hyperparameter_morgan_with_freq_v3.txt")
    os.remove(full_path + "/iteration_1/best_model_stats.txt")

    # Write the values to a new file
    evaluation_text = full_path + "/evaluation.csv"
    first_line = True
    if os.path.isfile(evaluation_text) == False:
        first_line = False
    with open(evaluation_text, "a") as e:
        if first_line == False:
            e.write('train_sample_size,recall_test,prec_test,left_test\n')
        for line in bms_contents:
            if 'Precision' in line:
                prec_test = line.split()[3][:-1]
            elif 'Recall' in line:
                rec_test = line.split()[3][:-1]
            elif 'Left' in line:
                lft_test = line.split()[4][:-1]      
        e.write(f"{training_size},{rec_test},{prec_test},{lft_test}\n")
    print("Done evaluation. Waiting for next sample size or terminating...")

if __name__ == '__main__':
    # Grab args
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", required=True, type=int)
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--project_path", required=True)
    parser.add_argument("--mode", required=True)

    # Parse the args
    args = parser.parse_args()

    # Run different modes
    if args.mode == "wait_phase_4":
        wait_phase_4(args.project_name, args.project_path)
    elif args.mode == "finished_iteration":
        finish_iteration(args.project_name, args.project_path, args.sample_size)
    elif args.mode == "add_train_num_mol":
        add_train_num_mol(args.project_name, args.project_path, args.sample_size)
