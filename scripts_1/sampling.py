from contextlib import closing
from multiprocessing import Pool
import pandas as pd
import numpy as np
import argparse
import glob
import time
import os

try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__

# For debugging purposes only:
def print(*args, **kwargs):
    __builtin__.print('\t sampling: ', end="")
    return __builtin__.print(*args, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument('-pt', '--project_name',required=True,help='Name of the DD project')
parser.add_argument('-fp', '--file_path',required=True,help='Path to the project directory, excluding project directory name')
parser.add_argument('-it', '--n_iteration',required=True,help='Number of current iteration')
parser.add_argument('-dd', '--data_directory',required=True,help='Path to directory containing the remaining molecules of the database; if first iteration, path to Morgan fingerprints of the database, if other iteration path to morgan_1024_predictions folder of the previous iteration')
parser.add_argument('-t_pos', '--tot_process',required=True,help='Number of CPUs to use for multiprocessing')
parser.add_argument('-tr_sz', '--train_size',required=True,help='Size of training set')
parser.add_argument('-vl_sz', '--val_size',required=True,help='Size of validation and test set')
io_args = parser.parse_args()

protein = io_args.project_name
file_path = io_args.file_path
n_it = int(io_args.n_iteration)
data_directory = io_args.data_directory
tot_process = int(io_args.tot_process)
tr_sz = int(io_args.train_size)
vl_sz = int(io_args.val_size)
rt_sz = tr_sz/vl_sz

print("Parsed Args:")
print(" - Iteration:", n_it)
print(" - Data Directory:", data_directory)
print(" - Training Size:", tr_sz)
print(" - Validation Size:", vl_sz)


def train_valid_test(file_name):
    sampling_start_time = time.time()
    f_name = file_name.split('/')[-1]
    mol_ct = pd.read_csv(data_directory+"/Mol_ct_file_updated_%s.csv"%protein, index_col=1)
    if n_it == 1:
        to_sample = int(mol_ct.loc[f_name].Sample_for_million/(rt_sz+2))
    else:
        to_sample = int(mol_ct.loc[f_name].Sample_for_million/3)

    total_len = int(mol_ct.loc[f_name].Number_of_Molecules)
    shuffle_array = np.linspace(0, total_len-1, total_len)
    seed = np.random.randint(0, 2**32)
    np.random.seed(seed=seed)
    np.random.shuffle(shuffle_array)

    if n_it == 1:
        train_ind = shuffle_array[:int(rt_sz*to_sample)]
        valid_ind = shuffle_array[int(to_sample*rt_sz):int(to_sample*(rt_sz+1))]
        test_ind = shuffle_array[int(to_sample*(rt_sz+1)):int(to_sample*(rt_sz+2))]
    else:
        train_ind = shuffle_array[:to_sample]
        valid_ind = shuffle_array[to_sample:to_sample*2]
        test_ind = shuffle_array[to_sample*2:to_sample*3]

    train_ind_dict = {}
    valid_ind_dict = {}
    test_ind_dict = {}

    train_set = open(file_path + '/' + protein + "/iteration_" + str(n_it) + "/train_set.txt", 'a')
    test_set = open(file_path + '/' + protein + "/iteration_" + str(n_it) + "/test_set.txt", 'a')
    valid_set = open(file_path + '/' + protein + "/iteration_" + str(n_it) + "/valid_set.txt", 'a')

    for i in train_ind:
        train_ind_dict[i] = 1
    for j in valid_ind:
        valid_ind_dict[j] = 1
    for k in test_ind:
        test_ind_dict[k] = 1

    # Opens the file and write the test, train, and valid files
    with open(file_name, 'r') as ref:
        for ind, line in enumerate(ref):
            molecule_id = line.strip().split(',')[0]
            if ind == 1:
                print("molecule_id:", molecule_id)

            # now we write to the train, test, and validation sets
            if ind in train_ind_dict.keys():
                train_set.write(molecule_id + '\n')
            elif ind in valid_ind_dict.keys():
                valid_set.write(molecule_id + '\n')
            elif ind in test_ind_dict.keys():
                test_set.write(molecule_id + '\n')

    train_set.close()
    valid_set.close()
    test_set.close()
    print("Process finished sampling in " + str(time.time()-sampling_start_time))

if __name__ == '__main__':
    try:
        os.mkdir(file_path+'/'+protein+"/iteration_"+str(n_it))
    except OSError:
        pass

    f_names = []
    for f in glob.glob(data_directory+'/*.txt'):
        f_names.append(f)

    t = time.time()
    print("Starting Processes...")
    with closing(Pool(np.min([tot_process, len(f_names)]))) as pool:
        pool.map(train_valid_test, f_names)

    print("Compressing smile file...")
    print("Sampling Complete - Total Time Taken:", time.time()-t)

