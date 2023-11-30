from multiprocessing import Pool
from functools import partial
from contextlib import closing
import argparse
import numpy as np
import pickle
import glob
import time
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-pt', '--project_name', required=True, help='Name of the DD project')
parser.add_argument('-it', '--n_iteration', required=True, help='Number of current iteration')
parser.add_argument('-smd', '--smile_directory', required=True, help='Path to SMILES directory of the database')
parser.add_argument('-t_pos', '--tot_process', required=True, help='Number of CPUs to use for multiprocessing')

io_args = parser.parse_args()
protein = io_args.project_name
n_it = int(io_args.n_iteration)
smile_directory = io_args.smile_directory
tot_process = int(io_args.tot_process)

ITER_PATH = protein + '/iteration_' + str(n_it)


def extract_smile(file_name, train, valid, test):
    # This function extracts the smiles from a file to write them to train, test, and valid files for model training.
    ref1 = open(ITER_PATH + '/smile/' + 'train_' + file_name.split('/')[-1], 'w')
    ref2 = open(ITER_PATH + '/smile/' + 'valid_' + file_name.split('/')[-1], 'w')
    ref3 = open(ITER_PATH + '/smile/' + 'test_' + file_name.split('/')[-1], 'w')

    df = pd.read_csv(file_name)
    ref1.write('r_i_docking_score' + ',' + 'ZINC_ID' + '\n')
    ref2.write('r_i_docking_score' + ',' + 'ZINC_ID' + '\n')
    ref3.write('r_i_docking_score' + ',' + 'ZINC_ID' + '\n')
    
    for i in range(df.shape[0]):
        score = df['score'][i]
        tmpp = df['Smiles'][i]
        line = str(score) + ',' + tmpp + '\n'

        if tmpp in train.keys():
            train[tmpp] += 1
            if train[tmpp] == 1: ref1.write(line)

        elif tmpp in valid.keys():
            valid[tmpp] += 1
            if valid[tmpp] == 1: ref2.write(line)

        elif tmpp in test.keys():
            test[tmpp] += 1
            if test[tmpp] == 1: ref3.write(line)

def alternate_concat(files):
    # Returns a list of the lines in a file
    with open(files, 'r') as ref:
        return ref.readlines()

def smile_duplicacy(f_name):
    # removes duplicate molec from the file 
    mol_list = {} # keeping track of which mol have been written
    ref1 = open(f_name[:-4] + '_updated.smi', 'a')
    with open(f_name, 'r') as ref:
        for line in ref:
            tmpp = line.strip().split()[-1]
            if tmpp not in mol_list: # avoiding duplicates
                mol_list[tmpp] = 1 
                ref1.write(line)
    os.remove(f_name)

def delete_all(files):
    os.remove(files)

def extracting_smiles(protein, n_it, smile_directory, tot_process):
    try:
        os.mkdir(ITER_PATH + '/smile')
    except: # catching exception for when the folder already exists
        pass

    files_smiles = [] # Getting the path to every smile file from docking
    for f in glob.glob(smile_directory + "/*.txt"):
        files_smiles.append(f)
    print(files_smiles)

    return_mols_per_file = []
    for j in range(3):
        ct = 0
        for i in range(len(return_mols_per_file)):
            ct += len(return_mols_per_file[i][j])
        print(ct)

    for k in range(3):
        t = time.time()
        for i in range(len(return_mols_per_file)):
            for j in range(i + 1, len(return_mols_per_file)):
                for keys in return_mols_per_file[i][k].keys():
                    if keys in return_mols_per_file[j][k]:
                        return_mols_per_file[j][k].pop(keys)
        print(time.time() - t)

    for j in range(3):
        ct = 0
        for i in range(len(return_mols_per_file)):
            ct += len(return_mols_per_file[i][j])
        print(ct)

    train = {}
    valid = {}
    test = {}
    for j in range(3):
        for i in range(len(return_mols_per_file)):
            for keys in return_mols_per_file[i][j]:
                if j == 0:
                    train[keys] = 0
                elif j == 1:
                    valid[keys] = 0
                elif j == 2:
                    test[keys] = 0

    all_train = {}
    all_valid = {}
    all_test = {}
    with open(ITER_PATH + "/train_set.txt", 'r') as ref:
        for line in ref:
            all_train[line.rstrip()] = 0

    with open(ITER_PATH + "/valid_set.txt", 'r') as ref:
        for line in ref:
            all_valid[line.rstrip()] = 0

    with open(ITER_PATH + "/test_set.txt", 'r') as ref:
        for line in ref:
            all_test[line.rstrip()] = 0

    for keys in train.keys():
        all_train.pop(keys)

    for keys in valid.keys():
        all_valid.pop(keys)

    for keys in test.keys():
        all_test.pop(keys)

    print(len(all_train), len(all_valid), len(all_test))

    t0 = time.time()
    with closing(Pool(np.min([tot_process, len(files_smiles)]))) as pool:
        pool.map(partial(extract_smile, train=all_train, valid=all_valid, test=all_test), files_smiles)
    print(time.time() - t0)

    all_to_delete = []
    for type_to in ['train', 'valid', 'test']:
        t = time.time()
        files = []
        for f in glob.glob(ITER_PATH + '/smile/' + type_to + '*'):
            files.append(f)
            all_to_delete.append(f)
        print(len(files))
        if len(files) == 0:
            print("Error in address above")
            break
        with closing(Pool(np.min([tot_process, len(files)]))) as pool:
            to_print = pool.map(alternate_concat, files)
        with open(ITER_PATH + '/smile/' + type_to + '_smiles_final.csv', 'w') as ref:
            for file_data in to_print:
                for line in file_data:
                    ref.write(line)
        to_print = []
        print(type_to, time.time() - t)

    f_names = []
    for f in glob.glob(ITER_PATH + '/smile/*final*'):
        f_names.append(f)


    with closing(Pool(np.min([tot_process, len(f_names)]))) as pool:
        pool.map(smile_duplicacy, f_names)

    with closing(Pool(np.min([tot_process, len(all_to_delete)]))) as pool:
        pool.map(delete_all, all_to_delete)
    print(time.time() - t0)

extracting_smiles(protein, n_it, smile_directory, tot_process)