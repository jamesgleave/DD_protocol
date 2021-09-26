import builtins as __builtin__
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
    __builtin__.print('\t simple_jobs: ', end="")
    return __builtin__.print(*args, **kwargs)


START_TIME = time.time()


parser = argparse.ArgumentParser()
parser.add_argument('-n_it','--iteration_no',required=True,help='Number of current iteration')
parser.add_argument('-mdd','--morgan_directory',required=True,help='Path to the Morgan fingerprint directory for the database')
parser.add_argument('-file_path','--file_path',required=True,help='Path to the project directory, including project directory name')
parser.add_argument('-nhp','--number_of_hyp',required=True,help='Number of hyperparameters')
parser.add_argument('-titr','--total_iterations',required=True,help='Desired total number of iterations')

parser.add_argument('-isl','--is_last',required=True,help='True/False for is this last iteration')

# adding parameter for where to save all the data to:
parser.add_argument('-save', '--save_path', required=False, default=None)

# allowing for variable number of molecules to validate and test from:
parser.add_argument('-n_mol', '--number_mol', required=False, default=1000000, help='Size of validation/test set to be used')

parser.add_argument('-pfm', '--percent_first_mols', required=False, default=-1, help='% of top scoring molecules to be considered as virtual hits in the first iteration (for standard DD run on 11 iterations, we recommend 1)')  # these two inputs must be percentages
parser.add_argument('-plm', '--percent_last_mols', required=False, default=-1, help='% of top scoring molecules to be considered as virtual hits in the last iteration (for standard DD run on 11 iterations, we recommend 0.01)')


# Pass the threshold
parser.add_argument('-ct', '--recall', required=False, default=0.9, help='Recall, [0,1] range, default value 0.9')

# Flag for switching between functions that determine how many mols to be left at the end of iteration 
#   if not provided it defaults to a linear dec
funct_flags = parser.add_mutually_exclusive_group(required=False)
funct_flags.add_argument('-expdec', '--exponential_dec', required=False, default=-1) # must pass in the base number
funct_flags.add_argument('-polydec', '--polynomial_dec', required=False, default=-1) # You must also pass in to what power for this flag

io_args, extra_args = parser.parse_known_args()
n_it = int(io_args.iteration_no)
mdd = io_args.morgan_directory
nhp = int(io_args.number_of_hyp)
isl = io_args.is_last
titr = int(io_args.total_iterations)
rec = float(io_args.recall)
protein = str(io_args.file_path).split('/')[-1]

num_molec = int(io_args.number_mol)

percent_first_mols = float(io_args.percent_first_mols)/100
percent_last_mols = float(io_args.percent_last_mols)/100

exponential_dec = int(io_args.exponential_dec)
polynomial_dec = int(io_args.polynomial_dec)

DATA_PATH = io_args.file_path   # Now == file_path/protein
SAVE_PATH = io_args.save_path
# if no save path is provided we just save it in the same location as the data
if SAVE_PATH is None: SAVE_PATH = DATA_PATH

# sums the first column and divides it by 1 million (this is our total database size)
t_mol = pd.read_csv(mdd+'/Mol_ct_file_%s.csv'%protein,header=None)[[0]].sum()[0]/1000000 # num of compounds in each file is mol_ct_file

cummulative = 0.25*n_it
dropout = [0.2, 0.5]
learn_rate = [0.0001]
bin_array = [2, 3]
wt = [2, 3]
if nhp < 144:
   bs = [256]
else:
    bs = [128, 256]
    
if nhp < 48:
    oss = [10]
elif nhp < 72:
    oss = [5, 10]
else:
    oss = [5, 10, 20]
    
if nhp < 24:
    num_units = [1500, 2000]
else:
    num_units = [100, 1500, 2000]

try:
    os.mkdir(SAVE_PATH+'/iteration_'+str(n_it)+'/simple_job')
except OSError: # catching file exists error
    pass

# Clearing up space from previous iteration
for f in glob.glob(SAVE_PATH+'/iteration_'+str(n_it)+'/simple_job/*'):
    os.remove(f)

scores_val = []
with open(DATA_PATH+'/iteration_'+str(1)+'/validation_labels.txt','r') as ref:
    ref.readline()  # first line is ignored
    for line in ref:
        scores_val.append(float(line.rstrip().split(',')[0]))

scores_val = np.array(scores_val)

first_mols = int(100*t_mol/13) if percent_first_mols == -1.0 else int(percent_first_mols * len(scores_val))

print(first_mols)

last_mols = 100 if percent_last_mols == -1.0 else int(percent_last_mols * len(scores_val))

print(last_mols)

if n_it==1:
    # 'good_mol' is the number of top scoring molecules to save at the end of the iteration
    good_mol = first_mols
else:
    if exponential_dec != -1:
        good_mol = int() #TODO: create functions for these
    elif polynomial_dec != -1:
        good_mol = int()
    else:
        good_mol = int(((last_mols-first_mols)*n_it + titr*first_mols-last_mols)/(titr-1))     # linear decrease as interations increase


print(isl)
#If this is the last iteration then we save only 100 molecules
if isl == 'True':
    good_mol = 100 if percent_last_mols == -1.0 else int(percent_last_mols * len(scores_val))

cf_start = np.mean(scores_val)  # the mean of all the docking scores (labels) of the validation set:
t_good = len(scores_val)

# we decrease the threshold value until we have our desired num of mol left.
while t_good > good_mol: 
    cf_start -= 0.005
    t_good = len(scores_val[scores_val<cf_start])

print('Threshold (cutoff):',cf_start)
print('Molec under threshold:', t_good)
print('Goal molec:', good_mol)
print('Total molec:', len(scores_val))

all_hyperparas = []

for o in oss:   # Over Sample Size
    for batch in bs:
        for nu in num_units:
            for do in dropout:
                for lr in learn_rate:
                    for ba in bin_array:
                        for w in wt:    # Weight
                            all_hyperparas.append([o,batch,nu,do,lr,ba,w,cf_start])

print('Total hyp:', len(all_hyperparas))

# Creating all the jobs for each hyperparameter combination:

other_args = ' '.join(extra_args) + '-rec {} -n_it {} -t_mol {} --data_path {} --save_path {} -n_mol {}'.format(rec, n_it, t_mol, DATA_PATH, SAVE_PATH, num_molec)
print(other_args)
count = 1
for i in range(len(all_hyperparas)):
    with open(SAVE_PATH+'/iteration_'+str(n_it)+'/simple_job/simple_job_'+str(count)+'.sh', 'w') as ref:
        ref.write('#!/bin/bash\n')
        cwd = os.getcwd()
        ref.write('cd {}/scripts_2\n'.format(cwd))
        hyp_args = '-os {} -bs {} -num_units {} -dropout {} -learn_rate {} -bin_array {} -wt {} -cf {}'.format(*all_hyperparas[i])
        ref.write('python -u progressive_docking.py ' + hyp_args + ' ' + other_args)
        ref.write("\n echo complete")
    count += 1
    
print('Runtime:', time.time() - START_TIME)
