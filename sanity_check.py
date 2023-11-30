import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-pt','--project_name',required=True,help='Name of project')
parser.add_argument('-it','--n_iteration',required=True,help='Number of current iteration')

io_args = parser.parse_args()
import time

protein = io_args.project_name
n_it = int(io_args.n_iteration)


def sanity_check(protein, n_it):

    old_dict = {}
    for i in range(1,n_it):
        with open(glob.glob(protein+'/iteration_'+str(i)+'/training_labels*')[-1]) as ref:
            ref.readline()
            for line in ref:
                tmpp = line.strip().split(',')[-1]
                old_dict[tmpp] = 1
        with open(glob.glob(protein+'/iteration_'+str(i)+'/validation_labels*')[-1]) as ref:
            ref.readline()
            for line in ref:
                tmpp = line.strip().split(',')[-1]
                old_dict[tmpp] = 1
        with open(glob.glob(protein+'/iteration_'+str(i)+'/testing_labels*')[-1]) as ref:
            ref.readline()
            for line in ref:
                tmpp = line.strip().split(',')[-1]
                old_dict[tmpp] = 1

    t=time.time()
    new_train = {}
    new_valid = {}
    new_test = {}
    with open(glob.glob(protein+'/iteration_'+str(n_it)+'/train_set*')[-1]) as ref:
        for line in ref:
            tmpp = line.strip().split(',')[0]
            new_train[tmpp] = 1
    with open(glob.glob(protein+'/iteration_'+str(n_it)+'/valid_set*')[-1]) as ref:
        for line in ref:
            tmpp = line.strip().split(',')[0]
            new_valid[tmpp] = 1
    with open(glob.glob(protein+'/iteration_'+str(n_it)+'/test_set*')[-1]) as ref:
        for line in ref:
            tmpp = line.strip().split(',')[0]
            new_test[tmpp] = 1
    print(time.time()-t)

    t=time.time()
    for keys in new_train.keys():
        if keys in new_valid.keys():
            new_valid.pop(keys)
        if keys in new_test.keys():
            new_test.pop(keys)
    for keys in new_valid.keys():
        if keys in new_test.keys():
            new_test.pop(keys)
    print(time.time()-t)

    for keys in old_dict.keys():
        if keys in new_train.keys():
            new_train.pop(keys)
        if keys in new_valid.keys():
            new_valid.pop(keys)
        if keys in new_test.keys():
            new_test.pop(keys)

    with open(protein+'/iteration_'+str(n_it)+'/train_set.txt','w') as ref:
        for keys in new_train.keys():
            ref.write(keys+'\n')
    with open(protein+'/iteration_'+str(n_it)+'/valid_set.txt','w') as ref:
        for keys in new_valid.keys():
            ref.write(keys+'\n')
    with open(protein+'/iteration_'+str(n_it)+'/test_set.txt','w') as ref:
        for keys in new_test.keys():
            ref.write(keys+'\n')

sanity_check(protein, n_it)