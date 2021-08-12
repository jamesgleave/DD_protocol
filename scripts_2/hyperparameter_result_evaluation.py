import builtins as __builtin__
import argparse

import pandas as pd
import numpy as np
import glob
import os
from ML.DDModel import DDModel
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,roc_curve, precision_score, recall_score
from shutil import copy2

import warnings
warnings.filterwarnings('ignore')


# For debugging purposes only:
def print(*args, **kwargs):
    __builtin__.print('\t eval_v2: ', end="")
    return __builtin__.print(*args, **kwargs)


parser = argparse.ArgumentParser()
parser.add_argument('-n_it','--n_iteration',required=True,help='Number of current iteration')
parser.add_argument('-d_path','--data_path',required=True,help='Path to project folder, including project folder name')
parser.add_argument('-mdd','--morgan_directory',required=True,help='Path to Morgan fingerprint directory for the database')
parser.add_argument('-ct', '--recall', required=False, default=0.9, help='Recall, [0,1] range, default value 0.9')

# adding parameter for where to save all the data to:
parser.add_argument('-s_path', '--save_path', required=False, default=None)

parser.add_argument('-n_mol', '--number_mol', required=False, default=3000000, help='Size of test/validation set to be used')

io_args = parser.parse_args()
n_iteration = int(io_args.n_iteration)
mdd = io_args.morgan_directory
num_molec = int(io_args.number_mol)
rec = float(io_args.recall)
protein = str(io_args.data_path).split('/')[-1]

DATA_PATH = io_args.data_path   # Now == file_path/protein
SAVE_PATH = io_args.save_path
# if no save path is provided we just save it in the same location as the data
if SAVE_PATH is None: SAVE_PATH = DATA_PATH


print("Done importing.")

# Gets the total number of molecules (SEE: simple_job_models.py, line 32)
total_mols = pd.read_csv(mdd+'/Mol_ct_file_%s.csv'%protein,header=None)[[0]].sum()[0]/1000000

# reading in the file created in progressive_docking.py (line 456)
hyperparameters = pd.read_csv(SAVE_PATH+'/iteration_'+str(n_iteration)+'/hyperparameter_morgan_with_freq_v3.csv',header=None)
 
# theses are also declared in progressive_docking.py
### TODO: add these columns in progressive_docking.py as a header instead of declaring them here (Line 456)
hyperparameters.columns = ['Model_no','Over_sampling','Batch_size','Learning_rate','N_layers','N_units','dropout',
                          'weight','cutoff','ROC_AUC','Pr_0_9','tot_left_0_9_mil','auc_te','pr_te','re_te','tot_left_0_9_mil_te','tot_positives']

# Converting total to per million
hyperparameters.tot_left_0_9_mil = hyperparameters.tot_left_0_9_mil/1000000
hyperparameters.tot_left_0_9_mil_te = hyperparameters.tot_left_0_9_mil_te/1000000   ## What are these? it is never used...

hyperparameters['re_vl/re_pr'] = rec/hyperparameters.re_te  # ratio of desired recall and the recall of the test set

print('hyp dataframe:', hyperparameters.head())

df_grouped_cf = hyperparameters.groupby('cutoff')  # Groups them according to cutoff values for calculations

cf_values = {}  # Cutoff values (thresholds for validation set virtual hits)

print('Got Hyperparams')
# Looping through each group and printing mean and std for that particular cuttoff value
for mini_df in df_grouped_cf:
    print(mini_df[0])   # the cutoff value for the group
    print(mini_df[1]['re_vl/re_pr'].mean())
    print(mini_df[1]['re_vl/re_pr'].std())
    cf_values[mini_df[0]] = mini_df[1]['re_vl/re_pr'].std()
    
print('cf_values:', cf_values)

model_to_use_with_cf = []   
ind_pr = []
for cf in cf_values:
    models = hyperparameters[hyperparameters.cutoff == cf]  # gets all models matching that cutoff
    n_models = len(models)
    thr = rec   # The recall for true positives ### TODO: make this a input for the script

    # Decreasing the threshold until a quarter of the models have recall values greater than it
    while len(models[models.re_te >= thr]) <= n_models//4:
        thr -= 0.01
    
    models = models[models.re_te >= thr]
    models = models.sort_values('pr_te', ascending=False)   # Sorting in descending order

    if cf_values[cf] < 0.01:    # Checks to see if std is less than 0.01
        model_to_use_with_cf.append([cf, models.Model_no[:1].values])
        ind_pr.append([cf, models.pr_te[:1].values])
    else:
        # When std is high we use 3 models to get a better idea of its perfomance as a whole
        model_to_use_with_cf.append([cf, models.Model_no[:3].values])
        ind_pr.append([cf, models.pr_te[:3].values])
        
print(model_to_use_with_cf)  # [ [cf_1, [model_no_1, ...]], [cf_2, [model_no_1, ...]] ... ]
print(ind_pr)  # printed for viewing progress?


def get_all_x_data(morgan_path, ID_labels): # ID_labels is a dataframe containing the zincIDs and their corresponding labels.
    train_set = np.zeros([num_molec,1024], dtype=bool) # using bool to save space
    train_id = []

    print('x data from:', morgan_path)
    with open(morgan_path,'r') as ref:
        line_no=0
        for line in ref:            
            mol_info=line.rstrip().split(',')
            train_id.append(mol_info[0])
            
            # "Decompressing" the information from the file about where the 1s are on the 1024 bit vector.
            bit_indicies = mol_info[1:] # array of indexes of the binary 1s in the 1024 bit vector representing the morgan fingerprint
            for elem in bit_indicies:
                train_set[line_no,int(elem)] = 1

            line_no+=1
    
    train_set = train_set[:line_no,:]

    print('Done...')
    train_pd = pd.DataFrame(data=train_set, dtype=np.uint8)
    train_pd['ZINC_ID'] = train_id

    score_col = ID_labels.columns.difference(['ZINC_ID'])[0]
    train_data = pd.merge(ID_labels, train_pd, how='inner',on=['ZINC_ID'])
    X_data = train_data[train_data.columns.difference(['ZINC_ID', score_col])].values   # input
    y_data = train_data[[score_col]].values    # labels
    return X_data, y_data


# This function gets all the zinc ids their corresponding labels into a pd.dataframe
def get_zinc_and_labels(zinc_path, labels_path):
    ids = []
    with open(zinc_path,'r') as ref:
        for line in ref:
            ids.append(line.split(',')[0])
    zincIDs = pd.DataFrame(ids, columns=['ZINC_ID'])

    labels_df = pd.read_csv(labels_path, header=0)
    combined_df = pd.merge(labels_df, zincIDs, how='inner', on=['ZINC_ID'])
    return combined_df.set_index('ZINC_ID')


main_path = DATA_PATH+'/iteration_1'
print('Geting zinc ids and labels')
# Creating the test, and validation data from the first iteration:
zinc_labels_valid = get_zinc_and_labels(main_path + '/morgan/valid_morgan_1024_updated.csv', main_path +'/validation_labels.txt')
zinc_labels_test = get_zinc_and_labels(main_path + '/morgan/test_morgan_1024_updated.csv', main_path +'/testing_labels.txt')

print('Generating test, and valid data')
# Getting the x data from the zinc ids (x=input, y=labels)
# decompresses the indexes to 1024 bit vector
X_valid, y_valid = get_all_x_data(main_path +'/morgan/valid_morgan_1024_updated.csv', zinc_labels_valid)
X_test, y_test = get_all_x_data(main_path +'/morgan/test_morgan_1024_updated.csv', zinc_labels_test)

print('Reducing models,', len(model_to_use_with_cf))
for i in range(len(model_to_use_with_cf)):
    cf = model_to_use_with_cf[i][0]
    n_good_mol = len([x for x in y_valid if x < cf]) # num of molec that are under the cutoff value
    print('\t','cf:',cf)

    # If not enough molecules exceed the cutoff then we only save one of the models and ignore the rest
    if n_good_mol <= 10000: 
        model_to_use_with_cf[i][1] = [model_to_use_with_cf[i][1][0]]    # Still maintaing the format: [ [cf_1, [model_no_1, ...]], [cf_2, [model_no_1, ...]] ... ]


cf_with_left = {}
main_thresholds = {}
all_sc = {}
path_to_model = SAVE_PATH+'/iteration_'+str(n_iteration)+'/all_models/'

print('Model_to_use_with_cf:', model_to_use_with_cf)
for i in range(len(model_to_use_with_cf)):
    cf = model_to_use_with_cf[i][0]

    # y_train<cf returns a bool array for this condition on each element
    y_test_cf = y_test<cf
    y_valid_cf = y_valid<cf

    models = []
    # loading the models matching the cutoff and appending them to the models list
    for mn in model_to_use_with_cf[i][-1]:
        print('\tLoading model:', path_to_model + '/model_'+str(mn))
        models.append(DDModel.load(path_to_model+'/model_'+str(mn)))
    print('num models:', len(models))
    
    prediction_valid = []
    scc = []
    for model in models:
        print('using valid as validation')
        model_pred = model.predict(X_valid)
        if model.output_activation == 'linear':
            # Converting back to binary values to get stats
            model_pred = model_pred < cf
        prediction_valid.append(model_pred)
        precision, recall, thresholds = precision_recall_curve(y_valid_cf, model_pred)
        scc.append([precision, recall, thresholds])

    tr = []
    for _, recall, thresholds in scc:
        # Getting the index positions for where the recall is greater than the specified value to append the threshold value that got it
        tr.append(thresholds[np.where(recall > rec)[0][-1]])
    main_thresholds[cf] = tr

    print('tr:', tr)

    prediction_test = []
    for model in models:
        model_pred = model.predict(X_test)
        if model.output_activation == 'linear':
            # Converting back to binary values to get stats
            model_pred = model_pred < cf
        prediction_test.append(model_pred)

    # Calculating the average prediction across the consensus of models. #TODO: change this when dealing with continuous
    avg_pred = np.zeros([len(prediction_test[0]),])
    for i in range(len(prediction_test)):
        # determining if the model prediction exceeds the threshold and adding the result (1 or 0)
        avg_pred += (prediction_test[i] >= tr[i]).reshape(-1,)

    print('avg_pred:', avg_pred)
    avg_pred = avg_pred > (len(models)//2)  # if greater than 50% of the models agree there would be a hit then that is our consensus value
    print('avg_pred:', avg_pred)

    if len(models) > 1:
        fpr_te_avg, tpr_te_avg, thresh_te_avg = roc_curve(y_test_cf, avg_pred)
    else:
        fpr_te_avg, tpr_te_avg, thresh_te_avg = roc_curve(y_test_cf, prediction_test[0])
    
    pr_te_avg = precision_score(y_test_cf, avg_pred)
    re_te_avg = recall_score(y_test_cf, avg_pred)   # TODO: make sure avg_pred is calc properly

    auc_te_avg = auc(fpr_te_avg,tpr_te_avg)
    pos_ct_orig = np.sum(y_test_cf)
    t_train_mol = len(y_test)   

    print(cf,re_te_avg,pr_te_avg,auc_te_avg,pos_ct_orig/t_train_mol)

    total_left_te = re_te_avg*pos_ct_orig/pr_te_avg*total_mols*1000000/t_train_mol  # molecules left
    all_sc[cf] = [cf,re_te_avg,pr_te_avg,auc_te_avg,total_left_te]
    cf_with_left[cf] = total_left_te

print(cf_with_left)

min_left_cf = total_mols*1000000
cf_to_use = 0
# cf_to_use is the cf for the lowest num molec
for key in cf_with_left:
    if cf_with_left[key] <= min_left_cf:
        min_left_cf = cf_with_left[key]
        cf_to_use = key

print('cf_to_use:', cf_to_use)
print('min_left_cf:', min_left_cf)
print('all_sc:', all_sc)

with open(SAVE_PATH+'/iteration_'+str(n_iteration)+'/best_model_stats.txt', 'w') as ref:
    cf,re,pr,auc,tot_le = all_sc[cf_to_use]
    m_string = "* Best Model Stats * \n"
    m_string += "_" * 20 + "\n"
    m_string += "- Model Cutoff: " + str(cf) + "\n"
    m_string += "- Model Precision: " + str(pr) + "\n"
    m_string += "- Model Recall: " + str(re) + "\n"
    m_string += "- Model Auc: " + str(auc) + "\n"
    m_string += "- Total Left Testing: " + str(tot_le) + "\n"

    ref.write(m_string)

try:
    os.mkdir(SAVE_PATH+'/iteration_'+str(n_iteration)+'/best_models')
except OSError:  # catching file exists error
    pass

for models_cf in model_to_use_with_cf:  # looping through the groups of models that match that cutoff
    if models_cf[0] == cf_to_use:
        count = 0
        # Looping through all the models in that group
        for mod_no in models_cf[-1]:
            with open(SAVE_PATH+'/iteration_'+str(n_iteration)+'/best_models/thresholds.txt', 'w') as ref:
                ref.write(str(mod_no)+','+str(main_thresholds[cf_to_use][count])+','+str(cf_to_use)+'\n')
            
            # Copying the specific models by their model_no to the best_models folder
            copy2(path_to_model+'/model_'+str(mod_no), SAVE_PATH+'/iteration_'+str(n_iteration) + '/best_models/')
            copy2(path_to_model+'/model_'+str(mod_no) + ".ddss", SAVE_PATH+'/iteration_'+str(n_iteration) + '/best_models/')

            count += 1


# Deleting all other models that are not in the model_to_use array
# placed at the bottom so we don't lose all our data each time we run phase 5 and it fails.
for f in glob.glob(SAVE_PATH+'/iteration_'+str(n_iteration)+'/all_models/*'):
    try:
        mn = int(f.split('/')[-1].split('_')[1])
    except:
        mn = int(f.split('/')[-1].split('_')[1].split('.')[0])
    found = False
    for models in model_to_use_with_cf:
        if mn in models[-1]:
            found = True
            break
    if not found and "." in f:
        os.remove(f)
