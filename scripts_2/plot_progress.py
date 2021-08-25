import argparse
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import glob
import os
from ML.DDModel import DDModel
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


parser = argparse.ArgumentParser()
parser.add_argument('-pr','--project',required=True,help='Location of project')
parser.add_argument('-sz_test','--size_test_set',required=True,help='Number of molecules in the test set')
parser.add_argument('-it_1','--start_iteration',required=True,help='Number of first iteration to analyze')
parser.add_argument('-it_2','--end_iteration',required=True,help='Number of last iteration to analyze')
parser.add_argument('-fo','--output_folder',required=True,help='Folder where to output the figures')

io_args = parser.parse_args()
path = io_args.project
num_molec = int(io_args.size_test_set)
it_1 = int(io_args.start_iteration)
it_2 = int(io_args.end_iteration)
path_out = io_args.output_folder

def get_zinc_and_labels(zinc_path, labels_path):
    ids = []
    with open(zinc_path,'r') as ref:
        for line in ref:
            ids.append(line.split(',')[0])
    zincIDs = pd.DataFrame(ids, columns=['ZINC_ID'])

    labels_df = pd.read_csv(labels_path, header=0)
    combined_df = pd.merge(labels_df, zincIDs, how='inner', on=['ZINC_ID'])
    return combined_df.set_index('ZINC_ID')

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

#Plot ROC curves

from matplotlib.pyplot import figure
figure(figsize=(5, 3), dpi=300)

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR')
plt.xlabel('FPR')

zinc_labels_test = get_zinc_and_labels(path + '/iteration_1/morgan/test_morgan_1024_updated.csv', path +'/iteration_1/testing_labels.txt')
X_test, y_test = get_all_x_data(path +'/iteration_1/morgan/test_morgan_1024_updated.csv', zinc_labels_test)

for i in range(it_1, it_2 + 1):
    mod_name = str(glob.glob(path + '/iteration_%i/best_models/model_*.ddss'%i)).split('/')[-1].split('.')[0]
    print('Now processing iteration %i'%i)
    model = DDModel.load(path +'/iteration_%i/best_models/%s'%(i, mod_name))
    thr = pd.read_csv(path +'/iteration_%i/best_models/thresholds.txt'%i, names=['n','prob','score'])
    prob = float(thr['prob'])
    score = float(thr['score'])
    y_test_cf = y_test<score
    model_pred = model.predict(X_test)
    fpr, tpr, threshold = roc_curve(y_test_cf, model_pred, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = 'AUC it. %i = %0.3f'%(i, roc_auc))

plt.legend(loc = 'lower right')    
plt.tight_layout()
plt.savefig(path_out + '/ROC.png')

#Plot number of molecules
it = []
count = []
for i in range(it_1, it_2 + 1):
    count_mol = pd.read_csv(path + '/iteration_%i/morgan_1024_predictions/passed_file_ct.txt'%i, names=['file','count'])
    total_it = count_mol['count'].sum()/1000000
    it.append(i)
    count.append(total_it)    

plt.plot(it, count, color='green', marker='o')
plt.xlim([it_1, it_2])
plt.ylim([0, max(count)+10])
plt.xticks(np.arange(it_1, it_2+0.01, step=1))
plt.ylabel('Remaining molecules, M')
plt.xlabel('Iteration')
plt.legend('',frameon=False)
plt.tight_layout()
plt.savefig(path_out + '/n_mol.png')
