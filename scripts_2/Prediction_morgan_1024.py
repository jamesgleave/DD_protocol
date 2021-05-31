import argparse
import glob
import os
import time
import warnings
import numpy as np
import pandas as pd
from ML.DDModel import DDModel

try:
    import __builtin__
except ImportError:
    # Python 3
    import builtins as __builtin__

# For debugging purposes only:
def print(*args, **kwargs):
    __builtin__.print('\t sampling: ', end="")
    return __builtin__.print(*args, **kwargs)

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-fn','--fn', required=True)
parser.add_argument('-protein', '--protein', required=True)
parser.add_argument('-it', '--it', required=True)
parser.add_argument('-file_path', '--file_path', required=True)
parser.add_argument('-mdd', '--morgan_directory', required=True)

io_args = parser.parse_args()
fn = io_args.fn
protein = str(io_args.protein)
it = int(io_args.it)
file_path = io_args.file_path
mdd = io_args.morgan_directory

# This debug feature will allow for speedy testing
DEBUG=False
def prediction_morgan(fname, models, thresh):   # TODO: improve runtime with parallelization across multiple nodes
    print("Starting Predictions...")
    t = time.time()
    per_time = 1000000
    n_features = 1024
    z_id = []
    X_set = np.zeros([per_time, n_features])
    total_passed = 0

    print("We are predicting from the file", fname, "located in", mdd)
    with open(mdd+'/'+fname,'r') as ref:
        no = 0
        for line in ref:
            tmp = line.rstrip().split(',')
            on_bit_vector = tmp[1:]
            z_id.append(tmp[0])
            for elem in on_bit_vector:
                X_set[no,int(elem)] = 1
            no+=1
            if no == per_time:
                X_set = X_set[:no, :]
                pred = []
                print("We are currently running line", line)
                print("(1) Predicting... Time elapsed:", time.time() - t, "seconds.")
                for model in models:
                    pred.append(model.predict(X_set))

                with open(file_path+'/iteration_'+str(it)+'/morgan_1024_predictions/'+fname, 'a') as ref:
                    for j in range(len(pred[0])):
                        is_pass = 0
                        for i,thr in enumerate(thresh):
                            if float(pred[i][j])>thr:
                                is_pass += 1
                        if is_pass >= 1:
                            total_passed += 1
                            line = z_id[j]+','+str(float(pred[i][j]))+'\n'
                            ref.write(line)
                X_set = np.zeros([per_time,n_features])
                z_id = []
                no = 0

                # With debug, we will only predict on 'per_time' molecules
                if DEBUG:
                    break

        if no != 0:
            X_set = X_set[:no,:]
            pred = []
            print("We are currently running line", line)
            print("(2) Predicting... Time elapsed:", time.time() - t, "seconds.")
            for model in models:
                pred.append(model.predict(X_set))
            with open(file_path+'/iteration_'+str(it)+'/morgan_1024_predictions/'+fname, 'a') as ref:
                for j in range(len(pred[0])):
                    is_pass = 0
                    for i,thr in enumerate(thresh):
                        if float(pred[i][j])>thr:
                            is_pass+=1
                    if is_pass>=1:
                        total_passed+=1
                        line = z_id[j]+','+str(float(pred[i][j]))+'\n'
                        ref.write(line)
    print("Prediction time:", time.time() - t)
    return total_passed


try:
    os.mkdir(file_path+'/iteration_'+str(it)+'/morgan_1024_predictions')
except OSError:
    print(file_path+'/iteration_'+str(it)+'/morgan_1024_predictions', "already exists")

thresholds = pd.read_csv(file_path+'/iteration_'+str(it)+'/best_models/thresholds.txt', header=None)
thresholds.columns = ['model_no', 'thresh', 'cutoff']

tr = []
models = []
for f in glob.glob(file_path+'/iteration_'+str(it)+'/best_models/model_*'):
    if "." not in f:    # skipping over the .ddss & .csv files
        mn = int(f.split('/')[-1].split('_')[1])
        tr.append(thresholds[thresholds.model_no == mn].thresh.iloc[0])
        models.append(DDModel.load(file_path+'/iteration_'+str(it)+'/best_models/model_'+str(mn)))

print("Number of models to predict:", len(models))
t = time.time()
returned = prediction_morgan(fn, models, tr)
print(time.time()-t)

with open(file_path+'/iteration_'+str(it)+'/morgan_1024_predictions/passed_file_ct.txt','a') as ref:
        ref.write(fn+','+str(returned)+'\n')
