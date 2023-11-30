import numpy as np
import pandas as pd
import glob

import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--folder", required=True,help='Path to project folder, including project folder name')
parser.add_argument('-n_it','--n_iteration',required=True,help='Number of current iteration')

io_args = parser.parse_args()
n_it = int(io_args.n_iteration)
protein = str(io_args.folder).split('/')[-1]

ITER_PATH =  protein + '/iteration_' + str(n_it)



files_remained = glob.glob(ITER_PATH + '/morgan_1024_predictions/chunk*')
print(files_remained)

smi_remained = []
for file in files_remained:
    df_remained = pd.read_csv(file, header = None)
    df_remained.columns = ['Smiles', 'prediction']
    smi_remained.append(df_remained['Smiles'])
    print(f'Smiles from {file} collected')

smi_remained_flat = [item for sublist in smi_remained for item in sublist]
smi_remained_flat = set(smi_remained_flat)
total_left = len(smi_remained_flat)

topN = [100, 1000, 10000, 50000]
lst_found, lst_lost = [], []
for i in topN:
    df_topN = pd.read_csv(f"D4_Glide_top{i}.csv")
    smi_topN = df_topN['Smiles'][:i]
    lst_intersect = set(smi_topN).intersection(smi_remained_flat)
    count_found = len(lst_intersect)
    count_lost = i - count_found
    lst_found.append(count_found) 
    lst_lost.append(count_lost)

df_analysis = pd.DataFrame(list(zip(topN, lst_found, lst_lost)), columns = ['TopN', 'Remained', 'Lost'])
df_analysis.to_csv(f'analysis_{protein[5:]}.csv')

print('Total remained: ', total_left)
print(df_analysis)
