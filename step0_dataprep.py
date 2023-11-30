import pandas as pd
import numpy as np
import os
import csv
import glob
import time
import pickle
from contextlib import closing
from multiprocessing import Pool
import multiprocessing
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from functools import partial
import shutil

smiles_lib = 'smiles_library'
if not os.path.exists(smiles_lib):
    os.mkdir(smiles_lib)

csvfiles = glob.glob("../../data/Enamine/enamine350-3/enamine350-3_*csv")

print('Copying files from data')
for fn in csvfiles:
    fn_split = fn.split('/')[-1]
    outname = fn_split.replace('.csv', '.txt')
    cmd = f"cp {fn} {smiles_lib}/chunk_{outname}"
    print(fn_split)
    os.system(cmd)


def morgan_fingp(fname, fn='morgan_library'):
    nbits=1024
    radius=2
    fsplit = fname.split('/')[-1]
    ref2  = open(fn+'/'+fsplit,'a')
    df_smi = pd.read_csv(fname)

    for i in range(df_smi.shape[0]):
        smile = df_smi['Smiles'][i]
        arg = np.zeros((1,))
        try:
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile),radius,nBits=nbits,useChirality=True),arg)

            ref2.write((',').join([smile]+[str(elem) for elem in np.where(arg==1)[0]]))
            ref2.write('\n')
        except:
            print(line)
            pass

files = []
for f in glob.glob(smiles_lib+'/chunk*.txt'):
    files.append(f)

fps_lib = 'morgan_library'
if not os.path.exists(fps_lib):
    os.mkdir(fps_lib)


print('Generating Morgan fingerprints')
t_f = len(files)
t = time.time()

with closing(Pool(np.min([multiprocessing.cpu_count(),10]))) as pool:
    pool.map(morgan_fingp,files)
    print(time.time()-t)

print('total time spent')
print(time.time()-t)


