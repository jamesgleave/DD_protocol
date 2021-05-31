import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('-pt','--project_name',required=True,help='name of project')
parser.add_argument('-fp','--file_path',required=True,help='path to the project folder')
parser.add_argument('-gf','--grid_file',required=True,help='docking grid file (zip)')
parser.add_argument('-n_it','--iteration_no',required=True,help='current iteration')
parser.add_argument('-g_in','--glide_input',required=True,help='template for glide input file')

io_args = parser.parse_args()
protein = io_args.project_name
file_path = io_args.file_path
n_it = int(io_args.iteration_no)
gf = io_args.grid_file
g_in = io_args.glide_input

if n_it!=-1:
    try:
       os.mkdir(file_path+'/'+protein+'/iteration_'+str(n_it)+'/docked')
    except:
       pass
else:
    try:
       os.mkdir(file_path+'/'+protein+'/after_iteration/to_dock'+'/docked')
    except:
       pass

if n_it!=-1:
    ligandfile = file_path+'/'+protein+'/iteration_'+str(n_it)+'/sdf/*'
else:
    ligandfile = file_path+'/'+protein+'/after_iteration/to_dock'+'/sdf/*'

for f in glob.glob(ligandfile):
    if n_it!=-1:
        name=f.split('/')[-1].split('_')[0]+'_docked'
    else:
        name=f.split('/')[-1].split('.')[0]+'_docked'
    if n_it!=-1:
        ptn = file_path+'/'+protein+'/iteration_'+str(n_it)+'/docked/'+name+'.in'
    else:
        ptn = file_path+'/'+protein+'/after_iteration/to_dock'+'/docked/'+name+'.in'
    ref1 =  open(ptn,'w')
    with open(g_in,'r') as ref:
         for line in ref:
             if 'GRIDFILE' in line:
                 ref1.write('GRIDFILE '+gf+'\n')
             elif 'LIGANDFILE' in line:
                 ref1.write('LIGANDFILE '+f+'\n')
             else:
                 ref1.write(line)
    ref1.close()
