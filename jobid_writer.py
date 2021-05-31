import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-pt','--project_name',required=True)
parser.add_argument('-fp','--file_path',required=True)
parser.add_argument('-n_it','--iteration_no',required=True)
### Use one param instead? 
parser.add_argument('-jid','--job_id',required=True)  # SLURM_JOB_NAME
parser.add_argument('-jn','--job_name',required=True) # SLURM_JOB_NAME.sh

io_args = parser.parse_args()
protein = io_args.project_name
file_path = io_args.file_path
n_it = int(io_args.iteration_no)
job_id = io_args.job_id
job_name = io_args.job_name

if n_it!=-1:   # creating the job directory
   try:
      os.mkdir(file_path+'/'+protein+'/iteration_'+str(n_it))
   except OSError: # file already exists
      pass
   with open(file_path+'/'+protein+'/iteration_'+str(n_it)+'/'+job_name,'w') as ref:
      ref.write(job_id+'\n')

else:    # When n_it == -1 we create a seperate directory (for jobs that occur after an iteration)
   try:
      os.mkdir(file_path+'/'+protein+'/after_iteration')
   except OSError:
      pass
   with open(file_path+'/'+protein+'/after_iteration'+'/'+job_name,'w') as ref:
      ref.write(job_id+'\n')

