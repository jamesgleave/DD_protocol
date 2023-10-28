#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH -n 3
#SBATCH -N 3
#SBATCH --mem=0 
#SBATCH --job-name=phase_2

t_nod=$2

file_path=`sed -n '1p' $3/$4/logs.txt`
protein=`sed -n '2p' $3/$4/logs.txt`

morgan_directory=`sed -n '4p' $3/$4/logs.txt`
smile_directory=`sed -n '5p' $3/$4/logs.txt`

cpu_part=$5

python jobid_writer.py -pt $protein -fp $file_path -n_it $1 -jid $SLURM_JOB_NAME -jn $SLURM_JOB_NAME.txt

cd $file_path/$protein/iteration_$1
mkdir sdf
for f in smile/*
do
   tmp="$(cut -d'/' -f2 <<<"$f")"
   tmp="$(cut -d'_' -f1 <<<"$tmp")"
   if [ $tmp = train ];then name=training;fi
   if [ $tmp = valid ];then name=validation;fi
   if [ $tmp = test ];then name=testing;fi
   echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

echo \$1
echo \$2
echo \$3
oeomega classic -in \$1 -out sdf/\$2\_sdf.sdf -maxconfs 1 -strictstereo false -mpi_np \$3 -log \$2.log -prefix \$2 -warts false">>$name'_'conf.sh

   sbatch -J $SLURM_JOB_NAME -p $cpu_part -c $t_nod $name'_'conf.sh $f $name $t_nod
done
wait

scancel $SLURM_JOBID
