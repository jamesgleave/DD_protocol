#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --job-name=phase_3

t_nod=$2

file_path=`sed -n '1p' $3/$4/logs.txt`
protein=`sed -n '2p' $3/$4/logs.txt`
grid_file=`sed -n '3p' $3/$4/logs.txt`

morgan_directory=`sed -n '4p' $3/$4/logs.txt`
smile_directory=`sed -n '5p' $3/$4/logs.txt`

cpu_part=$5

python jobid_writer.py -pt $protein -fp $file_path -n_it $1 -jid phase_3_new -jn phase_3.txt

cd $file_path/$protein/iteration_$1
mkdir docked
cd docked

for f in ../sdf/*.oeb.gz
do

    temp=$(echo $f | rev | cut -d'/' -f1 | rev)
    temp=$(echo $temp | cut -d'_' -f1)
    echo "#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

echo \$1
echo \$2
echo \$3

\$openeye fred -receptor \$1 -dbase \$2 -docked_molecule_file phase_3_\$3\_docked.sdf -hitlist_size 0 -mpi_np \$4 -prefix \$3
rm *undocked*sdf">>$temp'_'docking.sh
    
    sbatch -J $SLURM_JOB_NAME -p $cpu_part -c $t_nod $temp'_'docking.sh $grid_file $f $temp $t_nod

done
wait
 
scancel $SLURM_JOBID
