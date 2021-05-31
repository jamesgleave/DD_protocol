#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=prepare_states

name_in=$(echo $1 | cut -d'.' -f1)
name_out=$(echo $name_in | rev | cut -d'/' -f1 | rev)

echo "Calculating states for $name_in"

start=`date +%s`

mkdir -p $2

$openeye flipper -in $1 -out $name_in'_'isom.smi -warts
wait

$openeye tautomers -in $name_in'_'isom.smi -out $name_in'_'states.smi -maxtoreturn 1 -warts false
wait

rm $name_in'_'isom.smi
mv $name_in'_'states.smi $2/$name_out'.'txt

end=`date +%s`
runtime=$((end-start))
echo $runtime
