#! /bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=23:59:00
#SBATCH --job-name=xxxxq-yyyy

echo "Present working Directory is: `pwd`"
source /home/a/aspuru/abhinav/.bashrc
conda activate quantum_tequila

module load gcc

export PYTHONPATH=${PYTHONPATH}:/scratch/a/aspuru/abhinav/tequila/src/
VAR1="yyyy"
VAR2="random_unary"
VAR3="random"
if [ "$VAR1" = "$VAR2" ]; then
    echo "preparing random states"
    x=1
    while [ $x -le 100 ]
    do
      python GA.py >> output_${x}.txt
      x=$(( $x + 1 ))
    done
elif [ "$VAR1" = "$VAR3" ]; then
    echo "preparing product states"
    x=1
    while [ $x -le 100 ]
    do
      python GA.py >> output_${x}.txt
      x=$(( $x + 1 ))
    done
else
    echo "preparing yyyy"
    python GA.py > output_yyyy.txt
fi
