#!/bin/bash
#SBATCH -p defaultp # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -c 4 # number of cpus (cores)
#SBATCH --mem 150G # memory pool for all cores # 32GB for folktables ablation
#SBATCH --time 0-1:59:00 # time (D-HH:MM:SS)
#SBATCH --job-name=clt-data
#SBATCH -o world.%A_%a.%N.out # STDOUT
#SBATCH -e world.%A_%a.%N.err # STDERR
#SBATCH --requeue

SEED=${SLURM_ARRAY_TASK_ID:-0}
SEEDSTR=$( printf "%01d" $SEED )

hostname
date

module load python/3.8
source ~/twitter-env/bin/activate

export OMP_NUM_THREADS=4

ARG=( "$@" )
FILE="${ARG[0]}"

if [ -e ${FILE} ]
then
    echo ${FILE} exists
    stat -L -c "%a %G %U" ${FILE}
    cd ${HOME}/geo-twitter/
    srun python -u data-collector.py ${ARG[*]}
else
    echo ${FILE} does not exist
fi

date
