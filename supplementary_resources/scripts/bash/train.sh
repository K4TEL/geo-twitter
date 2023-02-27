#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=GTX1080Ti
#SBATCH --cpus-per-task=12
#SBATCH --export=NONE
#SBATCH --mem 100G # memory pool for all cores # 32GB for folktables ablation
#SBATCH --time 10-00:00:00 # time (D-HH:MM:SS)
#SBATCH --job-name=world-model
#SBATCH -o world-model.%A_%a.%N.out # STDOUT
#SBATCH -e world-model.%A_%a.%N.err # STDERR
#SBATCH --requeue

unset SLURM_EXPORT_ENV

SEED=${SLURM_ARRAY_TASK_ID:-0}
SEEDSTR=$( printf "%01d" $SEED )

hostname
date

module load python/3.8
module load cuda
source ${HOME}/twitter-env/bin/activate

export OMP_NUM_THREADS=12
export CUDA_VISIBLE_DEVICE=0

cd ${HOME}/geo-twitter/

ARG=( "$@" )
srun --cpu_bind=verbose python -u train_bert.py ${ARG[*]}

date
