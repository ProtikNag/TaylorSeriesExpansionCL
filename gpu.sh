#!/bin/sh
#SBATCH --job-name=er
#SBATCH -N 1         			## Compute None (Number of computers)
#SBATCH -n 24 	     			## CPU Cores
#SBATCH --gres=gpu:1 			## Run on 2 GPUs
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p dgx_aic

hostname
date

export CUDA_VISIBLE_DEVICES=0

module load cuda/12.3
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env/
python --version

cd /work/pnag/TaylorSeriesExpansionCL/
python main.py \
  --dataset CIFAR100 \
  --baseline ser \
  --levels 3 \
  --epochs 25 \
  --lr 0.01 \
  --batch-size 128 \
  --buffer-size 2000 \
  --catchup-epochs 10 \
  --group-size 3
