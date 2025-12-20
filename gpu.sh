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
  --dataset SplitMNIST \
  --baseline ser \
  --levels 2 3 4 5 \
  --debug \
  --epochs 10 \
  --lr 0.01 \
  --batch-size 64 \
  --buffer-size 50 \
  --catchup-epochs 5 \
  --seed 42

python main.py \
  --dataset SplitMNIST \
  --baseline er \
  --levels 2 3 4 5 \
  --debug \
  --epochs 10 \
  --lr 0.01 \
  --batch-size 64 \
  --buffer-size 50 \
  --catchup-epochs 5 \
  --seed 42

python main.py \
  --dataset SplitMNIST \
  --baseline der \
  --levels 2 3 4 5 \
  --debug \
  --epochs 10 \
  --lr 0.01 \
  --batch-size 64 \
  --buffer-size 50 \
  --catchup-epochs 5 \
  --seed 42