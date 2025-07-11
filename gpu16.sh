#!/bin/sh
#SBATCH --job-name=linear
#SBATCH -N 1         			## Compute None (Number of computers)
#SBATCH -n 24 	     			## CPU Cores 
#SBATCH --gres=gpu:2 			## Run on 2 GPUs
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
#SBATCH -p gpu-v100-16gb		

hostname
date

export CUDA_VISIBLE_DEVICES=0,1

module load cuda/12.3
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env/
python --version

cd /work/pnag/TaylorSeriesExpansionCL/
python main.py --group_size 2 --epochs 30 --lr 0.01 --methods linear

