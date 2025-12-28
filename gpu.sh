#!/bin/sh
#SBATCH --job-name=fl_comparison
#SBATCH -N 1         			## Compute None (Number of computers)
#SBATCH -n 24 	     			## CPU Cores
#SBATCH --gres=gpu:1 			## Run on 1 GPU
#SBATCH --output job%j.%N.out
#SBATCH --error job%j.%N.err
########SBATCH -p dgx_aic
#SBATCH -p gpu

hostname
date

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda/12.3
module load python3/anaconda/2023.9
source activate /work/pnag/envs/ml_env/
python --version

cd /work/pnag/TaylorSeriesExpansionCL/

echo "=============================================="
echo "FL Comparison Experiment: HTCL vs FedAvg vs FedProx"
echo "Dataset: SplitMNIST"
echo "Baselines: SER, DER"
echo "=============================================="

# Run FL comparison experiment on SplitMNIST with both SER and DER baselines
python run_fl_experiment.py \
  --dataset SplitMNIST \
  --baselines ser der \
  --levels 2 \
  --debug \
  --epochs 10 \
  --lr 0.01 \
  --batch-size 64 \
  --buffer-size 50 \
  --catchup-epochs 10 \
  --seed 42 \
  --num-perms 60

echo "=============================================="
echo "FL Comparison Experiment Completed!"
echo "Results saved to: ./results/splitmnist/fl_comparison/"
echo "=============================================="

date
