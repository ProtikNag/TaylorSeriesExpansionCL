#!/bin/sh
#SBATCH --job-name=htcl_experiment
#SBATCH -N 1         			## Compute Node (Number of computers)
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

# =============================================================================
# HTCL Experiment Configuration
# =============================================================================
# Available baselines: er, ser, der, ewc, icarl
# Available datasets: SplitMNIST, CIFAR100, 20Newsgroups, Cora
# =============================================================================

# Configuration variables (modify these as needed)
DATASET="SplitMNIST"
BASELINE="ser"  # Options: er, ser, der, ewc, icarl
LEVELS="2 3"
EPOCHS=10
LR=0.01
BATCH_SIZE=64
BUFFER_SIZE=50
CATCHUP_EPOCHS=10
SEED=42
NUM_PERMS=60

echo "=============================================="
echo "HTCL Experiment"
echo "Dataset: ${DATASET}"
echo "Baseline: ${BASELINE}"
echo "Hierarchy Levels: ${LEVELS}"
echo "=============================================="

# Run the experiment
python main.py \
  --dataset ${DATASET} \
  --baseline ${BASELINE} \
  --levels ${LEVELS} \
  --debug \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --batch-size ${BATCH_SIZE} \
  --buffer-size ${BUFFER_SIZE} \
  --catchup-epochs ${CATCHUP_EPOCHS} \
  --seed ${SEED} \
  --num-perms ${NUM_PERMS}

echo "=============================================="
echo "Experiment Completed!"
echo "Results saved to: ./results/${DATASET,,}/${BASELINE}/"
echo "=============================================="

date

# =============================================================================
# Example configurations for different baselines:
# =============================================================================
# 
# # Experience Replay (ER)
# python main.py --dataset SplitMNIST --baseline er --levels 2 3 --debug
#
# # Strong Experience Replay (SER)
# python main.py --dataset SplitMNIST --baseline ser --levels 2 3 --debug
#
# # Dark Experience Replay (DER)
# python main.py --dataset SplitMNIST --baseline der --levels 2 3 --debug
#
# # Elastic Weight Consolidation (EWC)
# python main.py --dataset SplitMNIST --baseline ewc --levels 2 3 --debug
#
# # iCaRL
# python main.py --dataset SplitMNIST --baseline icarl --levels 2 3 --debug
#
# =============================================================================
# Multi-baseline comparison (run sequentially):
# =============================================================================
#
# for BASELINE in er ser der ewc icarl; do
#   echo "Running ${BASELINE}..."
#   python main.py --dataset SplitMNIST --baseline ${BASELINE} --levels 2 3 --debug
# done
#
# =============================================================================
