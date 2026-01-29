# HTCL: Hierarchical Taylor Series Continual Learning

A modular framework for continual learning research that couples fast local adaptation with conservative, second-order global consolidation using Taylor series expansions.

## 🎯 Key Features

- **Multi-level hierarchy** for knowledge consolidation across temporal scales
- **Taylor series-based updates** using second-order approximations
- **Multiple baseline methods** (ER, SER, DER, EWC, iCaRL) with easy extensibility
- **Taylor-based catch-up mechanism** for global model synchronization (keeps best performing model)
- **Organized output structure** by dataset and baseline method
- **Publication-quality visualizations** with timing comparisons

## 📁 Project Structure

```
TaylorSeriesExpansionCL/
├── main.py                 # CLI entry point
├── diagnose.py             # Import diagnostic tool
├── requirements.txt        # Dependencies
├── README.md
├── .gitignore
└── htcl/                   # Main package
    ├── __init__.py
    ├── config/             # Configuration dataclasses
    ├── data/               # Dataset implementations
    ├── models/             # Neural network architectures
    ├── methods/            # CL algorithms
    │   ├── buffer.py       # Replay buffer
    │   ├── er.py           # Experience Replay baseline
    │   ├── ser.py          # Strong Experience Replay baseline
    │   ├── der.py          # Dark Experience Replay baseline
    │   ├── ewc.py          # Elastic Weight Consolidation baseline
    │   ├── icarl.py        # iCaRL baseline
    │   └── htcl.py         # HTCL implementation
    ├── utils/              # Helper functions
    ├── visualization/      # Plotting functions
    └── experiments/        # Experiment runners
```

## 📂 Output Organization

Results are automatically organized by dataset and baseline method:

```
results/
├── splitmnist/
│   ├── er/
│   │   ├── csv/
│   │   │   ├── er_results_SplitMNIST.csv
│   │   │   ├── htcl_er_L2_results_SplitMNIST.csv
│   │   │   └── htcl_er_L3_results_SplitMNIST.csv
│   │   ├── json/
│   │   ├── plots/
│   │   │   ├── png/
│   │   │   └── svg/
│   │   └── checkpoints/
│   ├── ser/
│   ├── der/
│   ├── ewc/
│   └── icarl/
├── cifar100/
│   ├── er/
│   ├── ser/
│   ├── der/
│   ├── ewc/
│   └── icarl/
└── ...
```

## 🚀 Quick Start

### Installation

```bash
cd TaylorSeriesExpansionCL
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run with ER baseline on SplitMNIST (debug mode)
python main.py --dataset SplitMNIST --baseline er --levels 2 3 --debug

# Run with SER baseline on CIFAR-100
python main.py --dataset CIFAR100 --baseline ser --levels 2 3 4 --debug

# Run with EWC baseline on SplitMNIST
python main.py --dataset SplitMNIST --baseline ewc --levels 2 3 --debug

# Run with iCaRL baseline on SplitMNIST
python main.py --dataset SplitMNIST --baseline icarl --levels 2 3 --debug

# List available baseline methods
python main.py --list-baselines

# Quick test
python main.py --quick-test --baseline er
```

### Full Experiment

```bash
# Full experiment on SplitMNIST with SER baseline
python main.py \
  --dataset SplitMNIST \
  --baseline ser \
  --levels 2 3 4 5 \
  --epochs 10 \
  --lr 0.01 \
  --batch-size 32 \
  --buffer-size 50 \
  --catchup-epochs 5 \
  --seed 42 \
  --output-dir ./results/ 
```

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Dataset to use: `SplitMNIST`, `CIFAR100`, `20Newsgroups`, `Cora` | `SplitMNIST` |
| `--baseline` | Baseline method: `er`, `ser`, `der`, `ewc`, `icarl` | `er` |
| `--levels` | Hierarchy levels to compare (e.g., `2 3 4 5`) | `2 3` |
| `--epochs` | Training epochs per task | `5` |
| `--lr` | Learning rate | `0.01` |
| `--batch-size` | Batch size for training | `32` |
| `--buffer-size` | Replay buffer capacity | `500` |
| `--group-size` | Tasks per group for permutation search | `2` |
| `--num-perms` | Number of canonical permutations to evaluate | `20` |
| `--catchup-epochs` | Taylor-based catch-up iterations | `2` |
| `--no-catchup` | Disable catch-up mechanism | `False` |
| `--debug` | Use smaller dataset for faster testing | `False` |
| `--output-dir` | Base output directory | `./results` |
| `--seed` | Random seed for reproducibility | `42` |
| `--list-baselines` | Show available baseline methods and exit | - |
| `--quick-test` | Run a minimal test to verify setup | - |

## 🔬 Available Baseline Methods

### Experience Replay (ER)
Standard replay-based continual learning with reservoir sampling.
```bash
python main.py --baseline er --dataset SplitMNIST --levels 2 3
```

### Strong Experience Replay (SER)
Enhanced replay with knowledge distillation from stored logits:
- Higher replay weight (beta=1.0)
- Temperature-scaled soft targets
- Combined CE + distillation loss

```bash
python main.py --baseline ser --dataset SplitMNIST --levels 2 3
```

### Dark Experience Replay (DER)
Stores and matches network logits throughout the optimization trajectory:
- MSE loss on stored logits
- Captures "dark knowledge" from training dynamics

Reference: Buzzega et al., "Dark Experience for General Continual Learning", NeurIPS 2020

```bash
python main.py --baseline der --dataset SplitMNIST --levels 2 3
```

### Elastic Weight Consolidation (EWC)
Regularization-based method using Fisher Information:
- Penalizes changes to important parameters
- Uses diagonal Fisher Information Matrix approximation
- L_total = L_current + (λ/2) * Σ F_i * (θ_i - θ*_i)²

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017

```bash
python main.py --baseline ewc --dataset SplitMNIST --levels 2 3
```

### iCaRL (Incremental Classifier and Representation Learning)
Exemplar-based method with herding selection and distillation:
- Nearest-mean-of-exemplars classification
- Herding-based exemplar selection
- Knowledge distillation from previous model

Reference: Rebuffi et al., "iCaRL: Incremental Classifier and Representation Learning", CVPR 2017

```bash
python main.py --baseline icarl --dataset SplitMNIST --levels 2 3
```

## 🐍 Python API

```python
from htcl import (
    run_hierarchy_experiment,
    get_mnist_config,
    list_baselines,
)

# See available baselines
print(list_baselines())  # ['er', 'ser', 'der', 'ewc', 'icarl']

# Configure experiment
config = get_mnist_config(debug=True)
config.htcl.catchup_enabled = True
config.htcl.catchup_epochs = 2

# Run with SER baseline
results = run_hierarchy_experiment(
    config=config,
    baseline="ser",                   # Select baseline method
    hierarchy_levels=[2, 3, 4, 5],
    create_visualizations=True,
)

# Access results
print(f"SER: {results['baseline_results']['summary']['mean_accuracy']:.2f}%")
for level, htcl_r in results['htcl_results_by_level'].items():
    print(f"SER+HTCL-L{level}: {htcl_r['summary']['mean_accuracy']:.2f}%")
```

## 🔧 Key Features Explained

### 1. Taylor-Based Global Model Catch-up

The global model uses conservative Taylor updates that can lag on recent tasks. Our catch-up mechanism maintains consistency by using the same Taylor update rule and automatically keeps the best performing model across iterations:

```python
config.htcl.catchup_enabled = True
config.htcl.catchup_epochs = 2  # Number of Taylor update iterations
```

### 2. Multi-Level Hierarchy

Compare different hierarchy depths to find the optimal configuration:

```python
results = run_hierarchy_experiment(
    config=config,
    baseline="ewc",  # Can use any baseline
    hierarchy_levels=[2, 3, 4, 5],  # Test 2 to 5 levels
)
```

### 3. Automatic Visualizations with Timing

All experiments generate publication-quality plots in PNG (300 DPI) and SVG:

- Task accuracy comparison (box plots)
- Hierarchy comparison (bar charts)  
- Task-order robustness (violin plots)
- Per-task accuracy trends (line plots)
- **Taskwise variance** (bar chart and heatmap)
- **Time comparison** (baseline vs HTCL variants)
- **Hierarchy time comparison** (by depth)
- Accuracy heatmaps (all permutations × tasks)
- Comprehensive 2x2 summary figure

## 📊 Supported Datasets

| Dataset | Tasks | Classes/Task | Domain |
|---------|-------|--------------|--------|
| SplitMNIST | 5 | 2 | Image |
| CIFAR-100 | 10 | 10 | Image |
| 20Newsgroups | 5 | 4 | Text |
| Cora | 3 | ~3 | Graph |

## 🛠 Troubleshooting

### Import errors
```bash
python diagnose.py  # Run diagnostic script
```

### Missing dependencies
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

### CUDA out of memory
- Use `--debug` flag for smaller datasets
- Reduce `--batch-size`
- Reduce `--buffer-size`

## 📖 Citation

```bibtex

```

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.2.0

## 📄 License

MIT License
