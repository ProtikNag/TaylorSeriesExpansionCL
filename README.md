# HTCL: Hierarchical Taylor Series Continual Learning

A modular, well-documented implementation of Hierarchical Taylor Series Continual Learning (HTCL) for research experiments.

## 🔬 Overview

HTCL addresses two key challenges in continual learning:
1. **Task-order sensitivity**: Performance varies dramatically based on the order tasks are learned
2. **Stability-plasticity dilemma**: Single models struggle to both retain old knowledge and adapt to new tasks

The framework implements:
- **Multi-level hierarchy**: Fast local adaptation with conservative global consolidation
- **Second-order Taylor updates**: Principled integration of local models into the global model
- **Global model catch-up**: Allows the global model to better adapt to recent tasks
- **Configurable L-level hierarchy**: Compare 2, 3, 4, 5+ level hierarchies

## 📁 Project Structure

```
htcl/
├── config/              # Configuration management
│   ├── __init__.py
│   └── config.py        # Dataclass-based configurations
├── data/                # Dataset implementations
│   ├── __init__.py
│   └── datasets.py      # ContinualSplitMNIST, ContinualCIFAR100, etc.
├── models/              # Neural network architectures
│   ├── __init__.py
│   └── architectures.py # SimpleResNet, SmallCNN, GCNNet, TextMLP
├── methods/             # Continual learning methods
│   ├── __init__.py
│   ├── buffer.py        # Replay buffer implementations
│   ├── er.py            # Experience Replay
│   └── htcl.py          # HTCL with Taylor updates & catch-up
├── utils/               # Utility functions
│   ├── __init__.py
│   └── helpers.py       # Evaluation, metrics, I/O
├── visualization/       # Plotting and visualization
│   ├── __init__.py
│   └── plots.py         # Publication-quality plots
├── experiments/         # Experiment runners
│   ├── __init__.py
│   └── runner.py        # High-level experiment functions
├── results/             # Output directory
│   ├── csv/            # CSV results
│   ├── plots/
│   │   ├── png/        # PNG plots
│   │   └── svg/        # SVG plots (vector format)
│   └── checkpoints/    # Model checkpoints
├── __init__.py          # Package initialization
├── main.py              # Command-line entry point
└── README.md            # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd htcl

# Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn torch-geometric
```

### Basic Usage

#### Quick Test
```bash
python main.py --quick-test
```

#### Run Hierarchy Comparison on SplitMNIST
```bash
python main.py --dataset SplitMNIST --levels 2 3 4 5 --debug
```

#### Run on CIFAR-100 (Full Dataset)
```bash
python main.py --dataset CIFAR100 --levels 2 3 --epochs 10
```

### Python API

```python
from htcl import run_hierarchy_experiment, get_mnist_config

# Get predefined configuration
config = get_mnist_config(debug=True)

# Run hierarchy comparison
results = run_hierarchy_experiment(
    config=config,
    hierarchy_levels=[2, 3, 4, 5],
    create_visualizations=True,
)

# Access results
print(f"ER Mean Accuracy: {results['er_results']['summary']['mean_accuracy']:.2f}%")
for level, htcl_results in results['htcl_results_by_level'].items():
    print(f"HTCL-L{level} Mean: {htcl_results['summary']['mean_accuracy']:.2f}%")
```

## 📊 Supported Datasets

| Dataset | Domain | Tasks | Classes/Task | Description |
|---------|--------|-------|--------------|-------------|
| SplitMNIST | Image | 5 | 2 | MNIST split into 5 binary classification tasks |
| CIFAR-100 | Image | 10 | 10 | 100 classes split into 10 tasks |
| 20Newsgroups | Text | 5 | 4 | TF-IDF features from newsgroup posts |
| Cora | Graph | 3 | ~3 | Citation network node classification |

## ⚙️ Configuration

### Command Line Options

```
--dataset       Dataset name (SplitMNIST, CIFAR100, 20Newsgroups, Cora)
--levels        Hierarchy levels to compare (e.g., 2 3 4 5)
--epochs        Training epochs per task
--lr            Learning rate
--batch-size    Batch size
--buffer-size   Replay buffer capacity
--group-size    Tasks per group for permutation search
--num-perms     Number of canonical permutations to evaluate (default: 20)
--catchup-epochs  Taylor-based catch-up iterations (default: 2)
--no-catchup    Disable catch-up mechanism
--debug         Use smaller dataset for faster testing
--output-dir    Output directory for results
--seed          Random seed for reproducibility
```

### Programmatic Configuration

```python
from htcl import ExperimentConfig, DataConfig, TrainingConfig, HTCLConfig

config = ExperimentConfig(
    data=DataConfig(
        name="SplitMNIST",
        num_tasks=5,
        batch_size=32,
        debug=True,
    ),
    training=TrainingConfig(
        num_epochs=5,
        learning_rate=0.01,
    ),
    htcl=HTCLConfig(
        num_levels=3,
        group_size=2,
        buffer_size=100,
        catchup_enabled=True,
        catchup_epochs=2,
    ),
    experiment_name="my_experiment",
)
```

## 🔧 Key Features

### 1. Taylor-Based Global Model Catch-up

The global model can lag behind on recent tasks due to conservative Taylor updates. Unlike traditional backpropagation-based fine-tuning, our catch-up mechanism uses the same Taylor series update rule to maintain consistency with HTCL's philosophy:

```python
config.htcl.catchup_enabled = True
config.htcl.catchup_epochs = 2        # Number of Taylor update iterations
```

The catch-up process:
1. Trains a temporary local model on recent tasks
2. Uses Taylor update to move global model toward this local model
3. Maintains the principled second-order update mechanism throughout

### 2. Multi-Level Hierarchy

Compare different hierarchy depths:

```python
from htcl import run_hierarchy_experiment

results = run_hierarchy_experiment(
    config=config,
    hierarchy_levels=[2, 3, 4, 5],  # Test 2 to 5 levels
)
```

### 3. Smart Canonical Permutations

For efficiency, HTCL generates only unique canonical permutations. Since permutations like `(0,1,2,3)` and `(1,0,3,2)` are equivalent under HTCL's grouping (only which tasks are grouped together matters), we avoid redundant computation:

```python
from htcl import generate_canonical_permutations

# Generate 20 unique permutations for 10 tasks with group_size=2
perms = generate_canonical_permutations(
    num_tasks=10,
    group_size=2,
    max_perms=20,
)
```

This significantly reduces compute time, especially for large numbers of tasks.

### 4. Automatic Visualization with Timing

All experiments automatically generate publication-quality plots in both PNG and SVG formats:

- Task accuracy comparison (box plots)
- Hierarchy comparison (bar charts)
- Task-order robustness (violin plots)
- Per-task accuracy trends (line plots)
- Comprehensive 2x2 summary figure
- **Time comparison** (ER baseline vs HTCL variants)
- **Hierarchy time comparison** (computation time by hierarchy depth)

## 📈 Output Format

### CSV Results
```
results/
├── csv/
│   ├── er_results_SplitMNIST.csv
│   ├── htcl_L2_results_SplitMNIST.csv
│   ├── htcl_L3_results_SplitMNIST.csv
│   └── ...
```

### Visualizations
```
results/plots/
├── png/
│   ├── task_accuracy_comparison_SplitMNIST.png
│   ├── hierarchy_comparison_SplitMNIST.png
│   ├── task_order_robustness_SplitMNIST.png
│   └── comprehensive_comparison_SplitMNIST.png
└── svg/
    ├── task_accuracy_comparison_SplitMNIST.svg
    └── ...
```

## 🧪 Running Experiments

### Single Dataset Experiment
```bash
# Debug mode (fast, small dataset)
python main.py --dataset SplitMNIST --levels 2 3 4 --debug --epochs 3

# Full experiment
python main.py --dataset CIFAR100 --levels 2 3 --epochs 10 --buffer-size 500
```

### Multiple Datasets
```python
from htcl.experiments import run_all_datasets_experiment

results = run_all_datasets_experiment(
    datasets=["SplitMNIST", "CIFAR100"],
    hierarchy_levels=[2, 3, 4],
    debug=True,
)
```

## 📚 Citation

If you use this code, please cite:

```bibtex
@inproceedings{htcl2025,
  title={Hierarchical Continual Learning via Taylor Series Expansion},
  author={Anonymous},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 1.12
- torchvision >= 0.13
- torch-geometric >= 2.0 (for Cora dataset)
- numpy >= 1.21
- pandas >= 1.3
- matplotlib >= 3.5
- seaborn >= 0.11
- scikit-learn >= 1.0

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🐛 Troubleshooting

### CUDA Out of Memory
- Use `--debug` mode for smaller datasets
- Reduce `--batch-size`
- Reduce `--buffer-size`

### Slow Training
- Use `--debug` mode
- Reduce `--num-perms` (number of permutations)
- Reduce `--epochs`

### Import Errors
- Ensure all dependencies are installed
- Run from the repository root directory
