#!/usr/bin/env python3
"""
Main entry point for HTCL experiments.

Usage:
    python main.py --dataset SplitMNIST --levels 2 3 4 5 --debug
    python main.py --dataset CIFAR100 --levels 2 3 --epochs 10
    python main.py --quick-test
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from htcl import (
    run_hierarchy_experiment,
    run_single_experiment,
    quick_test,
    get_mnist_config,
    get_cifar100_config,
    get_newsgroups_config,
    get_cora_config,
    ExperimentConfig,
    DataConfig,
    TrainingConfig,
    HTCLConfig,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="HTCL: Hierarchical Taylor Series Continual Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test to verify setup
  python main.py --quick-test
  
  # Run on SplitMNIST with hierarchy comparison
  python main.py --dataset SplitMNIST --levels 2 3 4 --debug
  
  # Run on CIFAR-100 (full dataset)
  python main.py --dataset CIFAR100 --levels 2 3 --epochs 10
  
  # Custom configuration
  python main.py --dataset SplitMNIST --levels 2 3 4 5 --epochs 5 --lr 0.01 --buffer-size 100
        """
    )
    
    parser.add_argument(
        "--dataset", type=str, default="SplitMNIST",
        choices=["SplitMNIST", "CIFAR100", "20Newsgroups", "Cora"],
        help="Dataset to use (default: SplitMNIST)"
    )
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[2, 3],
        help="Hierarchy levels to compare (default: 2 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: dataset-specific)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=100,
        help="Replay buffer size (default: 100)"
    )
    parser.add_argument(
        "--group-size", type=int, default=2,
        help="Tasks per group for permutation search (default: 2)"
    )
    parser.add_argument(
        "--num-perms", type=int, default=20,
        help="Number of canonical task permutations to evaluate (default: 20)"
    )
    parser.add_argument(
        "--catchup-epochs", type=int, default=2,
        help="Global model catch-up epochs (default: 2)"
    )
    parser.add_argument(
        "--no-catchup", action="store_true",
        help="Disable global model catch-up"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Use debug mode (smaller datasets)"
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=60,
        help="Samples per class in debug mode (default: 60)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Output directory (default: ./results)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run a quick test to verify setup"
    )
    parser.add_argument(
        "--single", action="store_true",
        help="Run single experiment instead of hierarchy comparison"
    )
    parser.add_argument(
        "--no-visualizations", action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()


def get_config_for_dataset(dataset: str, args) -> ExperimentConfig:
    """Get appropriate configuration for the dataset."""
    # Base configs by dataset
    config_factories = {
        "SplitMNIST": get_mnist_config,
        "CIFAR100": get_cifar100_config,
        "20Newsgroups": get_newsgroups_config,
        "Cora": get_cora_config,
    }
    
    # Get base config
    if dataset in config_factories:
        config = config_factories[dataset](debug=args.debug)
    else:
        config = ExperimentConfig(
            data=DataConfig.for_dataset(dataset, debug=args.debug)
        )
    
    # Override with command line arguments
    config.data.batch_size = args.batch_size
    config.data.seed = args.seed
    config.data.samples_per_class = args.samples_per_class
    
    config.training.num_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    config.htcl.buffer_size = args.buffer_size
    config.htcl.group_size = args.group_size
    config.htcl.num_permutations = args.num_perms
    config.htcl.catchup_enabled = not args.no_catchup
    config.htcl.catchup_epochs = args.catchup_epochs
    
    config.output_dir = args.output_dir
    config.verbose = not args.quiet
    config.experiment_name = f"htcl_{dataset}"
    
    return config


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  HTCL: Hierarchical Taylor Series Continual Learning")
    print("="*70)
    
    # Quick test mode
    if args.quick_test:
        print("\nRunning quick test...")
        results = quick_test(dataset="SplitMNIST", debug=True)
        print("\nQuick test completed successfully!")
        print(f"ER Mean: {results['er_results']['summary']['mean_accuracy']:.2f}%")
        print(f"HTCL Mean: {results['htcl_results']['summary']['mean_accuracy']:.2f}%")
        return
    
    # Get configuration
    config = get_config_for_dataset(args.dataset, args)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Debug mode: {args.debug}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Hierarchy levels: {args.levels}")
    print(f"  Canonical permutations: {args.num_perms}")
    print(f"  Catch-up enabled: {not args.no_catchup} (uses Taylor updates)")
    print(f"  Output: {args.output_dir}")
    
    # Run experiment
    if args.single:
        print("\nRunning single experiment...")
        results = run_single_experiment(config)
    else:
        print("\nRunning hierarchy comparison experiment...")
        results = run_hierarchy_experiment(
            config=config,
            hierarchy_levels=args.levels,
            create_visualizations=not args.no_visualizations,
        )
    
    print("\n" + "="*70)
    print("  Experiment completed!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}")
    
    if not args.no_visualizations:
        print(f"Plots saved to: {args.output_dir}/plots/")


if __name__ == "__main__":
    main()
