#!/usr/bin/env python3
"""
Main entry point for FL Comparison experiments.

Compares HTCL consolidation against FedAvg and FedProx on top of SER and DER baselines.

Run from the TaylorSeriesExpansionCL directory:
    python run_fl_experiment.py --dataset SplitMNIST --debug
    
Results are organized in:
    results/splitmnist/fl_comparison/
"""

import argparse
import sys
import os

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from htcl import (
    get_mnist_config,
    get_cifar100_config,
)
from htcl.experiments.fl_runner import run_fl_comparison_experiment, get_output_dir


def main():
    parser = argparse.ArgumentParser(
        description="FL Comparison: HTCL vs FedAvg vs FedProx",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="SplitMNIST",
        choices=["SplitMNIST", "CIFAR100"],
        help="Dataset to use"
    )
    
    # Base methods
    parser.add_argument(
        "--baselines", type=str, nargs="+", default=["ser", "der"],
        choices=["ser", "der"],
        help="Base CL methods to compare"
    )
    
    # Hierarchy
    parser.add_argument("--levels", type=int, default=2, help="Hierarchy levels")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50, help="Replay buffer size")
    parser.add_argument("--group-size", type=int, default=2, help="Tasks per group")
    parser.add_argument("--num-perms", type=int, default=40, help="Number of permutations")
    
    # HTCL specific
    parser.add_argument("--catchup-epochs", type=int, default=10, help="HTCL catch-up iterations")
    parser.add_argument("--no-catchup", action="store_true", help="Disable HTCL catch-up")
    
    # Modes
    parser.add_argument("--debug", action="store_true", help="Use smaller dataset")
    parser.add_argument("--no-visualizations", action="store_true", help="Skip visualizations")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./results", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Get config
    if args.dataset == "SplitMNIST":
        config = get_mnist_config(debug=args.debug)
    else:
        config = get_cifar100_config(debug=args.debug)
    
    # Override config with CLI args
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    config.data.batch_size = args.batch_size
    config.htcl.buffer_size = args.buffer_size
    config.htcl.group_size = args.group_size
    config.htcl.num_permutations = args.num_perms
    config.htcl.catchup_enabled = not args.no_catchup
    config.htcl.catchup_epochs = args.catchup_epochs
    config.output_dir = args.output_dir
    config.data.seed = args.seed
    
    # Calculate expected output path
    expected_output = get_output_dir(args.output_dir, args.dataset, "fl_comparison")
    
    print(f"\n{'=' * 70}")
    print("  FL Comparison: HTCL vs FedAvg vs FedProx")
    print(f"{'=' * 70}")
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Base methods: {[b.upper() for b in args.baselines]}")
    print(f"  Debug mode: {args.debug}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Hierarchy levels: {args.levels}")
    print(f"  Canonical permutations: {args.num_perms}")
    print(f"  HTCL Catch-up enabled: {not args.no_catchup}")
    print(f"  Output: {expected_output}")
    
    print("\nRunning FL comparison experiment...")
    print("This compares three consolidation methods:")
    print("  1. HTCL (Taylor series-based second-order consolidation)")
    print("  2. FedAvg (Simple weight averaging)")
    print("  3. FedProx (Weight averaging + proximal regularization)")
    print()
    
    results = run_fl_comparison_experiment(
        config=config,
        baselines=args.baselines,
        hierarchy_levels=args.levels,
        create_visualizations=not args.no_visualizations,
    )
    
    print("\n" + "=" * 70)
    print("  Experiment completed!")
    print("=" * 70)
    print(f"\nResults saved to: {expected_output}")
    
    if not args.no_visualizations:
        print(f"Plots saved to: {expected_output}/plots/")
        print(f"  - fl_taskwise_accuracy_{args.dataset}.png")
        print(f"  - fl_overall_accuracy_{args.dataset}.png")
        print(f"  - fl_overall_std_{args.dataset}.png")
        print(f"  - fl_overall_forgetting_{args.dataset}.png")
        print(f"  - fl_violin_comparison_{args.dataset}.png")
        print(f"  - fl_summary_comparison_{args.dataset}.png")


if __name__ == "__main__":
    main()
