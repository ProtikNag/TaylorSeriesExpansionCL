#!/usr/bin/env python3
"""
Main entry point for HTCL experiments.

Run from the TaylorSeriesExpansionCL directory:
    python main.py --dataset SplitMNIST --baseline er --levels 2 3 --debug
    python main.py --dataset CIFAR100 --baseline er --levels 2 3 4 --debug
    
Results are organized by dataset and baseline:
    results/splitmnist/er/
    results/splitmnist/er/
    results/cifar100/er/
    etc.
"""

import argparse
import sys
import os

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import from htcl
from htcl import (
    run_hierarchy_experiment,
    run_single_experiment,
    quick_test,
    list_baselines,
    get_mnist_config,
    get_cifar100_config,
    get_newsgroups_config,
    get_cora_config,
    BASELINE_REGISTRY,
)


def get_config_for_dataset(dataset_name: str, debug: bool = False):
    """Get the appropriate config for a dataset."""
    configs = {
        "SplitMNIST": get_mnist_config,
        "CIFAR100": get_cifar100_config,
        "20Newsgroups": get_newsgroups_config,
        "Cora": get_cora_config,
    }

    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(configs.keys())}")

    return configs[dataset_name](debug=debug)


def main():
    # Get available baselines for help text
    available_baselines = list(BASELINE_REGISTRY.keys())

    parser = argparse.ArgumentParser(
        description="HTCL: Hierarchical Taylor Series Continual Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="SplitMNIST",
        choices=["SplitMNIST", "CIFAR100", "20Newsgroups", "Cora"],
        help="Dataset to use"
    )

    # Baseline method
    parser.add_argument(
        "--baseline", type=str, default="er",
        choices=available_baselines,
        help=f"Baseline CL method ({', '.join(available_baselines)})"
    )

    # Hierarchy
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[2, 3],
        help="Hierarchy levels to compare"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=500, help="Replay buffer size")
    parser.add_argument("--group-size", type=int, default=2, help="Tasks per group")
    parser.add_argument("--num-perms", type=int, default=20, help="Number of canonical permutations")

    # Catch-up
    parser.add_argument("--catchup-epochs", type=int, default=2, help="Taylor catch-up iterations")
    parser.add_argument("--no-catchup", action="store_true", help="Disable catch-up")

    # Modes
    parser.add_argument("--debug", action="store_true", help="Use smaller dataset")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test")
    parser.add_argument("--single", action="store_true", help="Run single experiment")
    parser.add_argument("--no-visualizations", action="store_true", help="Skip visualizations")
    parser.add_argument("--list-baselines", action="store_true", help="List available baselines and exit")

    # Output
    parser.add_argument("--output-dir", type=str, default="./results", help="Base output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # List baselines mode
    if args.list_baselines:
        print("Available baseline methods:")
        for name in available_baselines:
            print(f"  - {name}")
        return

    # Quick test mode
    if args.quick_test:
        print(f"Running quick test with {args.baseline.upper()} baseline...")
        quick_test(baseline=args.baseline)
        return

    # Get config
    config = get_config_for_dataset(args.dataset, debug=args.debug)

    # Override config with CLI args
    config.training.num_epochs = args.epochs
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.data.batch_size = args.batch_size

    config.htcl.buffer_size = args.buffer_size
    config.htcl.group_size = args.group_size
    config.htcl.num_permutations = args.num_perms
    config.htcl.catchup_enabled = not args.no_catchup
    config.htcl.catchup_epochs = args.catchup_epochs

    config.output_dir = args.output_dir
    config.data.seed = args.seed

    # Calculate expected output path
    dataset_lower = args.dataset.lower().replace("-", "").replace("_", "")
    expected_output = os.path.join(args.output_dir, dataset_lower, args.baseline.lower())

    print(f"\n{'=' * 70}")
    print("  HTCL: Hierarchical Taylor Series Continual Learning")
    print(f"{'=' * 70}")

    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Baseline: {args.baseline.upper()}")
    print(f"  Debug mode: {args.debug}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Hierarchy levels: {args.levels}")
    print(f"  Canonical permutations: {args.num_perms}")
    print(f"  Catch-up enabled: {not args.no_catchup} (uses Taylor updates)")
    print(f"  Output: {expected_output}")

    # Run experiment
    if args.single:
        print("\nRunning single experiment...")
        results = run_single_experiment(
            config,
            baseline=args.baseline,
        )
    else:
        print("\nRunning hierarchy comparison experiment...")
        results = run_hierarchy_experiment(
            config=config,
            baseline=args.baseline,
            hierarchy_levels=args.levels,
            create_visualizations=not args.no_visualizations,
        )

    print("\n" + "=" * 70)
    print("  Experiment completed!")
    print("=" * 70)
    print(f"\nResults saved to: {expected_output}")

    if not args.no_visualizations:
        print(f"Plots saved to: {expected_output}/plots/")


if __name__ == "__main__":
    main()
