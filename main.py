#!/usr/bin/env python3
"""
Main entry point for HTCL experiments.

Run from the TaylorSeriesExpansionCL directory:
    python main.py --dataset SplitMNIST --levels 2 3 --debug
"""

import argparse
import sys
import os

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import from htcl
try:
    from htcl import (
        run_hierarchy_experiment,
        run_single_experiment,
        quick_test,
        get_mnist_config,
        get_cifar100_config,
        get_newsgroups_config,
        get_cora_config,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTrying to diagnose the issue...")
    
    # Try importing submodules one by one to find the problem
    try:
        from htcl import config
        print("  ✓ htcl.config imported successfully")
    except ImportError as e2:
        print(f"  ✗ htcl.config failed: {e2}")
    
    try:
        from htcl import data
        print("  ✓ htcl.data imported successfully")
    except ImportError as e2:
        print(f"  ✗ htcl.data failed: {e2}")
    
    try:
        from htcl import models
        print("  ✓ htcl.models imported successfully")
    except ImportError as e2:
        print(f"  ✗ htcl.models failed: {e2}")
    
    try:
        from htcl import utils
        print("  ✓ htcl.utils imported successfully")
    except ImportError as e2:
        print(f"  ✗ htcl.utils failed: {e2}")
    
    try:
        from htcl import methods
        print("  ✓ htcl.methods imported successfully")
    except ImportError as e2:
        print(f"  ✗ htcl.methods failed: {e2}")
    
    try:
        from htcl import visualization
        print("  ✓ htcl.visualization imported successfully")
    except ImportError as e2:
        print(f"  ✗ htcl.visualization failed: {e2}")
    
    try:
        from htcl import experiments
        print("  ✓ htcl.experiments imported successfully")
    except ImportError as e2:
        print(f"  ✗ htcl.experiments failed: {e2}")
    
    sys.exit(1)


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
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        print("Running quick test...")
        quick_test()
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
    
    print(f"\n{'='*70}")
    print("  HTCL: Hierarchical Taylor Series Continual Learning")
    print(f"{'='*70}")
    
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
