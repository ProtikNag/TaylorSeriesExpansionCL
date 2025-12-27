"""
Federated Learning Comparison Runner for HTCL.

This module runs experiments comparing HTCL consolidation against
federated learning methods (FedAvg, FedProx) on top of SER and DER baselines.

Usage:
    python -m htcl.experiments.fl_runner --dataset SplitMNIST --debug
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
import gc

from ..config import ExperimentConfig
from ..data import get_dataset
from ..models import get_model
from ..methods import (
    train_htcl,
    generate_canonical_permutations,
    run_ser_experiments,
    run_der_experiments,
)
from ..methods.fedavg import train_fedavg
from ..methods.fedprox import train_fedprox
from ..utils import set_seed, get_device, ResultsManager
from .fl_plots import create_fl_comparison_visualizations


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_output_dir(base_dir: str, dataset: str, experiment_type: str = "fl_comparison") -> str:
    """
    Generate organized output directory path.

    Args:
        base_dir: Base results directory
        dataset: Dataset name
        experiment_type: Type of experiment

    Returns:
        Path like "./results/splitmnist/fl_comparison/"
    """
    dataset_lower = dataset.lower().replace("-", "").replace("_", "")
    output_dir = os.path.join(base_dir, dataset_lower, experiment_type)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_fl_comparison_experiment(
    config: ExperimentConfig,
    baselines: List[str] = ["ser", "der"],
    hierarchy_levels: int = 2,
    save_results: bool = True,
    create_visualizations: bool = True,
) -> Dict[str, Any]:
    """
    Run experiments comparing HTCL vs FedAvg vs FedProx consolidation methods.

    Args:
        config: Experiment configuration
        baselines: Base CL methods to use ("ser", "der")
        hierarchy_levels: Number of hierarchy levels
        save_results: Whether to save results
        create_visualizations: Whether to create plots

    Returns:
        Complete results dictionary
    """
    set_seed(config.data.seed)
    device = get_device() if config.training.device == "cuda" else config.training.device

    # Get organized output directory
    output_dir = get_output_dir(config.output_dir, config.data.name, "fl_comparison")

    print(f"\n{'=' * 70}")
    print(f"FL Comparison Experiment: HTCL vs FedAvg vs FedProx")
    print(f"Dataset: {config.data.name}")
    print(f"Baselines: {[b.upper() for b in baselines]}")
    print(f"Hierarchy levels: {hierarchy_levels}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = get_dataset(
        name=config.data.name,
        data_dir=config.data.data_dir,
        num_tasks=config.data.num_tasks,
        batch_size=config.data.batch_size,
        seed=config.data.seed,
        debug=config.data.debug,
        samples_per_class=config.data.samples_per_class,
    )
    train_loaders, test_loaders = dataset.get_task_loaders()

    # Create model
    print("Creating model...")
    model = get_model(
        dataset=config.data.name,
        num_classes=config.model.num_classes_per_task,
    )

    # Generate canonical permutations
    num_tasks = len(train_loaders)
    max_perms = config.htcl.num_permutations if config.htcl.num_permutations else 20

    perms = generate_canonical_permutations(
        num_tasks=num_tasks,
        group_size=config.htcl.group_size,
        max_perms=max_perms,
        seed=config.data.seed,
    )

    print(f"Using {len(perms)} canonical permutations (group_size={config.htcl.group_size})")

    # Store all results
    all_results = {}

    for baseline in baselines:
        baseline_upper = baseline.upper()
        print(f"\n{'=' * 60}")
        print(f"Running experiments with {baseline_upper} as base method")
        print(f"{'=' * 60}")

        # 1. Run baseline alone (SER or DER)
        print(f"\n{'-' * 50}")
        print(f"Running {baseline_upper} baseline...")
        print(f"{'-' * 50}")

        if baseline == "ser":
            baseline_results = run_ser_experiments(
                model=model,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                perms=perms,
                buffer_size=config.htcl.buffer_size,
                num_epochs=config.training.num_epochs,
                lr=config.training.learning_rate,
                device=device,
                dataset=config.data.name,
                output_dir=output_dir,
                verbose=config.verbose,
            )
        else:  # der
            baseline_results = run_der_experiments(
                model=model,
                train_loaders=train_loaders,
                test_loaders=test_loaders,
                perms=perms,
                buffer_size=config.htcl.buffer_size,
                num_epochs=config.training.num_epochs,
                lr=config.training.learning_rate,
                device=device,
                dataset=config.data.name,
                output_dir=output_dir,
                verbose=config.verbose,
            )

        all_results[f"{baseline}_baseline"] = baseline_results

        # Clear GPU memory before next experiment
        clear_gpu_memory()

        # 2. Run HTCL
        print(f"\n{'-' * 50}")
        print(f"Running {baseline_upper} + HTCL-L{hierarchy_levels}...")
        print(f"{'-' * 50}")

        # Re-create model to ensure fresh start
        model = get_model(
            dataset=config.data.name,
            num_classes=config.model.num_classes_per_task,
        )

        htcl_results = train_htcl(
            model=model,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_levels=hierarchy_levels,
            group_size=config.htcl.group_size,
            num_epochs=config.training.num_epochs,
            lr=config.training.learning_rate,
            buffer_size=config.htcl.buffer_size,
            perms=perms,
            device=device,
            dataset=config.data.name,
            output_dir=output_dir,
            catchup_enabled=config.htcl.catchup_enabled,
            catchup_epochs=config.htcl.catchup_epochs,
            verbose=config.verbose,
        )

        all_results[f"{baseline}_htcl"] = htcl_results

        # Clear GPU memory
        clear_gpu_memory()

        # 3. Run FedAvg
        print(f"\n{'-' * 50}")
        print(f"Running {baseline_upper} + FedAvg-L{hierarchy_levels}...")
        print(f"{'-' * 50}")

        # Re-create model to ensure fresh start
        model = get_model(
            dataset=config.data.name,
            num_classes=config.model.num_classes_per_task,
        )

        fedavg_results = train_fedavg(
            model=model,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_levels=hierarchy_levels,
            group_size=config.htcl.group_size,
            num_epochs=config.training.num_epochs,
            lr=config.training.learning_rate,
            buffer_size=config.htcl.buffer_size,
            alpha=0.5,
            perms=perms,
            device=device,
            dataset=config.data.name,
            output_dir=output_dir,
            base_method=baseline,
            verbose=config.verbose,
        )

        all_results[f"{baseline}_fedavg"] = fedavg_results

        # Clear GPU memory
        clear_gpu_memory()

        # 4. Run FedProx
        print(f"\n{'-' * 50}")
        print(f"Running {baseline_upper} + FedProx-L{hierarchy_levels}...")
        print(f"{'-' * 50}")

        # Re-create model to ensure fresh start
        model = get_model(
            dataset=config.data.name,
            num_classes=config.model.num_classes_per_task,
        )

        fedprox_results = train_fedprox(
            model=model,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_levels=hierarchy_levels,
            group_size=config.htcl.group_size,
            num_epochs=config.training.num_epochs,
            lr=config.training.learning_rate,
            buffer_size=config.htcl.buffer_size,
            alpha=0.5,
            mu=0.01,
            perms=perms,
            device=device,
            dataset=config.data.name,
            output_dir=output_dir,
            base_method=baseline,
            verbose=config.verbose,
        )

        all_results[f"{baseline}_fedprox"] = fedprox_results

        # Clear GPU memory before next baseline
        clear_gpu_memory()

    # Compile results
    results = {
        "experiment_name": f"fl_comparison_{config.data.name}",
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "baselines": baselines,
        "hierarchy_levels": hierarchy_levels,
        "all_results": all_results,
    }

    # Save results
    if save_results:
        results_manager = ResultsManager(output_dir)
        results_path = results_manager.save_results(
            results, f"fl_comparison_{config.data.name}"
        )
        print(f"\nResults saved to: {results_path}")

    # Create visualizations
    if create_visualizations:
        print("\nCreating visualizations...")
        create_fl_comparison_visualizations(
            all_results=all_results,
            baselines=baselines,
            dataset=config.data.name,
            output_dir=output_dir,
            show=False,
        )

    # Print summary
    print_fl_comparison_summary(all_results, baselines)

    return results


def print_fl_comparison_summary(all_results: Dict[str, Any], baselines: List[str]):
    """Print a formatted summary of FL comparison results."""
    print("\n" + "=" * 80)
    print("FL COMPARISON SUMMARY")
    print("=" * 80)

    for baseline in baselines:
        baseline_upper = baseline.upper()
        print(f"\n{baseline_upper} Base Method:")
        print("-" * 60)
        print(f"{'Method':<25} {'Mean Acc (%)':<15} {'Std (%)':<12} {'Time (s)':<10}")
        print("-" * 60)

        methods = [
            (f"{baseline}_baseline", f"{baseline_upper} (Baseline)"),
            (f"{baseline}_htcl", f"{baseline_upper} + HTCL"),
            (f"{baseline}_fedavg", f"{baseline_upper} + FedAvg"),
            (f"{baseline}_fedprox", f"{baseline_upper} + FedProx"),
        ]

        for key, name in methods:
            if key in all_results:
                r = all_results[key]
                mean_acc = r['summary']['mean_accuracy']
                std_acc = r['summary']['std_accuracy']
                total_time = r.get('total_time_seconds', 0)
                print(f"{name:<25} {mean_acc:<15.2f} {std_acc:<12.2f} {total_time:<10.1f}")

    print("=" * 80)


def main():
    """Main entry point for FL comparison experiments."""
    import argparse
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from htcl import get_mnist_config, get_cifar100_config

    parser = argparse.ArgumentParser(
        description="FL Comparison: HTCL vs FedAvg vs FedProx",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset", type=str, default="SplitMNIST",
        choices=["SplitMNIST", "CIFAR100"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--baselines", type=str, nargs="+", default=["ser", "der"],
        choices=["ser", "der"],
        help="Base CL methods to compare"
    )
    parser.add_argument("--levels", type=int, default=2, help="Hierarchy levels")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer-size", type=int, default=50, help="Replay buffer size")
    parser.add_argument("--num-perms", type=int, default=40, help="Number of permutations")
    parser.add_argument("--catchup-epochs", type=int, default=10, help="HTCL catch-up iterations")
    parser.add_argument("--debug", action="store_true", help="Use smaller dataset")
    parser.add_argument("--output-dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-visualizations", action="store_true", help="Skip visualizations")

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
    config.htcl.num_permutations = args.num_perms
    config.htcl.catchup_epochs = args.catchup_epochs
    config.output_dir = args.output_dir
    config.data.seed = args.seed

    # Run experiment
    results = run_fl_comparison_experiment(
        config=config,
        baselines=args.baselines,
        hierarchy_levels=args.levels,
        create_visualizations=not args.no_visualizations,
    )

    print(f"\nExperiment completed!")
    print(f"Results saved to: {get_output_dir(args.output_dir, args.dataset, 'fl_comparison')}")


if __name__ == "__main__":
    main()