"""
Experiment runners for HTCL.

Provides easy-to-use functions for running complete experiments.
Results are organized by dataset and baseline method:
    results/{dataset}/{baseline}/
"""

import os
from typing import List, Dict, Any
from datetime import datetime

from ..config import ExperimentConfig
from ..data import get_dataset
from ..models import get_model
from ..methods import (
    train_htcl,
    generate_canonical_permutations,
    get_baseline_runner,
    BASELINE_REGISTRY,
)
from ..utils import set_seed, get_device, ResultsManager
from ..visualization import create_all_visualizations


def get_output_dir(base_dir: str, dataset: str, baseline: str) -> str:
    """
    Generate organized output directory path.
    
    Args:
        base_dir: Base results directory (e.g., "./results")
        dataset: Dataset name (e.g., "SplitMNIST")
        baseline: Baseline method name (e.g., "er", "er")
    
    Returns:
        Path like "./results/splitmnist/er/"
    """
    # Normalize names
    dataset_lower = dataset.lower().replace("-", "").replace("_", "")
    baseline_lower = baseline.lower()

    output_dir = os.path.join(base_dir, dataset_lower, baseline_lower)
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def run_hierarchy_experiment(
        config: ExperimentConfig,
        baseline: str = "er",
        hierarchy_levels: List[int] = [2, 3, 4, 5],
        save_results: bool = True,
        create_visualizations: bool = True,
) -> Dict[str, Any]:
    """
    Run experiments comparing different hierarchy levels.
    
    Args:
        config: Base experiment configuration
        baseline: Baseline method to use ("er", "er", etc.)
        hierarchy_levels: List of hierarchy levels to compare
        save_results: Whether to save results
        create_visualizations: Whether to create plots
    
    Returns:
        Complete results dictionary
    """
    set_seed(config.data.seed)
    device = get_device() if config.training.device == "cuda" else config.training.device

    # Get organized output directory
    output_dir = get_output_dir(config.output_dir, config.data.name, baseline)

    print(f"\n{'=' * 70}")
    print(f"Hierarchy Comparison Experiment: {config.experiment_name}")
    print(f"Baseline: {baseline.upper()}, Levels: {hierarchy_levels}")
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

    # Generate canonical permutations (unique under HTCL grouping)
    num_tasks = len(train_loaders)
    max_perms = config.htcl.num_permutations if config.htcl.num_permutations else 20

    perms = generate_canonical_permutations(
        num_tasks=num_tasks,
        group_size=config.htcl.group_size,
        max_perms=max_perms,
        seed=config.data.seed,
    )

    print(f"Using {len(perms)} canonical permutations (unique under group_size={config.htcl.group_size})")

    # Get the baseline runner
    baseline_runner = get_baseline_runner(baseline)

    # Run baseline
    print("\n" + "-" * 50)
    print(f"Running {baseline.upper()} baseline...")
    print("-" * 50)

    baseline_results = baseline_runner(
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

    # Run HTCL for each hierarchy level
    htcl_results_by_level = {}

    for num_levels in hierarchy_levels:
        print("\n" + "-" * 50)
        print(f"Running {baseline.upper()} + HTCL with L={num_levels} levels...")
        print("-" * 50)

        htcl_results = train_htcl(
            model=model,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_levels=num_levels,
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

        htcl_results_by_level[num_levels] = htcl_results

    # Compile results
    results = {
        "experiment_name": f"{config.experiment_name}_hierarchy_comparison",
        "baseline": baseline,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "hierarchy_levels": hierarchy_levels,
        "baseline_results": baseline_results,
        "htcl_results_by_level": htcl_results_by_level,
    }

    # Save results
    if save_results:
        results_manager = ResultsManager(output_dir)
        results_path = results_manager.save_results(
            results, f"{config.experiment_name}_hierarchy"
        )
        print(f"\nResults saved to: {results_path}")

    # Create visualizations
    if create_visualizations:
        print("\nCreating visualizations...")
        create_all_visualizations(
            er_results=baseline_results,  # Works with any baseline
            htcl_results_by_level=htcl_results_by_level,
            dataset=config.data.name,
            output_dir=output_dir,
            show=False,
        )

    # Print summary
    print("\n" + "=" * 70)
    print("HIERARCHY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<20} {'Mean Acc (%)':<15} {'Std (%)':<10} {'Time (s)':<10}")
    print("-" * 55)

    baseline_time = baseline_results.get('total_time_seconds', 0)
    print(f"{baseline.upper():<20} {baseline_results['summary']['mean_accuracy']:<15.2f} "
          f"{baseline_results['summary']['std_accuracy']:<10.2f} {baseline_time:<10.1f}")

    for level in hierarchy_levels:
        htcl_r = htcl_results_by_level[level]
        htcl_time = htcl_r.get('total_time_seconds', 0)
        print(f"{baseline.upper() + '+HTCL-L' + str(level):<20} {htcl_r['summary']['mean_accuracy']:<15.2f} "
              f"{htcl_r['summary']['std_accuracy']:<10.2f} {htcl_time:<10.1f}")
    print("=" * 70)

    return results


def list_baselines() -> List[str]:
    """Return list of available baseline methods."""
    return list(BASELINE_REGISTRY.keys())
