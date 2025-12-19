"""
Experiment runners for HTCL.

Provides easy-to-use functions for running complete experiments.
"""

import os
import json
import itertools
import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import torch

from ..config import ExperimentConfig, get_mnist_config, get_cifar100_config
from ..data import get_dataset
from ..models import get_model
from ..methods import run_er_experiments, train_htcl, run_hierarchy_comparison, generate_canonical_permutations
from ..utils import set_seed, get_device, ResultsManager
from ..visualization import create_all_visualizations


def run_single_experiment(
    config: ExperimentConfig,
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Experiment configuration
        save_results: Whether to save results to disk
    
    Returns:
        Dictionary with all results
    """
    set_seed(config.data.seed)
    device = get_device() if config.training.device == "cuda" else config.training.device
    
    print(f"\n{'='*70}")
    print(f"Experiment: {config.experiment_name}")
    print(f"Dataset: {config.data.name}, Device: {device}")
    print(f"{'='*70}\n")
    
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
    
    # Run ER experiments
    print("\n" + "-"*50)
    print("Running Experience Replay (ER) experiments...")
    print("-"*50)
    
    er_results = run_er_experiments(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        perms=perms,
        buffer_size=config.htcl.buffer_size,
        num_epochs=config.training.num_epochs,
        lr=config.training.learning_rate,
        beta=config.htcl.buffer_beta,
        device=device,
        dataset=config.data.name,
        output_dir=config.output_dir,
        verbose=config.verbose,
    )
    
    # Run HTCL experiments
    print("\n" + "-"*50)
    print(f"Running HTCL (L={config.htcl.num_levels}) experiments...")
    print("-"*50)
    
    htcl_results = train_htcl(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        num_levels=config.htcl.num_levels,
        group_size=config.htcl.group_size,
        num_epochs=config.training.num_epochs,
        lr=config.training.learning_rate,
        buffer_size=config.htcl.buffer_size,
        perms=perms,
        device=device,
        dataset=config.data.name,
        output_dir=config.output_dir,
        catchup_enabled=config.htcl.catchup_enabled,
        catchup_epochs=config.htcl.catchup_epochs,
        catchup_lr_factor=config.htcl.catchup_lr_factor,
        eta=config.htcl.eta,
        max_norm=config.htcl.max_norm,
        verbose=config.verbose,
    )
    
    # Compile results
    results = {
        "experiment_name": config.experiment_name,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "er_results": er_results,
        "htcl_results": htcl_results,
    }
    
    # Save results
    if save_results:
        results_manager = ResultsManager(config.output_dir)
        results_path = results_manager.save_results(results, config.experiment_name)
        print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"ER Mean Accuracy: {er_results['summary']['mean_accuracy']:.2f}% "
          f"(±{er_results['summary']['std_accuracy']:.2f}%)")
    print(f"HTCL Mean Accuracy: {htcl_results['summary']['mean_accuracy']:.2f}% "
          f"(±{htcl_results['summary']['std_accuracy']:.2f}%)")
    print("="*70)
    
    return results


def run_hierarchy_experiment(
    config: ExperimentConfig,
    hierarchy_levels: List[int] = [2, 3, 4, 5],
    save_results: bool = True,
    create_visualizations: bool = True,
) -> Dict[str, Any]:
    """
    Run experiments comparing different hierarchy levels.
    
    Args:
        config: Base experiment configuration
        hierarchy_levels: List of hierarchy levels to compare
        save_results: Whether to save results
        create_visualizations: Whether to create plots
    
    Returns:
        Complete results dictionary
    """
    set_seed(config.data.seed)
    device = get_device() if config.training.device == "cuda" else config.training.device
    
    print(f"\n{'='*70}")
    print(f"Hierarchy Comparison Experiment: {config.experiment_name}")
    print(f"Levels to compare: {hierarchy_levels}")
    print(f"{'='*70}\n")
    
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
    
    # Run ER baseline
    print("\n" + "-"*50)
    print("Running ER baseline...")
    print("-"*50)
    
    er_results = run_er_experiments(
        model=model,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        perms=perms,
        buffer_size=config.htcl.buffer_size,
        num_epochs=config.training.num_epochs,
        lr=config.training.learning_rate,
        device=device,
        dataset=config.data.name,
        output_dir=config.output_dir,
        verbose=config.verbose,
    )
    
    # Run HTCL for each hierarchy level
    htcl_results_by_level = {}
    
    for num_levels in hierarchy_levels:
        print("\n" + "-"*50)
        print(f"Running HTCL with L={num_levels} levels...")
        print("-"*50)
        
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
            output_dir=config.output_dir,
            catchup_enabled=config.htcl.catchup_enabled,
            catchup_epochs=config.htcl.catchup_epochs,
            verbose=config.verbose,
        )
        
        htcl_results_by_level[num_levels] = htcl_results
    
    # Compile results
    results = {
        "experiment_name": f"{config.experiment_name}_hierarchy_comparison",
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "hierarchy_levels": hierarchy_levels,
        "er_results": er_results,
        "htcl_results_by_level": htcl_results_by_level,
    }
    
    # Save results
    if save_results:
        results_manager = ResultsManager(config.output_dir)
        results_path = results_manager.save_results(
            results, f"{config.experiment_name}_hierarchy"
        )
        print(f"\nResults saved to: {results_path}")
    
    # Create visualizations
    if create_visualizations:
        print("\nCreating visualizations...")
        create_all_visualizations(
            er_results=er_results,
            htcl_results_by_level=htcl_results_by_level,
            dataset=config.data.name,
            output_dir=config.output_dir,
            show=False,
        )
    
    # Print summary
    print("\n" + "="*70)
    print("HIERARCHY COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Method':<15} {'Mean Acc (%)':<15} {'Std (%)':<10}")
    print("-"*40)
    print(f"{'ER':<15} {er_results['summary']['mean_accuracy']:<15.2f} "
          f"{er_results['summary']['std_accuracy']:<10.2f}")
    for level in hierarchy_levels:
        htcl_r = htcl_results_by_level[level]
        print(f"{'HTCL-L'+str(level):<15} {htcl_r['summary']['mean_accuracy']:<15.2f} "
              f"{htcl_r['summary']['std_accuracy']:<10.2f}")
    print("="*70)
    
    return results


def run_all_datasets_experiment(
    datasets: List[str] = ["SplitMNIST", "CIFAR100"],
    hierarchy_levels: List[int] = [2, 3],
    debug: bool = True,
    output_dir: str = "./results",
) -> Dict[str, Any]:
    """
    Run hierarchy comparison experiments on multiple datasets.
    
    Args:
        datasets: List of dataset names
        hierarchy_levels: Hierarchy levels to compare
        debug: Use debug mode (smaller datasets)
        output_dir: Output directory
    
    Returns:
        Results for all datasets
    """
    all_results = {}
    
    config_factories = {
        "SplitMNIST": get_mnist_config,
        "CIFAR100": get_cifar100_config,
    }
    
    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# Running experiments on {dataset_name}")
        print(f"{'#'*70}")
        
        if dataset_name in config_factories:
            config = config_factories[dataset_name](debug=debug)
        else:
            from ..config import DataConfig, ExperimentConfig
            config = ExperimentConfig(
                data=DataConfig.for_dataset(dataset_name, debug=debug),
                experiment_name=f"experiment_{dataset_name}",
            )
        
        config.output_dir = output_dir
        
        try:
            results = run_hierarchy_experiment(
                config=config,
                hierarchy_levels=hierarchy_levels,
                save_results=True,
                create_visualizations=True,
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error running {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def quick_test(dataset: str = "SplitMNIST", debug: bool = True) -> Dict[str, Any]:
    """
    Run a quick test experiment to verify the setup.
    
    Args:
        dataset: Dataset to test on
        debug: Use debug mode
    
    Returns:
        Test results
    """
    from ..config import ExperimentConfig, DataConfig, TrainingConfig, HTCLConfig
    
    config = ExperimentConfig(
        data=DataConfig.for_dataset(dataset, debug=debug),
        training=TrainingConfig(num_epochs=2),
        htcl=HTCLConfig(
            num_levels=2,
            num_permutations=2,
            buffer_size=50,
            catchup_epochs=1,
        ),
        experiment_name="quick_test",
        verbose=True,
    )
    
    return run_single_experiment(config, save_results=False)
