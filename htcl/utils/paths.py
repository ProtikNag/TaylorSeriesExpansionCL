"""
Path utilities for HTCL results organization.

This module provides consistent paths for saving results across all training functions.
Import and use these functions in htcl.py, er.py, ser.py, etc.

Usage:
    from ..utils.paths import get_results_paths, ensure_results_dirs
    
    # At the start of your experiment function
    paths = get_results_paths(output_dir)
    ensure_results_dirs(output_dir)
    
    # Save CSV
    csv_path = os.path.join(paths['csv'], f"results_{dataset}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save JSON
    json_path = os.path.join(paths['json'], f"summary_{dataset}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
"""

import os
from typing import Dict


def get_results_paths(output_dir: str) -> Dict[str, str]:
    """
    Get the standard subdirectory paths for a results directory.
    
    Args:
        output_dir: Base output directory (e.g., ./results/splitmnist/er)
    
    Returns:
        Dict with keys: 'csv', 'json', 'plots_png', 'plots_svg', 'checkpoints'
    """
    return {
        'csv': os.path.join(output_dir, 'csv'),
        'json': os.path.join(output_dir, 'json'),
        'plots_png': os.path.join(output_dir, 'plots', 'png'),
        'plots_svg': os.path.join(output_dir, 'plots', 'svg'),
        'checkpoints': os.path.join(output_dir, 'checkpoints'),
    }


def ensure_results_dirs(output_dir: str) -> Dict[str, str]:
    """
    Create all results subdirectories and return their paths.
    
    Args:
        output_dir: Base output directory
    
    Returns:
        Dict with paths (same as get_results_paths)
    """
    paths = get_results_paths(output_dir)
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def get_csv_path(output_dir: str, filename: str) -> str:
    """Get full path for a CSV file in the csv/ subdirectory."""
    if not filename.endswith('.csv'):
        filename = f"{filename}.csv"
    return os.path.join(output_dir, 'csv', filename)


def get_json_path(output_dir: str, filename: str) -> str:
    """Get full path for a JSON file in the json/ subdirectory."""
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    return os.path.join(output_dir, 'json', filename)


def get_plot_paths(output_dir: str, name: str) -> tuple:
    """
    Get paths for PNG and SVG versions of a plot.
    
    Returns:
        Tuple of (png_path, svg_path)
    """
    return (
        os.path.join(output_dir, 'plots', 'png', f"{name}.png"),
        os.path.join(output_dir, 'plots', 'svg', f"{name}.svg"),
    )
