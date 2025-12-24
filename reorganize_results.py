#!/usr/bin/env python3
"""
Utility script to reorganize existing HTCL results into proper directory structure.

Run this ONCE to fix the file organization:
    python reorganize_results.py

Or with a custom results directory:
    python reorganize_results.py --results-dir ./my_results
"""

import os
import shutil
import argparse
from pathlib import Path


def reorganize_directory(base_dir: str, dry_run: bool = False) -> dict:
    """
    Reorganize a single experiment directory.
    
    Moves:
    - .json files → json/
    - .csv files → csv/
    - .png files → plots/png/
    - .svg files → plots/svg/
    
    Args:
        base_dir: Directory to reorganize (e.g., ./results/splitmnist/er)
        dry_run: If True, only print what would be done without moving files
    
    Returns:
        Dict with counts of moved files by type
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"  Directory not found: {base_dir}")
        return {}
    
    # Create proper subdirectories
    csv_dir = base_path / "csv"
    json_dir = base_path / "json"
    plots_png_dir = base_path / "plots" / "png"
    plots_svg_dir = base_path / "plots" / "svg"
    checkpoints_dir = base_path / "checkpoints"
    
    if not dry_run:
        for d in [csv_dir, json_dir, plots_png_dir, plots_svg_dir, checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    counts = {"json": 0, "csv": 0, "png": 0, "svg": 0}
    
    # Walk through all files in the directory
    for root, dirs, files in os.walk(base_dir):
        root_path = Path(root)
        
        # Skip files already in the correct location
        if any(part in str(root_path) for part in ['csv', 'json', 'plots', 'checkpoints']):
            # But we might need to move files from old csv/ folder that are actually JSON
            if 'csv' in str(root_path):
                for file in files:
                    if file.endswith('.json'):
                        src = root_path / file
                        dst = json_dir / file
                        if src != dst:
                            if dry_run:
                                print(f"  Would move: {src} → {dst}")
                            else:
                                shutil.move(str(src), str(dst))
                                print(f"  Moved: {src} → {dst}")
                            counts["json"] += 1
            continue
        
        for file in files:
            src = root_path / file
            dst = None
            file_type = None
            
            if file.endswith('.json'):
                dst = json_dir / file
                file_type = "json"
            elif file.endswith('.csv'):
                dst = csv_dir / file
                file_type = "csv"
            elif file.endswith('.png'):
                dst = plots_png_dir / file
                file_type = "png"
            elif file.endswith('.svg'):
                dst = plots_svg_dir / file
                file_type = "svg"
            
            if dst and src != dst:
                if dry_run:
                    print(f"  Would move: {src} → {dst}")
                else:
                    # Check if file exists at destination
                    if dst.exists():
                        print(f"  Skipping (already exists): {dst}")
                        continue
                    shutil.move(str(src), str(dst))
                    print(f"  Moved: {src} → {dst}")
                counts[file_type] += 1
    
    return counts


def find_experiment_dirs(results_dir: str) -> list:
    """
    Find all experiment directories (dataset/baseline combinations).
    
    Expected structure:
        results/
        ├── splitmnist/
        │   ├── er/
        │   └── ser/
        ├── cifar100/
        │   ├── er/
        │   └── ser/
        ...
    """
    results_path = Path(results_dir)
    experiment_dirs = []
    
    if not results_path.exists():
        return []
    
    # Find all dataset directories
    for dataset_dir in results_path.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
            # Find all baseline directories within each dataset
            for baseline_dir in dataset_dir.iterdir():
                if baseline_dir.is_dir() and not baseline_dir.name.startswith('.'):
                    experiment_dirs.append(str(baseline_dir))
    
    return sorted(experiment_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize HTCL results into proper directory structure"
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="./results",
        help="Base results directory (default: ./results)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be done without actually moving files"
    )
    parser.add_argument(
        "--single", "-s",
        type=str,
        default=None,
        help="Reorganize a single directory instead of all experiments"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN - No files will be moved\n")
    
    if args.single:
        # Reorganize a single directory
        dirs_to_process = [args.single]
    else:
        # Find all experiment directories
        dirs_to_process = find_experiment_dirs(args.results_dir)
    
    if not dirs_to_process:
        print(f"No experiment directories found in: {args.results_dir}")
        print("\nExpected structure:")
        print("  results/")
        print("  ├── splitmnist/")
        print("  │   ├── er/")
        print("  │   └── ser/")
        print("  ├── cifar100/")
        print("  │   └── ...")
        return
    
    print(f"Found {len(dirs_to_process)} experiment directories to reorganize:\n")
    
    total_counts = {"json": 0, "csv": 0, "png": 0, "svg": 0}
    
    for exp_dir in dirs_to_process:
        print(f"Processing: {exp_dir}")
        counts = reorganize_directory(exp_dir, dry_run=args.dry_run)
        for k, v in counts.items():
            total_counts[k] += v
        print()
    
    print("=" * 60)
    if args.dry_run:
        print("DRY RUN SUMMARY - Would have moved:")
    else:
        print("REORGANIZATION COMPLETE - Moved:")
    print(f"  JSON files: {total_counts['json']}")
    print(f"  CSV files:  {total_counts['csv']}")
    print(f"  PNG files:  {total_counts['png']}")
    print(f"  SVG files:  {total_counts['svg']}")
    print("=" * 60)
    
    print("\nNew directory structure:")
    print("  <experiment>/")
    print("  ├── csv/           # Result CSV files")
    print("  ├── json/          # Metadata/summary JSON files")
    print("  ├── plots/")
    print("  │   ├── png/       # PNG plots")
    print("  │   └── svg/       # SVG plots")
    print("  └── checkpoints/   # Model checkpoints")


if __name__ == "__main__":
    main()
