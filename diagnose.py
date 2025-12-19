#!/usr/bin/env python3
"""
Diagnostic script to check HTCL package structure.
Run this to identify import issues.
"""

import sys
import os

def check_file_exists(path, name):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"  ✓ {name}")
        return True
    else:
        print(f"  ✗ {name} - MISSING!")
        return False

def main():
    print("="*60)
    print("HTCL Package Diagnostic")
    print("="*60)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    htcl_dir = os.path.join(script_dir, "htcl")
    
    print(f"\nScript directory: {script_dir}")
    print(f"Looking for htcl at: {htcl_dir}")
    
    # Check directory structure
    print("\n1. Checking directory structure...")
    
    all_ok = True
    
    # Main htcl package
    all_ok &= check_file_exists(os.path.join(htcl_dir, "__init__.py"), "htcl/__init__.py")
    
    # Submodules
    submodules = ["config", "data", "models", "methods", "utils", "visualization", "experiments"]
    for sub in submodules:
        all_ok &= check_file_exists(os.path.join(htcl_dir, sub, "__init__.py"), f"htcl/{sub}/__init__.py")
    
    # Key files
    print("\n2. Checking key files...")
    key_files = [
        ("htcl/config/config.py", "config/config.py"),
        ("htcl/data/datasets.py", "data/datasets.py"),
        ("htcl/models/architectures.py", "models/architectures.py"),
        ("htcl/methods/buffer.py", "methods/buffer.py"),
        ("htcl/methods/er.py", "methods/er.py"),
        ("htcl/methods/htcl.py", "methods/htcl.py"),
        ("htcl/utils/helpers.py", "utils/helpers.py"),
        ("htcl/visualization/plots.py", "visualization/plots.py"),
        ("htcl/experiments/runner.py", "experiments/runner.py"),
    ]
    
    for rel_path, name in key_files:
        all_ok &= check_file_exists(os.path.join(script_dir, rel_path), name)
    
    # Check Python path
    print("\n3. Checking Python path...")
    print(f"  Current directory: {os.getcwd()}")
    print(f"  sys.path includes script_dir: {script_dir in sys.path}")
    
    # Try imports
    print("\n4. Testing imports (may fail if dependencies not installed)...")
    
    # Add script directory to path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    import_tests = [
        ("htcl.config", "from htcl.config import ExperimentConfig"),
        ("htcl.data", "from htcl.data import get_dataset"),
        ("htcl.models", "from htcl.models import get_model"),
        ("htcl.utils", "from htcl.utils import set_seed"),
        ("htcl.methods", "from htcl.methods import run_er_experiments"),
        ("htcl.visualization", "from htcl.visualization import create_all_visualizations"),
        ("htcl.experiments", "from htcl.experiments import run_hierarchy_experiment"),
    ]
    
    for module_name, import_statement in import_tests:
        try:
            exec(import_statement)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  ✗ {module_name}: {e}")
            all_ok = False
    
    # Final test
    print("\n5. Testing main package import...")
    try:
        from htcl import run_hierarchy_experiment, get_mnist_config
        print("  ✓ from htcl import run_hierarchy_experiment, get_mnist_config")
    except ImportError as e:
        print(f"  ✗ Main import failed: {e}")
        all_ok = False
    
    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✓ All checks passed! Package structure is correct.")
    else:
        print("✗ Some checks failed. See above for details.")
        print("\nCommon fixes:")
        print("  1. Make sure all __init__.py files exist")
        print("  2. Run: pip install torch torchvision numpy pandas matplotlib seaborn")
        print("  3. Run from the TaylorSeriesExpansionCL directory")
    print("="*60)


if __name__ == "__main__":
    main()
