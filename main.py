# main.py

import argparse
import torch
from data import (
    ContinualCIFAR100,
    ContinualSplitMNIST,
    ContinualCora,
    Continual20Newsgroups,
)
from models import get_model
from methods.taylor import train_taylor
from methods.er import run_er_experiments
from utils import set_seed
import itertools
import random

set_seed(42)


def run_for_dataset(name, data_constructor, total_classes, num_tasks, batch_size, lr, epochs, group_size, buffer_size, device):
    """
    Runs Taylor + ER for a single dataset.
    """
    print("\n" + "=" * 80)
    print(f"Running dataset: {name}")
    print(f"  num_tasks={num_tasks}, total_classes={total_classes}, batch_size={batch_size}, lr={lr}")
    print("=" * 80 + "\n")

    # Instantiate dataset
    # data = data_constructor(num_tasks=num_tasks, batch_size=batch_size)
    data = data_constructor(num_tasks=num_tasks, batch_size=batch_size, debug=True, samples_per_class=1)

    num_classes_per_task = total_classes // num_tasks
    train_loaders, test_loaders = data.get_task_loaders()

    # ----- Taylor -----
    model_taylor = get_model(num_classes=num_classes_per_task, dataset=name)
    print(f"[{name}] Training Taylor method ({model_taylor.__class__.__name__})")

    num_tasks = len(train_loaders)
    task_indices = list(range(num_tasks))
    all_perms = list(itertools.permutations(task_indices))

    n = 1
    perms = random.sample(all_perms, n)

    train_taylor(
        model_taylor,
        train_loaders,
        test_loaders,
        group_size=group_size,
        num_epochs=epochs,
        lr=lr,
        buffer_size=buffer_size,
        perms=perms,
        device=device,
        dataset=name,
    )

    # ----- ER -----
    model_er = get_model(num_classes=num_classes_per_task, dataset=name)
    print(f"[{name}] Running ER experiments ({model_er.__class__.__name__})")

    run_er_experiments(
        model_er,
        train_loaders,
        test_loaders,
        buffer_size=buffer_size,
        num_epochs=epochs,
        lr=lr,
        perms=perms,
        device=device,
        dataset=name,
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # (dataset_name, constructor, total_classes, num_tasks, batch_size, lr)
    dataset_runs = [
        ("CIFAR100", ContinualCIFAR100, 100, 10, 128, 0.1, 500),
        ("SplitMNIST", ContinualSplitMNIST, 10, 5, 64, 0.01, 50),
        ("Cora", ContinualCora, 7, 3, 64, 0.01, 50),
        ("20Newsgroups", Continual20Newsgroups, 20, 5, 64, 1e-3, 100),
    ]

    for name, constructor, total_classes, num_tasks, batch_size, lr, buffer_size in dataset_runs:
        try:
            run_for_dataset(
                name=name,
                data_constructor=constructor,
                total_classes=total_classes,
                num_tasks=num_tasks,
                batch_size=batch_size,
                lr=lr,
                epochs=5,
                group_size=2,
                buffer_size=buffer_size,
                device=device,
            )
        except Exception as e:
            print(f"Error on dataset {name}: {e}")
            continue


if __name__ == "__main__":
    main()
