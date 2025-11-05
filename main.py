# main.py

import argparse
import torch
from data import ContinualCIFAR100, ContinualSplitMNIST
from models import get_model
from methods.taylor import train_taylor
from utils import set_seed
from methods.er import run_er_experiments

set_seed(42)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Running experiment with: {args}")

    # Dataset selector (future-proofing)
    if args.dataset == "CIFAR100":
        total_classes = 100
        data = ContinualCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size)
        # Debug mode for quick testing
        # data = ContinualCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size, debug=True, samples_per_class=3)
    elif args.dataset == "SplitMNIST":
        total_classes = 10
        data = ContinualSplitMNIST(num_tasks=args.num_tasks, batch_size=args.batch_size)
        # Debug mode for quick testing
        # data = ContinualSplitMNIST(num_tasks=args.num_tasks, batch_size=args.batch_size, debug=True, samples_per_class=200)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    num_classes_per_task = total_classes // args.num_tasks
    train_loaders, test_loaders = data.get_task_loaders()
    tag = f"T{args.num_tasks}_G{args.group_size}"

    model_taylor = get_model(num_classes=num_classes_per_task, dataset=args.dataset)
    # train_taylor(
    #     model_taylor, train_loaders, test_loaders,
    #     group_size=args.group_size, num_epochs=args.epochs,
    #     lr=args.lr, device=device
    # )

    model_er = get_model(num_classes=num_classes_per_task, dataset=args.dataset)
    run_er_experiments(
        model_er, train_loaders, test_loaders,
        buffer_size=1500, num_epochs=args.epochs,
        lr=args.lr, device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Experiment Runner")
    parser.add_argument("--dataset", type=str, default="CIFAR100", help="Dataset name (e.g., CIFAR100)")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to split the dataset into")
    parser.add_argument("--group_size", type=int, default=2, help="Group size for Taylor method")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per task/group")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()
    main(args)
