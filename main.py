# main.py

import argparse
import torch
import matplotlib.pyplot as plt
from data import ContinualCIFAR100, ContinualSplitMNIST
from models import get_model
from methods.taylor import train_taylor
from utils import compute_avg_accuracy, compute_avg_forgetting, save_model_and_metrics, load_model_and_metrics, set_seed
import numpy as np


set_seed(42)


def plot_results(method_names, acc_matrices, tag=""):
    plt.figure(figsize=(10, 5))
    for method, acc_matrix in zip(method_names, acc_matrices):
        x = list(range(1, len(acc_matrix) + 1))
        y = [np.mean(accs) for accs in acc_matrix]
        plt.plot(x, y, label=method)
    plt.xlabel("Task ID")
    plt.ylabel("Average Accuracy (%)")
    plt.title("Task-wise Average Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"accuracy_plot_{tag}.png")
    plt.close()


def print_metrics(name, acc_matrix):
    avg_acc = compute_avg_accuracy(acc_matrix)
    avg_forgetting = compute_avg_forgetting(acc_matrix)
    print(f"=== {name} ===")
    print(f"Accuracies   : {acc_matrix}")
    print(f"Avg Accuracy   : {avg_acc:.2f}%")
    print(f"Avg Forgetting : {avg_forgetting:.2f}%\n")
    return avg_acc, avg_forgetting


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
        # data = ContinualSplitMNIST(num_tasks=args.num_tasks, batch_size=args.batch_size, debug=True, samples_per_class=2)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    num_classes_per_task = total_classes // args.num_tasks
    train_loaders, test_loaders = data.get_task_loaders()
    tag = f"T{args.num_tasks}_G{args.group_size}"

    all_accs = []
    all_names = []


    if "taylor" in args.methods:
        model_taylor = get_model(num_classes=num_classes_per_task, dataset=args.dataset)
        model_taylor, acc_taylor = train_taylor(model_taylor, train_loaders, test_loaders,
                                                    group_size=args.group_size, num_epochs=args.epochs,
                                                    lr=args.lr, lambda_reg=args.lambda_reg, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual Learning Experiment Runner")
    parser.add_argument("--dataset", type=str, default="CIFAR100", help="Dataset name (e.g., CIFAR100)")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to split the dataset into")
    parser.add_argument("--group_size", type=int, default=2, help="Group size for Taylor method")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per task/group")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--ewc_lambda", type=float, default=500.0, help="EWC regularization strength")
    parser.add_argument("--lambda_reg", type=float, default=100.0, help="Taylor regularization strength")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for linear interpolation in linear update")
    parser.add_argument("--methods", nargs='+', default=["naive", "ewc", "taylor"],
                        help="Methods to run (choose any subset of: naive, ewc, taylor)")

    args = parser.parse_args()
    main(args)
