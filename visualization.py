import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_accuracy_matrix(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_taskwise_avg(acc_matrix):
    return [np.mean(accs) for accs in acc_matrix]


def print_accuracy_table(label, acc_matrix):
    print(f"\n=== Accuracy Matrix for {label} ===")
    for i, accs in enumerate(acc_matrix):
        accs_str = ", ".join([f"{acc:.2f}%" for acc in accs])
        print(f"After Group {i + 1} (Tasks 0 to {i}): [{accs_str}] → Avg: {np.mean(accs):.2f}%")


def plot_accuracy_curves(acc_dict, title="Task-wise Average Accuracy", save_path=None):
    plt.figure(figsize=(10, 6))
    for label, acc_matrix in acc_dict.items():
        avg_curve = compute_taskwise_avg(acc_matrix)
        x = list(range(1, len(avg_curve) + 1))
        plt.plot(x, avg_curve, marker='o', label=label)

    plt.xlabel("Task ID")
    plt.ylabel("Average Accuracy (%)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"\n📈 Plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    results_dir = "./results"
    acc_data = {}

    for fname in os.listdir(results_dir):
        if fname.endswith(".pkl"):
            label = fname.replace(".pkl", "")  # e.g., "Taylor_T10_G3"
            fpath = os.path.join(results_dir, fname)
            try:
                acc_matrix = load_accuracy_matrix(fpath)
                acc_data[label] = acc_matrix
                print_accuracy_table(label, acc_matrix)
            except Exception as e:
                print(f"Failed to load {fname}: {e}")

    if acc_data:
        plot_accuracy_curves(acc_data, title="Task-wise Average Accuracy Across Methods",
                             save_path="all_methods_comparison.png")
    else:
        print("No valid accuracy data found in './results'.")
