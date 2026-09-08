import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def analyze_entropy_and_repeat(analysis_data, save_fig_path=None):
    """绘制 entropy / repeat 分布图并保存（不展示）。"""
    if not save_fig_path:
        raise ValueError("save_fig_path is required for saving figure.")

    plt.figure(figsize=(16, 12))

    plt.subplot(1, 2, 1)
    entropy_data = analysis_data["entropy"].sort_values()
    plt.plot(entropy_data.values, "o", markersize=2, alpha=0.7, color="blue")
    step = max(1, len(analysis_data) // 10)
    plt.xticks(np.arange(0, len(analysis_data) + 1, step))
    plt.xlabel("Sample Index")
    plt.ylim(0, 6)
    plt.ylabel("Entropy (bits/char)")
    plt.title("Char Entropy (sorted ascending)")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    repeat_data = analysis_data["repeat"].sort_values(ascending=False)
    plt.plot(repeat_data.values, "o", markersize=2, alpha=0.7, color="green")
    plt.xticks(np.arange(0, len(analysis_data) + 1, step))
    plt.xlabel("Sample Index")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Repeat Value")
    plt.title("Repeat Values (Sorted High to Low)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_fig_path) or ".", exist_ok=True)
    plt.savefig(save_fig_path)
    print(f"[FIGURE SAVED] {save_fig_path}")
    plt.close()
