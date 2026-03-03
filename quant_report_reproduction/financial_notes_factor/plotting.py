"""
可视化模块 — 复现研报图表风格
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = ["#E74C3C", "#E67E22", "#95A5A6", "#3498DB", "#27AE60"]
SAVE_DIR = "/opt/cursor/artifacts"


def plot_group_nav(group_nav, benchmark_nav, title, save_name):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for g in group_nav.columns:
        ax.plot(group_nav.index, group_nav[g], label=f"Group {g}",
                color=COLORS[g - 1], linewidth=1.5)
    ax.plot(benchmark_nav.index, benchmark_nav.values, label="Benchmark",
            color="black", linewidth=1.5, linestyle="--")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("NAV")
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ic_series(ic_series, title, save_name):
    fig, ax = plt.subplots(figsize=(13, 4))
    colors = ["#27AE60" if v >= 0 else "#E74C3C" for v in ic_series.values]
    ax.bar(ic_series.index, ic_series.values, color=colors, alpha=0.7, width=20)
    avg_ic = ic_series.mean()
    ax.axhline(avg_ic, color="#2C3E50", linewidth=1.5, linestyle="--",
               label=f"Mean IC = {avg_ic:.4f}")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Rank IC")
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_excess_nav(excess_nav, long_short_nav, title, save_name):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(long_short_nav.index, long_short_nav.values,
            color="#E74C3C", linewidth=2, label="Long-Short")
    ax.plot(excess_nav.index, excess_nav.values,
            color="#27AE60", linewidth=2, label="Excess (G5 - Benchmark)")
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coverage(coverage, title, save_name):
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.fill_between(coverage.index, 0, coverage.values * 100,
                    alpha=0.4, color="#3498DB")
    ax.plot(coverage.index, coverage.values * 100,
            color="#2C3E50", linewidth=1)
    avg = coverage.mean() * 100
    ax.axhline(avg, color="#E74C3C", linestyle="--",
               label=f"Avg Coverage = {avg:.1f}%")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_performance_table(perf_df, title, save_name):
    fig, ax = plt.subplots(figsize=(14, max(2.5, len(perf_df) * 0.5 + 1.5)))
    ax.axis("off")

    display = perf_df.copy()
    display["ann_return"] = display["ann_return"].apply(lambda x: f"{x:.2%}")
    display["ann_volatility"] = display["ann_volatility"].apply(lambda x: f"{x:.2%}")
    display["max_drawdown"] = display["max_drawdown"].apply(lambda x: f"{x:.2%}")
    display.columns = ["Ann. Return", "Ann. Volatility", "Max Drawdown"]

    table = ax.table(
        cellText=display.values,
        rowLabels=display.index,
        colLabels=display.columns,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#ECF0F1")
            cell.set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_annual_returns(result, title, save_name):
    from .factor_test_framework import compute_annual_metrics

    g5_annual = compute_annual_metrics(result["group_returns"][5])
    bench_annual = compute_annual_metrics(result["benchmark_returns"])
    excess_annual = compute_annual_metrics(result["excess_returns"])
    ls_annual = compute_annual_metrics(result["long_short_returns"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    years = g5_annual.index
    x = np.arange(len(years))
    w = 0.35

    axes[0].bar(x - w / 2, g5_annual["return"].values * 100, w,
                label="Group 5 (Long)", color="#27AE60", alpha=0.8)
    axes[0].bar(x + w / 2, bench_annual["return"].values * 100, w,
                label="Benchmark", color="#95A5A6", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(years, rotation=45)
    axes[0].set_ylabel("Return (%)")
    axes[0].set_title("Group 5 vs Benchmark", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, color="black", linewidth=0.5)

    axes[1].bar(x - w / 2, ls_annual["return"].values * 100, w,
                label="Long-Short", color="#E74C3C", alpha=0.8)
    axes[1].bar(x + w / 2, excess_annual["return"].values * 100, w,
                label="Excess", color="#3498DB", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(years, rotation=45)
    axes[1].set_ylabel("Return (%)")
    axes[1].set_title("Long-Short & Excess", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].axhline(0, color="black", linewidth=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_factor_comparison(metrics_list, save_name):
    """多因子绩效对比柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = [m["name"] for m in metrics_list]
    x = np.arange(len(names))

    ic_vals = [m["ic_mean"] * 100 for m in metrics_list]
    axes[0].bar(x, ic_vals, color=COLORS[:len(names)], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("Rank IC (%)")
    axes[0].set_title("Monthly Rank IC Mean", fontsize=11)

    excess_vals = [m["excess_ann"] * 100 for m in metrics_list]
    axes[1].bar(x, excess_vals, color=COLORS[:len(names)], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("Ann. Excess (%)")
    axes[1].set_title("Ann. Excess Return", fontsize=11)

    dd_vals = [abs(m["excess_dd"]) * 100 for m in metrics_list]
    axes[2].bar(x, dd_vals, color=COLORS[:len(names)], alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[2].set_ylabel("Max DD (%)")
    axes[2].set_title("Excess Max Drawdown", fontsize=11)

    fig.suptitle("Factor Comparison Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, save_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
