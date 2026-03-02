"""
绩效评估与可视化模块

复现研报中的图表：
- 事件样本量统计
- 事件触发后收益表现
- 通道策略净值走势
- 信号叠加前后对比
- 绩效指标汇总表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from typing import Optional, Dict

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

COLORS = {
    "strategy": "#E74C3C",
    "benchmark": "#3498DB",
    "excess": "#27AE60",
    "low_signal": "#2ECC71",
    "high_signal": "#E74C3C",
    "combined": "#8E44AD",
    "gray": "#95A5A6",
}


def plot_event_sample_count(
    stats: pd.DataFrame,
    title: str = "Daily Event Trigger Count",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制每日事件触发数量（对应研报图表3/5）
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(stats.index, stats["n_triggered"], color=COLORS["gray"], alpha=0.7, width=2)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Stocks Triggered")

    avg = stats["n_triggered"].mean()
    ax.axhline(avg, color=COLORS["strategy"], linestyle="--", alpha=0.8,
               label=f"Average: {avg:.1f}")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_post_event_returns(
    avg_abs_ret: pd.Series,
    avg_excess_ret: pd.Series,
    title: str = "Post-Event Return Performance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制事件触发后收益表现（对应研报图表4/6）
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(avg_abs_ret.index, avg_abs_ret.values * 100,
            color=COLORS["strategy"], linewidth=2,
            label="Avg Cumulative Absolute Return")
    ax.plot(avg_excess_ret.index, avg_excess_ret.values * 100,
            color=COLORS["excess"], linewidth=2,
            label="Avg Cumulative Excess Return")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading Days After Signal Trigger")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=11)

    peak_idx = avg_excess_ret.idxmax() if len(avg_excess_ret) > 0 else 0
    if peak_idx > 0:
        peak_val = avg_excess_ret[peak_idx] * 100
        ax.annotate(
            f"Peak: Day {peak_idx}, {peak_val:.2f}%",
            xy=(peak_idx, peak_val),
            xytext=(peak_idx + 5, peak_val + 0.5),
            arrowprops=dict(arrowstyle="->", color=COLORS["excess"]),
            fontsize=10, color=COLORS["excess"],
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_channel_strategy_nav(
    strategy_result: pd.DataFrame,
    benchmark_cum: Optional[pd.Series] = None,
    title: str = "Channel Strategy NAV",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制通道策略净值走势（对应研报图表7/8/13/16/18）
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    nav = strategy_result["nav"]
    ax.plot(nav.index, nav.values, color=COLORS["strategy"],
            linewidth=2, label="Channel Strategy")

    if benchmark_cum is not None:
        bench_nav = benchmark_cum / benchmark_cum.iloc[0]
        common_idx = nav.index.intersection(bench_nav.index)
        ax.plot(common_idx, bench_nav.loc[common_idx].values,
                color=COLORS["benchmark"], linewidth=1.5,
                label="Benchmark (Equal-Weight)", alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_excess_return_comparison(
    excess_curves: Dict[str, pd.Series],
    title: str = "Excess Return Comparison (Before vs After Signal Combination)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制信号叠加前后超额收益净值走势对比（对应研报图表19/26/27）
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [COLORS["low_signal"], COLORS["combined"], COLORS["high_signal"],
              COLORS["benchmark"], COLORS["gray"]]

    for idx, (name, curve) in enumerate(excess_curves.items()):
        color = colors[idx % len(colors)]
        ax.plot(curve.index, curve.values, linewidth=2,
                label=name, color=color)

    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Excess Return (Cumulative)")
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_table(
    metrics_dict: Dict[str, dict],
    title: str = "Performance Metrics Summary",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制绩效指标汇总表（对应研报图表12/14/17/20/28/30/32）
    """
    rows = []
    for name, m in metrics_dict.items():
        rows.append({
            "Strategy": name,
            "Ann. Return": f"{m.get('excess_annual_return', 0):.2%}",
            "Ann. Volatility": f"{m.get('excess_annual_volatility', 0):.2%}",
            "Info Ratio": f"{m.get('excess_info_ratio', 0):.2f}",
            "Max Drawdown": f"{m.get('excess_max_drawdown', 0):.2%}",
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, max(2, len(rows) * 0.6 + 1.5)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#ECF0F1")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_signal_correlation_matrix(
    corr_matrix: pd.DataFrame,
    title: str = "Signal Correlation Matrix (Stock Pool Overlap)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制事件信号相关性矩阵热力图（对应研报图表21）
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)
    sns.heatmap(
        corr_matrix * 100,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        center=10,
        ax=ax,
        cbar_kws={"label": "Overlap %"},
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_full_report(
    all_figures: Dict[str, plt.Figure],
    save_dir: str = "/opt/cursor/artifacts",
) -> None:
    """保存所有图表到指定目录"""
    import os
    os.makedirs(save_dir, exist_ok=True)

    for name, fig in all_figures.items():
        filepath = os.path.join(save_dir, f"{name}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {filepath}")
