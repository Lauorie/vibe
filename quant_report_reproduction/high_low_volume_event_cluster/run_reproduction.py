#!/usr/bin/env python3
"""
国盛证券 - "量价淘金"选股因子系列研究（十五）
高/低位放量事件簇：正负向信号的有机结合

完整复现流程：
1. 日频数据下的高/低位放量事件表现 (研报第二节)
2. 高频数据下的高/低位放量事件簇 (研报第三节)
3. 信号筛选与合成 (研报第3.4节)
4. 低位放量+高位放量的有机结合 (研报第3.4.3节)
5. 绩效评估与可视化

由于无法获取 Wind/通联 的分钟级真实数据，
本脚本使用模拟数据验证方法论的正确性。
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from high_low_volume_event_cluster.data_utils import (
    generate_simulated_daily_data,
    generate_simulated_minute_data,
    get_benchmark_returns,
)
from high_low_volume_event_cluster.daily_signals import (
    detect_daily_low_volume_surge,
    detect_daily_high_volume_surge,
    compute_event_stats,
    compute_post_event_returns,
)
from high_low_volume_event_cluster.hf_signals import (
    identify_high_low_minutes_intraday_quantile,
    identify_high_low_minutes_intraday_meanstd,
    identify_volume_surge_intraday_quantile,
    identify_volume_surge_intraday_meanstd,
    signal_price_first_volume_second,
    signal_volume_first_price_second,
)
from high_low_volume_event_cluster.channel_strategy import (
    ChannelStrategy,
    compute_channel_strategy_metrics,
)
from high_low_volume_event_cluster.signal_screening import (
    compute_signal_overlap_correlation,
    composite_signal,
)
from high_low_volume_event_cluster.performance import (
    plot_event_sample_count,
    plot_post_event_returns,
    plot_channel_strategy_nav,
    plot_excess_return_comparison,
    plot_metrics_table,
    plot_signal_correlation_matrix,
)


SAVE_DIR = "/opt/cursor/artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)


def section_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================
# Step 1: Generate Simulated Data
# ============================================================
def step1_generate_data():
    section_header("Step 1: Generating Simulated Market Data")

    close_df, volume_df, open_df = generate_simulated_daily_data(
        n_stocks=100, n_days=2400, start_date="2016-01-04", seed=42
    )
    benchmark_returns = get_benchmark_returns(close_df)

    print(f"Stock universe: {close_df.shape[1]} stocks")
    print(f"Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
    print(f"Total trading days: {len(close_df)}")
    print(f"Benchmark avg daily return: {benchmark_returns.mean()*100:.4f}%")

    return close_df, volume_df, open_df, benchmark_returns


# ============================================================
# Step 2: Daily Frequency Event Signals (Section 2)
# ============================================================
def step2_daily_signals(close_df, volume_df, open_df, benchmark_returns):
    section_header("Step 2: Daily Frequency High/Low Volume Surge Events")

    print("Detecting daily low-position volume surge events...")
    low_signal = detect_daily_low_volume_surge(close_df, volume_df)
    low_stats = compute_event_stats(low_signal)
    total_low = low_signal.sum().sum()
    avg_daily_low = low_stats["n_triggered"].mean()
    print(f"  Total low-vol-surge events: {total_low:.0f}")
    print(f"  Avg daily triggers: {avg_daily_low:.2f}")

    print("\nDetecting daily high-position volume surge events...")
    high_signal = detect_daily_high_volume_surge(close_df, volume_df)
    high_stats = compute_event_stats(high_signal)
    total_high = high_signal.sum().sum()
    avg_daily_high = high_stats["n_triggered"].mean()
    print(f"  Total high-vol-surge events: {total_high:.0f}")
    print(f"  Avg daily triggers: {avg_daily_high:.2f}")

    print("\nComputing post-event returns for low-position events...")
    low_abs, low_excess = compute_post_event_returns(
        low_signal, open_df, close_df, benchmark_returns, max_horizon=60
    )
    if len(low_excess) > 0:
        peak_day = low_excess.idxmax()
        print(f"  Excess return peaks at day {peak_day}: {low_excess[peak_day]*100:.2f}%")

    print("\nComputing post-event returns for high-position events...")
    high_abs, high_excess = compute_post_event_returns(
        high_signal, open_df, close_df, benchmark_returns, max_horizon=60
    )
    if len(high_excess) > 0:
        min_day = high_excess.idxmin()
        print(f"  Excess return trough at day {min_day}: {high_excess[min_day]*100:.2f}%")

    print("\nPlotting daily signal charts...")

    fig1 = plot_event_sample_count(
        low_stats,
        title="Daily Low-Position Volume Surge: Event Count",
        save_path=os.path.join(SAVE_DIR, "fig03_daily_low_event_count.png")
    )
    plt.close(fig1)

    if len(low_abs) > 0:
        fig2 = plot_post_event_returns(
            low_abs, low_excess,
            title="Low-Position Volume Surge: Post-Event Returns",
            save_path=os.path.join(SAVE_DIR, "fig04_daily_low_post_event_returns.png")
        )
        plt.close(fig2)

    fig3 = plot_event_sample_count(
        high_stats,
        title="Daily High-Position Volume Surge: Event Count",
        save_path=os.path.join(SAVE_DIR, "fig05_daily_high_event_count.png")
    )
    plt.close(fig3)

    if len(high_abs) > 0:
        fig4 = plot_post_event_returns(
            high_abs, high_excess,
            title="High-Position Volume Surge: Post-Event Returns",
            save_path=os.path.join(SAVE_DIR, "fig06_daily_high_post_event_returns.png")
        )
        plt.close(fig4)

    print("\nRunning daily channel strategy for low-position signal...")
    engine = ChannelStrategy(n_channels=4, holding_period=20)
    low_result = engine.run(low_signal, open_df, close_df)
    low_metrics = compute_channel_strategy_metrics(low_result, benchmark_returns)
    print(f"  Strategy Ann. Return: {low_metrics['strategy_annual_return']:.2%}")
    print(f"  Excess Ann. Return: {low_metrics['excess_annual_return']:.2%}")
    print(f"  Excess Info Ratio: {low_metrics['excess_info_ratio']:.2f}")
    print(f"  Excess Max Drawdown: {low_metrics['excess_max_drawdown']:.2%}")

    bench_cum = (1 + benchmark_returns).cumprod()
    fig5 = plot_channel_strategy_nav(
        low_result, bench_cum,
        title="Daily Low-Volume-Surge: Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig07_daily_low_channel_nav.png")
    )
    plt.close(fig5)

    print("\nRunning daily channel strategy for high-position signal...")
    high_result = engine.run(high_signal, open_df, close_df)
    high_metrics = compute_channel_strategy_metrics(high_result, benchmark_returns)
    print(f"  Strategy Ann. Return: {high_metrics['strategy_annual_return']:.2%}")
    print(f"  Excess Ann. Return: {high_metrics['excess_annual_return']:.2%}")
    print(f"  Excess Info Ratio: {high_metrics['excess_info_ratio']:.2f}")
    print(f"  Excess Max Drawdown: {high_metrics['excess_max_drawdown']:.2%}")

    fig6 = plot_channel_strategy_nav(
        high_result, bench_cum,
        title="Daily High-Volume-Surge: Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig08_daily_high_channel_nav.png")
    )
    plt.close(fig6)

    return low_signal, high_signal, low_metrics, high_metrics


# ============================================================
# Step 3: High-Frequency Event Cluster (Section 3)
# ============================================================
def step3_hf_event_cluster(close_df, volume_df, open_df, benchmark_returns):
    section_header("Step 3: High-Frequency Event Cluster Construction")

    n_stocks_for_hf = 30
    stocks = close_df.columns[:n_stocks_for_hf].tolist()
    print(f"Generating minute-level data for {n_stocks_for_hf} stocks...")

    all_high_signals = {}
    all_low_signals = {}

    signal_configs = [
        ("intraday_quantile_PV", "intraday_quantile_vol", "PV"),
        ("intraday_meanstd_PV", "intraday_meanstd_vol", "PV"),
        ("intraday_quantile_VP", "intraday_quantile_vol", "VP"),
        ("intraday_meanstd_VP", "intraday_meanstd_vol", "VP"),
    ]

    for stock_idx, stock in enumerate(stocks):
        if stock_idx % 10 == 0:
            print(f"  Processing stock {stock_idx+1}/{n_stocks_for_hf}: {stock}")

        mc, mv = generate_simulated_minute_data(
            close_df[stock], volume_df[stock],
            n_minutes=240, seed=42 + stock_idx
        )

        is_high_q, is_low_q = identify_high_low_minutes_intraday_quantile(mc)
        is_high_m, is_low_m = identify_high_low_minutes_intraday_meanstd(mc)
        surge_q = identify_volume_surge_intraday_quantile(mv)
        surge_m = identify_volume_surge_intraday_meanstd(mv)

        h1, l1 = signal_price_first_volume_second(is_high_q, is_low_q, mv, surge_q)
        h2, l2 = signal_price_first_volume_second(is_high_m, is_low_m, mv, surge_m)
        h3, l3 = signal_volume_first_price_second(surge_q, mc)
        h4, l4 = signal_volume_first_price_second(surge_m, mc)

        config_results = [
            ("HF_intraQ_PV", h1, l1),
            ("HF_intraMS_PV", h2, l2),
            ("HF_intraQ_VP", h3, l3),
            ("HF_intraMS_VP", h4, l4),
        ]

        for sig_name, h_sig, l_sig in config_results:
            if sig_name not in all_high_signals:
                all_high_signals[sig_name] = {}
                all_low_signals[sig_name] = {}
            all_high_signals[sig_name][stock] = h_sig
            all_low_signals[sig_name][stock] = l_sig

    high_signal_dfs = {}
    low_signal_dfs = {}
    for name in all_high_signals:
        high_signal_dfs[name] = pd.DataFrame(all_high_signals[name])
        low_signal_dfs[name] = pd.DataFrame(all_low_signals[name])

    print(f"\nGenerated {len(high_signal_dfs)} HF signal types")
    for name, df in low_signal_dfs.items():
        avg_triggers = df.sum(axis=1).mean()
        print(f"  {name}: avg daily low triggers = {avg_triggers:.1f}")

    return high_signal_dfs, low_signal_dfs, stocks


# ============================================================
# Step 4: Signal Screening & Composite (Section 3.4)
# ============================================================
def step4_signal_composite(
    high_signal_dfs, low_signal_dfs, stocks,
    close_df, volume_df, open_df, benchmark_returns
):
    section_header("Step 4: Signal Screening & Composite Signal Construction")

    sub_close = close_df[stocks]
    sub_open = open_df[stocks]
    sub_bench = get_benchmark_returns(sub_close)

    engine = ChannelStrategy(n_channels=4, holding_period=20)

    print("Evaluating low-position volume surge signals...")
    low_metrics_all = {}
    for name, sig in low_signal_dfs.items():
        try:
            result = engine.run(sig, sub_open, sub_close)
            metrics = compute_channel_strategy_metrics(result, sub_bench)
            low_metrics_all[name] = metrics
            print(f"  {name}: Excess IR = {metrics['excess_info_ratio']:.2f}, "
                  f"Excess Return = {metrics['excess_annual_return']:.2%}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    print("\nEvaluating high-position volume surge signals...")
    high_metrics_all = {}
    for name, sig in high_signal_dfs.items():
        try:
            result = engine.run(sig, sub_open, sub_close)
            metrics = compute_channel_strategy_metrics(result, sub_bench)
            neg_ir = -metrics['excess_info_ratio']
            neg_ret = -metrics['excess_annual_return']
            high_metrics_all[name] = metrics
            print(f"  {name}: Negative Excess IR = {neg_ir:.2f}, "
                  f"Benchmark vs Strategy = {neg_ret:.2%}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    print("\n--- Constructing Composite Signals ---")

    low_selected = list(low_signal_dfs.keys())
    high_selected = list(high_signal_dfs.keys())

    low_composite = composite_signal(low_signal_dfs, low_selected, threshold_ratio=0.5)
    high_composite = composite_signal(high_signal_dfs, high_selected, threshold_ratio=0.5)

    low_comp_triggers = low_composite.sum(axis=1).mean()
    high_comp_triggers = high_composite.sum(axis=1).mean()
    print(f"\nLow composite signal: avg daily triggers = {low_comp_triggers:.1f}")
    print(f"High composite signal: avg daily triggers = {high_comp_triggers:.1f}")

    print("\n--- Low Composite Signal Channel Strategy ---")
    low_comp_result = engine.run(low_composite, sub_open, sub_close)
    low_comp_metrics = compute_channel_strategy_metrics(low_comp_result, sub_bench)
    print(f"  Strategy Ann. Return: {low_comp_metrics['strategy_annual_return']:.2%}")
    print(f"  Excess Ann. Return: {low_comp_metrics['excess_annual_return']:.2%}")
    print(f"  Excess Info Ratio: {low_comp_metrics['excess_info_ratio']:.2f}")
    print(f"  Excess Max Drawdown: {low_comp_metrics['excess_max_drawdown']:.2%}")

    bench_cum = (1 + sub_bench).cumprod()

    fig13 = plot_channel_strategy_nav(
        low_comp_result, bench_cum,
        title="Low-Volume-Surge Composite Signal: Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig13_low_composite_channel_nav.png")
    )
    plt.close(fig13)

    print("\n--- High Composite Signal Channel Strategy ---")
    high_comp_result = engine.run(high_composite, sub_open, sub_close)
    high_comp_metrics = compute_channel_strategy_metrics(high_comp_result, sub_bench)
    print(f"  Strategy Ann. Return: {high_comp_metrics['strategy_annual_return']:.2%}")
    print(f"  Benchmark vs Strategy: {-high_comp_metrics['excess_annual_return']:.2%}")
    print(f"  Negative Excess IR: {-high_comp_metrics['excess_info_ratio']:.2f}")

    fig16 = plot_channel_strategy_nav(
        high_comp_result, bench_cum,
        title="High-Volume-Surge Composite Signal: Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig16_high_composite_channel_nav.png")
    )
    plt.close(fig16)

    return (
        low_composite, high_composite,
        low_comp_result, high_comp_result,
        low_comp_metrics, high_comp_metrics,
        sub_open, sub_close, sub_bench
    )


# ============================================================
# Step 5: Combined Strategy (Section 3.4.3)
# ============================================================
def step5_combined_strategy(
    low_composite, high_composite,
    low_comp_result, low_comp_metrics,
    sub_open, sub_close, sub_bench
):
    section_header("Step 5: Combined Strategy (Low + High Volume Surge)")

    print("Running combined channel strategy (low signal + high signal exclusion)...")
    engine = ChannelStrategy(n_channels=4, holding_period=20)
    combined_result = engine.run(
        low_composite, sub_open, sub_close,
        exclude_signal_df=high_composite
    )
    combined_metrics = compute_channel_strategy_metrics(combined_result, sub_bench)

    print(f"\n{'Metric':<25} {'Low Only':<15} {'Low + High':<15}")
    print("-" * 55)
    print(f"{'Excess Ann. Return':<25} "
          f"{low_comp_metrics['excess_annual_return']:<15.2%} "
          f"{combined_metrics['excess_annual_return']:<15.2%}")
    print(f"{'Excess Ann. Volatility':<25} "
          f"{low_comp_metrics['excess_annual_volatility']:<15.2%} "
          f"{combined_metrics['excess_annual_volatility']:<15.2%}")
    print(f"{'Excess Info Ratio':<25} "
          f"{low_comp_metrics['excess_info_ratio']:<15.2f} "
          f"{combined_metrics['excess_info_ratio']:<15.2f}")
    print(f"{'Excess Max Drawdown':<25} "
          f"{low_comp_metrics['excess_max_drawdown']:<15.2%} "
          f"{combined_metrics['excess_max_drawdown']:<15.2%}")

    bench_cum = (1 + sub_bench).cumprod()
    fig18 = plot_channel_strategy_nav(
        combined_result, bench_cum,
        title="Combined (Low + High Exclusion): Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig18_combined_channel_nav.png")
    )
    plt.close(fig18)

    fig19 = plot_excess_return_comparison(
        {
            "Low-Volume-Surge Only": low_comp_metrics["excess_cumulative"],
            "Low + High Combined": combined_metrics["excess_cumulative"],
        },
        title="Excess Return: Before vs After Signal Combination",
        save_path=os.path.join(SAVE_DIR, "fig19_excess_comparison.png")
    )
    plt.close(fig19)

    metrics_table = {
        "Low-Volume-Surge Only": low_comp_metrics,
        "Low + High Combined": combined_metrics,
    }
    fig20 = plot_metrics_table(
        metrics_table,
        title="Performance Comparison: Before vs After Combining Signals",
        save_path=os.path.join(SAVE_DIR, "fig20_metrics_comparison.png")
    )
    plt.close(fig20)

    corr = compute_signal_overlap_correlation(low_composite, high_composite)
    print(f"\nLow-High composite signal overlap: {corr:.2%}")

    return combined_result, combined_metrics


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Guosheng Securities - Quant Research Report Reproduction")
    print("  'Volume-Price Gold Mining' Factor Series (XV)")
    print("  High/Low Position Volume Surge Event Cluster")
    print("=" * 70)

    close_df, volume_df, open_df, benchmark_returns = step1_generate_data()

    low_sig, high_sig, low_m, high_m = step2_daily_signals(
        close_df, volume_df, open_df, benchmark_returns
    )

    high_dfs, low_dfs, stocks = step3_hf_event_cluster(
        close_df, volume_df, open_df, benchmark_returns
    )

    (
        low_comp, high_comp,
        low_comp_result, high_comp_result,
        low_comp_metrics, high_comp_metrics,
        sub_open, sub_close, sub_bench
    ) = step4_signal_composite(
        high_dfs, low_dfs, stocks,
        close_df, volume_df, open_df, benchmark_returns
    )

    combined_result, combined_metrics = step5_combined_strategy(
        low_comp, high_comp,
        low_comp_result, low_comp_metrics,
        sub_open, sub_close, sub_bench
    )

    section_header("REPRODUCTION COMPLETE")
    print("All charts saved to:", SAVE_DIR)
    print("\nKey Results Summary:")
    print(f"  Daily low-vol-surge events: {low_sig.sum().sum():.0f} total")
    print(f"  Daily high-vol-surge events: {high_sig.sum().sum():.0f} total")
    print(f"  HF Low composite excess IR: {low_comp_metrics['excess_info_ratio']:.2f}")
    print(f"  HF High composite negative excess: {-high_comp_metrics['excess_annual_return']:.2%}")
    print(f"  Combined excess IR: {combined_metrics['excess_info_ratio']:.2f}")
    print(f"  Combined excess max DD: {combined_metrics['excess_max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
