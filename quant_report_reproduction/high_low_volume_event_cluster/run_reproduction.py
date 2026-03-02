#!/usr/bin/env python3
"""
国盛证券 - "量价淘金"选股因子系列研究（十五）
高/低位放量事件簇：正负向信号的有机结合

使用 akshare 真实 A 股数据的完整复现流程:
1. 日频数据下的高/低位放量事件表现 (研报第二节)
2. 高频数据下的高/低位放量事件簇 (研报第三节)
3. 信号筛选与合成 (研报第3.4节)
4. 低位放量+高位放量的有机结合 (研报第3.4.3节)
5. 绩效评估与可视化
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
    prepare_real_data,
    generate_realistic_minute_data,
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
    identify_high_low_minutes_rolling_quantile,
    identify_high_low_minutes_rolling_meanstd,
    identify_volume_surge_intraday_quantile,
    identify_volume_surge_intraday_meanstd,
    identify_volume_surge_rolling_quantile,
    identify_volume_surge_rolling_meanstd,
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
# Step 1: Load Real Data
# ============================================================
def step1_load_data():
    section_header("Step 1: Loading Real A-Share Market Data (akshare)")

    close_df, volume_df, open_df, amount_df, benchmark_returns = prepare_real_data(
        n_stocks=50,
        start_date="20150601",
        end_date="20251031",
    )

    print(f"\nData Summary:")
    print(f"  Stock universe: {close_df.shape[1]} stocks (CSI 800 constituents)")
    print(f"  Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
    print(f"  Total trading days: {len(close_df)}")
    print(f"  Benchmark avg daily return: {benchmark_returns.mean()*100:.4f}%")

    return close_df, volume_df, open_df, amount_df, benchmark_returns


# ============================================================
# Step 2: Daily Frequency Event Signals (Section 2)
# ============================================================
def step2_daily_signals(close_df, volume_df, open_df, benchmark_returns):
    section_header("Step 2: Daily High/Low Volume Surge Events (Report Section 2)")

    print("Detecting daily low-position volume surge events...")
    print("  Conditions: close <= 10th percentile (120d), volume > mean+1.5*std (120d)")
    low_signal = detect_daily_low_volume_surge(close_df, volume_df, lookback=120)
    low_stats = compute_event_stats(low_signal)
    total_low = int(low_signal.sum().sum())
    avg_daily_low = low_stats["n_triggered"].mean()
    print(f"  Total events: {total_low}")
    print(f"  Avg daily triggers: {avg_daily_low:.2f}")

    print("\nDetecting daily high-position volume surge events...")
    print("  Conditions: close >= 90th percentile (120d), volume > mean+1.5*std (120d)")
    high_signal = detect_daily_high_volume_surge(close_df, volume_df, lookback=120)
    high_stats = compute_event_stats(high_signal)
    total_high = int(high_signal.sum().sum())
    avg_daily_high = high_stats["n_triggered"].mean()
    print(f"  Total events: {total_high}")
    print(f"  Avg daily triggers: {avg_daily_high:.2f}")

    print("\nComputing post-event returns (next-day open to future close)...")
    low_abs, low_excess = compute_post_event_returns(
        low_signal, open_df, close_df, benchmark_returns, max_horizon=60
    )
    if len(low_excess) > 0:
        peak_day = low_excess.idxmax()
        print(f"  Low-vol-surge excess peaks at day {peak_day}: {low_excess[peak_day]*100:.2f}%")

    high_abs, high_excess = compute_post_event_returns(
        high_signal, open_df, close_df, benchmark_returns, max_horizon=60
    )
    if len(high_excess) > 0:
        min_day = high_excess.idxmin()
        print(f"  High-vol-surge excess trough at day {min_day}: {high_excess[min_day]*100:.2f}%")

    print("\nPlotting charts...")

    fig1 = plot_event_sample_count(
        low_stats,
        title="Fig.3: Daily Low-Position Volume Surge Event Count (CSI 800)",
        save_path=os.path.join(SAVE_DIR, "fig03_daily_low_event_count.png")
    )
    plt.close(fig1)

    if len(low_abs) > 0:
        fig2 = plot_post_event_returns(
            low_abs, low_excess,
            title="Fig.4: Low-Position Volume Surge Post-Event Returns",
            save_path=os.path.join(SAVE_DIR, "fig04_daily_low_post_event_returns.png")
        )
        plt.close(fig2)

    fig3 = plot_event_sample_count(
        high_stats,
        title="Fig.5: Daily High-Position Volume Surge Event Count (CSI 800)",
        save_path=os.path.join(SAVE_DIR, "fig05_daily_high_event_count.png")
    )
    plt.close(fig3)

    if len(high_abs) > 0:
        fig4 = plot_post_event_returns(
            high_abs, high_excess,
            title="Fig.6: High-Position Volume Surge Post-Event Returns",
            save_path=os.path.join(SAVE_DIR, "fig06_daily_high_post_event_returns.png")
        )
        plt.close(fig4)

    print("\nRunning daily channel strategy (4 channels, 20-day holding)...")
    engine = ChannelStrategy(n_channels=4, holding_period=20)

    low_result = engine.run(low_signal, open_df, close_df)
    low_metrics = compute_channel_strategy_metrics(low_result, benchmark_returns)
    print(f"\n  [Low-Vol-Surge Channel Strategy]")
    print(f"  Strategy Ann. Return:    {low_metrics['strategy_annual_return']:.2%}")
    print(f"  Excess Ann. Return:      {low_metrics['excess_annual_return']:.2%}")
    print(f"  Excess Info Ratio:       {low_metrics['excess_info_ratio']:.2f}")
    print(f"  Excess Max Drawdown:     {low_metrics['excess_max_drawdown']:.2%}")

    bench_cum = (1 + benchmark_returns).cumprod()
    fig5 = plot_channel_strategy_nav(
        low_result, bench_cum,
        title="Fig.7: Daily Low-Volume-Surge Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig07_daily_low_channel_nav.png")
    )
    plt.close(fig5)

    high_result = engine.run(high_signal, open_df, close_df)
    high_metrics = compute_channel_strategy_metrics(high_result, benchmark_returns)
    print(f"\n  [High-Vol-Surge Channel Strategy]")
    print(f"  Strategy Ann. Return:    {high_metrics['strategy_annual_return']:.2%}")
    print(f"  Excess Ann. Return:      {high_metrics['excess_annual_return']:.2%}")
    print(f"  Excess Info Ratio:       {high_metrics['excess_info_ratio']:.2f}")
    print(f"  Excess Max Drawdown:     {high_metrics['excess_max_drawdown']:.2%}")

    fig6 = plot_channel_strategy_nav(
        high_result, bench_cum,
        title="Fig.8: Daily High-Volume-Surge Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig08_daily_high_channel_nav.png")
    )
    plt.close(fig6)

    return low_signal, high_signal, low_metrics, high_metrics


# ============================================================
# Step 3: High-Frequency Event Cluster (Section 3)
# ============================================================
def step3_hf_event_cluster(close_df, volume_df, open_df, amount_df, benchmark_returns):
    section_header("Step 3: HF Event Cluster Construction (Report Section 3)")

    n_stocks_for_hf = min(20, close_df.shape[1])
    vol_rank = volume_df.mean().sort_values(ascending=False)
    stocks = vol_rank.index[:n_stocks_for_hf].tolist()
    print(f"Generating minute-level data for top-{n_stocks_for_hf} liquid stocks...")

    all_high_signals = {}
    all_low_signals = {}

    signal_type_names = [
        "HF_intraQ_PV_vol",
        "HF_intraMS_PV_vol",
        "HF_intraQ_VP_vol",
        "HF_intraMS_VP_vol",
        "HF_intraQ_PV_amt",
        "HF_intraMS_PV_amt",
        "HF_rollQ_PV_vol",
        "HF_rollMS_PV_vol",
    ]
    for name in signal_type_names:
        all_high_signals[name] = {}
        all_low_signals[name] = {}

    for stock_idx, stock in enumerate(stocks):
        if stock_idx % 10 == 0:
            print(f"  Processing stock {stock_idx+1}/{n_stocks_for_hf}: {stock}")

        s_close = close_df[stock].dropna()
        s_volume = volume_df[stock].reindex(s_close.index).fillna(0)
        s_amount = amount_df[stock].reindex(s_close.index).fillna(0) if stock in amount_df.columns else s_volume.copy()

        mc, mv = generate_realistic_minute_data(
            s_close, s_volume, n_minutes=240, seed=42 + stock_idx
        )
        _, ma = generate_realistic_minute_data(
            s_close, s_amount, n_minutes=240, seed=1000 + stock_idx
        )

        is_high_q, is_low_q = identify_high_low_minutes_intraday_quantile(mc)
        is_high_m, is_low_m = identify_high_low_minutes_intraday_meanstd(mc, n_std=3.0)
        surge_q_vol = identify_volume_surge_intraday_quantile(mv)
        surge_m_vol = identify_volume_surge_intraday_meanstd(mv, n_std=3.0)
        surge_q_amt = identify_volume_surge_intraday_quantile(ma)
        surge_m_amt = identify_volume_surge_intraday_meanstd(ma, n_std=3.0)

        h1, l1 = signal_price_first_volume_second(is_high_q, is_low_q, mv, surge_q_vol)
        h2, l2 = signal_price_first_volume_second(is_high_m, is_low_m, mv, surge_m_vol)
        h3, l3 = signal_volume_first_price_second(surge_q_vol, mc)
        h4, l4 = signal_volume_first_price_second(surge_m_vol, mc)
        h5, l5 = signal_price_first_volume_second(is_high_q, is_low_q, ma, surge_q_amt)
        h6, l6 = signal_price_first_volume_second(is_high_m, is_low_m, ma, surge_m_amt)

        is_high_rq, is_low_rq = identify_high_low_minutes_rolling_quantile(mc, lookback_days=20)
        surge_rq_vol = identify_volume_surge_rolling_quantile(mv, lookback_days=20)
        h7, l7 = signal_price_first_volume_second(is_high_rq, is_low_rq, mv, surge_rq_vol)

        is_high_rm, is_low_rm = identify_high_low_minutes_rolling_meanstd(mc, lookback_days=20, n_std=1.5)
        surge_rm_vol = identify_volume_surge_rolling_meanstd(mv, lookback_days=20, n_std=1.5)
        h8, l8 = signal_price_first_volume_second(is_high_rm, is_low_rm, mv, surge_rm_vol)

        results = [
            ("HF_intraQ_PV_vol", h1, l1),
            ("HF_intraMS_PV_vol", h2, l2),
            ("HF_intraQ_VP_vol", h3, l3),
            ("HF_intraMS_VP_vol", h4, l4),
            ("HF_intraQ_PV_amt", h5, l5),
            ("HF_intraMS_PV_amt", h6, l6),
            ("HF_rollQ_PV_vol", h7, l7),
            ("HF_rollMS_PV_vol", h8, l8),
        ]

        for sig_name, h_sig, l_sig in results:
            all_high_signals[sig_name][stock] = h_sig
            all_low_signals[sig_name][stock] = l_sig

    high_signal_dfs = {}
    low_signal_dfs = {}
    for name in signal_type_names:
        if all_high_signals[name]:
            high_signal_dfs[name] = pd.DataFrame(all_high_signals[name])
            low_signal_dfs[name] = pd.DataFrame(all_low_signals[name])

    print(f"\nGenerated {len(high_signal_dfs)} HF signal types")
    for name, df in low_signal_dfs.items():
        avg_triggers = df.sum(axis=1).mean()
        print(f"  {name}: avg daily low triggers = {avg_triggers:.1f}, "
              f"high triggers = {high_signal_dfs[name].sum(axis=1).mean():.1f}")

    return high_signal_dfs, low_signal_dfs, stocks


# ============================================================
# Step 4: Signal Screening & Composite (Section 3.4)
# ============================================================
def step4_signal_composite(
    high_signal_dfs, low_signal_dfs, stocks,
    close_df, volume_df, open_df, benchmark_returns
):
    section_header("Step 4: Signal Screening & Composite (Report Section 3.4)")

    sub_close = close_df[stocks]
    sub_open = open_df[stocks]
    sub_bench = get_benchmark_returns(sub_close)

    engine = ChannelStrategy(n_channels=4, holding_period=20)

    print("--- Evaluating Low-Position Volume Surge Signals ---")
    low_metrics_all = {}
    for name, sig in low_signal_dfs.items():
        try:
            result = engine.run(sig, sub_open, sub_close)
            metrics = compute_channel_strategy_metrics(result, sub_bench)
            low_metrics_all[name] = metrics
            print(f"  {name}: Excess IR={metrics['excess_info_ratio']:.2f}, "
                  f"Excess Ret={metrics['excess_annual_return']:.2%}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    print("\n--- Evaluating High-Position Volume Surge Signals ---")
    high_metrics_all = {}
    for name, sig in high_signal_dfs.items():
        try:
            result = engine.run(sig, sub_open, sub_close)
            metrics = compute_channel_strategy_metrics(result, sub_bench)
            high_metrics_all[name] = metrics
            neg_ir = -metrics['excess_info_ratio']
            neg_ret = -metrics['excess_annual_return']
            print(f"  {name}: Negative IR={neg_ir:.2f}, "
                  f"Benchmark vs Strategy={neg_ret:.2%}")
        except Exception as e:
            print(f"  {name}: Error - {e}")

    print("\n--- Constructing Composite Signals ---")

    low_selected = list(low_signal_dfs.keys())
    high_selected = list(high_signal_dfs.keys())

    low_composite = composite_signal(low_signal_dfs, low_selected, threshold_ratio=0.5)
    high_composite = composite_signal(high_signal_dfs, high_selected, threshold_ratio=0.5)

    low_triggers = low_composite.sum(axis=1).mean()
    high_triggers = high_composite.sum(axis=1).mean()
    print(f"Low composite: avg daily triggers = {low_triggers:.1f}")
    print(f"High composite: avg daily triggers = {high_triggers:.1f}")

    print("\n--- Low Composite Signal Channel Strategy ---")
    low_comp_result = engine.run(low_composite, sub_open, sub_close)
    low_comp_metrics = compute_channel_strategy_metrics(low_comp_result, sub_bench)
    print(f"  Strategy Ann. Return:    {low_comp_metrics['strategy_annual_return']:.2%}")
    print(f"  Excess Ann. Return:      {low_comp_metrics['excess_annual_return']:.2%}")
    print(f"  Excess Ann. Volatility:  {low_comp_metrics['excess_annual_volatility']:.2%}")
    print(f"  Excess Info Ratio:       {low_comp_metrics['excess_info_ratio']:.2f}")
    print(f"  Excess Max Drawdown:     {low_comp_metrics['excess_max_drawdown']:.2%}")

    bench_cum = (1 + sub_bench).cumprod()

    fig13 = plot_channel_strategy_nav(
        low_comp_result, bench_cum,
        title="Fig.13: Low-Volume-Surge Composite Signal Channel Strategy",
        save_path=os.path.join(SAVE_DIR, "fig13_low_composite_nav.png")
    )
    plt.close(fig13)

    fig14 = plot_metrics_table(
        {"Low Composite Signal": low_comp_metrics},
        title="Fig.14: Low-Volume-Surge Composite Signal Metrics",
        save_path=os.path.join(SAVE_DIR, "fig14_low_composite_metrics.png")
    )
    plt.close(fig14)

    print("\n--- High Composite Signal Channel Strategy ---")
    high_comp_result = engine.run(high_composite, sub_open, sub_close)
    high_comp_metrics = compute_channel_strategy_metrics(high_comp_result, sub_bench)
    print(f"  Strategy Ann. Return:    {high_comp_metrics['strategy_annual_return']:.2%}")
    print(f"  Benchmark vs Strategy:   {-high_comp_metrics['excess_annual_return']:.2%}")
    print(f"  Negative Excess IR:      {-high_comp_metrics['excess_info_ratio']:.2f}")
    print(f"  Excess Max Drawdown:     {high_comp_metrics['excess_max_drawdown']:.2%}")

    fig16 = plot_channel_strategy_nav(
        high_comp_result, bench_cum,
        title="Fig.16: High-Volume-Surge Composite Signal Channel Strategy",
        save_path=os.path.join(SAVE_DIR, "fig16_high_composite_nav.png")
    )
    plt.close(fig16)

    fig17 = plot_metrics_table(
        {"High Composite Signal": high_comp_metrics},
        title="Fig.17: High-Volume-Surge Composite Signal Metrics",
        save_path=os.path.join(SAVE_DIR, "fig17_high_composite_metrics.png")
    )
    plt.close(fig17)

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
    section_header("Step 5: Combined Strategy — Low + High Exclusion (Section 3.4.3)")

    engine = ChannelStrategy(n_channels=4, holding_period=20)

    print("Running combined channel strategy...")
    print("  Rule: Low-vol-surge positive screen + High-vol-surge negative exclusion")
    combined_result = engine.run(
        low_composite, sub_open, sub_close,
        exclude_signal_df=high_composite
    )
    combined_metrics = compute_channel_strategy_metrics(combined_result, sub_bench)

    print(f"\n{'Metric':<25} {'Low Only':<15} {'Low+High':<15} {'Change':<15}")
    print("-" * 70)

    def fmt_compare(key, pct=True):
        v1 = low_comp_metrics[key]
        v2 = combined_metrics[key]
        if pct:
            return f"{v1:.2%}", f"{v2:.2%}", f"{(v2-v1):.2%}"
        else:
            return f"{v1:.2f}", f"{v2:.2f}", f"{(v2-v1):+.2f}"

    for label, key, pct in [
        ("Excess Ann. Return", "excess_annual_return", True),
        ("Excess Ann. Volatility", "excess_annual_volatility", True),
        ("Excess Info Ratio", "excess_info_ratio", False),
        ("Excess Max Drawdown", "excess_max_drawdown", True),
    ]:
        v1, v2, diff = fmt_compare(key, pct)
        print(f"  {label:<25} {v1:<15} {v2:<15} {diff:<15}")

    bench_cum = (1 + sub_bench).cumprod()

    fig18 = plot_channel_strategy_nav(
        combined_result, bench_cum,
        title="Fig.18: Combined (Low+High Exclusion) Channel Strategy NAV",
        save_path=os.path.join(SAVE_DIR, "fig18_combined_channel_nav.png")
    )
    plt.close(fig18)

    fig19 = plot_excess_return_comparison(
        {
            "Low-Vol-Surge Only": low_comp_metrics["excess_cumulative"],
            "Low + High Combined": combined_metrics["excess_cumulative"],
        },
        title="Fig.19: Excess Return Before vs After Signal Combination",
        save_path=os.path.join(SAVE_DIR, "fig19_excess_comparison.png")
    )
    plt.close(fig19)

    fig20 = plot_metrics_table(
        {
            "Low-Vol-Surge Only (Excess)": low_comp_metrics,
            "Low+High Combined (Excess)": combined_metrics,
        },
        title="Fig.20: Performance Comparison Before vs After Combining Signals",
        save_path=os.path.join(SAVE_DIR, "fig20_metrics_comparison.png")
    )
    plt.close(fig20)

    corr = compute_signal_overlap_correlation(low_composite, high_composite)
    print(f"\n  Low-High signal overlap: {corr:.2%}")

    return combined_result, combined_metrics


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Guosheng Securities — Quant Research Report Reproduction")
    print("  'Volume-Price Gold Mining' Factor Series (XV)")
    print("  High/Low Position Volume Surge Event Cluster")
    print("  Data Source: akshare (Real A-Share Market Data)")
    print("=" * 70)

    close_df, volume_df, open_df, amount_df, benchmark_returns = step1_load_data()

    low_sig, high_sig, low_m, high_m = step2_daily_signals(
        close_df, volume_df, open_df, benchmark_returns
    )

    high_dfs, low_dfs, stocks = step3_hf_event_cluster(
        close_df, volume_df, open_df, amount_df, benchmark_returns
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

    section_header("REPRODUCTION COMPLETE — ALL RESULTS WITH REAL DATA")
    print(f"Charts saved to: {SAVE_DIR}")
    print(f"\nKey Results (CSI 800 Constituents, Real Data):")
    print(f"  Daily low-vol-surge events:  {int(low_sig.sum().sum()):,}")
    print(f"  Daily high-vol-surge events: {int(high_sig.sum().sum()):,}")
    print(f"  HF low composite excess IR:  {low_comp_metrics['excess_info_ratio']:.2f}")
    print(f"  HF high composite neg excess:{-high_comp_metrics['excess_annual_return']:.2%}")
    print(f"  Combined excess ann. return: {combined_metrics['excess_annual_return']:.2%}")
    print(f"  Combined excess IR:          {combined_metrics['excess_info_ratio']:.2f}")
    print(f"  Combined excess max DD:      {combined_metrics['excess_max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
