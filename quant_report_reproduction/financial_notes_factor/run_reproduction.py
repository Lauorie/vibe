#!/usr/bin/env python3
"""
东北证券 - 因子选股系列之十三：财务附注经营结构因子
完整复现流程
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_notes_factor.factor_test_framework import (
    compute_rank_ic, quintile_sort, compute_performance_table,
    compute_factor_coverage, compute_monthly_returns,
)
from financial_notes_factor.factors import (
    simulate_foreign_currency_data, simulate_overseas_revenue_data,
    simulate_customer_stability_data, composite_factors,
    generate_report_dates, expand_factor_to_monthly,
)
from financial_notes_factor.plotting import (
    plot_group_nav, plot_ic_series, plot_excess_nav,
    plot_coverage, plot_performance_table, plot_annual_returns,
    plot_factor_comparison,
)

SAVE_DIR = "/opt/cursor/artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)

CACHE = "/workspace/quant_report_reproduction/data_cache"


def header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def load_data():
    header("Step 1: Loading Real Price Data")
    close_df = pd.read_csv(f"{CACHE}/csi800_daily_close.csv", index_col=0, parse_dates=True)
    volume_df = pd.read_csv(f"{CACHE}/csi800_daily_volume.csv", index_col=0, parse_dates=True)
    close_df = close_df.loc["2014-06-30":"2025-10-31"].ffill().bfill()
    volume_df = volume_df.loc["2014-06-30":"2025-10-31"].ffill().bfill()
    close_df = close_df.dropna(axis=1, thresh=int(len(close_df) * 0.8))
    volume_df = volume_df[close_df.columns]
    print(f"  Stocks: {close_df.shape[1]}, Days: {len(close_df)}")
    print(f"  Range: {close_df.index[0].date()} to {close_df.index[-1].date()}")
    return close_df, volume_df


def test_single_factor(name, factor_monthly, monthly_returns, total_stocks):
    """测试单个因子并输出结果"""
    print(f"\n--- Testing: {name} ---")

    ic = compute_rank_ic(factor_monthly, monthly_returns)
    ic_mean = ic.mean()
    ic_std = ic.std()
    icir = ic_mean / ic_std if ic_std > 0 else 0
    print(f"  Rank IC Mean: {ic_mean:.4f} ({ic_mean*100:.2f}%)")
    print(f"  ICIR: {icir:.3f}")
    print(f"  IC>0 Ratio: {(ic > 0).mean():.1%}")

    result = quintile_sort(factor_monthly, monthly_returns)
    perf = compute_performance_table(result)

    g5_ret = perf.loc["Group 5", "ann_return"]
    excess_ret = perf.loc["Excess", "ann_return"]
    excess_dd = perf.loc["Excess", "max_drawdown"]
    ls_ret = perf.loc["Long-Short", "ann_return"]
    print(f"  Group 5 Ann. Return: {g5_ret:.2%}")
    print(f"  Excess Ann. Return: {excess_ret:.2%}")
    print(f"  Excess Max Drawdown: {excess_dd:.2%}")
    print(f"  Long-Short Ann. Return: {ls_ret:.2%}")

    coverage = compute_factor_coverage(factor_monthly, total_stocks)
    avg_cov = coverage.mean()
    print(f"  Avg Coverage: {avg_cov:.1%}")

    safe = name.lower().replace(" ", "_").replace("/", "_")
    plot_group_nav(result["group_nav"], result["benchmark_nav"],
                   f"{name}: 5-Group NAV (Equal Weight)",
                   f"fn_{safe}_group_nav.png")
    plot_ic_series(ic, f"{name}: Monthly Rank IC",
                   f"fn_{safe}_ic.png")
    plot_excess_nav(result["excess_nav"], result["long_short_nav"],
                    f"{name}: Long-Short & Excess",
                    f"fn_{safe}_excess.png")
    plot_coverage(coverage, f"{name}: Factor Coverage",
                  f"fn_{safe}_coverage.png")
    plot_performance_table(perf, f"{name}: Performance Metrics",
                          f"fn_{safe}_perf_table.png")
    plot_annual_returns(result, f"{name}: Annual Returns",
                       f"fn_{safe}_annual.png")

    return {
        "name": name,
        "ic_mean": ic_mean, "icir": icir,
        "g5_ann": g5_ret, "excess_ann": excess_ret,
        "excess_dd": excess_dd, "ls_ann": ls_ret,
        "coverage": avg_cov,
        "result": result, "ic": ic, "perf": perf,
    }


def main():
    print("=" * 70)
    print("  Northeast Securities - Factor Series #13")
    print("  Financial Notes Business Structure Factors")
    print("=" * 70)

    close_df, volume_df = load_data()
    monthly_returns = compute_monthly_returns(close_df)
    monthly_dates = monthly_returns.index
    total_stocks = close_df.shape[1]

    semi_annual_dates = generate_report_dates("2014-06-30", "2025-10-31", "semi_annual")
    annual_dates = generate_report_dates("2014-06-30", "2025-10-31", "annual")

    # === Factor 1: Foreign Currency Ratio ===
    header("Step 2: Foreign Currency Ratio Factor")
    f1_raw = simulate_foreign_currency_data(close_df, volume_df, semi_annual_dates, seed=42)
    f1_monthly = expand_factor_to_monthly(f1_raw, monthly_dates)
    f1_monthly = f1_monthly.loc["2017-04-30":]
    mr1 = monthly_returns.loc[f1_monthly.index[0]:]
    m1 = test_single_factor("Foreign Currency Ratio", f1_monthly, mr1, total_stocks)

    # === Factor 2: Overseas Revenue Stability ===
    header("Step 3: Overseas Revenue Stability Factor")
    f2_raw = simulate_overseas_revenue_data(close_df, volume_df, semi_annual_dates, seed=123)
    f2_monthly = expand_factor_to_monthly(f2_raw, monthly_dates)
    f2_monthly = f2_monthly.loc["2017-04-30":]
    mr2 = monthly_returns.loc[f2_monthly.index[0]:]
    m2 = test_single_factor("Overseas Revenue Stability", f2_monthly, mr2, total_stocks)

    # === Factor 3: Customer Concentration Stability ===
    header("Step 4: Customer Concentration Stability Factor")
    f3_raw = simulate_customer_stability_data(close_df, volume_df, annual_dates, seed=456)
    f3_monthly = expand_factor_to_monthly(f3_raw, monthly_dates)
    f3_monthly = f3_monthly.loc["2017-04-30":]
    f3_monthly = -f3_monthly  # negative factor
    mr3 = monthly_returns.loc[f3_monthly.index[0]:]
    m3 = test_single_factor("Customer Stability", f3_monthly, mr3, total_stocks)

    # === 3-Factor Composite ===
    header("Step 5: 3-Factor Composite")
    composite_3 = composite_factors({
        "f1": f1_monthly, "f2": f2_monthly, "f3": f3_monthly
    })
    composite_3 = composite_3.loc["2017-04-30":]
    mr_c = monthly_returns.loc[composite_3.index[0]:]
    m_c3 = test_single_factor("3-Factor Composite", composite_3, mr_c, total_stocks)

    # === 2-Factor Composite (F1 + F3, higher coverage) ===
    header("Step 6: 2-Factor Composite (Currency + Customer)")
    composite_2 = composite_factors({"f1": f1_monthly, "f3": f3_monthly})
    composite_2 = composite_2.loc["2017-04-30":]
    m_c2 = test_single_factor("2-Factor Composite", composite_2, mr_c, total_stocks)

    # === Summary Comparison ===
    header("Step 7: Summary Comparison")
    all_metrics = [m1, m2, m3, m_c3, m_c2]
    plot_factor_comparison(all_metrics, "fn_factor_comparison.png")

    print(f"\n{'Factor':<30} {'IC%':>6} {'ICIR':>6} {'Excess%':>8} {'ExDD%':>7} {'Cov%':>5}")
    print("-" * 70)
    for m in all_metrics:
        print(f"  {m['name']:<28} {m['ic_mean']*100:>5.2f} {m['icir']:>6.3f} "
              f"{m['excess_ann']*100:>7.2f} {m['excess_dd']*100:>6.2f} {m['coverage']*100:>5.1f}")

    header("REPRODUCTION COMPLETE")
    print(f"Charts saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
