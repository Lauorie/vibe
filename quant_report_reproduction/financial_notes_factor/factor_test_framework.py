"""
标准因子测试框架

实现研报中的因子测试方法论：
- 5分组回测（等权 / 市值加权）
- Rank IC / ICIR 序列
- 多空净值、超额净值
- 分年度收益统计
- 因子宽度（覆盖度）
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional


def compute_rank_ic(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
) -> pd.Series:
    """
    计算每期 Rank IC (Spearman rank correlation)

    Args:
        factor: 因子值 (index=date, columns=stocks)
        forward_returns: 下期收益率

    Returns:
        ic_series: Rank IC 时间序列
    """
    common_dates = factor.index.intersection(forward_returns.index)
    ic_list = []
    for date in common_dates:
        f = factor.loc[date].dropna()
        r = forward_returns.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < 30:
            continue
        corr, _ = stats.spearmanr(f[common], r[common])
        ic_list.append({"date": date, "ic": corr})

    if not ic_list:
        return pd.Series(dtype=float)
    df = pd.DataFrame(ic_list).set_index("date")
    return df["ic"]


def quintile_sort(
    factor: pd.DataFrame,
    forward_returns: pd.DataFrame,
    n_groups: int = 5,
    weight_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    因子分组回测

    Args:
        factor: 因子值 (index=date, columns=stocks)
        forward_returns: 下期收益率
        n_groups: 分组数量
        weight_df: 市值权重 (None=等权)

    Returns:
        dict with group_returns, long_short, excess, metrics
    """
    common_dates = sorted(factor.index.intersection(forward_returns.index))
    group_returns = {g: [] for g in range(1, n_groups + 1)}
    benchmark_returns = []
    dates_used = []

    for date in common_dates:
        f = factor.loc[date].dropna()
        r = forward_returns.loc[date].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_groups * 5:
            continue

        f_vals = f[common]
        r_vals = r[common]

        ranks = f_vals.rank(method="first")
        quantiles = pd.qcut(ranks, n_groups, labels=False) + 1

        for g in range(1, n_groups + 1):
            mask = quantiles == g
            stocks_in_group = mask[mask].index

            if weight_df is not None and date in weight_df.index:
                w = weight_df.loc[date].reindex(stocks_in_group).fillna(0)
                w_sum = w.sum()
                if w_sum > 0:
                    w = w / w_sum
                else:
                    w = pd.Series(1.0 / len(stocks_in_group), index=stocks_in_group)
                group_returns[g].append((r_vals[stocks_in_group] * w).sum())
            else:
                group_returns[g].append(r_vals[stocks_in_group].mean())

        if weight_df is not None and date in weight_df.index:
            w_all = weight_df.loc[date].reindex(common).fillna(0)
            w_all = w_all / w_all.sum() if w_all.sum() > 0 else pd.Series(1.0 / len(common), index=common)
            benchmark_returns.append((r_vals * w_all).sum())
        else:
            benchmark_returns.append(r_vals.mean())

        dates_used.append(date)

    group_ret_df = pd.DataFrame(group_returns, index=dates_used)
    benchmark_ret = pd.Series(benchmark_returns, index=dates_used, name="benchmark")

    long_short_ret = group_ret_df[n_groups] - group_ret_df[1]
    long_short_ret.name = "long_short"
    excess_ret = group_ret_df[n_groups] - benchmark_ret
    excess_ret.name = "excess"

    group_nav = (1 + group_ret_df).cumprod()
    benchmark_nav = (1 + benchmark_ret).cumprod()
    long_short_nav = (1 + long_short_ret).cumprod()
    excess_nav = (1 + excess_ret).cumprod()

    return {
        "group_returns": group_ret_df,
        "group_nav": group_nav,
        "benchmark_returns": benchmark_ret,
        "benchmark_nav": benchmark_nav,
        "long_short_returns": long_short_ret,
        "long_short_nav": long_short_nav,
        "excess_returns": excess_ret,
        "excess_nav": excess_nav,
        "dates": dates_used,
    }


def compute_annual_metrics(ret_series: pd.Series) -> pd.DataFrame:
    """计算分年度收益"""
    ret_series = ret_series.copy()
    ret_series.index = pd.DatetimeIndex(ret_series.index)
    years = ret_series.index.year.unique()
    rows = []
    for y in sorted(years):
        yr = ret_series[ret_series.index.year == y]
        ann_ret = (1 + yr).prod() - 1
        rows.append({"year": y, "return": ann_ret})
    return pd.DataFrame(rows).set_index("year")


def compute_performance_table(result: Dict, n_groups: int = 5) -> pd.DataFrame:
    """
    生成研报风格的绩效指标表

    Returns:
        DataFrame: 各组 + 基准 + 多空 + 超额 的年化收益/波动/回撤
    """
    ann_factor = 12
    rows = []

    for g in range(1, n_groups + 1):
        ret = result["group_returns"][g]
        ann_ret = (1 + ret).prod() ** (ann_factor / len(ret)) - 1
        ann_vol = ret.std() * np.sqrt(ann_factor)
        nav = (1 + ret).cumprod()
        dd = (nav / nav.expanding().max() - 1).min()
        rows.append({
            "group": f"Group {g}", "ann_return": ann_ret,
            "ann_volatility": ann_vol, "max_drawdown": dd,
        })

    bench = result["benchmark_returns"]
    ann_ret_b = (1 + bench).prod() ** (ann_factor / len(bench)) - 1
    ann_vol_b = bench.std() * np.sqrt(ann_factor)
    nav_b = (1 + bench).cumprod()
    dd_b = (nav_b / nav_b.expanding().max() - 1).min()
    rows.append({
        "group": "Benchmark", "ann_return": ann_ret_b,
        "ann_volatility": ann_vol_b, "max_drawdown": dd_b,
    })

    ls = result["long_short_returns"]
    ann_ret_ls = (1 + ls).prod() ** (ann_factor / len(ls)) - 1
    ann_vol_ls = ls.std() * np.sqrt(ann_factor)
    nav_ls = (1 + ls).cumprod()
    dd_ls = (nav_ls / nav_ls.expanding().max() - 1).min()
    rows.append({
        "group": "Long-Short", "ann_return": ann_ret_ls,
        "ann_volatility": ann_vol_ls, "max_drawdown": dd_ls,
    })

    ex = result["excess_returns"]
    ann_ret_ex = (1 + ex).prod() ** (ann_factor / len(ex)) - 1
    ann_vol_ex = ex.std() * np.sqrt(ann_factor)
    nav_ex = (1 + ex).cumprod()
    dd_ex = (nav_ex / nav_ex.expanding().max() - 1).min()
    rows.append({
        "group": "Excess", "ann_return": ann_ret_ex,
        "ann_volatility": ann_vol_ex, "max_drawdown": dd_ex,
    })

    return pd.DataFrame(rows).set_index("group")


def compute_factor_coverage(factor: pd.DataFrame, total_stocks: int) -> pd.Series:
    """计算因子覆盖度（宽度）"""
    coverage = factor.notna().sum(axis=1) / total_stocks
    coverage.name = "coverage"
    return coverage


def compute_monthly_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """从日频收盘价计算月度收益率"""
    monthly_close = close_df.resample("ME").last()
    monthly_returns = monthly_close.pct_change()
    return monthly_returns.iloc[1:]
