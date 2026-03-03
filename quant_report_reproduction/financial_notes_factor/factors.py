"""
三个财务附注经营结构因子实现

1. 外币资金占货币资金比（截面）: factor = 1 - RMB_cash / total_cash
2. 境外业务收入占比稳定性（时序）: factor = ratio / std(ratio)_{t=1,...,6}
3. 主要客户销售收入占比稳定性（时序）: factor = std(top1_ratio)_{t=1,...,3}

由于财务附注数据仅在 Wind 等专业终端可获取，本模块提供：
- 完整的因子计算逻辑
- 基于公司特征的合理模拟数据生成
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ============================================================
# Factor 1: 外币资金占货币资金比
# ============================================================

def calc_foreign_currency_ratio(
    rmb_cash: pd.DataFrame,
    total_cash: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算外币资金占货币资金比

    factor = 1 - RMB_cash / total_cash
    大于1的值clip为1
    """
    ratio = 1 - rmb_cash / total_cash
    ratio = ratio.clip(upper=1.0)
    return ratio


def simulate_foreign_currency_data(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    report_dates: pd.DatetimeIndex,
    seed: int = 42,
) -> pd.DataFrame:
    """
    基于股票特征模拟外币资金占比因子

    逻辑：大市值、高成交额的公司更可能有境外业务，外币占比更高。
    半年更新（4月末、8月末），覆盖度约80%。
    """
    rng = np.random.default_rng(seed)
    stocks = close_df.columns
    n_stocks = len(stocks)

    avg_volume = volume_df.mean()
    vol_rank = avg_volume.rank(pct=True)

    factor_dict = {}
    for date in report_dates:
        base = vol_rank.values * 0.3 + rng.beta(2, 5, size=n_stocks) * 0.5
        base = np.clip(base, 0, 1)

        mask = rng.random(n_stocks) < 0.80
        values = pd.Series(np.where(mask, base, np.nan), index=stocks)
        factor_dict[date] = values

    return pd.DataFrame(factor_dict).T


# ============================================================
# Factor 2: 境外业务收入占比稳定性
# ============================================================

def calc_overseas_revenue_stability(
    overseas_revenue_ratio: pd.DataFrame,
    lookback_periods: int = 6,
) -> pd.DataFrame:
    """
    计算境外业务收入占比稳定性

    factor = ratio / std(ratio)_{t=1,...,lookback_periods}
    """
    ratio_std = overseas_revenue_ratio.rolling(
        lookback_periods, min_periods=3
    ).std()
    factor = overseas_revenue_ratio / ratio_std
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor


def simulate_overseas_revenue_data(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    report_dates: pd.DatetimeIndex,
    seed: int = 123,
) -> pd.DataFrame:
    """
    模拟境外收入占比稳定性因子

    逻辑：境外收入稳定的公司经营更稳健。覆盖度约40%。
    """
    rng = np.random.default_rng(seed)
    stocks = close_df.columns
    n_stocks = len(stocks)

    base_ratio = rng.beta(2, 8, size=n_stocks)
    stability = rng.uniform(0.3, 3.0, size=n_stocks)

    factor_dict = {}
    for i, date in enumerate(report_dates):
        noise = rng.normal(0, 0.05, size=n_stocks)
        current_ratio = np.clip(base_ratio + noise * (i % 4 - 1.5) * 0.1, 0, 1)
        ratio_std = np.abs(rng.normal(0.05, 0.03, size=n_stocks))
        ratio_std = np.clip(ratio_std, 0.01, None)
        factor_vals = current_ratio / ratio_std * stability

        mask = rng.random(n_stocks) < 0.40
        values = pd.Series(np.where(mask, factor_vals, np.nan), index=stocks)
        factor_dict[date] = values

    return pd.DataFrame(factor_dict).T


# ============================================================
# Factor 3: 主要客户销售收入占比稳定性
# ============================================================

def calc_customer_concentration_stability(
    top1_customer_ratio: pd.DataFrame,
    lookback_years: int = 3,
) -> pd.DataFrame:
    """
    计算主要客户销售收入占比稳定性

    factor = std(top1_ratio)_{t=1,...,lookback_years}
    注意：这是负向因子（IC为负），std越低 → 越稳定 → 收益越好
    """
    factor = top1_customer_ratio.rolling(lookback_years, min_periods=2).std()
    return factor


def simulate_customer_stability_data(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    report_dates: pd.DatetimeIndex,
    seed: int = 456,
) -> pd.DataFrame:
    """
    模拟主要客户占比稳定性因子

    逻辑：客户集中度越稳定，公司下游需求越稳定。
    年度更新（8月末），覆盖度约60%。此为负向因子。
    """
    rng = np.random.default_rng(seed)
    stocks = close_df.columns
    n_stocks = len(stocks)

    base_stability = rng.exponential(0.05, size=n_stocks)

    factor_dict = {}
    for i, date in enumerate(report_dates):
        noise = rng.exponential(0.02, size=n_stocks)
        factor_vals = base_stability + noise

        mask = rng.random(n_stocks) < 0.60
        values = pd.Series(np.where(mask, factor_vals, np.nan), index=stocks)
        factor_dict[date] = values

    return pd.DataFrame(factor_dict).T


# ============================================================
# Factor Composite
# ============================================================

def composite_factors(
    factors: dict,
    method: str = "equal_weight",
) -> pd.DataFrame:
    """
    因子等权复合

    Args:
        factors: {name: factor_df}
        method: 'equal_weight'

    Returns:
        composite factor DataFrame
    """
    factor_list = list(factors.values())

    common_dates = factor_list[0].index
    common_stocks = factor_list[0].columns
    for f in factor_list[1:]:
        common_dates = common_dates.intersection(f.index)
        common_stocks = common_stocks.intersection(f.columns)

    standardized = []
    for f in factor_list:
        f_sub = f.loc[common_dates, common_stocks]
        f_rank = f_sub.rank(axis=1, pct=True)
        standardized.append(f_rank)

    if method == "equal_weight":
        composite = sum(standardized) / len(standardized)
    else:
        composite = sum(standardized) / len(standardized)

    return composite


def generate_report_dates(
    start: str = "2014-06-30",
    end: str = "2025-10-31",
    freq: str = "semi_annual",
) -> pd.DatetimeIndex:
    """
    生成财报发布日期

    semi_annual: 每年 4月末 + 8月末
    annual: 每年 8月末
    """
    dates = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        if freq == "semi_annual":
            d1 = pd.Timestamp(f"{year}-04-30")
            d2 = pd.Timestamp(f"{year}-08-31")
            if d1 >= pd.Timestamp(start) and d1 <= pd.Timestamp(end):
                dates.append(d1)
            if d2 >= pd.Timestamp(start) and d2 <= pd.Timestamp(end):
                dates.append(d2)
        elif freq == "annual":
            d = pd.Timestamp(f"{year}-08-31")
            if d >= pd.Timestamp(start) and d <= pd.Timestamp(end):
                dates.append(d)
    return pd.DatetimeIndex(dates)


def expand_factor_to_monthly(
    factor: pd.DataFrame,
    monthly_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    将半年/年度因子扩展到月频（前向填充）
    """
    factor_monthly = factor.reindex(
        factor.index.union(monthly_dates)
    ).sort_index().ffill()
    return factor_monthly.reindex(monthly_dates)
