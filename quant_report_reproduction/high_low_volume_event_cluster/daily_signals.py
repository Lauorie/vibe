"""
日频事件信号模块

实现研报第二节：日频数据下的高/低位放量事件定义与检测

低位放量事件条件:
  (1) 当日收盘价 <= 过去120个交易日收盘价的10%分位数
  (2) 当日成交量 > 过去120个交易日成交量的均值 + 1.5倍标准差

高位放量事件条件:
  (1) 当日收盘价 >= 过去120个交易日收盘价的90%分位数
  (2) 当日成交量 > 过去120个交易日成交量的均值 + 1.5倍标准差
"""

import numpy as np
import pandas as pd
from typing import Tuple


def detect_daily_low_volume_surge(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    lookback: int = 120,
    price_quantile: float = 0.10,
    volume_n_std: float = 1.5,
) -> pd.DataFrame:
    """
    检测日频低位放量事件

    Args:
        close_df: 收盘价 (index=date, columns=stocks)
        volume_df: 成交量
        lookback: 回看窗口（交易日数）
        price_quantile: 价格低位分位数阈值
        volume_n_std: 成交量放大的标准差倍数

    Returns:
        signal_df: 布尔型 DataFrame, True表示触发低位放量信号
    """
    price_threshold = close_df.rolling(lookback, min_periods=lookback).quantile(price_quantile)
    vol_mean = volume_df.rolling(lookback, min_periods=lookback).mean()
    vol_std = volume_df.rolling(lookback, min_periods=lookback).std()
    vol_threshold = vol_mean + volume_n_std * vol_std

    is_low_price = close_df <= price_threshold
    is_high_volume = volume_df > vol_threshold

    signal = is_low_price & is_high_volume

    return signal


def detect_daily_high_volume_surge(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    lookback: int = 120,
    price_quantile: float = 0.90,
    volume_n_std: float = 1.5,
) -> pd.DataFrame:
    """
    检测日频高位放量事件

    Args:
        close_df: 收盘价
        volume_df: 成交量
        lookback: 回看窗口
        price_quantile: 价格高位分位数阈值
        volume_n_std: 成交量放大的标准差倍数

    Returns:
        signal_df: 布尔型 DataFrame, True表示触发高位放量信号
    """
    price_threshold = close_df.rolling(lookback, min_periods=lookback).quantile(price_quantile)
    vol_mean = volume_df.rolling(lookback, min_periods=lookback).mean()
    vol_std = volume_df.rolling(lookback, min_periods=lookback).std()
    vol_threshold = vol_mean + volume_n_std * vol_std

    is_high_price = close_df >= price_threshold
    is_high_volume = volume_df > vol_threshold

    signal = is_high_price & is_high_volume

    return signal


def compute_event_stats(
    signal_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    统计事件触发的基本信息

    Returns:
        DataFrame with columns: [date, n_triggered]
    """
    daily_count = signal_df.sum(axis=1)
    stats = pd.DataFrame({
        "date": daily_count.index,
        "n_triggered": daily_count.values,
    })
    stats.set_index("date", inplace=True)
    return stats


def compute_post_event_returns(
    signal_df: pd.DataFrame,
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    max_horizon: int = 60,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算事件触发后的累积绝对收益和超额收益

    研报方法：若当日收盘触发事件，以次日开盘价为起点，
    考察未来max_horizon个交易日的收益表现。

    Returns:
        avg_abs_cum_ret: 平均累积绝对收益 (index=horizon)
        avg_excess_cum_ret: 平均累积超额收益
    """
    dates = signal_df.index.tolist()
    stocks = signal_df.columns.tolist()

    forward_returns = open_df.pct_change().shift(-1)

    all_abs_paths = []
    all_excess_paths = []

    for i, date in enumerate(dates):
        triggered_stocks = signal_df.columns[signal_df.loc[date]].tolist()
        if not triggered_stocks or i + 1 >= len(dates):
            continue

        start_idx = i + 1
        end_idx = min(start_idx + max_horizon, len(dates))

        if end_idx - start_idx < 5:
            continue

        for stock in triggered_stocks:
            entry_price = open_df.iloc[start_idx][stock]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            future_close = close_df.iloc[start_idx:end_idx][stock]
            abs_ret = (future_close / entry_price - 1).values

            bench_ret = benchmark_returns.iloc[start_idx:end_idx].values
            bench_cum = np.cumprod(1 + bench_ret) - 1

            excess_ret = abs_ret - bench_cum[:len(abs_ret)]

            padded_abs = np.full(max_horizon, np.nan)
            padded_excess = np.full(max_horizon, np.nan)
            n = len(abs_ret)
            padded_abs[:n] = abs_ret
            padded_excess[:n] = excess_ret

            all_abs_paths.append(padded_abs)
            all_excess_paths.append(padded_excess)

    if not all_abs_paths:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    abs_arr = np.array(all_abs_paths)
    excess_arr = np.array(all_excess_paths)

    avg_abs = np.nanmean(abs_arr, axis=0)
    avg_excess = np.nanmean(excess_arr, axis=0)

    horizon = np.arange(1, max_horizon + 1)
    return (
        pd.Series(avg_abs, index=horizon, name="avg_abs_cum_ret"),
        pd.Series(avg_excess, index=horizon, name="avg_excess_cum_ret"),
    )
