"""
数据工具模块

提供数据加载、模拟数据生成、股票池管理等功能。
由于原研报使用 Wind/通联数据（付费），本模块提供：
1. 模拟数据生成器（用于验证方法论正确性）
2. 通用数据接口（可替换为实际数据源）
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


def generate_simulated_daily_data(
    n_stocks: int = 200,
    n_days: int = 2400,
    start_date: str = "2016-01-04",
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    生成模拟日频数据（收盘价、成交量、开盘价）

    Returns:
        close_df: 收盘价 DataFrame (index=date, columns=stock_codes)
        volume_df: 成交量 DataFrame
        open_df: 开盘价 DataFrame
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start_date, periods=n_days, freq="B")
    stock_codes = [f"SH{600000 + i:06d}" for i in range(n_stocks)]

    mu = rng.uniform(-0.0002, 0.0005, size=n_stocks)
    sigma = rng.uniform(0.015, 0.035, size=n_stocks)

    daily_returns = rng.normal(
        loc=mu, scale=sigma, size=(n_days, n_stocks)
    )
    market_factor = rng.normal(0, 0.012, size=(n_days, 1))
    daily_returns += market_factor * 0.6

    init_prices = rng.uniform(5, 50, size=n_stocks)
    close_prices = init_prices * np.exp(np.cumsum(daily_returns, axis=0))

    base_volume = rng.uniform(5e6, 5e7, size=n_stocks)
    volume_noise = rng.lognormal(0, 0.5, size=(n_days, n_stocks))
    volume = base_volume * volume_noise

    abs_ret = np.abs(daily_returns)
    volume *= (1 + abs_ret * 10)

    open_noise = rng.normal(0, 0.003, size=(n_days, n_stocks))
    open_prices = close_prices * np.exp(open_noise)

    close_df = pd.DataFrame(close_prices, index=dates, columns=stock_codes)
    volume_df = pd.DataFrame(volume, index=dates, columns=stock_codes)
    open_df = pd.DataFrame(open_prices, index=dates, columns=stock_codes)

    return close_df, volume_df, open_df


def generate_simulated_minute_data(
    daily_close: pd.Series,
    daily_volume: pd.Series,
    n_minutes: int = 240,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于日频数据生成单只股票某日的模拟分钟数据

    Args:
        daily_close: 该股票的日频收盘价 Series (index=date)
        daily_volume: 该股票的日频成交量 Series (index=date)
        n_minutes: 每日分钟数

    Returns:
        minute_close: 分钟收盘价 DataFrame (index=date, columns=minute_idx)
        minute_volume: 分钟成交量 DataFrame
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily_close)

    minute_close_list = []
    minute_volume_list = []

    for i in range(n_days):
        prev_close = daily_close.iloc[i - 1] if i > 0 else daily_close.iloc[0] * 0.99
        target_close = daily_close.iloc[i]

        drift = np.log(target_close / prev_close) / n_minutes
        noise = rng.normal(0, 0.001, size=n_minutes)
        returns = drift + noise

        minute_prices = prev_close * np.exp(np.cumsum(returns))
        scale_factor = target_close / minute_prices[-1]
        minute_prices *= scale_factor

        total_vol = daily_volume.iloc[i]
        u_shape = np.array([
            1.5 - 0.8 * np.sin(np.pi * j / n_minutes)
            for j in range(n_minutes)
        ])
        vol_noise = rng.lognormal(0, 0.3, size=n_minutes)
        raw_vol = u_shape * vol_noise
        minute_vol = raw_vol / raw_vol.sum() * total_vol

        minute_close_list.append(minute_prices)
        minute_volume_list.append(minute_vol)

    minute_close = pd.DataFrame(
        minute_close_list, index=daily_close.index,
        columns=[f"min_{j+1}" for j in range(n_minutes)]
    )
    minute_volume = pd.DataFrame(
        minute_volume_list, index=daily_volume.index,
        columns=[f"min_{j+1}" for j in range(n_minutes)]
    )

    return minute_close, minute_volume


def generate_batch_minute_data(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    stock_list: Optional[List[str]] = None,
    n_minutes: int = 240,
    seed: int = 42,
) -> Tuple[dict, dict]:
    """
    批量生成多只股票的分钟数据

    Returns:
        minute_close_dict: {stock_code: minute_close_df}
        minute_volume_dict: {stock_code: minute_volume_df}
    """
    if stock_list is None:
        stock_list = close_df.columns.tolist()

    minute_close_dict = {}
    minute_volume_dict = {}

    for idx, stock in enumerate(stock_list):
        mc, mv = generate_simulated_minute_data(
            close_df[stock], volume_df[stock],
            n_minutes=n_minutes, seed=seed + idx
        )
        minute_close_dict[stock] = mc
        minute_volume_dict[stock] = mv

    return minute_close_dict, minute_volume_dict


def get_benchmark_returns(
    close_df: pd.DataFrame,
    method: str = "equal_weight",
) -> pd.Series:
    """
    计算基准指数收益率

    Args:
        close_df: 股票收盘价 DataFrame
        method: 'equal_weight' = 等权指数
    """
    daily_returns = close_df.pct_change()
    if method == "equal_weight":
        benchmark = daily_returns.mean(axis=1)
    else:
        benchmark = daily_returns.mean(axis=1)
    benchmark.name = "benchmark"
    return benchmark


def get_trading_calendar(
    close_df: pd.DataFrame,
) -> pd.DatetimeIndex:
    """获取交易日历"""
    return close_df.index


def get_weekly_rebalance_dates(
    trading_calendar: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """获取每周末（周五）的调仓日期"""
    dates = pd.Series(trading_calendar)
    week_end = dates.groupby(dates.dt.isocalendar().week).last()
    return pd.DatetimeIndex(week_end.values)
