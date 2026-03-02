"""
高频事件信号模块

实现研报第三节：高频数据下的高/低位放量事件簇

三步流程：
1. 事件识别：高/低位的定义 + 放量的定义
2. 信号定义：先看价后看量 / 先看量后看价
3. 信号筛选与合成

高/低位识别维度：
  - 对比方式：日内固定对比 / 日间滚动对比(20日)
  - 判定阈值：分位数法(90%/10%) / 均值±N标准差法(N=3日内, N=1.5日间)

放量识别维度：
  - 何种量：成交量 / 成交金额 / 成交笔数 / 单笔成交金额
  - 何人的量：整体 / 大单 / 小单 (模拟)
  - 何方向：整体 / 主买 / 主卖 (模拟)
  - 对比方式：同高/低位识别
  - 判定阈值：同高/低位识别
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from itertools import product


# ============================================================
# Part 1: 事件识别 — 高/低位标记
# ============================================================

def identify_high_low_minutes_intraday_quantile(
    minute_close: pd.DataFrame,
    high_quantile: float = 0.90,
    low_quantile: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    日内固定对比 + 分位数法 标记价格处于高/低位的分钟

    Args:
        minute_close: (index=date, columns=minute_idx)
        high_quantile: 高位分位数阈值
        low_quantile: 低位分位数阈值

    Returns:
        is_high: 布尔型 DataFrame
        is_low: 布尔型 DataFrame
    """
    high_thresh = minute_close.quantile(high_quantile, axis=1)
    low_thresh = minute_close.quantile(low_quantile, axis=1)

    is_high = minute_close.ge(high_thresh, axis=0)
    is_low = minute_close.le(low_thresh, axis=0)

    return is_high, is_low


def identify_high_low_minutes_intraday_meanstd(
    minute_close: pd.DataFrame,
    n_std: float = 3.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    日内固定对比 + 均值±N标准差法 标记价格处于高/低位的分钟
    研报指出日内对比时 N=3
    """
    row_mean = minute_close.mean(axis=1)
    row_std = minute_close.std(axis=1)

    high_thresh = row_mean + n_std * row_std
    low_thresh = row_mean - n_std * row_std

    is_high = minute_close.ge(high_thresh, axis=0)
    is_low = minute_close.le(low_thresh, axis=0)

    return is_high, is_low


def identify_high_low_minutes_rolling_quantile(
    minute_close: pd.DataFrame,
    lookback_days: int = 20,
    high_quantile: float = 0.90,
    low_quantile: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    日间滚动对比(20日) + 分位数法 标记价格处于高/低位的分钟

    将过去lookback_days天的所有分钟价格展平后计算分位数阈值
    """
    dates = minute_close.index
    n_minutes = minute_close.shape[1]
    is_high_list = []
    is_low_list = []

    for i in range(len(dates)):
        start = max(0, i - lookback_days + 1)
        historical = minute_close.iloc[start:i + 1].values.flatten()
        historical = historical[~np.isnan(historical)]

        if len(historical) < n_minutes:
            is_high_list.append(np.full(n_minutes, False))
            is_low_list.append(np.full(n_minutes, False))
            continue

        h_thresh = np.quantile(historical, high_quantile)
        l_thresh = np.quantile(historical, low_quantile)

        today = minute_close.iloc[i].values
        is_high_list.append(today >= h_thresh)
        is_low_list.append(today <= l_thresh)

    is_high = pd.DataFrame(is_high_list, index=dates, columns=minute_close.columns)
    is_low = pd.DataFrame(is_low_list, index=dates, columns=minute_close.columns)

    return is_high, is_low


def identify_high_low_minutes_rolling_meanstd(
    minute_close: pd.DataFrame,
    lookback_days: int = 20,
    n_std: float = 1.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    日间滚动对比(20日) + 均值±N标准差法
    研报指出日间对比时 N=1.5
    """
    dates = minute_close.index
    n_minutes = minute_close.shape[1]
    is_high_list = []
    is_low_list = []

    for i in range(len(dates)):
        start = max(0, i - lookback_days + 1)
        historical = minute_close.iloc[start:i + 1].values.flatten()
        historical = historical[~np.isnan(historical)]

        if len(historical) < n_minutes:
            is_high_list.append(np.full(n_minutes, False))
            is_low_list.append(np.full(n_minutes, False))
            continue

        mu = np.mean(historical)
        sigma = np.std(historical)

        today = minute_close.iloc[i].values
        is_high_list.append(today >= mu + n_std * sigma)
        is_low_list.append(today <= mu - n_std * sigma)

    is_high = pd.DataFrame(is_high_list, index=dates, columns=minute_close.columns)
    is_low = pd.DataFrame(is_low_list, index=dates, columns=minute_close.columns)

    return is_high, is_low


# ============================================================
# Part 2: 事件识别 — 放量标记
# ============================================================

def identify_volume_surge_intraday_quantile(
    minute_volume: pd.DataFrame,
    quantile: float = 0.90,
) -> pd.DataFrame:
    """日内固定对比 + 分位数法 标记放量分钟"""
    thresh = minute_volume.quantile(quantile, axis=1)
    return minute_volume.ge(thresh, axis=0)


def identify_volume_surge_intraday_meanstd(
    minute_volume: pd.DataFrame,
    n_std: float = 3.0,
) -> pd.DataFrame:
    """日内固定对比 + 均值+N标准差法（日内N=3）"""
    row_mean = minute_volume.mean(axis=1)
    row_std = minute_volume.std(axis=1)
    thresh = row_mean + n_std * row_std
    return minute_volume.ge(thresh, axis=0)


def identify_volume_surge_rolling_quantile(
    minute_volume: pd.DataFrame,
    lookback_days: int = 20,
    quantile: float = 0.90,
) -> pd.DataFrame:
    """日间滚动对比 + 分位数法"""
    dates = minute_volume.index
    n_minutes = minute_volume.shape[1]
    is_surge_list = []

    for i in range(len(dates)):
        start = max(0, i - lookback_days + 1)
        historical = minute_volume.iloc[start:i + 1].values.flatten()
        historical = historical[~np.isnan(historical)]

        if len(historical) < n_minutes:
            is_surge_list.append(np.full(n_minutes, False))
            continue

        thresh = np.quantile(historical, quantile)
        is_surge_list.append(minute_volume.iloc[i].values >= thresh)

    return pd.DataFrame(is_surge_list, index=dates, columns=minute_volume.columns)


def identify_volume_surge_rolling_meanstd(
    minute_volume: pd.DataFrame,
    lookback_days: int = 20,
    n_std: float = 1.5,
) -> pd.DataFrame:
    """日间滚动对比 + 均值+N标准差法（日间N=1.5）"""
    dates = minute_volume.index
    n_minutes = minute_volume.shape[1]
    is_surge_list = []

    for i in range(len(dates)):
        start = max(0, i - lookback_days + 1)
        historical = minute_volume.iloc[start:i + 1].values.flatten()
        historical = historical[~np.isnan(historical)]

        if len(historical) < n_minutes:
            is_surge_list.append(np.full(n_minutes, False))
            continue

        mu = np.mean(historical)
        sigma = np.std(historical)
        thresh = mu + n_std * sigma
        is_surge_list.append(minute_volume.iloc[i].values >= thresh)

    return pd.DataFrame(is_surge_list, index=dates, columns=minute_volume.columns)


# ============================================================
# Part 3: 信号定义 — 高/低位与放量结合
# ============================================================

def signal_price_first_volume_second(
    is_high: pd.DataFrame,
    is_low: pd.DataFrame,
    minute_volume: pd.DataFrame,
    is_volume_surge: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """
    先看价、后看量：
    先识别价格处于高/低位的时间点，
    再检查这些时间点的平均成交量是否满足放量条件

    Returns:
        high_vol_signal: 每日是否触发高位放量 (index=date, dtype=bool)
        low_vol_signal: 每日是否触发低位放量
    """
    dates = is_high.index
    high_signals = []
    low_signals = []

    vol_quantile_90 = minute_volume.quantile(0.90, axis=1)

    for i in range(len(dates)):
        high_mask = is_high.iloc[i].values
        low_mask = is_low.iloc[i].values
        vol_vals = minute_volume.iloc[i].values

        if high_mask.sum() > 0:
            avg_vol_at_high = vol_vals[high_mask].mean()
            high_signals.append(avg_vol_at_high >= vol_quantile_90.iloc[i])
        else:
            high_signals.append(False)

        if low_mask.sum() > 0:
            avg_vol_at_low = vol_vals[low_mask].mean()
            low_signals.append(avg_vol_at_low >= vol_quantile_90.iloc[i])
        else:
            low_signals.append(False)

    return (
        pd.Series(high_signals, index=dates, name="high_vol_surge"),
        pd.Series(low_signals, index=dates, name="low_vol_surge"),
    )


def signal_volume_first_price_second(
    is_volume_surge: pd.DataFrame,
    minute_close: pd.DataFrame,
    high_quantile: float = 0.90,
    low_quantile: float = 0.10,
) -> Tuple[pd.Series, pd.Series]:
    """
    先看量、后看价（研报示例方法）：
    先识别放量时间点，再检查这些时间点的平均价格是否处于高/低位

    Returns:
        high_vol_signal: 每日是否触发高位放量
        low_vol_signal: 每日是否触发低位放量
    """
    dates = is_volume_surge.index
    high_signals = []
    low_signals = []

    price_high_thresh = minute_close.quantile(high_quantile, axis=1)
    price_low_thresh = minute_close.quantile(low_quantile, axis=1)

    for i in range(len(dates)):
        surge_mask = is_volume_surge.iloc[i].values
        close_vals = minute_close.iloc[i].values

        if surge_mask.sum() > 0:
            avg_price_at_surge = close_vals[surge_mask].mean()
            high_signals.append(avg_price_at_surge >= price_high_thresh.iloc[i])
            low_signals.append(avg_price_at_surge <= price_low_thresh.iloc[i])
        else:
            high_signals.append(False)
            low_signals.append(False)

    return (
        pd.Series(high_signals, index=dates, name="high_vol_surge"),
        pd.Series(low_signals, index=dates, name="low_vol_surge"),
    )


# ============================================================
# Part 4: 批量信号生产
# ============================================================

def batch_produce_signals_single_stock(
    minute_close: pd.DataFrame,
    minute_volume: pd.DataFrame,
    minute_amount: Optional[pd.DataFrame] = None,
) -> Dict[str, Tuple[pd.Series, pd.Series]]:
    """
    对单只股票批量生产所有组合的高/低位放量信号

    通过排列组合不同的：
    - 高/低位识别方式 (4种)
    - 放量识别方式 (4种)
    - 信号结合方式 (2种：先看价后看量 / 先看量后看价)

    Returns:
        signals: {signal_name: (high_signal, low_signal)}
    """
    if minute_amount is None:
        minute_amount = minute_volume.copy()

    price_methods = {
        "intraday_quantile": identify_high_low_minutes_intraday_quantile,
        "intraday_meanstd": identify_high_low_minutes_intraday_meanstd,
        "rolling_quantile": lambda mc: identify_high_low_minutes_rolling_quantile(mc, 20),
        "rolling_meanstd": lambda mc: identify_high_low_minutes_rolling_meanstd(mc, 20),
    }

    volume_methods = {
        "vol_intraday_quantile": lambda mv: identify_volume_surge_intraday_quantile(mv),
        "vol_intraday_meanstd": lambda mv: identify_volume_surge_intraday_meanstd(mv),
        "vol_rolling_quantile": lambda mv: identify_volume_surge_rolling_quantile(mv, 20),
        "vol_rolling_meanstd": lambda mv: identify_volume_surge_rolling_meanstd(mv, 20),
    }

    volume_inputs = {
        "volume": minute_volume,
        "amount": minute_amount,
    }

    signals = {}

    for (pm_name, pm_func), (vm_name, vm_func), (vi_name, vi_data) in product(
        price_methods.items(), volume_methods.items(), volume_inputs.items()
    ):
        is_high, is_low = pm_func(minute_close) if "quantile" in pm_name or "meanstd" in pm_name else pm_func(minute_close)
        is_surge = vm_func(vi_data)

        sig_name_pv = f"PV_{pm_name}_{vm_name}_{vi_name}"
        high_pv, low_pv = signal_price_first_volume_second(
            is_high, is_low, vi_data, is_surge
        )
        signals[sig_name_pv] = (high_pv, low_pv)

        sig_name_vp = f"VP_{pm_name}_{vm_name}_{vi_name}"
        high_vp, low_vp = signal_volume_first_price_second(
            is_surge, minute_close
        )
        signals[sig_name_vp] = (high_vp, low_vp)

    return signals


def batch_produce_signals_multi_stock(
    minute_close_dict: dict,
    minute_volume_dict: dict,
    signal_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    对多只股票批量生产信号

    Returns:
        high_signals: {signal_name: DataFrame(index=date, columns=stocks, dtype=bool)}
        low_signals: {signal_name: DataFrame(index=date, columns=stocks, dtype=bool)}
    """
    stocks = list(minute_close_dict.keys())

    first_stock = stocks[0]
    all_signals = batch_produce_signals_single_stock(
        minute_close_dict[first_stock],
        minute_volume_dict[first_stock],
    )

    if signal_names is None:
        signal_names = list(all_signals.keys())

    high_signals = {name: {} for name in signal_names}
    low_signals = {name: {} for name in signal_names}

    for stock in stocks:
        stock_signals = batch_produce_signals_single_stock(
            minute_close_dict[stock],
            minute_volume_dict[stock],
        )
        for name in signal_names:
            if name in stock_signals:
                high_sig, low_sig = stock_signals[name]
                high_signals[name][stock] = high_sig
                low_signals[name][stock] = low_sig

    high_df_dict = {}
    low_df_dict = {}
    for name in signal_names:
        if high_signals[name]:
            high_df_dict[name] = pd.DataFrame(high_signals[name])
            low_df_dict[name] = pd.DataFrame(low_signals[name])

    return high_df_dict, low_df_dict
