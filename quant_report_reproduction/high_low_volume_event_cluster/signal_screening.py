"""
信号筛选与合成模块

实现研报 3.4 节的信号筛选与合成流程：

1. 分时段筛选：
   - 第一阶段 (2016-2018)：按超额信息比率和低相关性筛选
   - 第二阶段 (2019-2021)：进一步筛选（2022+ 为样本外）

2. 相关性检验：
   - 信号之间的平均相关性（股票池重合度）
   - 通道策略最大回撤时段的相关性

3. 综合信号合成：
   - 若某只股票某日同时触发半数及以上信号 → 触发综合信号
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def compute_signal_overlap_correlation(
    signal_a: pd.DataFrame,
    signal_b: pd.DataFrame,
) -> float:
    """
    计算两个信号之间的股票池重合度

    日相关性 = 两股票池交集数量 / min(信号A股票池数量, 信号B股票池数量)
    返回回测期内的平均相关性
    """
    common_dates = signal_a.index.intersection(signal_b.index)
    if len(common_dates) == 0:
        return 0.0

    correlations = []
    for date in common_dates:
        pool_a = set(signal_a.columns[signal_a.loc[date]])
        pool_b = set(signal_b.columns[signal_b.loc[date]])

        if len(pool_a) == 0 or len(pool_b) == 0:
            continue

        overlap = len(pool_a & pool_b)
        min_size = min(len(pool_a), len(pool_b))
        correlations.append(overlap / min_size)

    return np.mean(correlations) if correlations else 0.0


def compute_pairwise_correlations(
    signals: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """计算所有信号两两之间的股票池重合度矩阵"""
    names = list(signals.keys())
    n = len(names)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=names, columns=names)

    for i in range(n):
        for j in range(i + 1, n):
            corr = compute_signal_overlap_correlation(
                signals[names[i]], signals[names[j]]
            )
            corr_matrix.iloc[i, j] = corr
            corr_matrix.iloc[j, i] = corr

    return corr_matrix


def screen_signals_by_period(
    signal_metrics: Dict[str, dict],
    signal_correlations: pd.DataFrame,
    metric_key: str = "excess_info_ratio",
    min_ir: float = 0.5,
    max_avg_corr: float = 0.3,
    top_n: int = 5,
) -> List[str]:
    """
    按绩效和相关性筛选信号

    Args:
        signal_metrics: {signal_name: metrics_dict}
        signal_correlations: 相关性矩阵
        metric_key: 用于排序的绩效指标
        min_ir: 最低信息比率要求
        max_avg_corr: 最大平均相关性要求
        top_n: 最终保留信号数量

    Returns:
        selected: 筛选后的信号名称列表
    """
    qualified = {
        name: metrics[metric_key]
        for name, metrics in signal_metrics.items()
        if metrics.get(metric_key, 0) >= min_ir
    }

    if not qualified:
        sorted_all = sorted(
            signal_metrics.items(),
            key=lambda x: x[1].get(metric_key, 0),
            reverse=True
        )
        return [name for name, _ in sorted_all[:top_n]]

    sorted_signals = sorted(qualified.items(), key=lambda x: x[1], reverse=True)

    selected = []
    for name, ir in sorted_signals:
        if len(selected) >= top_n:
            break

        if not selected:
            selected.append(name)
            continue

        avg_corr = np.mean([
            signal_correlations.loc[name, s]
            for s in selected
            if name in signal_correlations.index and s in signal_correlations.columns
        ])

        if avg_corr <= max_avg_corr:
            selected.append(name)

    return selected


def two_stage_screening(
    signals: Dict[str, pd.DataFrame],
    open_df: pd.DataFrame,
    close_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    stage1_start: str = "2016-01-01",
    stage1_end: str = "2018-12-31",
    stage2_start: str = "2019-01-01",
    stage2_end: str = "2021-12-31",
    signal_type: str = "low",
    top_n: int = 5,
) -> List[str]:
    """
    两阶段筛选流程

    Args:
        signals: {signal_name: signal_df}
        signal_type: "low" (低位放量, 正向) 或 "high" (高位放量, 负向)
        top_n: 每阶段保留的信号数量
    """
    from .channel_strategy import ChannelStrategy, compute_channel_strategy_metrics

    s1_signals = {
        name: sig.loc[stage1_start:stage1_end]
        for name, sig in signals.items()
    }
    s1_open = open_df.loc[stage1_start:stage1_end]
    s1_close = close_df.loc[stage1_start:stage1_end]
    s1_bench = benchmark_returns.loc[stage1_start:stage1_end]

    engine = ChannelStrategy()
    s1_metrics = {}
    for name, sig in s1_signals.items():
        try:
            result = engine.run(sig, s1_open, s1_close)
            metrics = compute_channel_strategy_metrics(result, s1_bench)
            if signal_type == "high":
                metrics["excess_info_ratio"] = -metrics.get("excess_info_ratio", 0)
                metrics["excess_annual_return"] = -metrics.get("excess_annual_return", 0)
            s1_metrics[name] = metrics
        except Exception:
            continue

    s1_corr = compute_pairwise_correlations(s1_signals)
    s1_selected = screen_signals_by_period(
        s1_metrics, s1_corr, top_n=min(top_n * 2, len(s1_metrics))
    )

    if not s1_selected:
        return list(signals.keys())[:top_n]

    s2_signals = {
        name: signals[name].loc[stage2_start:stage2_end]
        for name in s1_selected if name in signals
    }
    s2_open = open_df.loc[stage2_start:stage2_end]
    s2_close = close_df.loc[stage2_start:stage2_end]
    s2_bench = benchmark_returns.loc[stage2_start:stage2_end]

    s2_metrics = {}
    for name, sig in s2_signals.items():
        try:
            result = engine.run(sig, s2_open, s2_close)
            metrics = compute_channel_strategy_metrics(result, s2_bench)
            if signal_type == "high":
                metrics["excess_info_ratio"] = -metrics.get("excess_info_ratio", 0)
                metrics["excess_annual_return"] = -metrics.get("excess_annual_return", 0)
            s2_metrics[name] = metrics
        except Exception:
            continue

    s2_corr = compute_pairwise_correlations(s2_signals)
    s2_selected = screen_signals_by_period(
        s2_metrics, s2_corr, top_n=top_n
    )

    return s2_selected


def composite_signal(
    signals: Dict[str, pd.DataFrame],
    selected_names: List[str],
    threshold_ratio: float = 0.5,
) -> pd.DataFrame:
    """
    合成综合信号

    规则：若某只股票某日同时触发 >= threshold_ratio * 信号总数 的信号，
    则视为触发综合信号

    Returns:
        composite_df: 综合信号 DataFrame (bool)
    """
    selected = [signals[n] for n in selected_names if n in signals]
    if not selected:
        return pd.DataFrame()

    common_dates = selected[0].index
    common_stocks = selected[0].columns
    for sig in selected[1:]:
        common_dates = common_dates.intersection(sig.index)
        common_stocks = common_stocks.intersection(sig.columns)

    count = pd.DataFrame(0, index=common_dates, columns=common_stocks)
    for sig in selected:
        count += sig.loc[common_dates, common_stocks].astype(int)

    threshold = int(np.ceil(len(selected) * threshold_ratio))
    return count >= threshold
