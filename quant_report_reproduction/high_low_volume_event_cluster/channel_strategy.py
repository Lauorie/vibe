"""
资金通道策略回测引擎

实现研报中的资金通道策略（Capital Channel Strategy）：

(1) 设置 N 个资金通道，每个通道持股周期为 holding_period 个交易日
(2) 每周末，回看过去 5 个交易日，所有曾触发入场信号的股票 → 目标股票池
(3) 下周初开盘时，在空闲通道中等权买入，持有 holding_period 天后平仓
(4) 计算各通道净值并求和
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple


class ChannelStrategy:
    """
    资金通道策略回测引擎

    Parameters:
        n_channels: 资金通道数量，默认4
        holding_period: 每个通道持股周期（交易日数），默认20
        lookback_days: 每周末回看天数，默认5
    """

    def __init__(
        self,
        n_channels: int = 4,
        holding_period: int = 20,
        lookback_days: int = 5,
    ):
        self.n_channels = n_channels
        self.holding_period = holding_period
        self.lookback_days = lookback_days

    def run(
        self,
        signal_df: pd.DataFrame,
        open_df: pd.DataFrame,
        close_df: pd.DataFrame,
        exclude_signal_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        运行资金通道策略

        Args:
            signal_df: 入场信号 (index=date, columns=stocks, dtype=bool)
            open_df: 开盘价
            close_df: 收盘价
            exclude_signal_df: 剔除信号（高位放量负向剔除）

        Returns:
            result_df: DataFrame with columns:
                - channel_returns: 各通道收益
                - portfolio_value: 组合净值
                - daily_return: 日收益率
                - weekly_holdings: 每周持股数量
        """
        trading_dates = signal_df.index.tolist()
        stocks = signal_df.columns.tolist()

        rebalance_dates = self._get_weekly_rebalance_dates(trading_dates)

        channels = [
            {"status": "idle", "stocks": [], "entry_date": None,
             "entry_prices": {}, "days_held": 0}
            for _ in range(self.n_channels)
        ]

        portfolio_values = []
        daily_returns_list = []
        weekly_holdings = []

        date_idx = {d: i for i, d in enumerate(trading_dates)}

        for t, date in enumerate(trading_dates):
            for ch in channels:
                if ch["status"] == "holding":
                    ch["days_held"] += 1
                    if ch["days_held"] >= self.holding_period:
                        ch["status"] = "idle"
                        ch["stocks"] = []
                        ch["entry_prices"] = {}
                        ch["days_held"] = 0

            if date in rebalance_dates and t + 1 < len(trading_dates):
                target_stocks = self._get_target_stocks(
                    signal_df, trading_dates, t, exclude_signal_df
                )

                next_date = trading_dates[t + 1]

                for ch in channels:
                    if ch["status"] == "idle" and len(target_stocks) > 0:
                        ch["status"] = "holding"
                        ch["stocks"] = target_stocks.copy()
                        ch["entry_date"] = next_date
                        ch["entry_prices"] = {
                            s: open_df.loc[next_date, s]
                            for s in target_stocks
                            if s in open_df.columns and not pd.isna(open_df.loc[next_date, s])
                        }
                        ch["stocks"] = list(ch["entry_prices"].keys())
                        ch["days_held"] = 0
                        break

            total_value = 0
            n_active = 0
            for ch in channels:
                if ch["status"] == "holding" and ch["stocks"]:
                    ch_value = 0
                    for s in ch["stocks"]:
                        if s in close_df.columns and date in close_df.index:
                            entry_p = ch["entry_prices"].get(s, np.nan)
                            if not np.isnan(entry_p) and entry_p > 0:
                                current_p = close_df.loc[date, s]
                                if not pd.isna(current_p):
                                    ch_value += current_p / entry_p
                    if len(ch["stocks"]) > 0:
                        ch_value /= len(ch["stocks"])
                        total_value += ch_value
                        n_active += 1

            for ch in channels:
                if ch["status"] == "idle":
                    total_value += 1.0
                    n_active += 1

            if n_active > 0:
                portfolio_values.append(total_value / self.n_channels)
            else:
                portfolio_values.append(1.0)

            if date in rebalance_dates:
                n_holding = sum(
                    len(ch["stocks"]) for ch in channels if ch["status"] == "holding"
                )
                weekly_holdings.append(n_holding)

        nav = pd.Series(portfolio_values, index=trading_dates, name="nav")
        nav = nav / nav.iloc[0]

        daily_ret = nav.pct_change().fillna(0)
        daily_ret.name = "daily_return"

        result = pd.DataFrame({
            "nav": nav,
            "daily_return": daily_ret,
        })

        return result

    def _get_weekly_rebalance_dates(self, trading_dates: list) -> set:
        """获取每周最后一个交易日作为调仓日"""
        dates = pd.DatetimeIndex(trading_dates)
        df = pd.DataFrame({"date": dates})
        df["year_week"] = df["date"].dt.isocalendar().week.values
        df["year"] = df["date"].dt.year

        rebalance = df.groupby(["year", "year_week"])["date"].last()
        return set(rebalance.values)

    def _get_target_stocks(
        self,
        signal_df: pd.DataFrame,
        trading_dates: list,
        current_idx: int,
        exclude_signal_df: Optional[pd.DataFrame] = None,
    ) -> List[str]:
        """
        回看过去 lookback_days 天，获取触发信号的目标股票池
        可选：用 exclude_signal_df 进行负向剔除
        """
        start_idx = max(0, current_idx - self.lookback_days + 1)
        lookback_dates = trading_dates[start_idx:current_idx + 1]

        triggered = set()
        for d in lookback_dates:
            if d in signal_df.index:
                stocks_triggered = signal_df.columns[signal_df.loc[d]].tolist()
                triggered.update(stocks_triggered)

        if exclude_signal_df is not None:
            excluded = set()
            for d in lookback_dates:
                if d in exclude_signal_df.index:
                    stocks_excluded = exclude_signal_df.columns[
                        exclude_signal_df.loc[d]
                    ].tolist()
                    excluded.update(stocks_excluded)
            triggered -= excluded

        return list(triggered)


def compute_channel_strategy_metrics(
    strategy_result: pd.DataFrame,
    benchmark_returns: pd.Series,
) -> dict:
    """
    计算通道策略绩效指标

    Returns:
        dict with: annual_return, annual_volatility, info_ratio,
                   max_drawdown, excess metrics
    """
    nav = strategy_result["nav"]
    strat_ret = strategy_result["daily_return"]

    common_idx = strat_ret.index.intersection(benchmark_returns.index)
    strat_ret = strat_ret.loc[common_idx]
    bench_ret = benchmark_returns.loc[common_idx]

    ann_factor = 252

    strat_ann_ret = (1 + strat_ret).prod() ** (ann_factor / len(strat_ret)) - 1
    strat_ann_vol = strat_ret.std() * np.sqrt(ann_factor)

    bench_cum = (1 + bench_ret).cumprod()
    bench_ann_ret = (1 + bench_ret).prod() ** (ann_factor / len(bench_ret)) - 1

    excess_ret = strat_ret - bench_ret
    excess_cum = (1 + excess_ret).cumprod()
    excess_ann_ret = (1 + excess_ret).prod() ** (ann_factor / len(excess_ret)) - 1
    excess_ann_vol = excess_ret.std() * np.sqrt(ann_factor)
    info_ratio = excess_ann_ret / excess_ann_vol if excess_ann_vol > 0 else 0

    excess_peak = excess_cum.expanding().max()
    excess_dd = excess_cum / excess_peak - 1
    excess_max_dd = excess_dd.min()

    strat_cum = (1 + strat_ret).cumprod()
    strat_peak = strat_cum.expanding().max()
    strat_dd = strat_cum / strat_peak - 1
    strat_max_dd = strat_dd.min()

    return {
        "strategy_annual_return": strat_ann_ret,
        "strategy_annual_volatility": strat_ann_vol,
        "strategy_max_drawdown": strat_max_dd,
        "benchmark_annual_return": bench_ann_ret,
        "excess_annual_return": excess_ann_ret,
        "excess_annual_volatility": excess_ann_vol,
        "excess_info_ratio": info_ratio,
        "excess_max_drawdown": excess_max_dd,
        "excess_cumulative": excess_cum,
        "strategy_cumulative": strat_cum,
        "benchmark_cumulative": bench_cum,
    }
