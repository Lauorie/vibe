"""
数据工具模块 — 基于 akshare 获取真实 A 股数据

数据源:
- akshare: 免费 A 股数据（日频 OHLCV）
- 中证 800 成份股 (000906.SH)

研报参数:
- 回测区间: 2016/01/01 - 2025/10/31
- 股票池: 中证 800 成份股
- 基准: 中证 800 等权指数
"""

import os
import time
import numpy as np
import pandas as pd
import akshare as ak
from typing import Optional, Tuple, List
from tqdm import tqdm


CACHE_DIR = "/workspace/quant_report_reproduction/data_cache"


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_csi800_constituents() -> List[str]:
    """获取中证800当前成份股代码列表"""
    cache_path = os.path.join(CACHE_DIR, "csi800_constituents.csv")
    _ensure_cache_dir()

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, dtype=str)
        return df["code"].tolist()

    df = ak.index_stock_cons(symbol="000906")
    codes = df["品种代码"].tolist()

    pd.DataFrame({"code": codes}).to_csv(cache_path, index=False)
    return codes


def fetch_stock_daily(
    code: str,
    start_date: str = "20150601",
    end_date: str = "20251031",
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    获取单只股票日频数据（前复权），带重试
    """
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="qfq"
            )
            if df is None or len(df) == 0:
                return None

            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low",
                "成交量": "volume", "成交额": "amount",
                "换手率": "turnover",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df[["open", "close", "high", "low", "volume", "amount"]].astype(float)
            return df
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 + attempt * 2)
            else:
                return None


def fetch_all_stocks_daily(
    stock_list: List[str],
    start_date: str = "20150601",
    end_date: str = "20251031",
    cache_name: str = "csi800_daily",
    sleep_interval: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    批量获取股票日频数据，支持缓存

    Returns:
        close_df, volume_df, open_df, amount_df
    """
    _ensure_cache_dir()
    cache_prefix = os.path.join(CACHE_DIR, cache_name)

    close_path = f"{cache_prefix}_close.csv"
    volume_path = f"{cache_prefix}_volume.csv"
    open_path = f"{cache_prefix}_open.csv"
    amount_path = f"{cache_prefix}_amount.csv"

    if all(os.path.exists(p) for p in [close_path, volume_path, open_path, amount_path]):
        print("Loading cached data...")
        close_df = pd.read_csv(close_path, index_col=0, parse_dates=True)
        volume_df = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        open_df = pd.read_csv(open_path, index_col=0, parse_dates=True)
        amount_df = pd.read_csv(amount_path, index_col=0, parse_dates=True)
        print(f"  Loaded {close_df.shape[1]} stocks, {len(close_df)} days")
        return close_df, volume_df, open_df, amount_df

    print(f"Fetching daily data for {len(stock_list)} stocks...")
    all_close = {}
    all_volume = {}
    all_open = {}
    all_amount = {}

    for i, code in enumerate(tqdm(stock_list, desc="Fetching stocks")):
        df = fetch_stock_daily(code, start_date, end_date)
        if df is not None and len(df) > 100:
            all_close[code] = df["close"]
            all_volume[code] = df["volume"]
            all_open[code] = df["open"]
            all_amount[code] = df["amount"]

        time.sleep(sleep_interval)

    close_df = pd.DataFrame(all_close)
    volume_df = pd.DataFrame(all_volume)
    open_df = pd.DataFrame(all_open)
    amount_df = pd.DataFrame(all_amount)

    close_df.to_csv(close_path)
    volume_df.to_csv(volume_path)
    open_df.to_csv(open_path)
    amount_df.to_csv(amount_path)

    print(f"  Fetched {close_df.shape[1]} stocks, {len(close_df)} days")
    print(f"  Data cached to {cache_prefix}_*.parquet")

    return close_df, volume_df, open_df, amount_df


def fetch_csi800_index_daily(
    start_date: str = "20150601",
    end_date: str = "20251031",
) -> pd.DataFrame:
    """获取中证800指数日频数据"""
    cache_path = os.path.join(CACHE_DIR, "csi800_index_daily.csv")
    _ensure_cache_dir()

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = ak.stock_zh_index_daily(symbol="sh000906")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.loc[start_date:end_date]
    df.to_csv(cache_path)
    return df


def prepare_real_data(
    n_stocks: int = 50,
    start_date: str = "20150601",
    end_date: str = "20251031",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    准备真实数据的主入口（从缓存加载）
    """
    _ensure_cache_dir()
    cache_prefix = os.path.join(CACHE_DIR, "csi800_daily")

    close_path = f"{cache_prefix}_close.csv"
    volume_path = f"{cache_prefix}_volume.csv"
    open_path = f"{cache_prefix}_open.csv"
    amount_path = f"{cache_prefix}_amount.csv"

    if all(os.path.exists(p) for p in [close_path, volume_path, open_path, amount_path]):
        print("Loading cached real data...")
        close_df = pd.read_csv(close_path, index_col=0, parse_dates=True)
        volume_df = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        open_df = pd.read_csv(open_path, index_col=0, parse_dates=True)
        amount_df = pd.read_csv(amount_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(
            "Data cache not found. Run download_data.py first to fetch data."
        )

    close_df = close_df.loc["2016-01-01":"2025-10-31"]
    volume_df = volume_df.loc["2016-01-01":"2025-10-31"]
    open_df = open_df.loc["2016-01-01":"2025-10-31"]
    amount_df = amount_df.loc["2016-01-01":"2025-10-31"]

    close_df = close_df.dropna(axis=1, thresh=int(len(close_df) * 0.8))
    common_stocks = close_df.columns
    volume_df = volume_df[common_stocks]
    open_df = open_df[common_stocks]
    amount_df = amount_df[common_stocks]

    close_df = close_df.ffill().bfill()
    volume_df = volume_df.ffill().bfill()
    open_df = open_df.ffill().bfill()
    amount_df = amount_df.ffill().bfill()

    if n_stocks < len(close_df.columns):
        stocks = close_df.columns[:n_stocks].tolist()
        close_df = close_df[stocks]
        volume_df = volume_df[stocks]
        open_df = open_df[stocks]
        amount_df = amount_df[stocks]

    benchmark_returns = get_benchmark_returns(close_df)

    print(f"  Loaded {close_df.shape[1]} stocks, {len(close_df)} trading days")
    print(f"  Date range: {close_df.index[0].date()} to {close_df.index[-1].date()}")

    return close_df, volume_df, open_df, amount_df, benchmark_returns


def get_benchmark_returns(
    close_df: pd.DataFrame,
    method: str = "equal_weight",
) -> pd.Series:
    """计算等权基准收益率"""
    daily_returns = close_df.pct_change()
    if method == "equal_weight":
        benchmark = daily_returns.mean(axis=1)
    else:
        benchmark = daily_returns.mean(axis=1)
    benchmark.name = "benchmark"
    return benchmark


def get_trading_calendar(close_df: pd.DataFrame) -> pd.DatetimeIndex:
    return close_df.index


def get_weekly_rebalance_dates(trading_calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.Series(trading_calendar)
    week_end = dates.groupby(dates.dt.isocalendar().week).last()
    return pd.DatetimeIndex(week_end.values)


# ============================================================
# 分钟级数据模拟（基于真实日频数据特征生成）
# ============================================================

def generate_realistic_minute_data(
    daily_close: pd.Series,
    daily_volume: pd.Series,
    daily_high: Optional[pd.Series] = None,
    daily_low: Optional[pd.Series] = None,
    n_minutes: int = 240,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于真实日频数据特征生成分钟级数据

    使用真实的 OHLCV 约束来生成合理的日内分布，
    保证分钟级价格波动与真实日频 high/low/close 一致。
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily_close)

    minute_close_list = []
    minute_volume_list = []

    for i in range(n_days):
        prev_close = daily_close.iloc[i - 1] if i > 0 else daily_close.iloc[0] * 0.99
        target_close = daily_close.iloc[i]

        if daily_high is not None and daily_low is not None:
            day_high = daily_high.iloc[i]
            day_low = daily_low.iloc[i]
        else:
            day_range = abs(target_close - prev_close) * 2
            day_high = max(prev_close, target_close) + day_range * 0.3
            day_low = min(prev_close, target_close) - day_range * 0.3

        if np.isnan(target_close) or np.isnan(prev_close) or prev_close <= 0:
            minute_close_list.append(np.full(n_minutes, np.nan))
            minute_volume_list.append(np.full(n_minutes, np.nan))
            continue

        drift = np.log(target_close / prev_close) / n_minutes
        intraday_vol = abs(np.log(day_high / max(day_low, 0.01))) / (4 * np.sqrt(n_minutes))
        if intraday_vol < 1e-6:
            intraday_vol = 0.001

        noise = rng.normal(0, intraday_vol, size=n_minutes)
        returns = drift + noise

        minute_prices = prev_close * np.exp(np.cumsum(returns))
        scale_factor = target_close / minute_prices[-1]
        minute_prices *= scale_factor

        minute_prices = np.clip(minute_prices, day_low * 0.98, day_high * 1.02)

        total_vol = daily_volume.iloc[i]
        if np.isnan(total_vol) or total_vol <= 0:
            total_vol = 1e6

        u_shape = np.array([
            1.8 - 1.0 * np.sin(np.pi * j / n_minutes) + 0.5 * np.exp(-j / 10) + 0.5 * np.exp(-(n_minutes - j) / 20)
            for j in range(n_minutes)
        ])
        vol_noise = rng.lognormal(0, 0.4, size=n_minutes)
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
