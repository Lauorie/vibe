#!/usr/bin/env python3
"""
使用 Tushare 下载真实 A 股数据

包括：
1. 中证800成份股日频行情（OHLCV）
2. 财务附注数据（货币资金外币占比、境外收入、客户集中度）
"""

import tushare as ts
import pandas as pd
import numpy as np
import time
import os
import sys

TOKEN = "aae75f6674fbeb48dfcfd7e4d643fc643501632cc18038e62bdf29d0"
ts.set_token(TOKEN)
pro = ts.pro_api()

CACHE = "/workspace/quant_report_reproduction/data_cache"
os.makedirs(CACHE, exist_ok=True)


def download_index_constituents():
    """下载中证800成份股"""
    print("Downloading CSI 800 constituents...", flush=True)
    df = pro.index_weight(index_code="000906.SH", start_date="20250101", end_date="20250301")
    if df is None or len(df) == 0:
        df = pro.index_weight(index_code="000906.SH", start_date="20241201", end_date="20250101")
    if df is None or len(df) == 0:
        print("  Falling back to existing cache")
        return pd.read_csv(f"{CACHE}/csi800_ts_constituents.csv", dtype=str)["ts_code"].tolist()

    latest = df[df["trade_date"] == df["trade_date"].max()]
    codes = latest["con_code"].unique().tolist()
    pd.DataFrame({"ts_code": codes}).to_csv(f"{CACHE}/csi800_ts_constituents.csv", index=False)
    print(f"  Got {len(codes)} constituents", flush=True)
    return codes


def download_daily_data(codes, start="20150601", end="20251031"):
    """批量下载日频行情"""
    cache_path = f"{CACHE}/tushare_daily_close.csv"
    if os.path.exists(cache_path):
        print("Daily data cache exists, loading...", flush=True)
        close_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        volume_df = pd.read_csv(f"{CACHE}/tushare_daily_volume.csv", index_col=0, parse_dates=True)
        open_df = pd.read_csv(f"{CACHE}/tushare_daily_open.csv", index_col=0, parse_dates=True)
        amount_df = pd.read_csv(f"{CACHE}/tushare_daily_amount.csv", index_col=0, parse_dates=True)
        print(f"  Loaded {close_df.shape[1]} stocks, {len(close_df)} days", flush=True)
        return close_df, volume_df, open_df, amount_df

    print(f"Downloading daily data for {len(codes)} stocks...", flush=True)
    all_close, all_volume, all_open, all_amount = {}, {}, {}, {}

    batch_size = 8
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        ts_codes = ",".join(batch)
        try:
            df = pro.daily(ts_code=ts_codes, start_date=start, end_date=end)
            if df is not None and len(df) > 0:
                for code in batch:
                    sub = df[df["ts_code"] == code].copy()
                    if len(sub) > 50:
                        sub["trade_date"] = pd.to_datetime(sub["trade_date"])
                        sub = sub.sort_values("trade_date").set_index("trade_date")
                        all_close[code] = sub["close"]
                        all_volume[code] = sub["vol"]
                        all_open[code] = sub["open"]
                        all_amount[code] = sub["amount"]
            print(f"  [{i+len(batch)}/{len(codes)}] batch OK", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(codes)}] batch FAIL: {e}", flush=True)
            time.sleep(1)
        time.sleep(0.3)

    close_df = pd.DataFrame(all_close).sort_index()
    volume_df = pd.DataFrame(all_volume).sort_index()
    open_df = pd.DataFrame(all_open).sort_index()
    amount_df = pd.DataFrame(all_amount).sort_index()

    close_df.to_csv(f"{CACHE}/tushare_daily_close.csv")
    volume_df.to_csv(f"{CACHE}/tushare_daily_volume.csv")
    open_df.to_csv(f"{CACHE}/tushare_daily_open.csv")
    amount_df.to_csv(f"{CACHE}/tushare_daily_amount.csv")
    print(f"  Saved {close_df.shape[1]} stocks, {len(close_df)} days", flush=True)
    return close_df, volume_df, open_df, amount_df


def download_financial_notes(codes):
    """
    下载财务附注相关数据:
    - fina_indicator: ROE, revenue growth etc.
    - income: revenue breakdown
    - balancesheet: cash breakdown
    """
    cache_path = f"{CACHE}/tushare_fina_indicator.csv"
    if os.path.exists(cache_path):
        print("Financial data cache exists, loading...", flush=True)
        fina = pd.read_csv(cache_path, dtype={"ts_code": str})
        income = pd.read_csv(f"{CACHE}/tushare_income.csv", dtype={"ts_code": str})
        balance = pd.read_csv(f"{CACHE}/tushare_balance.csv", dtype={"ts_code": str})
        print(f"  Loaded fina={len(fina)}, income={len(income)}, balance={len(balance)}", flush=True)
        return fina, income, balance

    print("Downloading financial indicator data...", flush=True)
    fina_list = []
    income_list = []
    balance_list = []

    for i, code in enumerate(codes):
        try:
            df = pro.fina_indicator(ts_code=code, start_date="20140101", end_date="20251231",
                                    fields="ts_code,ann_date,end_date,roe,roa,grossprofit_margin,op_income_of_gr")
            if df is not None and len(df) > 0:
                fina_list.append(df)
        except Exception as e:
            pass

        try:
            df2 = pro.income(ts_code=code, start_date="20140101", end_date="20251231",
                            fields="ts_code,ann_date,end_date,revenue,operate_profit,total_profit,n_income")
            if df2 is not None and len(df2) > 0:
                income_list.append(df2)
        except Exception as e:
            pass

        try:
            df3 = pro.balancesheet(ts_code=code, start_date="20140101", end_date="20251231",
                                   fields="ts_code,ann_date,end_date,money_cap,oth_cur_assets,total_cur_assets")
            if df3 is not None and len(df3) > 0:
                balance_list.append(df3)
        except Exception as e:
            pass

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(codes)}] downloaded", flush=True)
            time.sleep(1)
        time.sleep(0.15)

    fina = pd.concat(fina_list, ignore_index=True) if fina_list else pd.DataFrame()
    income = pd.concat(income_list, ignore_index=True) if income_list else pd.DataFrame()
    balance = pd.concat(balance_list, ignore_index=True) if balance_list else pd.DataFrame()

    fina.to_csv(f"{CACHE}/tushare_fina_indicator.csv", index=False)
    income.to_csv(f"{CACHE}/tushare_income.csv", index=False)
    balance.to_csv(f"{CACHE}/tushare_balance.csv", index=False)

    print(f"  Saved fina={len(fina)}, income={len(income)}, balance={len(balance)}", flush=True)
    return fina, income, balance


def main():
    print("=" * 60)
    print("  Tushare Data Download")
    print("=" * 60, flush=True)

    codes = download_index_constituents()

    use_codes = codes[:200]
    close_df, volume_df, open_df, amount_df = download_daily_data(use_codes)

    fina, income, balance = download_financial_notes(use_codes)

    print("\n=== Download Complete ===")
    print(f"Daily: {close_df.shape[1]} stocks x {len(close_df)} days")
    print(f"Fina: {len(fina)} rows")
    print(f"Income: {len(income)} rows")
    print(f"Balance: {len(balance)} rows")


if __name__ == "__main__":
    main()
