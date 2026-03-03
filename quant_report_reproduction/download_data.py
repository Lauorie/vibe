#!/usr/bin/env python3
"""Download stock data from akshare and cache it."""
import akshare as ak
import pandas as pd
import time
import os
import sys

CACHE_DIR = "/workspace/quant_report_reproduction/data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

cons = pd.read_csv(f"{CACHE_DIR}/csi800_constituents.csv", dtype=str)
stocks = cons["code"].tolist()[:50]
print(f"Downloading {len(stocks)} stocks...", flush=True)

all_close, all_volume, all_open, all_amount = {}, {}, {}, {}

for i, code in enumerate(stocks):
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date="20160101", end_date="20251031", adjust="qfq"
            )
            if df is not None and len(df) > 100:
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "收盘": "close",
                    "最高": "high", "最低": "low",
                    "成交量": "volume", "成交额": "amount",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
                all_close[code] = df["close"].astype(float)
                all_volume[code] = df["volume"].astype(float)
                all_open[code] = df["open"].astype(float)
                all_amount[code] = df["amount"].astype(float)
                print(f"  [{i+1}/{len(stocks)}] {code}: {len(df)} rows", flush=True)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  [{i+1}/{len(stocks)}] {code}: SKIP", flush=True)
    time.sleep(0.3)

print(f"\nGot {len(all_close)} stocks", flush=True)

close_df = pd.DataFrame(all_close)
volume_df = pd.DataFrame(all_volume)
open_df = pd.DataFrame(all_open)
amount_df = pd.DataFrame(all_amount)

close_df.to_csv(f"{CACHE_DIR}/csi800_daily_close.csv")
volume_df.to_csv(f"{CACHE_DIR}/csi800_daily_volume.csv")
open_df.to_csv(f"{CACHE_DIR}/csi800_daily_open.csv")
amount_df.to_csv(f"{CACHE_DIR}/csi800_daily_amount.csv")
print(f"Saved: {close_df.shape[1]} stocks, {len(close_df)} days", flush=True)
