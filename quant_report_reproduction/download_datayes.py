#!/usr/bin/env python3
"""
通联数据 DataYes REST API — 下载真实 A 股数据
"""

import requests
import pandas as pd
import numpy as np
import time
import os

CACHE = "/workspace/quant_report_reproduction/data_cache"
os.makedirs(CACHE, exist_ok=True)
BASE_URL = "https://api.wmcloud.com/data/v1/api"

# 200只主流A股（通联数据格式）
STOCK_LIST = [
    # 上证50核心
    "600519.XSHG","601318.XSHG","600036.XSHG","600276.XSHG","601166.XSHG",
    "600900.XSHG","601398.XSHG","600030.XSHG","600887.XSHG","601888.XSHG",
    "600309.XSHG","601012.XSHG","600585.XSHG","601688.XSHG","600809.XSHG",
    "601899.XSHG","600690.XSHG","600048.XSHG","601668.XSHG","601288.XSHG",
    "600000.XSHG","601601.XSHG","601857.XSHG","600016.XSHG","601088.XSHG",
    "601390.XSHG","600050.XSHG","600104.XSHG","601766.XSHG","600031.XSHG",
    "600029.XSHG","601006.XSHG","601628.XSHG","600028.XSHG","601225.XSHG",
    "600196.XSHG","601919.XSHG","600436.XSHG","601138.XSHG","603259.XSHG",
    "600570.XSHG","601669.XSHG","600406.XSHG","600150.XSHG","603288.XSHG",
    "600763.XSHG","601211.XSHG","600600.XSHG","600741.XSHG","600588.XSHG",
    # 深证蓝筹
    "000001.XSHE","000858.XSHE","000333.XSHE","000725.XSHE","000651.XSHE",
    "002594.XSHE","000002.XSHE","002415.XSHE","000063.XSHE","002304.XSHE",
    "000338.XSHE","002475.XSHE","000776.XSHE","000568.XSHE","002236.XSHE",
    "000166.XSHE","002027.XSHE","000895.XSHE","002714.XSHE","000661.XSHE",
    "002352.XSHE","000538.XSHE","002241.XSHE","002142.XSHE","000100.XSHE",
    "002050.XSHE","000876.XSHE","002008.XSHE","000423.XSHE","002032.XSHE",
    "000625.XSHE","002230.XSHE","000983.XSHE","002311.XSHE","000157.XSHE",
    "002410.XSHE","000800.XSHE","002466.XSHE","002044.XSHE","000792.XSHE",
    "002001.XSHE","000960.XSHE","002007.XSHE","000703.XSHE","002202.XSHE",
    "000009.XSHE","000069.XSHE","000513.XSHE","002129.XSHE","000768.XSHE",
    # 中证500/1000补充
    "600332.XSHG","600346.XSHG","600398.XSHG","600426.XSHG","600438.XSHG",
    "600456.XSHG","600460.XSHG","600486.XSHG","600489.XSHG","600516.XSHG",
    "600547.XSHG","600566.XSHG","600567.XSHG","600583.XSHG","600584.XSHG",
    "600637.XSHG","600660.XSHG","600674.XSHG","600703.XSHG","600729.XSHG",
    "600745.XSHG","600765.XSHG","600779.XSHG","600801.XSHG","600848.XSHG",
    "600862.XSHG","600871.XSHG","600886.XSHG","600893.XSHG","600905.XSHG",
    "600918.XSHG","600919.XSHG","600926.XSHG","600958.XSHG","600989.XSHG",
    "601009.XSHG","601016.XSHG","601066.XSHG","601077.XSHG","601100.XSHG",
    "601111.XSHG","601117.XSHG","601155.XSHG","601162.XSHG","601186.XSHG",
    "601198.XSHG","601229.XSHG","601233.XSHG","601236.XSHG","601238.XSHG",
    "000301.XSHE","000400.XSHE","000408.XSHE","000425.XSHE","000488.XSHE",
    "000519.XSHE","000528.XSHE","000537.XSHE","000547.XSHE","000559.XSHE",
    "000563.XSHE","000581.XSHE","000596.XSHE","000598.XSHE","000600.XSHE",
    "000627.XSHE","000629.XSHE","000630.XSHE","000636.XSHE","000650.XSHE",
    "000671.XSHE","000681.XSHE","000686.XSHE","000709.XSHE","000712.XSHE",
    "000717.XSHE","000728.XSHE","000729.XSHE","000738.XSHE","000739.XSHE",
    "000750.XSHE","000761.XSHE","000778.XSHE","000783.XSHE","000786.XSHE",
    "000789.XSHE","000807.XSHE","000825.XSHE","000830.XSHE","000831.XSHE",
    "000837.XSHE","000839.XSHE","000848.XSHE","000860.XSHE","000869.XSHE",
    "000877.XSHE","000878.XSHE","000883.XSHE","000887.XSHE","000893.XSHE",
]


def get_token():
    token = "91c0bbee46a09cb339b4e0d3577950ee6ab63941e71101f2a35660ad1878fb62"
    print(f"Token: {token[:8]}...", flush=True)
    return token


def api_get(token, endpoint, params):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=60)
        if r.status_code == 200:
            j = r.json()
            return j.get("data", []) if j.get("retCode") == 1 else []
        return []
    except:
        return []


def download_daily(token, codes, start="20160101", end="20251031"):
    cache_path = f"{CACHE}/datayes_daily_close.csv"
    if os.path.exists(cache_path):
        print("Loading cached daily data...", flush=True)
        c = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        v = pd.read_csv(f"{CACHE}/datayes_daily_volume.csv", index_col=0, parse_dates=True)
        o = pd.read_csv(f"{CACHE}/datayes_daily_open.csv", index_col=0, parse_dates=True)
        a = pd.read_csv(f"{CACHE}/datayes_daily_amount.csv", index_col=0, parse_dates=True)
        print(f"  {c.shape[1]} stocks, {len(c)} days", flush=True)
        return c, v, o, a

    print(f"Downloading daily data for {len(codes)} stocks...", flush=True)
    all_records = []

    batch_size = 10
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        batch_str = ",".join(batch)
        data = api_get(token, "market/getMktEqudAdj.json", {
            "secID": batch_str, "beginDate": start, "endDate": end,
            "field": "secID,tradeDate,openPrice,closePrice,highestPrice,lowestPrice,turnoverVol,turnoverValue"
        })
        all_records.extend(data)
        done = min(i + batch_size, len(codes))
        if done % 50 == 0 or done == len(codes):
            print(f"  [{done}/{len(codes)}] records={len(all_records)}", flush=True)
        time.sleep(0.5)

    if not all_records:
        print("  No data!", flush=True)
        return None, None, None, None

    df = pd.DataFrame(all_records)
    df["tradeDate"] = pd.to_datetime(df["tradeDate"])

    close_df = df.pivot(index="tradeDate", columns="secID", values="closePrice").sort_index()
    volume_df = df.pivot(index="tradeDate", columns="secID", values="turnoverVol").sort_index()
    open_df = df.pivot(index="tradeDate", columns="secID", values="openPrice").sort_index()
    amount_df = df.pivot(index="tradeDate", columns="secID", values="turnoverValue").sort_index()

    close_df.to_csv(f"{CACHE}/datayes_daily_close.csv")
    volume_df.to_csv(f"{CACHE}/datayes_daily_volume.csv")
    open_df.to_csv(f"{CACHE}/datayes_daily_open.csv")
    amount_df.to_csv(f"{CACHE}/datayes_daily_amount.csv")

    print(f"  Saved {close_df.shape[1]} stocks, {len(close_df)} days", flush=True)
    return close_df, volume_df, open_df, amount_df


def download_balance_sheet(token, codes):
    """下载资产负债表数据（货币资金等）"""
    cache_path = f"{CACHE}/datayes_balance.csv"
    if os.path.exists(cache_path):
        print("Loading cached balance sheet...", flush=True)
        return pd.read_csv(cache_path)

    print("Downloading balance sheet data...", flush=True)
    all_records = []
    batch_size = 10
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        data = api_get(token, "fundamental/getFdmtBSAllLatestA.json", {
            "secID": ",".join(batch), "beginDate": "20140101", "endDate": "20251231",
            "field": "secID,endDate,publishDate,monetaryCap,workcap,TShEquity,totalAssets,totalCurAssets"
        })
        all_records.extend(data)
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(codes):
            print(f"  [{min(i+batch_size, len(codes))}/{len(codes)}] balance={len(all_records)}", flush=True)
        time.sleep(0.3)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    if len(df) > 0:
        df.to_csv(cache_path, index=False)
    print(f"  Balance sheet: {len(df)} rows", flush=True)
    return df


def download_income_statement(token, codes):
    """下载利润表数据（收入结构）"""
    cache_path = f"{CACHE}/datayes_income.csv"
    if os.path.exists(cache_path):
        print("Loading cached income statement...", flush=True)
        return pd.read_csv(cache_path)

    print("Downloading income statement data...", flush=True)
    all_records = []
    batch_size = 10
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        data = api_get(token, "fundamental/getFdmtISAllLatestA.json", {
            "secID": ",".join(batch), "beginDate": "20140101", "endDate": "20251231",
            "field": "secID,endDate,publishDate,revenue,operateProfit,totalProfit,NIncome,operatingCost"
        })
        all_records.extend(data)
        if (i + batch_size) % 100 == 0 or i + batch_size >= len(codes):
            print(f"  [{min(i+batch_size, len(codes))}/{len(codes)}] income={len(all_records)}", flush=True)
        time.sleep(0.3)

    df = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    if len(df) > 0:
        df.to_csv(cache_path, index=False)
    print(f"  Income statement: {len(df)} rows", flush=True)
    return df


def main():
    print("=" * 60)
    print("  DataYes Data Download (200 A-share stocks)")
    print("=" * 60, flush=True)

    token = get_token()
    codes = STOCK_LIST

    close_df, volume_df, open_df, amount_df = download_daily(token, codes)

    balance = download_balance_sheet(token, codes)
    income = download_income_statement(token, codes)

    print("\n=== Download Complete ===")
    if close_df is not None:
        print(f"  Daily: {close_df.shape[1]} stocks x {len(close_df)} days")
    print(f"  Balance: {len(balance)} rows")
    print(f"  Income: {len(income)} rows")


if __name__ == "__main__":
    main()
