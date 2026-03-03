#!/usr/bin/env python3
"""下载通联数据真实财务报表"""

import requests
import pandas as pd
import time
import os

TOKEN = "91c0bbee46a09cb339b4e0d3577950ee6ab63941e71101f2a35660ad1878fb62"
BASE = "https://api.wmcloud.com/data/v1/api"
CACHE = "/workspace/quant_report_reproduction/data_cache"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

STOCK_FILE = f"{CACHE}/datayes_daily_close.csv"
stocks = pd.read_csv(STOCK_FILE, index_col=0, nrows=0).columns.tolist()
print(f"Stocks: {len(stocks)}")


def api_get(endpoint, params):
    try:
        r = requests.get(f"{BASE}/{endpoint}", headers=HEADERS, params=params, timeout=30)
        if r.status_code == 200:
            j = r.json()
            return j.get("data", []) if j.get("retCode") == 1 else []
    except:
        pass
    return []


def download_bs():
    """下载资产负债表"""
    cache = f"{CACHE}/datayes_bs.csv"
    if os.path.exists(cache):
        print(f"BS cache exists: {cache}")
        return pd.read_csv(cache)

    print("Downloading balance sheet...", flush=True)
    all_rows = []
    for i in range(0, len(stocks), 5):
        batch = stocks[i:i+5]
        data = api_get("fundamental/getFdmtBS.json", {
            "secID": ",".join(batch),
            "beginDate": "20140101", "endDate": "20251231",
            "reportType": "A",
            "field": "secID,publishDate,endDate,cashCEquiv,AR,inventories,TCA,fixedAssets,TAssets,TCL,TLiab,TShEquity,retainedEarnings,paidInCapital"
        })
        all_rows.extend(data)
        if (i+5) % 50 == 0 or i+5 >= len(stocks):
            print(f"  [{min(i+5,len(stocks))}/{len(stocks)}] bs={len(all_rows)}", flush=True)
        time.sleep(0.4)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df.to_csv(cache, index=False)
    print(f"  Total: {len(df)} rows", flush=True)
    return df


def download_is():
    """下载利润表"""
    cache = f"{CACHE}/datayes_is.csv"
    if os.path.exists(cache):
        print(f"IS cache exists: {cache}")
        return pd.read_csv(cache)

    print("Downloading income statement...", flush=True)
    all_rows = []
    for i in range(0, len(stocks), 5):
        batch = stocks[i:i+5]
        data = api_get("fundamental/getFdmtIS.json", {
            "secID": ",".join(batch),
            "beginDate": "20140101", "endDate": "20251231",
            "reportType": "A",
            "field": "secID,publishDate,endDate,tRevenue,revenue,COGS,sellExp,adminExp,operateProfit,TProfit,NIncome,NIncomeAttrP,basicEPS"
        })
        all_rows.extend(data)
        if (i+5) % 50 == 0 or i+5 >= len(stocks):
            print(f"  [{min(i+5,len(stocks))}/{len(stocks)}] is={len(all_rows)}", flush=True)
        time.sleep(0.4)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df.to_csv(cache, index=False)
    print(f"  Total: {len(df)} rows", flush=True)
    return df


def download_cf():
    """下载现金流量表"""
    cache = f"{CACHE}/datayes_cf.csv"
    if os.path.exists(cache):
        print(f"CF cache exists: {cache}")
        return pd.read_csv(cache)

    print("Downloading cash flow statement...", flush=True)
    all_rows = []
    for i in range(0, len(stocks), 5):
        batch = stocks[i:i+5]
        data = api_get("fundamental/getFdmtCF.json", {
            "secID": ",".join(batch),
            "beginDate": "20140101", "endDate": "20251231",
            "reportType": "A",
            "field": "secID,publishDate,endDate,CFrSaleGS,CFrOthOperworka,CPayGS,CPaidIBCF,CPaidStworkaff,CPaidTax,NCFOperworka,CPaidFA,NCFInvestA,procShworkaCapworka,CPaidDivProfworka,NCFFinworka,NCE,CCEBegworka"
        })
        all_rows.extend(data)
        if (i+5) % 50 == 0 or i+5 >= len(stocks):
            print(f"  [{min(i+5,len(stocks))}/{len(stocks)}] cf={len(all_rows)}", flush=True)
        time.sleep(0.4)

    df = pd.DataFrame(all_rows)
    if len(df) > 0:
        df.to_csv(cache, index=False)
    print(f"  Total: {len(df)} rows", flush=True)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  Download Financial Statements (DataYes)")
    print("=" * 60)
    bs = download_bs()
    is_ = download_is()
    cf = download_cf()
    print(f"\nDone: BS={len(bs)}, IS={len(is_)}, CF={len(cf)}")
