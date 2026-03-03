"""
三个财务附注经营结构因子 — 基于通联数据真实财务报表

因子定义（原文）:
1. 外币资金占货币资金比 = 1 - RMB_cash / total_cash
2. 境外收入占比稳定性 = ratio / std(ratio)_{t=1,...,6}
3. 主要客户销售收入占比稳定性 = std(top1_ratio)_{t=1,...,3}

由于通联数据主表不直接提供"外币资金币种明细"和"境外收入地区拆分"等
附注级字段（需Wind终端），本模块使用真实财务数据构建等效代理因子:
1. 代理因子1: 经营活动现金流/货币资金 (现金充裕度，类似外币资金反映的境外经营强度)
2. 代理因子2: 营业收入/总资产 稳定性 (收入资产比稳定性，反映收入结构的稳健程度)
3. 代理因子3: 毛利率波动率 (经营结构稳定性，反映主要客户/产品结构的稳定)
"""

import numpy as np
import pandas as pd
import os
from typing import Optional

CACHE = "/workspace/quant_report_reproduction/data_cache"


def load_real_financial_data():
    """加载通联数据真实财务报表"""
    bs = pd.read_csv(f"{CACHE}/datayes_bs.csv")
    is_ = pd.read_csv(f"{CACHE}/datayes_is.csv")
    cf = pd.read_csv(f"{CACHE}/datayes_cf.csv")
    return bs, is_, cf


def _pivot_financial(df, value_col, date_col="endDate", id_col="secID"):
    """将长表转为宽表 (index=endDate, columns=secID)"""
    if value_col not in df.columns:
        return pd.DataFrame()
    sub = df[[id_col, date_col, value_col]].dropna(subset=[value_col])
    sub = sub.drop_duplicates(subset=[id_col, date_col], keep="last")
    pivot = sub.pivot(index=date_col, columns=id_col, values=value_col)
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()


# ============================================================
# Factor 1: 现金充裕度因子 (代理外币资金占比)
# ============================================================

def calc_factor1_cash_intensity(bs_df, cf_df):
    """
    因子1: 经营现金流 / 货币资金

    经济逻辑: 原文的"外币资金占比"反映境外经营强度。
    作为代理, 我们用"经营活动现金流入/货币资金"来衡量现金的活跃程度,
    高比值说明公司现金周转活跃、经营强度高。

    使用真实数据: cashCEquiv(货币资金) 来自资产负债表,
    CFrSaleGS(销售商品提供劳务收到的现金) 来自现金流量表
    """
    cash = _pivot_financial(bs_df, "cashCEquiv")
    if "CFrSaleGS" in cf_df.columns:
        cf_sales = _pivot_financial(cf_df, "CFrSaleGS")
    else:
        return pd.DataFrame()

    common = cash.columns.intersection(cf_sales.columns)
    common_dates = cash.index.intersection(cf_sales.index)

    if len(common) == 0 or len(common_dates) == 0:
        return pd.DataFrame()

    factor = cf_sales.loc[common_dates, common] / cash.loc[common_dates, common].replace(0, np.nan)
    factor = factor.clip(-10, 10)
    return factor


# ============================================================
# Factor 2: 收入结构稳定性因子 (代理境外收入占比稳定性)
# ============================================================

def calc_factor2_revenue_stability(bs_df, is_df):
    """
    因子2: (revenue / TAssets) / std(revenue / TAssets)_{历史}

    经济逻辑: 原文的"境外收入占比稳定性"衡量收入结构的时序稳健性。
    作为代理, 我们用"营业收入/总资产"的当期值除以历史波动率,
    高比值说明收入资产效率高且稳定。

    使用真实数据: revenue(营业收入) 来自利润表,
    TAssets(总资产) 来自资产负债表
    """
    revenue = _pivot_financial(is_df, "revenue")
    total_assets = _pivot_financial(bs_df, "TAssets")

    common = revenue.columns.intersection(total_assets.columns)
    common_dates = revenue.index.intersection(total_assets.index)

    if len(common) == 0 or len(common_dates) == 0:
        return pd.DataFrame()

    ratio = revenue.loc[common_dates, common] / total_assets.loc[common_dates, common].replace(0, np.nan)
    ratio_std = ratio.expanding(min_periods=2).std()
    factor = ratio / ratio_std.replace(0, np.nan)
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.clip(-20, 20)
    return factor


# ============================================================
# Factor 3: 毛利率稳定性因子 (代理主要客户占比稳定性)
# ============================================================

def calc_factor3_margin_stability(is_df):
    """
    因子3: std(毛利率)_{历史3期}

    经济逻辑: 原文的"主要客户销售收入占比稳定性"反映下游需求稳定性。
    毛利率稳定是客户结构和产品结构稳定的直接体现——
    如果主要客户/产品结构不变, 毛利率通常也稳定。

    负向因子: std越低 = 越稳定 = 收益越好

    使用真实数据: revenue(营业收入) 和 COGS(营业成本) 来自利润表
    """
    revenue = _pivot_financial(is_df, "revenue")
    cogs = _pivot_financial(is_df, "COGS")

    common = revenue.columns.intersection(cogs.columns)
    common_dates = revenue.index.intersection(cogs.index)

    if len(common) == 0 or len(common_dates) == 0:
        return pd.DataFrame()

    gross_margin = (revenue.loc[common_dates, common] - cogs.loc[common_dates, common]) / \
                   revenue.loc[common_dates, common].replace(0, np.nan)

    factor = gross_margin.expanding(min_periods=2).std()
    return factor


# ============================================================
# Utility functions
# ============================================================

def composite_factors(factors: dict, method: str = "equal_weight") -> pd.DataFrame:
    """因子等权复合（排名百分比标准化后等权）"""
    factor_list = list(factors.values())

    common_dates = factor_list[0].index
    common_stocks = factor_list[0].columns
    for f in factor_list[1:]:
        common_dates = common_dates.intersection(f.index)
        common_stocks = common_stocks.intersection(f.columns)

    if len(common_dates) == 0 or len(common_stocks) == 0:
        return pd.DataFrame()

    standardized = []
    for f in factor_list:
        f_sub = f.loc[common_dates, common_stocks]
        f_rank = f_sub.rank(axis=1, pct=True)
        standardized.append(f_rank)

    return sum(standardized) / len(standardized)


def generate_report_dates(start, end, freq="semi_annual"):
    """生成财报发布日期"""
    dates = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        if freq == "semi_annual":
            for m, d in [("04", "30"), ("08", "31")]:
                dt = pd.Timestamp(f"{year}-{m}-{d}")
                if dt >= pd.Timestamp(start) and dt <= pd.Timestamp(end):
                    dates.append(dt)
        elif freq == "annual":
            dt = pd.Timestamp(f"{year}-08-31")
            if dt >= pd.Timestamp(start) and dt <= pd.Timestamp(end):
                dates.append(dt)
    return pd.DatetimeIndex(dates)


def expand_factor_to_monthly(factor, monthly_dates):
    """半年/年度因子扩展到月频（前向填充）"""
    combined = factor.index.union(monthly_dates)
    expanded = factor.reindex(combined).sort_index().ffill()
    return expanded.reindex(monthly_dates)
