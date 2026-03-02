# Skill: 因子构建与分析研报复现

## 适用场景

当研报涉及 alpha 因子构建、因子有效性检验、多因子模型、因子合成等主题时使用本 skill。

## QuantsPlaybook 参考实现索引

### 量价类因子
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 高频价量相关性 CPV | `B-因子构建类/高频价量相关性*/` | 日内分钟价量相关 → PV_corr_avg/std/trend → 反转增强 |
| 买卖压力 | `B-因子构建类/基于量价关系度量股票的买卖压力/` | 量价关系度量买卖压力 |
| 振幅因子 | `B-因子构建类/振幅因子的隐藏结构/` | 振幅因子拆解与隐藏结构 |
| 行业量价轮动 | `B-因子构建类/行业有效量价因子与行业轮动策略/` | 行业级量价因子 + ETF轮动 (qlib) |

### 动量/反转类因子
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 高质量动量 | `B-因子构建类/高质量动量因子选股/` | F-Score + 动量交叉 |
| A股动量构造 | `B-因子构建类/A股市场中如何构造动量因子？/` | 多种动量因子构造方法 |
| 球队硬币因子 | `B-因子构建类/个股动量效应的识别及球队硬币因子/` | 动量效应识别 |

### 微观结构类因子
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 筹码分布 | `B-因子构建类/筹码因子/` | 换手率衰减 + 三角/均匀分布 → 筹码集中度 (numba加速) |
| 聪明钱 | `B-因子构建类/聪明钱因子模型的2.0版本/` | 机构/聪明钱资金流改进 |

### 网络/关联类因子
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 网络中心度 | `B-因子构建类/股票网络与网络中心度因子研究/` | SCC(空间) + TCC(时序) + CC(复合) 网络因子 |
| 隔夜日间网络 | `B-因子构建类/基于隔夜与日间的网络关系因子/` | 领先滞后网络 + DLESC聚类 |

### 行为金融类因子
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 凸显度 STR | `B-因子构建类/凸显理论STR因子/` | 凸显理论(Salience Theory)因子 |
| 处置效应 CGO | `B-因子构建类/处置效应因子/` | 资本利得突出量 CGO + 风险偏好 |

### 其他
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 分析师金股 | `B-因子构建类/金股增强策略/` | Beta分布建模分析师推荐成功率 |
| 企业生命周期 | `B-因子构建类/企业生命周期/` | 生命周期阶段因子 |
| 多因子增强 | `B-因子构建类/多因子指数增强/` | 多因子合成 + 指数增强 |
| 因子择时 | `B-因子构建类/因子择时/` | 因子动量/拥挤度择时 |

## 标准化工作流

### Phase 1：数据准备

```python
import pandas as pd
import numpy as np

# 1. 确定股票池和回测区间
# 标准池：全A股，剔除ST/停牌/次新股（上市不足60-180天）
# 回测区间：通常 2010-2020 或更长

# 2. 获取所需行情数据
# 日频：OHLCV、换手率、总市值、流通市值
# 分钟频（如需）：用于高频因子

# 3. 获取辅助数据（按需）
# 行业分类（申万/中信）、财务数据、指数成分
```

### Phase 2：因子计算

根据研报的因子定义，选择对应的计算模式：

#### 模式A：截面因子（最常见）

```python
# 对每个截面日期（通常月末/调仓日），计算所有股票的因子值
# 关键函数模式：
def calc_factor(date, stock_pool, data):
    """计算单个截面的因子值"""
    # 1. 获取该日期前N天的数据窗口
    # 2. 按因子公式计算
    # 3. 返回 Series(index=stock_code, values=factor_value)
    pass

# 循环所有截面日期
factor_df = pd.DataFrame()
for date in rebalance_dates:
    factor = calc_factor(date, stock_pool, data)
    factor_df[date] = factor
# 转为 MultiIndex(date, stock) 的 Series
factor_series = factor_df.stack()
factor_series.index.names = ['date', 'asset']
```

#### 模式B：时序因子

```python
# 对每只股票计算时序因子值（如动量、波动率）
# 关键：使用 rolling/expanding 操作
factor = data.groupby('stock').apply(
    lambda x: x['close'].pct_change(20)  # 示例：20日动量
)
```

#### 模式C：高频因子

```python
# 使用日内分钟数据计算
# 参考 CPV因子.ipynb：
# 1. 获取每日分钟级价量数据
# 2. 计算日内价量相关系数
# 3. 跨日聚合（均值/标准差/趋势）
from scipy.stats import pearsonr
```

#### 模式D：网络因子

```python
# 参考 股票网络中心度因子.ipynb + src/factor_algo.py：
# 1. 计算股票间收益相关矩阵（滚动窗口）
# 2. 构建网络邻接矩阵（阈值/Top-K）
# 3. 计算中心度指标（degree/betweenness/eigenvector/closeness）
# SignalMaker.utils.sliding_window 可用于滑动窗口
```

### Phase 3：因子预处理

```python
# 标准预处理流程（几乎所有因子研报都需要）：

# 1. 去极值（Winsorize / MAD）
def winsorize_mad(series, n=5):
    median = series.median()
    mad = (series - median).abs().median()
    upper = median + n * 1.4826 * mad
    lower = median - n * 1.4826 * mad
    return series.clip(lower, upper)

# 2. 标准化（Z-Score）
def standardize(series):
    return (series - series.mean()) / series.std()

# 3. 中性化（行业 + 市值）
# 回归法：factor ~ industry_dummies + ln(market_cap)
# 残差即为中性化后的因子值
import statsmodels.api as sm
```

### Phase 4：因子有效性检验

```python
# ===== 4.1 IC 分析 =====
# IC = rank_corr(factor, forward_return)
# 标准工具：alphalens
import alphalens

factor_data = alphalens.utils.get_clean_factor_and_forward_returns(
    factor=factor_series,
    prices=prices,
    quantiles=5,
    periods=(1, 5, 10, 20)
)

# IC 均值、IC 标准差、ICIR、IC>0 占比
ic = alphalens.performance.factor_information_coefficient(factor_data)
ic_summary = pd.DataFrame({
    'IC_mean': ic.mean(),
    'IC_std': ic.std(),
    'ICIR': ic.mean() / ic.std(),
    'IC>0': (ic > 0).mean()
})

# ===== 4.2 分层回测 =====
# 按因子值分5组，计算各组收益
alphalens.tears.create_returns_tear_sheet(factor_data)

# ===== 4.3 换手率分析 =====
alphalens.tears.create_turnover_tear_sheet(factor_data)

# ===== 4.4 Fama-MacBeth 回归（可选）=====
# 截面回归检验因子收益率的显著性
```

### Phase 5：因子合成（多因子场景）

```python
# 方式1：等权合成
composite_factor = sum(standardize(f) for f in factors) / len(factors)

# 方式2：IC加权
ic_weights = ic_means / ic_means.abs().sum()
composite_factor = sum(w * standardize(f) for w, f in zip(ic_weights, factors))

# 方式3：ICIR加权
icir_weights = icir_values / icir_values.abs().sum()

# 方式4：最大化IC_IR（优化法）
# 参考 B-因子构建类/多因子指数增强/
```

### Phase 6：策略回测

```python
# 基于因子的选股策略回测
# 通常使用分层多空策略或 Top/Bottom 策略

# 方式1：使用 hugos_toolkit
from hugos_toolkit.BackTestReport.performance import strategy_performance

# 方式2：使用 qlib workflow
# 参考 B-因子构建类/筹码因子/scr/qlib_workflow.py

# 方式3：自定义回测
# 每月调仓，等权/市值加权
# 计算组合收益 → empyrical 性能指标
from empyrical import annual_return, sharpe_ratio, max_drawdown, calmar_ratio
```

### Phase 7：输出报告

```python
# 标准输出内容：
# 1. 因子描述统计（均值、标准差、偏度、峰度）
# 2. IC 序列图 + IC 累计图
# 3. 分层净值曲线
# 4. 多空组合净值曲线
# 5. 分年度收益统计表
# 6. 换手率统计
# 7. 行业分布分析

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

## 关键注意事项

1. **因子方向**：确认因子值与预期收益的单调性方向（正向/负向），必要时取负
2. **数据频率**：确认因子计算频率（日频/周频/月频）与调仓频率的匹配
3. **极端值处理**：MAD 去极值优于百分位去极值（对离群值更稳健）
4. **IC 衰减**：检验 IC 在不同持有期的衰减模式，确定最优调仓频率
5. **换手率**：高换手率因子需考虑交易成本影响
6. **行业暴露**：检查因子是否有明显的行业偏向
7. **市值暴露**：大多数 alpha 因子与市值相关，需中性化后再检验
8. **多重共线性**：多因子合成前检查因子间相关性
9. **样本外检验**：区分 in-sample 和 out-of-sample 结果

## 因子公式速查

| 因子类型 | 典型公式 | 数据需求 |
|---------|---------|---------|
| 动量 | `ret_N = close / close.shift(N) - 1` | 日频收盘价 |
| 反转 | `ret_short = close / close.shift(5) - 1` (取负) | 日频收盘价 |
| 波动率 | `vol = ret.rolling(N).std()` | 日频收益率 |
| 换手率 | `turnover.rolling(N).mean()` | 日频换手率 |
| 特质波动率 | FF3残差的标准差 | 日频收益 + FF3因子 |
| 价量相关 | `pearsonr(price_min, volume_min)` per day | 分钟频价量 |
| 筹码集中度 | 换手率衰减加权的价格分布 | 日频OHLCV+换手率 |
| 网络中心度 | 相关矩阵 → 邻接矩阵 → centrality | 日频收益率 |
