# Skill: 基本面选股研报复现

## 适用场景

当研报涉及基本面选股、财务质量评分、价值投资筛选等主题时使用本 skill。

## QuantsPlaybook 参考实现

| 主题 | 路径 | 核心方法 |
|------|------|---------|
| Piotroski F-Score | `QuantsPlaybook/A-量化基本面/华泰FFScore/FFScore.ipynb` | 9指标/5指标评分 + 低PB筛选 + 随机森林 |
| 大师选股法则 | `QuantsPlaybook/A-量化基本面/申万大师系列十三/` | 多维度价值筛选（PB/PE/股息率/现金流/负债率） |
| 企业生命周期 | `QuantsPlaybook/B-因子构建类/企业生命周期/` | 生命周期阶段划分 + 因子有效性分析 |

## 标准化工作流

### Phase 1：数据准备

```python
import pandas as pd
import numpy as np

# 1. 定义股票池
# 标准筛选：全A股，剔除ST、停牌、上市不足180天
# 参考：FFScore.ipynb 中的 Filter_Stocks 类

# 2. 获取财务数据
# 核心财务指标：ROA, ROE, CFO, 资产负债率, 流动比率, 毛利率, 周转率等
# 注意 point-in-time（避免未来数据）：使用报告期+滞后期
# 参考：申万大师系列中的 get_near_fundamental() 和 OffsetRptdateN()

# 3. 获取行情数据
# 月频/季频收盘价、PB、PE、股息率
# 参考：FFScore.ipynb 中的 get_factor_price(), get_factor_pb_ratio()
```

### Phase 2：因子/信号构建

根据研报方法论选择构建方式：

#### 方式A：评分法（F-Score 类）

```python
# 参考 FFScore.ipynb
# 标准 F-Score 9指标:
# 盈利能力(4)：ROA>0, CFO>0, ΔROA>0, ACCRUAL<0
# 杠杆/流动性(3)：ΔLEVER<0, ΔLIQUID>0, EQ_OFFER≤0
# 经营效率(2)：ΔMARGIN>0, ΔTURN>0

# FFScore 5指标（华泰改进版）:
# ROE>0, ΔROE>0, ΔCATURN>0, ΔTURN>0, ΔLEVER<0

# 计算每个指标的二元得分（满足=1，不满足=0）
# 总分 = 各指标得分之和

# 进阶：使用 F13_Score（13个原始指标）+ RandomForest 特征重要性
```

#### 方式B：条件筛选法（大师系列类）

```python
# 参考 申万大师系列十三
# 定义多个筛选条件，如：
# - PB < 阈值
# - 股息率 > 市场均值
# - PE < 市场均值
# - 负债率 < 33%
# - 价格/自由现金流 < 市场均值的80%

# get_criteria_df()：对每个截面日期应用筛选条件
# 也可用打分法：score_indicators() + add_group()
```

### Phase 3：分层回测与因子分析

```python
# 1. PB分层（如适用）
# 按PB分5组，分析各组收益特征
# 参考：FFScore.ipynb 中 PB quintile 分析

# 2. 因子分层回测
# 按因子得分分组（如 FScore 0-1 vs 8-9）
# 计算各组的月度/年度收益

# 3. 使用 alphalens 进行标准化因子分析
import alphalens
# factor_data = alphalens.utils.get_clean_factor_and_forward_returns(...)
# alphalens.tears.create_full_tear_sheet(factor_data)

# 4. 业绩归因
# 使用 empyrical 计算：年化收益、波动率、Sharpe、最大回撤、IR、Alpha
from empyrical import annual_return, sharpe_ratio, max_drawdown
```

### Phase 4：策略构建与回测

```python
# 使用 hugos_toolkit 进行回测（如适用）
from hugos_toolkit.BackTestReport.performance import strategy_performance
from hugos_toolkit.BackTestReport.tear import analysis_rets

# 或自定义回测框架
# 参考：FFScore.ipynb 中的 Strategy_performance()
# 关键指标：年化收益、年化波动、Sharpe、最大回撤、IR、Alpha、Beta
```

### Phase 5：输出与可视化

```python
# 标准输出：
# 1. 因子/策略绩效汇总表
# 2. 累计收益曲线（vs 基准）
# 3. 分年度收益统计
# 4. 最大回撤分析
# 5. 月度收益热力图

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

## 关键注意事项

1. **Point-in-time 数据**：财务数据必须使用报告发布日期+滞后期，避免使用未来数据
2. **股票池筛选**：每个调仓日动态更新，剔除 ST/停牌/次新股
3. **调仓频率**：通常月频或季频，与财报发布频率匹配
4. **行业分布**：检查策略组合的行业集中度
5. **小市值偏差**：注意基本面因子可能天然偏向小市值

## 常用数据源

| 数据源 | 用途 | 获取方式 |
|--------|------|---------|
| JQData | 行情 + 财务 | `jqdatasdk` |
| Tushare | 行情 + 财务 | `tushare` (需token) |
| Qlib | 行情 | `qlib` (需本地数据) |
| Wind | 行情 + 财务 | Wind API (付费) |
| 本地CSV | 任意 | `pd.read_csv()` |
