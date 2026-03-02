# Skill: 择时策略研报复现

## 适用场景

当研报涉及指数择时、趋势判断、买卖信号生成、技术分析、市场情绪等主题时使用本 skill。

## QuantsPlaybook 参考实现索引

### 经典择时指标
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| RSRS 阻力支撑 | `C-择时类/RSRS择时指标/` | high~low 回归斜率 β → z-score → R²修正 |
| QRS | `C-择时类/QRS择时信号/` | zscore(β) × R² (`SignalMaker.qrs.QRSCreator`) |
| ICU 均线 | `C-择时类/ICU均线/` | Repeated Median 稳健回归均线 |
| 鳄鱼线 | `C-择时类/基于鳄鱼线*/` | Williams Alligator + AO + Fractal + MACD |
| 均线通道 | `C-择时类/均线交叉结合通道突破*/` | MA交叉 + 通道突破 |
| 低延迟趋势线 | `C-择时类/低延迟趋势线*/` | 低延迟趋势线 |
| 扩散指标 | `C-择时类/扩散指标/` | NH-NL% 扩散 + 快慢线 |
| 趋与势 | `C-择时类/趋与势的量化定义研究/` | 归一化位移 + MA符号序列 |

### 信号处理类
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| HHT 择时 | `C-择时类/结合改进HHT模型*/` | EMD分解 + Hilbert变换 + ML分类 (`SignalMaker.hht_signal`) |
| 小波分析 | `C-择时类/小波分析/` | 小波分解 + Hilbert/SVM |

### 量价择时
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 价量共振 | `C-择时类/成交量的奥秘*/` | 价格能量×成交量能量 → 市场状态 |
| 特征分布建模(一) | `C-择时类/特征分布建模择时/` | 净买入 + HMA 交叉信号 |
| 特征分布建模(二) | `C-择时类/特征分布建模择时系列之二/` | AMA5/AMA100 + 双峰分布 + 贝叶斯优化 |
| 日内动量 ETF | `C-择时类/另类ETF交易策略*/` | 噪音区域突破 (`SignalMaker.noise_area.NoiseArea`) |

### 波动率/统计类
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 波动率择时 | `C-择时类/择时视角下的波动率因子/` | FF3残差波动率 + 1-R² |
| 高阶矩择时 | `C-择时类/指数高阶矩择时/` | 偏度、峰度预测 |
| 时变 Sharpe | `C-择时类/时变夏普/` | 宏观变量 → 条件 Sharpe |
| C-VIX | `C-择时类/C-VIX*/` | 50ETF期权 → 中国版VIX |
| 牛熊指标 | `C-择时类/CSVC框架及熊牛指标/` | 波动率+换手率构建牛熊 |

### 情绪/资金类
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 投资者情绪 | `C-择时类/投资者情绪指数择时模型/` | GSISI = Spearman(行业β, 指数) |
| 北向资金 | `C-择时类/北向资金*/` | 北向资金流预测 |
| 羊群效应 CCK | `C-择时类/基于CCK模型*/` | CSAD ~ |R_m| + R_m² |

### 技术形态
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 技术形态识别 | `C-择时类/技术分析算法框架与实战/` | 头肩顶/底、三角形、矩形等 |
| 圆弧底识别 | `C-择时类/技术分析算法框架与实战二/` | 核回归 + 局部极值 → 圆弧底 |

### 其他
| 主题 | 路径 | 核心方法 |
|------|------|---------|
| 点位效率 | `C-择时类/基于点位效率理论*/` | 自动波段划分 + 趋势预测 |
| 相对强弱 | `C-择时类/基于相对强弱下单向波动差值应用/` | 波动率-收益 Granger因果 |
| 顶底信号 | `C-择时类/行业指数顶部和底部信号/` | NH-NL% 锚定效应 → 顶底 |
| Trader-Company | `C-择时类/Trader-Company*/` | Trader集成模型 + Company管理 |

## 标准化工作流

### Phase 1：数据准备

```python
import pandas as pd
import numpy as np

# 1. 确定择时标的
# 常见标的：沪深300、中证500、上证50、创业板指、行业ETF
# 数据需求：OHLCV（日频/分钟频，视策略而定）

# 2. 获取行情数据
# 日频数据通常足够大多数择时策略
# 分钟频数据：用于日内择时和高频信号

# 3. 获取辅助数据（按需）
# 宏观数据：M1、国债收益率、PMI 等（时变Sharpe）
# 期权数据：50ETF期权（VIX）
# 资金流数据：北向资金、融资融券
# 行业数据：行业指数收益率（情绪指标）
```

### Phase 2：信号构建

根据研报方法论选择信号构建模式：

#### 模式A：回归/统计类信号（RSRS/QRS）

```python
# ===== RSRS =====
# 参考 C-择时类/RSRS择时指标/py/RSRS.ipynb
import statsmodels.api as sm

def calc_rsrs(high, low, window=18):
    """high ~ low 回归斜率 β"""
    betas = []
    for i in range(window, len(high)):
        y = high[i-window:i]
        X = sm.add_constant(low[i-window:i])
        model = sm.OLS(y, X).fit()
        betas.append(model.params[1])
    return pd.Series(betas)

# z-score 标准化
rsrs_zscore = (rsrs - rsrs.rolling(600).mean()) / rsrs.rolling(600).std()
# R² 修正
rsrs_signal = rsrs_zscore * r_squared

# ===== QRS =====
# 直接使用 SignalMaker
from SignalMaker.qrs import QRSCreator
qrs = QRSCreator(low_df, high_df)
signal = qrs.fit(regression_window=18, zscore_window=600, n=2)
```

#### 模式B：信号处理类（HHT/小波）

```python
# ===== HHT =====
# 使用 SignalMaker
from SignalMaker.hht_signal import get_ht_signal, get_hht_signal, parallel_apply

# 简单 HT 信号
ht_signal = get_ht_signal(close, ma_period=60, ht_period=30)

# HHT 信号（EMD + HT）
hht_signal = parallel_apply(close, window=60, imf_index=2, max_imf=9, method='EMD', n_jobs=-1)

# ===== 小波 =====
import pywt
coeffs = pywt.wavedec(close, 'db4', level=4)
# 选取特定分量重构
```

#### 模式C：技术指标类（鳄鱼线/MACD/均线）

```python
# ===== 鳄鱼线 =====
from SignalMaker.alligator_indicator_timing import (
    get_alligator_signal, get_ao_indicator_signal,
    get_fractal_signal, get_macd_signal, evaluate_signals
)

alligator = get_alligator_signal(close_df, periods=(13,8,5), lag=(8,5,3))
ao = get_ao_indicator_signal(high_df, low_df, window=3)
fractal = get_fractal_signal(close_df, high_df, low_df)
macd = get_macd_signal(close_df)

# 综合评估
combined = signals_df.apply(evaluate_signals, axis=1)

# ===== 均线类 =====
import talib
ma_short = talib.SMA(close, timeperiod=5)
ma_long = talib.SMA(close, timeperiod=20)
signal = np.where(ma_short > ma_long, 1, -1)
```

#### 模式D：量价类信号

```python
# ===== 价量共振 =====
# 参考 C-择时类/成交量的奥秘*/scr/create_signal.py
# 价格能量 = BMA_today / BMA_(today-N)
# 成交量能量 = AMA5 / AMA_long
# 共振 = 价格能量 × 成交量能量

# ===== VMACD-MTM =====
from SignalMaker.vmacd_mtm import calc_vmacd_mtm
vmacd = calc_vmacd_mtm(volume, period=60)

# ===== 日内动量（ETF） =====
from SignalMaker.noise_area import NoiseArea
na = NoiseArea(ohlcv_data)
signal_df = na.fit(window=14)
```

#### 模式E：情绪/宏观类信号

```python
# ===== 投资者情绪 GSISI =====
# 参考 C-择时类/投资者情绪指数择时模型/
# 1. 计算各行业对指数的滚动 β
# 2. 计算 β 排名与指数涨跌排名的 Spearman 相关
# 3. GSISI 即为该相关系数

# ===== CCK 羊群效应 =====
# CSAD = (1/N) Σ|R_i - R_m|
# CSAD = α + β₁|R_m| + β₂R_m² + ε
# β₂ < 0 → 羊群效应
```

### Phase 3：信号处理与阈值确定

```python
# 1. 信号平滑（避免频繁交易）
signal_smooth = signal.rolling(window=3).mean()

# 2. 阈值确定
# 方式A：固定阈值（如 RSRS > 0.7 做多，< -0.7 做空）
# 方式B：滚动分位数（如 > 80%分位做多，< 20%分位做空）
# 方式C：参数优化
from skopt import gp_minimize  # 贝叶斯优化

# 3. 信号转换为仓位
def signal_to_position(signal, open_threshold, close_threshold):
    position = pd.Series(0, index=signal.index)
    for i in range(1, len(signal)):
        if signal.iloc[i] >= open_threshold:
            position.iloc[i] = 1
        elif signal.iloc[i] <= close_threshold:
            position.iloc[i] = 0
        else:
            position.iloc[i] = position.iloc[i-1]
    return position
```

### Phase 4：回测

```python
# ===== 方式1：使用 hugos_toolkit（推荐） =====
from hugos_toolkit.BackTestTemplate import get_backtesting
from hugos_toolkit.BackTestTemplate.bt_strategy import SignalStrategy

# 需要将信号添加到 data feed 中
result = get_backtesting(
    data=data,
    name='HS300',
    strategy=SignalStrategy,
    begin_dt='2010-01-01',
    end_dt='2023-12-31',
    commission=0.0003,
    stamp_duty=0.001,
    slippage_perc=0.001
)

# 生成报告
from hugos_toolkit.BackTestReport.tear import analysis_rets, analysis_trade
report = analysis_rets(price, result.result, benchmark_rets)
trade_report = analysis_trade(price, result.result)

# ===== 方式2：使用 Backtrader =====
import backtrader as bt

# ===== 方式3：使用 vectorbt =====
import vectorbt as vbt

# ===== 方式4：使用 empyrical（简单净值计算） =====
from empyrical import annual_return, sharpe_ratio, max_drawdown, calmar_ratio

# 计算策略收益
strategy_returns = position.shift(1) * daily_returns
metrics = {
    '年化收益': annual_return(strategy_returns),
    '年化波动': strategy_returns.std() * np.sqrt(252),
    'Sharpe': sharpe_ratio(strategy_returns),
    '最大回撤': max_drawdown(strategy_returns),
    'Calmar': calmar_ratio(strategy_returns)
}
```

### Phase 5：样本内外分析

```python
# 择时策略容易过拟合，务必区分样本内外

# 1. 时间切分
in_sample = strategy_returns['2010':'2018']
out_sample = strategy_returns['2019':]

# 2. CSCV 过拟合检验（可选）
# 参考 C-择时类/CSVC框架及熊牛指标/py/CSCV回测过拟合概率分析框架.ipynb

# 3. 参数稳健性
# 在参数邻域内检验策略绩效的稳定性
```

### Phase 6：输出报告

```python
# 标准输出：
# 1. 信号序列图（信号值 + 指数走势 + 多空区间标记）
# 2. 策略净值曲线（vs 基准买入持有）
# 3. 分年度收益对比表
# 4. 最大回撤期间标记
# 5. 信号统计（多空频率、持仓天数分布、胜率）
# 6. 样本内外对比

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用 hugos_toolkit 的可视化
from hugos_toolkit.VectorbtStylePlotting.plotting import (
    plot_cumulative, plot_underwater, plot_drawdowns,
    plot_annual_returns, plot_monthly_heatmap
)
```

## 关键注意事项

1. **前视偏差**：信号计算只能使用当前及历史数据，position 需 shift(1) 后再乘以收益
2. **交易成本**：择时策略换手率通常较高，需考虑手续费和滑点
3. **过拟合风险**：参数越多越容易过拟合，优先使用少参数策略
4. **市场状态**：策略在趋势市/震荡市表现可能差异大，需分别检验
5. **信号延迟**：注意信号计算是否引入了不必要的延迟
6. **开盘/收盘价**：明确使用开盘价还是收盘价执行交易

## SignalMaker 快速参考

```python
# QRS 择时
from SignalMaker.qrs import QRSCreator

# HHT 择时
from SignalMaker.hht_signal import get_ht_signal, get_hht_signal, parallel_apply

# 日内动量/噪音区域
from SignalMaker.noise_area import NoiseArea

# 鳄鱼线 + AO + Fractal + MACD
from SignalMaker.alligator_indicator_timing import (
    get_alligator_signal, get_ao_indicator_signal,
    get_fractal_signal, get_macd_signal, evaluate_signals
)

# VMACD 动量
from SignalMaker.vmacd_mtm import calc_vmacd_mtm
```
