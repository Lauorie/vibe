# Skill: QuantsPlaybook 工具包 API 参考

## 适用场景

在使用任何其他 skill 进行研报复现时，同时参考本 skill 以使用 hugos_toolkit 和 SignalMaker 中的工具函数。

## 一、hugos_toolkit

路径：`QuantsPlaybook/hugos_toolkit/`

### 1.1 BackTestTemplate — 回测引擎

```python
from hugos_toolkit.BackTestTemplate import get_backtesting
```

#### `get_backtesting(data, name, strategy, begin_dt, end_dt, **kwargs)`
主回测入口，基于 Backtrader。

| 参数 | 类型 | 说明 |
|------|------|------|
| `data` | DataFrame | OHLCV 数据 |
| `name` | str | 标的名称 |
| `strategy` | bt.Strategy | 策略类 |
| `begin_dt` | str | 开始日期 |
| `end_dt` | str | 结束日期 |
| `mulit_add_data` | bool | 是否多标的 |
| `slippage_perc` | float | 滑点比例 |
| `commission` | float | 手续费率 |
| `stamp_duty` | float | 印花税率 |
| `show_log` | bool | 是否打印日志 |

返回：`namedtuple(result, cerebro)`

#### SignalStrategy — 信号驱动策略

```python
from hugos_toolkit.BackTestTemplate.bt_strategy import SignalStrategy
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `open_threshold` | 0.301 | 开仓信号阈值 |
| `close_threshold` | -0.301 | 平仓信号阈值 |
| `show_log` | True | 日志开关 |

需要数据中包含 `GSISI` 列作为信号源。使用 `AddSignalData` 加载含信号的数据。

#### AddSignalData — 含信号数据源

```python
from hugos_toolkit.BackTestTemplate.backtest_engine import AddSignalData
```

在标准 OHLCV 基础上增加 `GSISI` 信号线。

---

### 1.2 BackTestReport — 绩效报告

#### strategy_performance — 策略绩效汇总

```python
from hugos_toolkit.BackTestReport.performance import strategy_performance, information_ratio
```

`strategy_performance(returns, mark_benchmark="benchmark", periods="daily")`

返回 DataFrame 包含：年化收益、累计收益、年化波动、Sharpe、最大回撤、Sortino、Calmar、IR、Alpha、Beta。

#### 回撤分析

```python
from hugos_toolkit.BackTestReport.timeseries import gen_drawdown_table, get_top_drawdowns
```

- `gen_drawdown_table(returns, top=10)` → 前N大回撤的详细表格
- `get_top_drawdowns(returns, top=10)` → 前N大回撤的 (peak, valley, recovery)

#### 回测报告生成

```python
from hugos_toolkit.BackTestReport.tear import (
    analysis_rets,      # 收益分析报告
    analysis_trade,     # 交易分析报告
    get_backtest_report,       # 绩效报告
    create_trade_report_table,  # 交易统计表
    get_transactions_frame,     # 交易记录
    get_trade_flag             # 买卖信号标记
)
```

##### `analysis_rets(price, result, benchmark_rets, use_widgets=False)`
生成完整收益分析报告，包含：
- 风险收益指标表
- 累计收益曲线
- 回撤分析
- 水下曲线
- 年度收益
- 月度热力图
- 月度分布

##### `analysis_trade(price, result, use_widgets=False)`
生成交易分析报告，包含：
- 交易统计表（总交易数、胜率、盈亏比等）
- PnL 散点图
- 订单标记图
- 持仓区间图

---

### 1.3 VectorbtStylePlotting — Plotly 可视化

```python
from hugos_toolkit.VectorbtStylePlotting.plotting import (
    plot_cumulative,      # 累计收益曲线
    plot_underwater,      # 水下曲线
    plot_drawdowns,       # 回撤标记
    plot_annual_returns,  # 年度收益柱状图
    plot_monthly_heatmap, # 月度收益热力图
    plot_monthly_dist,    # 月度收益分布
    plot_orders,          # 订单标记
    plot_position,        # 持仓区间
    plot_pnl,             # PnL散点
    plot_against,         # 双线对比(红绿填充)
    plot_table            # DataFrame 渲染为表格
)
```

#### 常用调用示例

```python
# 累计收益对比
fig = plot_cumulative(
    strategy_rets,
    benchmark_rets,
    start_value=1,
    fill_to_benchmark=True,
    use_widgets=False
)

# 水下曲线
fig = plot_underwater(strategy_rets, use_widgets=False)

# 最大回撤标记
fig = plot_drawdowns(strategy_rets, top_n=5, start_value=1, use_widgets=False)

# 月度热力图
fig = plot_monthly_heatmap(strategy_rets, use_widgets=False)
```

---

### 1.4 utils — 通用工具

```python
from hugos_toolkit.utils import sliding_window
```

`sliding_window(arr, window, step=1)` — 滑动窗口生成器

---

## 二、SignalMaker

路径：`QuantsPlaybook/SignalMaker/`

### 2.1 QRS 择时信号

```python
from SignalMaker.qrs import QRSCreator
```

#### QRSCreator(low_df, high_df)

| 方法 | 说明 |
|------|------|
| `fit(regression_window=18, zscore_window=600, n=2, adjust_regulation=False, use_simple_beta=False)` | 生成 QRS 信号 DataFrame |
| `calc_simple_signal(regression_window, zscore_window)` | 简单 β 信号 |
| `calc_zscore_beta(regression_window, zscore_window)` | Z-score β 信号 |
| `calc_regulation(regression_window, n)` | R^n 修正系数 |

**输入**：low_df, high_df 为 DataFrame（index=date, columns=assets）
**输出**：QRS 信号 DataFrame

---

### 2.2 HHT 择时信号

```python
from SignalMaker.hht_signal import (
    get_ht_signal,          # HT 信号
    get_hht_signal,         # HHT 信号（EMD+HT）
    parallel_apply,         # 并行 HHT 计算
    decompose_signal,       # EMD/VMD 分解
    calculate_instantaneous_phase,  # 瞬时相位
    get_ht_binary_signal    # HT 二值信号
)
```

#### 常用函数

| 函数 | 参数 | 说明 |
|------|------|------|
| `get_ht_signal(data, ma_period=60, ht_period=30)` | close Series | MA平滑→差分→HT |
| `get_hht_signal(data, hht_period=60, imf_index=2, max_imf=9, method='EMD')` | close Series | EMD→选IMF→HT |
| `parallel_apply(close, window, imf_index, max_imf, method, n_jobs)` | close Series | 滚动并行HHT |

---

### 2.3 NoiseArea 日内动量

```python
from SignalMaker.noise_area import NoiseArea
```

#### NoiseArea(ohlcv)

| 方法 | 说明 |
|------|------|
| `fit(window=14)` | 返回 DataFrame(ubound, signal, lbound) |
| `calculate_intraday_vwap()` | VWAP |
| `calculate_sigma(window=14)` | 滚动波动率 |
| `calculate_bound(window=14, method='U'/'L')` | 上/下界 |

**输入**：OHLCV DataFrame，需包含 `code`, `trade_time`, `open`, `close`, `volume`

---

### 2.4 鳄鱼线及相关指标

```python
from SignalMaker.alligator_indicator_timing import (
    get_alligator_signal,       # 鳄鱼线信号
    get_ao_indicator_signal,    # AO指标信号
    get_fractal_signal,         # 分形信号
    get_macd_signal,            # MACD信号
    evaluate_signals,           # 综合评估
    get_north_money_signal,     # 北向资金信号
    calculate_alligator_indicator,  # 鳄鱼线计算
    calculate_ao                # AO计算
)
```

#### 常用函数

| 函数 | 输入 | 输出 |
|------|------|------|
| `get_alligator_signal(close_df, periods=(13,8,5), lag=(8,5,3))` | close DataFrame | 1=多, -1=空, NaN=中性 |
| `get_ao_indicator_signal(high_df, low_df, window=3)` | high/low DataFrame | AO信号 |
| `get_fractal_signal(close_df, high_df, low_df)` | 三个 DataFrame | 分形信号 |
| `get_macd_signal(close_df)` | close DataFrame | MACD信号 |
| `evaluate_signals(row)` | 信号 Series | 综合信号 |

---

### 2.5 VMACD-MTM

```python
from SignalMaker.vmacd_mtm import calc_vmacd_mtm
```

`calc_vmacd_mtm(volume, period=60)` — 成交量 MACD 动量

**输入**：volume Series 或 DataFrame
**输出**：VMACD-MTM 信号

---

### 2.6 通用工具

```python
from SignalMaker.utils import sliding_window
```

`sliding_window(arr, window, step=1)` — 内存高效的滑动窗口生成器

---

## 三、常用第三方库速查

| 库 | 用途 | 关键函数/类 |
|----|------|------------|
| `alphalens` | 因子分析 | `get_clean_factor_and_forward_returns`, `create_full_tear_sheet` |
| `empyrical` | 绩效指标 | `annual_return`, `sharpe_ratio`, `max_drawdown`, `calmar_ratio`, `sortino_ratio` |
| `backtrader` | 回测框架 | `bt.Cerebro`, `bt.Strategy`, `bt.feeds.PandasData` |
| `vectorbt` | 向量化回测 | `vbt.Portfolio.from_signals()` |
| `qlib` | 量化平台 | `qlib.init()`, DataHandlerLP, Backtest workflow |
| `talib` | 技术指标 | `SMA`, `EMA`, `MACD`, `RSI`, `BBANDS` 等 |
| `statsmodels` | 统计模型 | `OLS`, `WLS`, `add_constant` |
| `scipy.optimize` | 优化 | `minimize`, `differential_evolution` |
| `scipy.stats` | 统计分布 | `pearsonr`, `spearmanr`, `norm`, `t` |
| `PyEMD` | 经验模态分解 | `EMD`, `EEMD`, `CEEMDAN` |
| `pywt` | 小波变换 | `wavedec`, `waverec` |
| `sklearn` | 机器学习 | `RandomForestClassifier`, `cross_val_score` |
| `lightgbm` | 梯度提升 | `LGBMClassifier`, `LGBMRegressor` |
| `torch` | 深度学习 | `nn.Module`, `optim.Adam` |
| `plotly` | 交互可视化 | `go.Figure`, `go.Scatter` |
| `numba` | JIT 加速 | `@jit(nopython=True)` |

## 四、标准 matplotlib 中文配置

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')  # 或 'ggplot'
```

## 五、数据源配置

### JQData
```python
import jqdatasdk as jq
jq.auth('username', 'password')
```

### Tushare
```python
import tushare as ts
ts.set_token('your_token')
pro = ts.pro_api()
```

### Qlib
```python
import qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
```

## 六、sys.path 配置

QuantsPlaybook 中的 notebook 通常需要将父目录加入 sys.path 以导入本地模块：

```python
import sys
sys.path.insert(0, '/workspace/QuantsPlaybook')
# 这样可以 import hugos_toolkit 和 SignalMaker
```
