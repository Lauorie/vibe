# Skill: 组合优化研报复现

## 适用场景

当研报涉及组合优化、资产配置、风险平价、均值方差、目标波动率、深度学习组合等主题时使用本 skill。

## QuantsPlaybook 参考实现索引

| 主题 | 路径 | 核心方法 |
|------|------|---------|
| MTL-TSMOM | `D-组合优化/MLT_TSMOM/` | 多任务学习：主任务=权重预测(Softmax)，辅助任务=波动率预测；损失=-Sharpe+λ×vol_corr |
| 差分进化优化 | `D-组合优化/DE算法下的组合优化/` | DE算法优化组合权重；"次优"理论：样本内最优≠样本外最优 |

## 标准化工作流

### Phase 1：数据准备

```python
import pandas as pd
import numpy as np

# 1. 确定资产池
# 常见场景：
# - 大类资产配置：股票/债券/商品/货币 指数或ETF
# - FOF：基金产品
# - 行业轮动：行业指数或行业ETF
# - 个股组合：选股策略输出的候选池

# 2. 获取收益率数据
# 标的池日频/周频/月频收益率
# 参考 MLT_TSMOM: 使用 ETF (518880, 513100, 159915, 510300)
# 参考 DE: 使用 JQData 获取

# 3. 计算波动率特征（如需）
# 参考 MLT_TSMOM/src/data_processor.py:
# - CTC波动率: close-to-close
# - Parkinson波动率: (high-low)^2
# - Garman-Klass波动率: 0.5*(high-low)^2 - (2ln2-1)*(close-open)^2
# - Rogers-Satchell波动率
# - Yang-Zhang波动率
```

### Phase 2：优化目标定义

根据研报方法论选择优化目标：

#### 目标A：最大化 Sharpe

```python
def neg_sharpe(weights, returns_matrix):
    """优化目标：负Sharpe（用于最小化）"""
    portfolio_return = (returns_matrix @ weights).mean() * 252
    portfolio_vol = (returns_matrix @ weights).std() * np.sqrt(252)
    return -portfolio_return / portfolio_vol
```

#### 目标B：最小化波动率

```python
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
```

#### 目标C：风险平价

```python
def risk_parity_objective(weights, cov_matrix):
    """各资产风险贡献相等"""
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    marginal_risk = cov_matrix @ weights
    risk_contrib = weights * marginal_risk / portfolio_vol
    target_risk = portfolio_vol / len(weights)
    return np.sum((risk_contrib - target_risk) ** 2)
```

#### 目标D：目标波动率（Target Volatility Scaling）

```python
# 参考 MLT_TSMOM
def target_vol_scaling(weights, returns, target_vol=0.1):
    """根据预测波动率缩放仓位"""
    realized_vol = returns.rolling(20).std() * np.sqrt(252)
    scaling = target_vol / realized_vol
    return weights * scaling.clip(0, 2)  # 限制杠杆
```

### Phase 3：优化方法选择

#### 方法A：传统优化（scipy）

```python
from scipy.optimize import minimize

n_assets = len(asset_names)
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和=1
]
bounds = [(0, 1)] * n_assets  # 做多约束

result = minimize(
    neg_sharpe,
    x0=np.ones(n_assets) / n_assets,  # 等权初始
    args=(returns_matrix,),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
optimal_weights = result.x
```

#### 方法B：差分进化（DE）

```python
# 参考 D-组合优化/DE算法下的组合优化/py/DE_algorithm.py
from scipy.optimize import differential_evolution

# 或使用 scikit-opt
# from sko.DE import DE

def objective(weights):
    weights = weights / weights.sum()  # 归一化
    ret = (returns @ weights).mean() * 252
    vol = (returns @ weights).std() * np.sqrt(252)
    return -ret / vol  # 负Sharpe

bounds = [(0, 1)] * n_assets
result = differential_evolution(objective, bounds, seed=42, maxiter=1000)
```

#### 方法C：深度学习（PyTorch）

```python
# 参考 D-组合优化/MLT_TSMOM/src/module.py
import torch
import torch.nn as nn

class PortfolioNetwork(nn.Module):
    def __init__(self, n_features, n_assets, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 主任务：权重预测
        self.weight_head = nn.Linear(hidden_dim, n_assets)
        # 辅助任务：波动率预测（可选）
        self.vol_head = nn.Linear(hidden_dim, n_assets)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        weights = torch.softmax(self.weight_head(h), dim=-1)
        vol_pred = torch.relu(self.vol_head(h))
        return weights, vol_pred

# 损失函数：-Sharpe + λ × vol_prediction_loss
def portfolio_loss(weights, returns, vol_pred, vol_true, lam=0.1):
    portfolio_returns = (weights * returns).sum(dim=-1)
    sharpe = portfolio_returns.mean() / portfolio_returns.std()
    vol_loss = nn.functional.mse_loss(vol_pred, vol_true)
    return -sharpe + lam * vol_loss
```

#### 方法D：梯度无关优化

```python
# 参考 gradient_free_optimizers
from gradient_free_optimizers import RandomSearchOptimizer

search_space = {f'w_{i}': np.linspace(0, 1, 100) for i in range(n_assets)}

def objective(para):
    weights = np.array([para[f'w_{i}'] for i in range(n_assets)])
    weights = weights / weights.sum()
    ret = (returns @ weights).mean() * 252
    vol = (returns @ weights).std() * np.sqrt(252)
    return ret / vol  # 正Sharpe（最大化）

opt = RandomSearchOptimizer(search_space)
opt.search(objective, n_iter=1000)
```

### Phase 4：滚动优化与回测

```python
# 大多数组合优化需要滚动窗口
def rolling_optimization(returns, lookback=252, rebalance_freq=21):
    """滚动优化回测"""
    dates = returns.index
    portfolio_returns = []

    for i in range(lookback, len(dates), rebalance_freq):
        # 样本内数据
        train_returns = returns.iloc[i-lookback:i]

        # 优化权重
        weights = optimize(train_returns)  # 选择上述任一方法

        # 样本外收益（持有到下次调仓）
        end = min(i + rebalance_freq, len(dates))
        oos_returns = returns.iloc[i:end]
        port_ret = (oos_returns @ weights)
        portfolio_returns.append(port_ret)

    return pd.concat(portfolio_returns)

# "次优"理论检验
# 参考 DE算法: 比较样本内最优 vs 次优(如等权) 在样本外的表现
```

### Phase 5：绩效评估

```python
from empyrical import (
    annual_return, annual_volatility, sharpe_ratio,
    max_drawdown, calmar_ratio, sortino_ratio
)

# 标准绩效指标
metrics = {
    '年化收益': annual_return(portfolio_returns),
    '年化波动': annual_volatility(portfolio_returns),
    'Sharpe': sharpe_ratio(portfolio_returns),
    '最大回撤': max_drawdown(portfolio_returns),
    'Calmar': calmar_ratio(portfolio_returns),
    'Sortino': sortino_ratio(portfolio_returns),
}

# 与基准比较
# 常用基准：等权组合、60/40组合、买入持有

# 权重分析
# 1. 权重时序变化图
# 2. 换手率统计
# 3. 各资产平均权重
```

### Phase 6：输出报告

```python
# 标准输出：
# 1. 最优权重柱状图 / 权重时序面积图
# 2. 组合净值曲线（vs 等权 vs 其他基准）
# 3. 有效前沿图（均值-方差平面）
# 4. 分年度收益表
# 5. 风险贡献分解图
# 6. 回撤分析
# 7. 滚动Sharpe对比

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用 hugos_toolkit 可视化
from hugos_toolkit.VectorbtStylePlotting.plotting import (
    plot_cumulative, plot_underwater, plot_drawdowns
)
```

## 关键注意事项

1. **估计误差**：协方差矩阵估计不稳定，考虑使用收缩估计 (Ledoit-Wolf)
2. **样本内过拟合**：优化结果在样本外可能大幅衰减，检验"次优"理论
3. **约束条件**：现实中需添加权重上下限、换手率约束、行业暴露约束等
4. **交易成本**：频繁调仓的交易成本可能吞噬优化收益
5. **目标波动率缩放**：需要波动率预测，预测误差会影响仓位
6. **深度学习方法**：需要大量数据，小样本场景优先使用传统方法
7. **回看偏差**：滚动优化中的 lookback 窗口长度本身也是参数

## 波动率估计方法速查

| 方法 | 公式简述 | 数据需求 |
|------|---------|---------|
| CTC | `std(ln(C_t/C_{t-1}))` | 收盘价 |
| Parkinson | `(H-L)²/(4ln2)` | 最高最低价 |
| Garman-Klass | `0.5(H-L)² - (2ln2-1)(C-O)²` | OHLC |
| Rogers-Satchell | `(H-C)(H-O) + (L-C)(L-O)` | OHLC |
| Yang-Zhang | `σ²_overnight + k×σ²_CTC + (1-k)×σ²_RS` | OHLC |
