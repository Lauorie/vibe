# AGENTS.md

## Skills

以下 skills 用于券商金工研报复现。当用户输入一篇量化研报时，首先读取路由器 skill 识别研报类型，然后调用对应的子 skill 执行复现。

| Skill 文件 | 适用场景 |
|-----------|---------|
| `.cursor/skills/quant_research_router.md` | **首先读取** — 当用户输入量化研报时，识别研报类型并路由到对应子 skill |
| `.cursor/skills/skill_fundamental_stock_selection.md` | 研报涉及基本面选股、F-Score、价值投资、财务质量评分 |
| `.cursor/skills/skill_factor_construction.md` | 研报涉及 alpha 因子构建、IC 分析、多因子模型、因子合成 |
| `.cursor/skills/skill_timing_strategy.md` | 研报涉及择时策略、趋势判断、技术分析、市场情绪 |
| `.cursor/skills/skill_portfolio_optimization.md` | 研报涉及组合优化、资产配置、风险平价、深度学习组合 |
| `.cursor/skills/skill_toolkit_reference.md` | **辅助参考** — hugos_toolkit 和 SignalMaker 的 API 文档 |

## Cursor Cloud specific instructions

### 项目概述

本仓库包含 QuantsPlaybook（`/workspace/QuantsPlaybook/`），一个券商金工研报复现的量化研究集合，涵盖 100+ 策略，分为四大类：
- **A-量化基本面**：基本面选股（F-Score、价值投资）
- **B-因子构建类**：因子构建与分析（量价、动量、筹码、网络、行为金融等）
- **C-择时类**：择时策略（RSRS、HHT、鳄鱼线、波动率、情绪等）
- **D-组合优化**：组合优化（深度学习、差分进化）

核心工具包：
- **hugos_toolkit**：回测引擎 (Backtrader)、绩效报告、Plotly 可视化
- **SignalMaker**：择时信号生成器（QRS、HHT、NoiseArea、鳄鱼线、VMACD）

### 研报复现工作流

1. 用户输入研报 → 读取 `quant_research_router.md` 识别类型
2. 路由到对应 skill → 按标准化工作流执行
3. 同时参考 `skill_toolkit_reference.md` 使用工具函数
4. 在 QuantsPlaybook 中查找最接近的参考实现
5. 输出标准化分析报告（IC分析/净值曲线/回撤/年度统计等）

### 使用工具包时的注意事项

- 导入 hugos_toolkit / SignalMaker 前需将 QuantsPlaybook 加入 sys.path：
  ```python
  import sys
  sys.path.insert(0, '/workspace/QuantsPlaybook')
  ```
- QuantsPlaybook 中的 notebook 使用 JQData / Tushare 作为数据源，这些需要付费账号。在无数据源的环境中，可使用本地 CSV 或其他免费数据替代。
- matplotlib 中文显示需配置字体：`plt.rcParams['font.sans-serif'] = ['SimHei']`
