"""工业点源作为决策主体，与煤电机组联合优化。

在本模块之前，2 552 个工业点源只作为输入存在：`builders/industry.py` 把它们聚类成
390 个分行业 hub，`builders/water_quota.py` 把它们的现状取水从流域预留中扣除，以便煤电
机组争用这份预留。`optimization/` 里从没有代码读过它们。它们的减排是外生的——为零——
而由此形成的流域余量整份给了煤电。

这里联合的是：

1. **CO2 管网。** 工业捕集的 CO2 在 hub 自己的节点进入煤电 hub 所用的同一张管网图，
   争用同样的边容量，封存在同样的汇里，受同样的注入能力上限约束。每个工业 hub 也有
   到其最近几个汇的直连候选弧，规则与煤电 hub 相同。
2. **流域取水上限。** 工业厂址用水与工业捕集的用水惩罚，记在煤电同样从中取水的
   `residual_m3_per_year` 上。
3. **氢供应。** 走氢路线的 hub 从为煤电侧掺烧供氨的同一批绿氨节点取氢，匹配半径相同，
   付各节点自己的出厂 LCOH 加运输费——所以两类用氢方在量上相互竞争，工业付的是边际
   价格，而不是全国供给加权均价。
4. **排放目标。** 每个部门组一条残余排放上限（钢铁、水泥、化工，以及煤电机组所在的
   电力组），读自 `inputs/sector_targets_<source>.csv`。有上限时碳价通常为零；不为零时，
   工业残余排放与煤电残余排放按完全相同的方式计收碳价。

每个 hub 有三条路线——`unabated`、`ccs`、`h2`——其中 `h2` 只在 `SECTOR_HAS_H2_ROUTE`
为真时开放（所以水泥与电炉钢要么捕集、要么不改造，对工艺 CO2 排放源而言这正是正确答案）。
路线份额是 [0, 1] 上的连续量，并且跨规划年必须单调：装了捕集的 hub 不能再拆掉它，
也不能把捕集换成氢。

成本口径（作者 2026-09-22 的决定；与煤电侧一贯采用的口径相同）。每条工业路线都按下式计价：

    capex   = 单位改造 capex x 建成能力，只在路线份额增量上计一次
              （份额单调，所以增量就是新建存量）
    annual  = 固定运维（每年为该 capex 的一个比例）
            + 能耗与耗材，按模型自己的煤价与电价计
            + （氢路线）相对现有工艺的非氢运行差额
            + （氢路线）求解器里按供氢链路购买的氢

并在规划期末把每笔 capex 的未折旧部分抵扣回来（求解器中的 `salvage_credit`）。目标函数
不再含任何平准化每吨成本。2026-09-10 至 2026-09-22 期间，ACCA21 的平准化捕集成本被拆成
按 CRF 反年化的资本部分与年度余项；这样做把平准化数字隐含的资本回收假设保留在了一个
本身就是要决定何时建设的模型里。ACCA21 的区间保留下来，作交叉核对
（`constants_industry.levelised_capture_cost_cny_per_t`）。

CCS 路线（`constants_industry.INDUSTRY_CCS_*`）：按每吨年捕集能力计的 capex 取自中国项目
备案（水泥 1 150、钢铁 1 000、高浓度化工 450 CNY/(t/a)），固定运维 5%/a；再生蒸汽由厂内
燃煤锅炉产生，按 hub 所在省的煤价计价；电按情景电价；耗材 15 或 5 CNY/t。蒸汽的 CO2
直接放空，并从该路线的减排量中扣除（2026-09-22 之前把 ACCA21 单位成本视为已含能耗，
也没有放空任何 CO2）。煤电侧的学习曲线照旧缩放 capex（固定运维随之缩放）。

氢路线（`INDUSTRY_H2_ROUTE_*`）：按每吨年产品产能计的 capex（H2-DRI 竖炉 + 电炉 3 500；
合成氨 / 甲醇的氢接入 500 CNY/(t/a)），固定运维 3.5%/a，以及由文献溢价锚点反推的非氢
运行差额：

    opex_delta = premium_ref - k * P_ref - capex * (CRF(r, life) + fom)

这样锚点在它自己的参考氢价下被精确复现，再通过 hub 的氢强度 k 移到实际支付的氢价上。
钢铁的 `opex_delta` 为负（省下的焦炭与高炉 opex 超过电炉电费）；求解器里的地板
`annual + hydrogen purchase >= 0` 表达的是：换路线不可能比继续运行资本已沉没的现有工艺
更便宜；它仍把省下的化石原料全额计为收益，所以氢路线的采用量仍是上界。

仍然存在的已知偏差：
* 电解用水不计（10-22 L/kg H2）：它属于电解槽所在的流域，煤电侧的掺氨也同样不计。
* 产量按外生指数变化（设置了 `industry_output_index_source` 时为 TIMES CN60）；工业没有
  厂级的退役决策。

文件分工（2026-09-23 从本文件拆出，本文件只留设计说明并转导出，调用方的 import 不用改）：
`industry_inputs.py` 输入准备（hub 表、氢链路）；`industry_matrices.py` 逐年系数
（`IndustryYearData`）；`model_industry.py` 变量、约束与一次性 capex 表达式（`IndustryPayload`）。
"""
from __future__ import annotations

from .industry_inputs import IndustryInputs, prepare_industry, prepare_industry_h2_links
from .industry_matrices import (
    CCS,
    H2,
    ROUTE_INDEX,
    UNABATED,
    IndustryYearData,
    basin_membership,
    industry_year_data,
)
from .model_industry import (
    IndustryPayload,
    add_industry_monotonicity,
    add_industry_year,
    industry_capex_expr,
)

__all__ = [
    "CCS",
    "H2",
    "ROUTE_INDEX",
    "UNABATED",
    "IndustryInputs",
    "IndustryPayload",
    "IndustryYearData",
    "add_industry_monotonicity",
    "add_industry_year",
    "basin_membership",
    "industry_capex_expr",
    "industry_year_data",
    "prepare_industry",
    "prepare_industry_h2_links",
]
