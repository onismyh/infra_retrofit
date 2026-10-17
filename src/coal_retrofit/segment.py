"""模型分段号：结果落盘时写进 result.json 的 `resolved.model_segment`，读结果时与当前代码比对。

CLAUDE.md 二.7 把 `_indtree` 结果按模型口径分段，段与段之间不得相减。凡改变模型口径（变量、约束、目标函数、
参数缺省值、输入表）的 PR 都把 `MODEL_SEGMENT` 加一，并在二.7 记下新段的边界。2026-10-10 之前的结果没有这个字段，
都早于第 9 段，读时一律当作旧结果拒绝。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

# 第 9 段：绿氢共享上限、碳目标两入口、海上管道按海上段计、封存注入上限按 Fan 2025 逐格累加与全国部署上限、
# 煤电成本增量口径与运维取值（掺氨按 capex 比例、基线 110 元/MWh）。
# 第 10 段（2026-10-11）：生物质供给表改用 Wang 2023 情景 4（农林残余物，不含能源作物）。
# 第 11 段（2026-10-11）：绿氢节点松弛罚额按 NH3_H2_RATIO 折到氢（1000 / 0.18 元/kg H2，此前 1000 元/kg H2）。
# 第 12 段（2026-10-12）：水表重建：cwatm 的 ipsl / mpi / mri 偏差因子按实际驱动的历史文件（ISIMIP issue #34），
# ukesm 取其余四个 GCM 的均值；粗化水节点坐标随之微移，所有情景的供水距离与输入摘要都变。
# 第 13 段（2026-10-12）：粗化水节点坐标改按模拟径流（local_runoff）加权（此前按校正后的可用量），水节点与供水链路表重建。
# 第 14 段（2026-10-13）：ISIMIP 网格面积的地球半径改用 `spatial.EARTH_RADIUS_KM`（此前 6 371 km），水表重建（耗水与模拟径流
# 放大 2.76 ppm）；流域余量取整到 m3；煤电发电量直接按利用小时算，不经容量因子往返。
# 第 15 段（2026-10-14）：删全国封存部署上限（各汇只受 Fan 2025 聚类之和的年注入与累计上限）；松弛与缺口变量由
# `allow_slack` 控制、缺省关（硬约束）；成本对账绝对差不到 1 元的类别不计。
# 第 16 段（2026-10-16）：管道改 ChinaCCS 细 / 中 / 粗三档（3.25 / 19.69 / 83.954 Mt/a），根数不设上限，删每边在役新增上限与
# 建成、新增两个二元标志，寿命 30 → 40 a；流域余量把煤电闭式循环与空冷的现状取水也从预留里扣出（此前计两次），水表重建；
# 删掺烧档位单调（煤电可降档）与全国生物质上限，掺烧档位选择变量只在整数 hub 下建，其余删掉的约束是被别的约束蕴含的冗余行；
# 水泥再生蒸汽的燃煤锅炉烟气进捕集岛（净减排 0.708 → 0.876）；GEM 机组表按有来源的改正改 21 行（18 台冷却方式、2 台省份、
# 1 台投产年），plants.csv 重建；直流冷却标定只数公报年及以前投产的机组（因子 0.497 → 0.547）。2026-10-17 复核后工业扣出只计
# 公报年在运的点源、煤电扣出的利用小时按厂址所在省取，water_basin_caps.csv 完整重建（本段尚未求解，不另加段）。
MODEL_SEGMENT = 16


class StaleResultError(RuntimeError):
    """结果的模型分段号与当前代码不同：旧口径的结果不能拿来出图、对账或与新结果相减。"""


def result_segment(result: dict[str, Any]) -> int | None:
    """result.json 记的分段号；没有记（2026-10-10 之前落盘）为 None。"""
    value = (result.get("resolved") or {}).get("model_segment")
    return None if value is None else int(value)


def check_segment(result: dict[str, Any], path: Path | str) -> None:
    """分段号不是 `MODEL_SEGMENT` 就抛 `StaleResultError`，提示重解。"""
    found = result_segment(result)
    if found != MODEL_SEGMENT:
        shown = "未记录（早于第 9 段）" if found is None else f"第 {found} 段"
        raise StaleResultError(
            f"{path}：结果的模型分段号为 {shown}，当前代码为第 {MODEL_SEGMENT} 段（CLAUDE.md 二.7），"
            "旧口径的结果不能用，须用当前代码重解"
        )
