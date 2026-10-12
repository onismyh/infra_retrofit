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
MODEL_SEGMENT = 12


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
