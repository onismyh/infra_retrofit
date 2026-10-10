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
MODEL_SEGMENT = 9


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
