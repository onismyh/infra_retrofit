"""封存汇与候选管网的输入变体（2026-10-16 起：作者要求 300 km 的两种距离聚类都建、都算）。

汇表与管网是配套的三张表：`storage_hubs`、`pipeline_nodes`、`pipeline_candidate_edges`（管网节点里的汇 id 就是汇表的 id）。
缺省变体（空串）读不带后缀的三张表，即栅格连通域加海上盆地合并的 89 个汇与在它上面建的 v9.2 管网；变体 `<阈值>km_<linkage>`
读同名加后缀的三张表（如 `storage_hubs_300km_complete.csv`）：

    python scripts/build_storage_inputs.py --cluster-km 300 --linkage complete     # 写 storage_hubs_300km_complete.csv
    python scripts/build_network_inputs.py --sinks 300km_complete                   # 读它，写同后缀的两张管网表

情景字段 `OptimizationScenario.sink_variant` 选变体（`scenarios.CHOICES` 只认 `SINK_VARIANTS`），命令行 `run --sinks`。
换了汇表而没在它上面重建管网，`optimization.network.build_runtime_network` 会因汇不在管网节点里报错。
不同变体是不同的输入版本（CLAUDE.md 二.6），结果不得相减（`scripts/check_run_provenance.py --pair` 记 failure）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Final

# 模型认的变体：空串为缺省（连通域），其余为 `cluster_variant` 拼出的距离聚类。加变体就在这里加一项。
SINK_VARIANTS: Final[tuple[str, ...]] = ("", "300km_complete", "300km_average")
# 随汇变体换文件的输入表（不含 .csv）
SINK_INPUT_STEMS: Final[tuple[str, ...]] = ("storage_hubs", "pipeline_nodes", "pipeline_candidate_edges")


def cluster_variant(cluster_km: float, linkage: str) -> str:
    """按距离聚类建出的汇表的变体名 `<阈值>km_<linkage>`，阈值按 `:g` 写（300.0 写成 300）。"""
    return f"{cluster_km:g}km_{linkage}"


def check_sink_variant(variant: str) -> str:
    """*variant* 须在 `SINK_VARIANTS` 里，否则报错；原样返回。"""
    if variant not in SINK_VARIANTS:
        known = "、".join(repr(item) for item in SINK_VARIANTS)
        raise ValueError(f"汇变体只能是 {known}，得到 {variant!r}（新变体先加进 sink_inputs.SINK_VARIANTS）")
    return variant


def sink_input_path(inputs_dir: Path, stem: str, variant: str = "") -> Path:
    """汇变体 *variant* 的输入表 *stem*（`SINK_INPUT_STEMS` 之一）：缺省 `<stem>.csv`，否则 `<stem>_<variant>.csv`。"""
    if stem not in SINK_INPUT_STEMS:
        raise ValueError(f"{stem!r} 不是随汇变体换文件的输入表 {SINK_INPUT_STEMS}")
    check_sink_variant(variant)
    return Path(inputs_dir) / (f"{stem}_{variant}.csv" if variant else f"{stem}.csv")


def require_sink_inputs(inputs_dir: Path, variant: str = "") -> None:
    """汇变体 *variant* 的三张表都在，否则报 FileNotFoundError 并给建表命令（汇表在栅格上建，管网要在本机读油气干线走廊建）。"""
    missing = [path.name for stem in SINK_INPUT_STEMS if not (path := sink_input_path(inputs_dir, stem, variant)).exists()]
    if not missing:
        return
    if variant:
        km, _, linkage = variant.partition("km_")
        how = (f"先运行 scripts/build_storage_inputs.py --cluster-km {km} --linkage {linkage}，"
               f"再运行 scripts/build_network_inputs.py --sinks {variant}")
    else:
        how = "先运行 scripts/build_storage_inputs.py 与 scripts/build_network_inputs.py"
    raise FileNotFoundError(f"{Path(inputs_dir)} 里没有汇变体 {variant or '（缺省）'} 的 {'、'.join(missing)}：{how}")
