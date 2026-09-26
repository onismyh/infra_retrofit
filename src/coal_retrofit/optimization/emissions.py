from __future__ import annotations


def captured_fraction(
    pathway: str,
    capture_rate: float,
    biomass_blend: float = 0.0,
    ammonia_blend: float = 0.0,
) -> float:
    """每单位电厂基线排放中，送去运输 / 封存的物理 CO2。"""
    pathway_name = pathway.strip().lower()
    if pathway_name in {"ccs", "beccs"}:
        return float(capture_rate)
    return 0.0


def reduction_fraction(
    pathway: str,
    capture_rate: float,
    biomass_blend: float = 0.0,
    ammonia_blend: float = 0.0,
) -> float:
    """每单位电厂基线排放的净减排量。

    BECCS 把物理捕集与净核算分开：
    捕集的 CO2 为 eta * E，而净减排还包括生物质替代和被捕集的生物源部分，
    在模型的等碳强度假设下合计为 eta + beta。
    """
    pathway_name = pathway.strip().lower()
    if pathway_name == "unabated":
        return 0.0
    if pathway_name == "retire":
        return 1.0
    if pathway_name == "ccs":
        return float(capture_rate)
    if pathway_name == "biomass":
        return float(biomass_blend)
    if pathway_name == "beccs":
        return float(capture_rate) + float(biomass_blend)
    if pathway_name == "ammonia":
        return float(ammonia_blend)
    raise KeyError(pathway)


def residual_fraction(
    pathway: str,
    capture_rate: float,
    biomass_blend: float = 0.0,
    ammonia_blend: float = 0.0,
) -> float:
    """每单位电厂基线排放的净残余排放。"""
    return 1.0 - reduction_fraction(
        pathway,
        capture_rate,
        biomass_blend=biomass_blend,
        ammonia_blend=ammonia_blend,
    )


# 档位下标与整数之差的容许值。独热档位的 select 是二元变量，各有整数容差（IntFeasTol 缺省 1e-5，
# 本仓库未改），Σ l·select 与整数之差至多约 5 × 1e-5（最高档 5 乘整数容差）；连续 hub 的混合一般远大于它。
_LEVEL_INT_TOL = 1e-3


def blend_level_to_ratio(level: object, levels: tuple[float, ...]) -> float:
    """把整数掺烧档位下标（0 = 不掺，l = 第 l 档）换算为物理掺烧比例。

    只对独热档位的结果成立（v9 结果、`hub_decisions_continuous=False`）。连续 hub 下
    `blend_level = Σ l·select` 是档位下标的加权和：非整数时对应不到任何一档，恰为整数时也可能是
    几档的混合，都换算不出比例；有效比例看 `plant_detail` 的 `*_blend_ratio` 列。
    下标不是整数或越界时报错。此前把它原样当比例返回（2.5 → 250%）。
    """
    if level is None:
        return 0.0
    value = float(level)
    level_idx = int(round(value))
    if abs(value - level_idx) > _LEVEL_INT_TOL or not 0 <= level_idx <= len(levels):
        raise ValueError(
            f"掺烧档位下标 {value!r} 不是 0..{len(levels)} 的整数，换算不出掺烧比例："
            "连续 hub 的结果请读 plant_detail 的 *_blend_ratio 列，没有这几列的连续 hub 结果需重解"
        )
    return 0.0 if level_idx == 0 else float(levels[level_idx - 1])
