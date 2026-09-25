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


def blend_level_to_ratio(level: object, levels: tuple[float, ...]) -> float:
    """把求解器的掺烧档位下标转换为物理掺烧比例。"""
    if level is None:
        return 0.0
    value = float(level)
    if value <= 0.0:
        return 0.0
    level_idx = int(round(value))
    if abs(value - level_idx) <= 1e-6 and 1 <= level_idx <= len(levels):
        return float(levels[level_idx - 1])
    return value
