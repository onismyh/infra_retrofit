from __future__ import annotations


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
