from __future__ import annotations


def captured_fraction(
    pathway: str,
    capture_rate: float,
    biomass_blend: float = 0.0,
    ammonia_blend: float = 0.0,
) -> float:
    """Physical CO2 sent to transport/storage per baseline plant emission."""
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
    """Net emission reduction per baseline plant emission.

    BECCS separates physical capture from net accounting:
    captured CO2 is eta * E, while net reduction includes biomass substitution
    and the captured biogenic fraction, giving eta + beta under the model's
    equal-carbon-intensity assumption.
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
    """Net residual emissions per baseline plant emission."""
    return 1.0 - reduction_fraction(
        pathway,
        capture_rate,
        biomass_blend=biomass_blend,
        ammonia_blend=ammonia_blend,
    )


def blend_level_to_ratio(level: object, levels: tuple[float, ...]) -> float:
    """Convert a solver blend-level index into a physical blend ratio."""
    if level is None:
        return 0.0
    value = float(level)
    if value <= 0.0:
        return 0.0
    level_idx = int(round(value))
    if abs(value - level_idx) <= 1e-6 and 1 <= level_idx <= len(levels):
        return float(levels[level_idx - 1])
    return value
