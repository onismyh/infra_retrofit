"""分省缺省表：输入表省名的别名、分省煤价与分省煤电利用小时。

`OptimizationAssumptions` 的 `province_coal_cost_cny_per_gj`、`province_operating_hours` 缺省为 `PROVINCE_COAL_COST_CNY_PER_GJ`、
`PROVINCE_OPERATING_HOURS` 的副本（情景可整张替换），结果表的省名归一（`optimization.results_regions`）也用别名表 `PROVINCE_NAME_ALIASES`。
"""
from __future__ import annotations

# 输入表里与分省煤价表（`PROVINCE_COAL_COST_CNY_PER_GJ`）写法不同的省名，
# 读入时（`prepare_industry`、`_prepare_plants`）由 `OptimizationAssumptions.canonical_provinces`
# 换成煤价表的写法。只收有证据的：仓库根 `inputs/industry_hubs.csv`（`a303f05`）的 28 个合成氨、水泥、甲醇 hub
# 把内蒙古写作拼音 "Neimenggu"，2026-09-25（PR #3 合入）前查不到煤价，退回 `coal_fuel_cost_cny_per_gj`。
PROVINCE_NAME_ALIASES: dict[str, str] = {"Neimenggu": "Inner Mongolia"}


# 分省煤价，CNY/GJ = An et al. 2025（Nat Commun）SI Table 2 的 USD/GJ × 7。原表北京无煤价（9.92 $/GJ 是气价，
# 2026-09-23 前误记为 69.4），京津两行气价、生物质价与潜力相同，取天津 5.51（38.6）；内蒙古取东、西两行 2.63 / 2.49 的均值。
PROVINCE_COAL_COST_CNY_PER_GJ: dict[str, float] = {
    "Anhui": 42.8, "Beijing": 38.6, "Chongqing": 42.6,
    "Fujian": 40.0, "Gansu": 37.0, "Guangdong": 41.5,
    "Guangxi": 47.9, "Guizhou": 34.6, "Hainan": 33.6,
    "Hebei": 30.8, "Heilongjiang": 35.6, "Henan": 42.1,
    "Hubei": 46.2, "Hunan": 48.5, "Inner Mongolia": 17.9,
    "Jiangsu": 40.7, "Jiangxi": 48.2, "Jilin": 39.1,
    "Liaoning": 37.5, "Ningxia": 27.4, "Qinghai": 33.9,
    "Shaanxi": 31.0, "Shandong": 41.5, "Shanghai": 38.1,
    "Shanxi": 28.2, "Sichuan": 45.4, "Tianjin": 38.6,
    "Xinjiang": 16.8, "Yunnan": 29.6, "Zhejiang": 39.4,
}


# 分省年运行小时数（计算发电量时替代统一的 capacity_factor）
# 来源：中国电力企业联合会（China Electricity Council）统计，分省煤电平均利用小时数
PROVINCE_OPERATING_HOURS: dict[str, float] = {
    "Inner Mongolia": 5031.5, "Zhejiang": 5535.3, "Xinjiang": 5281.2,
    "Jiangsu": 5015.5, "Fujian": 5126.2, "Anhui": 5082.1,
    "Hainan": 5030.9, "Ningxia": 4970.9, "Shanghai": 4954.8,
    "Sichuan": 4996.4, "Chongqing": 4794.2, "Guangdong": 4831.7,
    "Guangxi": 4786.6, "Jiangxi": 4766.4, "Guizhou": 4747.1,
    "Shanxi": 4629.5, "Shaanxi": 4538.0, "Yunnan": 4482.2,
    "Hebei": 4455.1, "Shandong": 4361.1, "Tianjin": 4347.5,
    "Gansu": 4189.6, "Qinghai": 4097.0, "Hunan": 3906.9,
    "Hubei": 3749.6, "Jilin": 3621.4, "Henan": 3599.6,
    "Heilongjiang": 3478.5, "Liaoning": 3100.7, "Beijing": 779.2,
}
