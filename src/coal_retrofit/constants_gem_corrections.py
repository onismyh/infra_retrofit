"""GEM 机组表（`GEM_with_Cooling_Technology_July2025.xlsx`）的逐台改正：有公开来源、与 GEM 字段冲突的值（2026-10-16 起）。

`builders.plants.apply_unit_corrections` 在读入 GEM 后按机组号改，改前核对原值（GEM 更新后原值变了就报错，免得悄悄改错）。
只收来源点名本厂、且与机组对得上的：A = 原文明确点名本厂本机组；B = 原文点名本厂，机组对应靠容量与年份推断。来源都只取得
检索摘要（本环境打不开原页），引句是摘要转述，采用前作者可再核原页。查过而没有改的疑点（盘山 1、2 号的机型、神头第一
发电厂的 600 / 660 MW、元宝山的机组年份、三峡南疆的分台投产年）没有可用来源，不在表里，见 README §0 的 2026-10-16 条。
冷却方式只分湿冷循环、直流、空冷三类（`constants.cooling_class`），直接空冷与间接空冷都记 `Air`。
"""
from __future__ import annotations

from typing import Final, NamedTuple


class UnitCorrection(NamedTuple):
    unit_id: str
    field: str        # plants_unit.csv 的列名
    gem_value: str    # GEM 原值（按字符串比）
    value: str        # 改后的值
    grade: str        # A / B
    source: str       # 来源 URL 与摘要里的原句大意


_WUAN = ("A", "https://www.power-eng.com/coal/plant-decommissioning/coal-fired-power-plant-cooling-tower-awarded-to-spx-corp/："
         "SPX 2011 年为 2×300 MW 的大唐武安电厂（河北）供间接空冷塔（indirect dry cooling tower），预计 2012 年初投运")
_DATONG = ("A", "https://finance.sina.com.cn/roll/20050428/094540486.shtml：大同二电厂二期工程设计安装两台 60 万千瓦直接空冷"
           "机组，我国首台 60 万千瓦直接空冷机组 2005-04-21 移交生产")
_SHANGDU = ("B", "https://www.industrialinfo.com/news/article/huaneng-shangdu-coal-fired-power-plant-begins-operations--119186、"
            "…/china-huaneng-group-begins-commercial-operation-of-two-thermal-power-projects--124019：上都一期为两台 600 MW "
            "亚临界空冷机组（two 600-MW subcritical air-cooling units），2006 年投产")
_JIMSAR = ("B", "https://cs.com.cn/ssgs/gsxw/201805/t20180529_5815176.html：双良节能 2018-05-28 签《大唐吉木萨尔热电联产一期"
           "（2×350MW）工程间接空冷系统设备采购合同》")
_LUNTAI = ("A", "https://www.industrialinfo.com/news/article/jepc1-wins-construction-bid-for-huaneng-luntai-cogeneration-project--239421："
           "一期建两台 350 MW 超临界燃煤间接空冷热电机组（two 350-MW supercritical, coal-fired, indirect, air-cooling "
           "cogeneration units）")
_AKSU = ("B", "https://www.industrialinfo.com/news/article/zheneng-powers-aksu-cogeneration-project-receives-approval--241430："
         "两台 350 MW 超临界空冷抽凝热电机组（two 350-MW, supercritical air-cooling extraction and condensing units）")
_SANXIA = ("B", "https://finance.eastmoney.com/a/202605163739588824.html（人民财讯 2026-05-16）：三峡新疆若羌新能源基地煤电项目"
           "（6 台 660 MW 超超临界）1 号间冷塔 X 柱合拢；塔有编号，按全厂间接空冷计（只点名 1 号塔，其余 5 台是推断）")
_BOZHOU = ("B", "https://www.gem.wiki/Guoshen_Bozhou_power_station：项目在博尔塔拉州博乐市（页面把省写成内蒙古，是页面错误）；"
           "表中坐标 44.87N 82.06E 也在博乐")
_ZHONGWEI = ("B", "https://www.enerdata.net/publications/daily-energy-news/chn-energy-commissions-first-660-mw-unit-26-gw-coal-fired-"
             "project-china.html：国能中卫 4×660 MW 首台于 2026 年 3 月前后投运（2026-03-02 新闻稿）；"
             "https://www.stdaily.com/web/gdxw/2026-10/02/content_591358.html：6 号机 2026-10-02 通过 168 小时，4 台全部投产")


def _cooling(unit_ids: tuple[str, ...], gem_value: str, evidence: tuple[str, str]) -> tuple[UnitCorrection, ...]:
    return tuple(UnitCorrection(unit, "cooling_technology", gem_value, "Air", *evidence) for unit in unit_ids)


GEM_UNIT_CORRECTIONS: Final[tuple[UnitCorrection, ...]] = (
    *_cooling(("G100000102200", "G100000102201"), "once-through", _WUAN),       # 大唐武安 2×300 MW，2012
    *_cooling(("G100000102249", "G100000102250"), "recirculating", _DATONG),    # 大同二电厂二期 2×600 MW，2005
    *_cooling(("G100000104513", "G100000104514"), "recirculating", _SHANGDU),   # 上都一期 2×600 MW，2006
    *_cooling(("G100000102091", "G100000102092"), "recirculating", _JIMSAR),    # 大唐吉木萨尔 2×350 MW，2026
    *_cooling(("G100000104442", "G100000104443"), "recirculating", _LUNTAI),    # 华能轮台 2×350 MW，2016
    *_cooling(("G100000100098", "G100000100099"), "recirculating", _AKSU),      # 浙能阿克苏 2×350 MW，2016 / 2017
    *_cooling(
        ("G100001027684", "G100001027686", "G100001027687", "G100001027688", "G100001027689", "G100001027690"),
        "recirculating", _SANXIA,
    ),                                                                          # 三峡南疆 6×660 MW，2027
    UnitCorrection("G100001000209", "province", "Inner Mongolia", "Xinjiang", *_BOZHOU),   # 国神博州 2×660 MW，2026
    UnitCorrection("G100001000210", "province", "Inner Mongolia", "Xinjiang", *_BOZHOU),
    UnitCorrection("G100000115553", "commission_year", "2025", "2026", *_ZHONGWEI),        # 国能中卫扩建 660 MW
)
