"""`scripts/validate_basin_runoff.py` 的成员键与同列保护：导入脚本模块本身，私有名字改名会在这里先被抓到。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    alias = "validate_basin_runoff_under_test"
    spec = importlib.util.spec_from_file_location(alias, SCRIPTS / "validate_basin_runoff.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, alias, module)
    spec.loader.exec_module(module)
    return module


def test_member_tag_keeps_the_gcm_for_scenario_and_historical_files(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load(monkeypatch)
    future = Path("cwatm_gfdl-esm4_w5e5_ssp126_2015soc-from-histsoc_default_qtot_global_monthly_2015_2100.nc")
    hist = Path("watergap2-2e_ukesm1-0-ll_w5e5_historical_histsoc_default_qtot_global_monthly_1850_2014.nc")
    assert script._member_tag(future) == "cwatm|gfdl-esm4|126"
    assert script._member_tag(hist) == "wgap|ukesm1-0-ll|historical"


def test_frames_by_member_refuses_two_files_on_one_column_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _load(monkeypatch)
    monkeypatch.setattr(script, "basin_runoff", lambda path, window: pytest.fail("must not read files when keys collide"))
    same_member = [Path("a") / "cwatm_gfdl-esm4_w5e5_ssp126_x.nc", Path("b") / "cwatm_gfdl-esm4_w5e5_ssp126_y.nc"]
    with pytest.raises(ValueError, match="collided") as info:
        script._frames_by_member(same_member, (2015, 2024))
    assert "x.nc" in str(info.value) and "y.nc" in str(info.value)
    # 键不同就逐文件读，列按键命名
    monkeypatch.setattr(script, "basin_runoff", lambda path, window: path.name)
    frames = script._frames_by_member([same_member[0], Path("cwatm_mri-esm2-0_w5e5_ssp126_z.nc")], (2015, 2024))
    assert frames == {"cwatm|gfdl-esm4|126": "cwatm_gfdl-esm4_w5e5_ssp126_x.nc", "cwatm|mri-esm2-0|126": "cwatm_mri-esm2-0_w5e5_ssp126_z.nc"}
