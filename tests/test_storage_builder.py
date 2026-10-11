"""汇表构建的入参处理：海上盆地膨胀像素数在读 Fan 栅格之前就查（2026-10-11 起缺省 6），0 为不合并、不进盆地合并。"""
from __future__ import annotations

import pandas as pd
import pytest

from coal_retrofit.builders import storage
from coal_retrofit.builders.storage import build_storage_hub_dataframe
from coal_retrofit.paths import ProjectPaths


def test_negative_dilation_is_rejected_before_any_raster_is_read(tmp_path) -> None:
    # 临时目录里没有栅格：报的是入参错误而不是打不开文件，说明校验在读栅格之前
    with pytest.raises(ValueError, match="须 >= 0"):
        build_storage_hub_dataframe(ProjectPaths(tmp_path), offshore_basin_dilation_px=-1)


def _toy_bodies(storage_type: str) -> pd.DataFrame:
    dsa = storage_type == "dsa"
    return pd.DataFrame({
        "storage_type": [storage_type],
        "storage_dsa_mt": [100.0 if dsa else 0.0],
        "storage_eor_mt": [0.0 if dsa else 10.0],
        "storage_all_mt": [100.0 if dsa else 10.0],
        "injectivity_dsa_avg_mtpa": [1.0 if dsa else 0.0],
        "injectivity_eor_avg_mtpa": [0.0 if dsa else 0.1],
        "pixel_count": [4],
        "longitude": [120.0],
        "latitude": [30.0],
    })


def test_zero_dilation_skips_the_offshore_basin_merge(monkeypatch, tmp_path) -> None:
    """0 不得进 `_merge_offshore_by_basin`：scipy 的 binary_dilation 对 iterations <= 0 是"膨胀到不再变化"，会并成一个盆地。"""
    calls: list[int] = []

    def fake_merge(paths: ProjectPaths, hubs: pd.DataFrame, storage_path: object, dilation_px: int) -> pd.DataFrame:
        calls.append(dilation_px)
        return hubs

    monkeypatch.setattr(storage, "_extract_storage_nodes",
                        lambda storage_path, injectivity_path, storage_type: _toy_bodies(storage_type))
    monkeypatch.setattr(storage, "_merge_offshore_by_basin", fake_merge)
    monkeypatch.setattr(storage, "_flag_offshore", lambda paths, hubs: pd.Series(False, index=hubs.index))
    paths = ProjectPaths(tmp_path)
    kept = build_storage_hub_dataframe(paths, offshore_basin_dilation_px=0)
    assert calls == []
    assert kept["storage_hub_id"].tolist() == ["S001", "S002"]   # dsa、eor 各一个封存体，原样保留
    build_storage_hub_dataframe(paths, offshore_basin_dilation_px=6)
    assert calls == [6]
