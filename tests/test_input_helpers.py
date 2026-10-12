"""`input_tables.read_input_csv` 与 `spatial.within_radius`：读表检查的报错文字，半径配对的下标、距离与排名顺序。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from coal_retrofit.optimization.input_tables import read_input_csv
from coal_retrofit.spatial import haversine_km, within_radius


def test_read_input_csv_messages_and_passthrough(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    with pytest.raises(FileNotFoundError) as missing_file:
        read_input_csv(path, ("a", "b"), not_found="; run scripts/build_x.py.")
    assert str(missing_file.value) == f"{path} not found; run scripts/build_x.py."
    path.write_text("a,c\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError) as missing_column:
        read_input_csv(path, ("b", "a"), not_found="")
    assert str(missing_column.value) == f"{path} lacks required columns ['b']"
    table = read_input_csv(path, ("a",), not_found="")
    assert list(table.columns) == ["a", "c"] and table["c"].tolist() == [2]  # 不裁列


def test_within_radius_indices_distances_and_ranking() -> None:
    lons = np.array([110.5, 110.0, 110.2, 108.0, 110.0])  # 最后两个点约 193、167 km，圈外
    lats = np.array([30.0, 30.0, 30.0, 30.0, 31.5])
    expected = haversine_km(110.0, 30.0, lons, lats)
    matched, distances = within_radius(110.0, 30.0, lons, lats, 100.0)
    assert matched.tolist() == [0, 1, 2]  # 缺省按下标升序
    assert np.array_equal(distances, expected[[0, 1, 2]])
    ranked, ranked_distances = within_radius(110.0, 30.0, lons, lats, 100.0, ranked=True)
    assert ranked.tolist() == [1, 2, 0]  # 按距离升序：0、约 19、约 48 km
    assert np.array_equal(ranked_distances, expected[[1, 2, 0]])
    empty, empty_distances = within_radius(100.0, 20.0, lons, lats, 1.0)
    assert empty.shape == (0,) and empty_distances.shape == (0,)
