"""候选管网构建：Delaunay 三角化失败时回退到 k=3 近邻并记一条 warning；其他异常照常抛出。"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.spatial import QhullError

from coal_retrofit.builders import network as net


def test_triangulation_falls_back_to_knn_on_qhull_error(monkeypatch, caplog) -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])  # 共线点，Qhull 会退化

    def _raise_qhull(*args, **kwargs):
        raise QhullError("QH6154 Qhull precision error")

    monkeypatch.setattr(net, "Delaunay", _raise_qhull)
    with caplog.at_level(logging.WARNING, logger=net.logger.name):
        pairs = net._triangulation_candidate_pairs(points)
    assert pairs == net._knn_candidate_pairs(points, k=3)
    assert any("Delaunay 三角化失败" in record.message for record in caplog.records)


def test_triangulation_other_errors_propagate(monkeypatch) -> None:
    def _raise_runtime(*args, **kwargs):
        raise RuntimeError("not a qhull failure")

    monkeypatch.setattr(net, "Delaunay", _raise_runtime)
    with pytest.raises(RuntimeError):
        net._triangulation_candidate_pairs(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
