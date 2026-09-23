"""输入摘要按本次实际读取的目录计算（此前按源码位置推仓库根，求解树的运行会记错）。"""
from __future__ import annotations

from coal_retrofit.optimization.solver_provenance import _DIGEST_FILES, _input_digest


def _write_inputs(root, pipeline_nodes: str):
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "pipeline_nodes.csv").write_text(pipeline_nodes, encoding="utf-8")
    (inputs / "plants.csv").write_text("plant_id\nP1\n", encoding="utf-8")
    return inputs


def test_digest_follows_the_inputs_that_were_read(tmp_path) -> None:
    a = _input_digest(_write_inputs(tmp_path / "tree_a", "node_id\nN1\n"))
    b = _input_digest(_write_inputs(tmp_path / "tree_b", "node_id\nN1\nN2\n"))
    assert a["digest_plants"] == b["digest_plants"]
    # 两棵树只差管网：摘要必须能区分，这正是仓库根 v7 与 _indtree 的差别所在。
    assert a["digest_pipeline_nodes"] != b["digest_pipeline_nodes"]
    # 缺的表记 None，不报错。
    assert a["digest_storage_hubs"] is None
    assert a["input_dir"].endswith("tree_a/inputs")
    assert {f"digest_{key}" for key, _ in _DIGEST_FILES} <= set(a)
