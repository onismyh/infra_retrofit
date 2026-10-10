# 已求解结果（作者本机求解后拷入）

求解在作者本机做（`python -m coal_retrofit run <情景>`），结果写在本机的 `_indtree/results/`（不入库）。要在库里出图、对账，
把一次求解的两样东西原样拷到这里再提交：

```
_indtree/results/<结果名>.json   ->  results/solved/<结果名>.json
_indtree/results/<结果名>/       ->  results/solved/<结果名>/
```

不要拷 `.sol`、求解日志以外的临时文件。`<结果名>.json` 的 `resolved.model_segment` 是求解时代码的模型分段号
（`src/coal_retrofit/segment.py`）：与当前代码不同的结果，出图脚本（`scripts/plot_style.read_result_json`）与
`scripts/check_run_provenance.py --pair --results results/solved` 都会拒绝（CLAUDE.md 二.7：段与段之间不得相减）。
换了模型分段之后，这里的旧结果要么删掉，要么重解覆盖。

出图脚本缺省读这里：`python scripts/plot_fig3_power_pathways.py --scenario <结果名>`；`--scenario` 给路径时读那里，
例如本机直接读求解树 `--scenario _indtree/results/<结果名>`。
