# -*- coding: utf-8 -*-
"""在求解树上重绘全部图，并把图与关键结果 CSV 归档到图幅版本目录。

    python scripts/render_version.py --version <版本>

求解树 <tree> 固定是 `_bootstrap.ROOT`（`_indtree`）：出图脚本只读这棵树，所以没有 `--tree`，另指一棵树，
归档的数据就不是画图用的那套。`--version` 必填（原缺省 `v9`，会把 `_indtree` 的图拷进 v9 的归档）。
v9 / v9.1 的图只能在历史里（`a303f05`）重绘，当前版本不再维护复现步骤（README §0）。

做四件事：
1. 给了 `--freeze-tag` 时，把版本目录里**已有的** PDF/PNG 挪到 <version>/frozen_<tag>/ ——它们画自另一套输入
   （v9 原图来自已丢失的 103 汇求解树），不能与新求解的图并排放在同一目录里
   （CLAUDE.md §二.6 / §五：不同输入版本不得混用）。只在树里真有新图时才挪。
2. 逐个运行 scripts/plot_fig*.py（缺省参数：中英两版、主情景；脚本经 `_bootstrap.ROOT` 读求解树的
   inputs/results，工作目录也设为求解树），输出与退出码记入 <version>/render.log，
   失败的照样记下来——README 里"哪些图不能画"直接从这里抄。
3. 把 <tree>/results/figures/ 根目录下第 2 步画出的 PDF/PNG（按修改时间判断；上一轮留下的旧图不拷，
   子目录 main/、extended/ 里旧脚本画的图与 save_fig 没删掉的临时文件 .*.tmp.* 也不拷）拷到 <version>/figures/；
   `--skip-plots` 时不画图，根目录下的图全拷（子目录照旧不拷）。
4. 把 <tree>/results/<scen>/*.csv、<scen>.json、求解日志与输入指纹拷到 <version>/data/，
   图背后的数字不再只存在于会话临时目录（v8/v9 的求解树就是这么丢的）。

不写 README——版本 README 的数字必须由人读过图之后填。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import REPO_ROOT, ROOT


def run_plots(tree: Path, log_path: Path, timeout: int) -> list[tuple[str, int, float]]:
    scripts = sorted((REPO_ROOT / "scripts").glob("plot_fig*.py"))
    rows = []
    with log_path.open("w", encoding="utf-8") as log:
        for script in scripts:
            t0 = time.time()
            try:
                proc = subprocess.run([sys.executable, str(script)], cwd=str(tree),
                                      capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=timeout)
                code, out = proc.returncode, proc.stdout + proc.stderr
            except subprocess.TimeoutExpired as exc:
                # 超时时抓到的输出在 POSIX 上不管 text=True 都是 bytes（Windows 上是 str），统一成 str 再写日志，否则日志里是 b'...'
                partial = "".join(p.decode("utf-8", errors="replace") if isinstance(p, bytes) else (p or "")
                                  for p in (exc.stdout, exc.stderr))
                code, out = -9, f"TIMEOUT after {timeout}s\n{partial}"
            dt = time.time() - t0
            rows.append((script.name, code, dt))
            log.write(f"===== {script.name} exit={code} {dt:.0f}s =====\n{out}\n")
            log.flush()
            print(f"  {'ok ' if code == 0 else 'FAIL'} {script.name} ({dt:.0f}s)")
    return rows


def tree_figures(tree: Path, since: float | None) -> list[Path]:
    """<tree>/results/figures/ 下（不含子目录）的 PDF/PNG；给了 since 就只要这之后写出的（本次运行画的）。
    以 "." 开头的是 save_fig 的临时文件（出图中断时可能留下），不收。"""
    src = tree / "results" / "figures"
    if not src.is_dir():
        return []
    return [f for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".pdf") and not f.name.startswith(".")
            and (since is None or f.stat().st_mtime >= since)]


def freeze_old(dest: Path, tag: str) -> int:
    """把版本目录里已有的图挪进 frozen_<tag>/，保持 figures/extended 子目录结构。"""
    n = 0
    for sub in ("figures", "extended"):
        d = dest / sub
        if not d.is_dir():
            continue
        for f in list(d.iterdir()):
            if f.is_file() and f.suffix.lower() in (".png", ".pdf"):
                target = dest / f"frozen_{tag}" / sub
                target.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(target / f.name))
                n += 1
    return n


def copy_figures(figs: list[Path], dest: Path) -> int:
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    for f in figs:
        shutil.copy2(f, dest / "figures" / f.name)
    return len(figs)


def copy_results(tree: Path, dest: Path) -> list[str]:
    res = tree / "results"
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    scen = []
    for js in sorted(res.glob("*.json")):
        name = js.stem
        folder = res / name
        if not folder.is_dir():
            continue
        out = data / name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(js, data / js.name)
        for f in folder.glob("*.csv"):
            shutil.copy2(f, out / f.name)
        scen.append(name)
    if (res / "campaign.log").exists():
        shutil.copy2(res / "campaign.log", data / "campaign.log")
    for f in res.glob("*.solve.log"):
        shutil.copy2(f, data / f.name)
    # 输入版本的指纹：汇与候选边的行数是 README 第一行要写的数字。
    for name in ("storage_hubs.csv", "pipeline_candidate_edges.csv", "pipeline_nodes.csv"):
        f = tree / "inputs" / name
        if f.exists():
            shutil.copy2(f, data / name)
    return scen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--freeze-tag", default="",
                        help="已有旧图挪到 <version>/frozen_<tag>/；缺省不挪（v9 当年用 103sink_lost_tree）")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    tree = ROOT
    dest = REPO_ROOT / "results" / "figures" / args.version
    dest.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows = [] if args.skip_plots else run_plots(tree, dest / "render.log", args.timeout)
    figs = tree_figures(tree, since=None if args.skip_plots else started)
    n_frozen = 0
    if figs and args.freeze_tag and not (dest / f"frozen_{args.freeze_tag}").exists():
        n_frozen = freeze_old(dest, args.freeze_tag)
    n_fig = copy_figures(figs, dest)
    scen = copy_results(tree, dest)
    print(f"old figures frozen: {n_frozen}; new figures copied: {n_fig}; scenarios archived: {scen}")
    failed = [r for r in rows if r[1] != 0]
    print(f"plot scripts: {len(rows)} run, {len(failed)} failed")
    for name, code, _ in failed:
        print(f"  FAIL {name} exit={code}")


if __name__ == "__main__":
    main()
