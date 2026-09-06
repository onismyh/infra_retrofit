# -*- coding: utf-8 -*-
"""在隔离求解树上重绘全部图，并把图与关键结果 CSV 归档到图幅版本目录。

    python scripts/render_version.py --tree _v9tree --version v9

做四件事：
1. 把版本目录里**已有的** PDF/PNG 挪到 <version>/frozen_<tag>/ ——它们画自另一套输入
   （v9 原图来自已丢失的 103 汇求解树），不能与新求解的图并排放在同一目录里
   （CLAUDE.md §二.6 / §五：不同输入版本不得混用）。只在树里真有新图时才挪。
2. 逐个运行 <tree>/scripts/plot_fig*.py 与 plot_ed*.py（脚本按 __file__ 定位 ROOT，
   所以读到的是求解树自己的 inputs/results），输出与退出码记入 <version>/render.log，
   失败的照样记下来——README 里"哪些图不能画"直接从这里抄。
3. 把 <tree>/results/figures/{main,extended,根目录} 的 PDF/PNG 拷到
   <version>/{figures,extended}/。
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

ROOT = Path(__file__).resolve().parents[1]
# 已被 plot_ed_multihop_matching.py 取代的旧 ED14（年份 × 口径六张管网图）；
# plot_fig3_attribution 是 BASE 单情景的成本归因，不在 v9 图序列里，且要 8 条 annual/wgap_370
# 臂才画得全，不随版本归档。
SKIP = {"plot_ed_source_sink_matching_years.py", "plot_fig3_attribution.py"}


def run_plots(tree: Path, log_path: Path, timeout: int) -> list[tuple[str, int, float]]:
    scripts = sorted(p for p in (tree / "scripts").glob("plot_*.py")
                     if (p.name.startswith("plot_fig") or p.name.startswith("plot_ed"))
                     and p.name not in SKIP)
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
                code, out = -9, f"TIMEOUT after {timeout}s\n{exc.stdout or ''}{exc.stderr or ''}"
            dt = time.time() - t0
            rows.append((script.name, code, dt))
            log.write(f"===== {script.name} exit={code} {dt:.0f}s =====\n{out}\n")
            log.flush()
            print(f"  {'ok ' if code == 0 else 'FAIL'} {script.name} ({dt:.0f}s)")
    return rows


def tree_figures(tree: Path) -> list[tuple[Path, str]]:
    src = tree / "results" / "figures"
    out = []
    for sub, target in (("main", "figures"), ("", "figures"), ("extended", "extended")):
        d = src / sub if sub else src
        if not d.is_dir():
            continue
        out += [(f, target) for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in (".png", ".pdf")]
    return out


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


def copy_figures(figs: list[tuple[Path, str]], dest: Path) -> int:
    for f, target in figs:
        (dest / target).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / target / f.name)
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
    parser.add_argument("--tree", default="_v9tree")
    parser.add_argument("--version", default="v9")
    parser.add_argument("--freeze-tag", default="103sink_lost_tree",
                        help="已有旧图挪到 <version>/frozen_<tag>/；空字符串 = 不挪")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    tree = (ROOT / args.tree).resolve()
    dest = ROOT / "results" / "figures" / args.version
    dest.mkdir(parents=True, exist_ok=True)

    rows = [] if args.skip_plots else run_plots(tree, dest / "render.log", args.timeout)
    figs = tree_figures(tree)
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
