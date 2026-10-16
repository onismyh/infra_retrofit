"""重绘全部图，并把图、画图用的结果与输入指纹归档到图幅版本目录。

    python scripts/render_version.py --version <版本> [--timeout 秒] [--skip-plots] [--force]

出图脚本读求解树 <tree>（固定是 `_bootstrap.ROOT`，即 `_indtree`）的 inputs/，结果图读入库的 results/solved/
（`plot_io.SOLVED_DIR`），图写到 <tree>/results/figures/。所以没有 `--tree`：另指一棵树，归档的就不是画图用的那套。
`--version` 必填（原缺省 `v9`，会把 `_indtree` 的图拷进 v9 的归档）。
v9 / v9.1 的图只能在历史里（`a303f05`）重绘，当前版本不再维护复现步骤（README §0）。

做三件事（2026-10-13 删去给 v9 旧图用的 `--freeze-tag`：版本目录不要与另一套输入画的图混放，CLAUDE.md §五）：
1. 逐个运行 scripts/plot_fig*.py（缺省参数：中英两版、主情景；工作目录设为求解树），输出与退出码记入 <version>/render.log，
   失败的照样记下来——README 里"哪些图不能画"直接从这里抄。
2. 把 <tree>/results/figures/ 根目录下第 1 步画出的 PDF/PNG（按修改时间判断；上一轮留下的旧图不拷，
   子目录 main/、extended/ 里旧脚本画的图与 save_fig 没删掉的临时文件 .*.tmp.* 也不拷）拷到 <version>/figures/；
   `--skip-plots` 时不画图，根目录下的图全拷（子目录照旧不拷）。
3. 把 results/solved/ 下每个结果的 <结果名>.json、<结果名>/ 里的文件（九张表与结果工作簿），
   连同 <tree>/inputs/ 的输入指纹拷到 <version>/data/，图背后的数字不再只存在于会话临时目录（v8/v9 的求解树就是这么丢的）。
   <version>/data/ 被 .gitignore 的 `data/` 一行忽略，只留在本机；入库的是 results/solved/ 本身。
   2026-10-11 之前拷的是 <tree>/results/ 下的结果与表（不含工作簿），与出图读的 results/solved/ 不是同一份。

版本目录下已有 figures/ 或 data/ 且不为空时拒绝运行（2026-10-16 起）：重跑同一版本，这次失败的图与已从 results/solved/ 删掉的结果
会以上一轮的旧文件留在目录里，图与数据不再同源（CLAUDE.md §五）。`--force` 先清掉这两个目录与 render.log 再归档；README.md 不动。

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
from plot_io import SOLVED_DIR  # 结果图读的结果目录（plot_io 不依赖 matplotlib）

from coal_retrofit.sink_inputs import SINK_INPUT_STEMS, SINK_VARIANTS, sink_input_path


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


def copy_figures(figs: list[Path], dest: Path) -> int:
    (dest / "figures").mkdir(parents=True, exist_ok=True)
    for f in figs:
        shutil.copy2(f, dest / "figures" / f.name)
    return len(figs)


def copy_results(tree: Path, solved: Path, dest: Path) -> list[str]:
    """把 *solved* 下有同名目录的 `<结果名>.json` 连同目录里的文件，和 *tree* 的输入指纹拷到 `<dest>/data/`；
    返回拷了的结果名。"""
    data = dest / "data"
    data.mkdir(parents=True, exist_ok=True)
    scen = []
    for js in sorted(solved.glob("*.json")):
        name = js.stem
        folder = solved / name
        if not folder.is_dir():
            continue
        out = data / name
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(js, data / js.name)
        for f in folder.iterdir():
            if f.is_file():
                shutil.copy2(f, out / f.name)
        scen.append(name)
    # 输入版本的指纹：汇与候选边的行数是 README 第一行要写的数字。汇变体（`coal_retrofit.sink_inputs`）的三张表有就一并拷。
    for variant in SINK_VARIANTS:
        for stem in SINK_INPUT_STEMS:
            f = sink_input_path(tree / "inputs", stem, variant)
            if f.exists():
                shutil.copy2(f, data / f.name)
    return scen


ARCHIVED = ("figures", "data", "render.log")


def prepare_dest(dest: Path, force: bool) -> None:
    """建版本目录。里面已有 `ARCHIVED` 的任一项（figures/、data/ 非空，或 render.log）时拒绝，`force` 为真则先删掉它们。

    只动这三项：版本目录的 README.md 是人写的，不碰。
    """
    dest.mkdir(parents=True, exist_ok=True)
    stale = [name for name in ARCHIVED
             if (dest / name).is_file() or ((dest / name).is_dir() and any((dest / name).iterdir()))]
    if not stale:
        return
    if not force:
        raise SystemExit(
            f"{dest} 下已有 {', '.join(stale)}：上一轮归档的图与数据会与这次的混在一起，不再同源（CLAUDE.md §五）。"
            "换一个 --version，或加 --force 先清掉这三项再归档（README.md 不动）"
        )
    for name in stale:
        target = dest / name
        shutil.rmtree(target) if target.is_dir() else target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="图幅版本目录名，写到 results/figures/<version>/")
    parser.add_argument("--timeout", type=int, default=1800, help="每个出图脚本的时限（秒），超时记为失败")
    parser.add_argument("--skip-plots", action="store_true", help="不画图，只归档根目录下已有的图")
    parser.add_argument("--force", action="store_true",
                        help="版本目录下已有 figures/、data/ 或 render.log 时先清掉再归档（缺省拒绝运行）")
    args = parser.parse_args()
    tree = ROOT
    dest = REPO_ROOT / "results" / "figures" / args.version
    prepare_dest(dest, args.force)

    started = time.time()
    rows = [] if args.skip_plots else run_plots(tree, dest / "render.log", args.timeout)
    figs = tree_figures(tree, since=None if args.skip_plots else started)
    n_fig = copy_figures(figs, dest)
    scen = copy_results(tree, SOLVED_DIR, dest)
    print(f"new figures copied: {n_fig}; scenarios archived: {scen}")
    failed = [r for r in rows if r[1] != 0]
    print(f"plot scripts: {len(rows)} run, {len(failed)} failed")
    for name, code, _ in failed:
        print(f"  FAIL {name} exit={code}")


if __name__ == "__main__":
    main()
