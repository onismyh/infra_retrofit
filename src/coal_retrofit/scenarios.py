"""情景登记表：读仓库根 `scenarios/*.toml`，解析继承，校验并转换成两个参数 dataclass 的覆盖项。

每个情景是一张 TOML 表 `[名]`，可写的键：

- `extends`：父情景名，可以多层。`scenario`、`assumptions` 两节按字段合并，子情景的字段覆盖父情景的。
- `abstract = true`：只作基底，不能求解，不出现在 `list` 与 `EXPERIMENTS` 里。
- `tree`：求解树，相对登记目录的上一级（缺省登记目录是仓库根的 `scenarios/`，即相对仓库根；也可写绝对路径）。
  求解读 `<tree>/inputs`、`<tree>/data`，写 `<tree>/results`。随 `extends` 继承；能求解的情景必须有。
- `note`：说明文字，只给 `list`、`show` 显示。不继承，不进 dataclass。
- `[名.scenario]`、`[名.assumptions]`：`OptimizationScenario`、`OptimizationAssumptions` 的字段覆盖。
  字段名写错、写错节、类型不对都直接报错。数组读成元组，整数写在浮点字段里读成浮点。
  字典型字段（如 `province_coal_cost_cny_per_gj`）整张替换，不与缺省或父情景按键合并：只写一个省，其余省就不在
  表里了（与 `dataclasses.replace` 相同）。
  取值有限的字段（`CHOICES`：`water_mode`、`water_season`、`warm_start`、`mip_focus`）只许写模型认的值。

`experiment_id`、`description` 由运行器取结果名，`solver_threads`、`solver_time_limit` 是运行选项、
由命令行 `--threads`、`--time-limit` 给，这四个字段不许写进登记表。

情景名同时是结果文件名（`<tree>/results/<名>.json` 与同名目录），只许字母、数字、`_`、`.`、`-`，以字母或数字
开头，不以 `.` 或 `.json`（不分大小写）结尾；只差大小写的也算重名。Windows 的文件名不分大小写，还会去掉目录名
末尾的点，这两种名字的结果文件会互相覆盖；`X.json` 的结果目录就是 `X` 的 result.json，两者谁后跑谁在求解完
之后写不出结果。
"""
from __future__ import annotations

import difflib
import json
import re
import tomllib
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .optimization.scenario import OptimizationAssumptions, OptimizationScenario

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_DIR = REPO_ROOT / "scenarios"

_SECTIONS: dict[str, type] = {"scenario": OptimizationScenario, "assumptions": OptimizationAssumptions}
_HINTS: dict[str, dict[str, Any]] = {name: typing.get_type_hints(cls) for name, cls in _SECTIONS.items()}
_ENTRY_KEYS = ("extends", "abstract", "tree", "note", *_SECTIONS)
NAME_RE = re.compile(r"(?!.*(?i:\.json)$)[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?")
NAME_RULE = "只许字母、数字、_ . -，以字母或数字开头，不以 . 或 .json（不分大小写）结尾"

# 由运行器按结果名与命令行给出的字段；写进登记表会被静默改掉，所以直接拒绝。
RESERVED_FIELDS: dict[str, str] = {
    "experiment_id": "取结果名",
    "description": "取结果名",
    "solver_threads": "用命令行 --threads",
    "solver_time_limit": "用命令行 --time-limit",
}

# scenario 节里只认这几个值的字段，读登记表、`--set` 与兼容环境变量时就查。`water_season`、`water_mode` 模型按值比较，
# 写错不报错，而是静默走另一支：`water_season = "dyr"` 按全年算，`water_mode` 拼错就按有水约束的 baseline 族算。
# `warm_start` 运行器另有兜底检查，`mip_focus`（Gurobi 的取值范围）越界时 Gurobi 设参数才报错，这里都提前报错。
CHOICES: dict[str, tuple[Any, ...]] = {
    "water_mode": ("no_water", "base_water", "grid_supply", "high_water_stress"),
    "water_season": ("annual", "dry"),
    "warm_start": ("none", "lp_relax"),
    "mip_focus": (0, 1, 2, 3),
}


class ScenarioRegistryError(ValueError):
    """登记表或 `--set` 写错：情景名、键名、节名、字段类型、继承关系。"""


@dataclass(frozen=True)
class ScenarioSpec:
    """解析后的一个情景：两节覆盖项已按继承合并，值已转换成字段类型。"""

    name: str
    source: Path
    tree: Path | None
    scenario: Mapping[str, Any]
    assumptions: Mapping[str, Any]
    note: str = ""
    parent: str | None = None
    abstract: bool = False


@dataclass(frozen=True)
class Registry:
    """一个登记目录下全部情景（含抽象基底），按文件名与文件内的先后排列。"""

    directory: Path
    specs: Mapping[str, ScenarioSpec]

    def runnable(self) -> list[str]:
        """能求解的情景名。"""
        return [name for name, spec in self.specs.items() if not spec.abstract]

    def get(self, name: str) -> ScenarioSpec:
        """按名取一个能求解的情景；没有或是抽象基底时报错。"""
        spec = self.specs.get(name)
        if spec is None:
            clash = self.name_clash(name)
            hint = f"；是不是 {clash}？（情景名区分大小写）" if clash else _suggest(name, self.runnable())
            raise ScenarioRegistryError(f"登记表里没有情景 {name!r}{hint}")
        if spec.abstract:
            raise ScenarioRegistryError(f"{name} 是抽象基底（abstract = true），不能直接求解")
        return spec

    def name_clash(self, name: str) -> str | None:
        """登记表里与 *name* 只差大小写（或完全相同）的情景名；没有则 None。"""
        folded = name.casefold()
        return next((other for other in self.specs if other.casefold() == folded), None)

    def lineage(self, name: str) -> list[str]:
        """从本情景到最顶层父情景的名字链。"""
        chain = [name]
        while (parent := self.specs[chain[-1]].parent) is not None:
            chain.append(parent)
        return chain


def load_registry(directory: Path = DEFAULT_REGISTRY_DIR) -> Registry:
    """读 `directory` 下全部 `*.toml` 并解析；任何一个情景写错都报错，不只是要跑的那个。"""
    directory = Path(directory).absolute()
    files = sorted(directory.glob("*.toml"))
    if not files:
        raise ScenarioRegistryError(
            f"{directory} 下没有 *.toml 情景文件（缺省读仓库根的 scenarios/；包要以可编辑方式安装：pip install -e .）"
        )
    raw: dict[str, tuple[dict[str, Any], Path]] = {}
    folded: dict[str, str] = {}
    for path in files:
        with path.open("rb") as fh:
            try:
                data = tomllib.load(fh)
            except tomllib.TOMLDecodeError as exc:
                raise ScenarioRegistryError(f"{path.name}：TOML 语法错误：{exc}") from exc
        for name, entry in data.items():
            where = f"{path.name} [{name}]"
            other = folded.get(name.casefold())
            if other is not None:
                same = "重复" if other == name else f"只差大小写（{other}），结果文件会互相覆盖"
                raise ScenarioRegistryError(f"{where}：情景名与 {raw[other][1].name} 里的{same}")
            if not NAME_RE.fullmatch(name):
                raise ScenarioRegistryError(f"{where}：情景名{NAME_RULE}")
            if not isinstance(entry, dict):
                raise ScenarioRegistryError(f"{where}：顶层只能是情景表 [名]，不能是单个值")
            unknown = [key for key in entry if key not in _ENTRY_KEYS]
            if unknown:
                raise ScenarioRegistryError(
                    f"{where}：不认识的键 {unknown[0]!r}{_suggest(unknown[0], _ENTRY_KEYS)}"
                    f"（可写的键：{'、'.join(_ENTRY_KEYS)}）"
                )
            raw[name] = (entry, path)
            folded[name.casefold()] = name
    base = directory.parent
    specs = {name: _resolve(name, raw, base, ()) for name in raw}
    return Registry(directory=directory, specs=specs)


def _resolve(name: str, raw: Mapping[str, tuple[dict[str, Any], Path]], base: Path,
             stack: tuple[str, ...]) -> ScenarioSpec:
    """解析一个情景：先解析父情景，再用本情景的键覆盖。"""
    entry, path = raw[name]
    where = f"{path.name} [{name}]"
    if name in stack:
        raise ScenarioRegistryError(f"{where}：extends 成环：{' -> '.join((*stack, name))}")
    parent_name = entry.get("extends")
    if parent_name is None:
        tree: Path | None = None
        scenario: dict[str, Any] = {}
        assumptions: dict[str, Any] = {}
    else:
        if not isinstance(parent_name, str) or parent_name not in raw:
            raise ScenarioRegistryError(f"{where}：extends 的 {parent_name!r} 不在登记表里{_suggest(str(parent_name), raw)}")
        parent = _resolve(parent_name, raw, base, (*stack, name))
        tree, scenario, assumptions = parent.tree, dict(parent.scenario), dict(parent.assumptions)
    abstract = _typed(entry, "abstract", bool, where, default=False)
    note = _typed(entry, "note", str, where, default="")
    tree_text = _typed(entry, "tree", str, where, default=None)
    if tree_text is not None:
        tree = base / tree_text
    for section, target in (("scenario", scenario), ("assumptions", assumptions)):
        values = entry.get(section, {})
        if not isinstance(values, dict):
            raise ScenarioRegistryError(f"{where}：{section} 应是一张表 [{name}.{section}]")
        for key, value in values.items():
            target[key] = convert_field(section, key, value, f"{where} {section}.{key}")
    if not abstract and tree is None:
        raise ScenarioRegistryError(f"{where}：没有 tree（自己或父情景都没写），不知道读哪一套输入")
    return ScenarioSpec(name=name, source=path, tree=tree, scenario=scenario, assumptions=assumptions,
                        note=note, parent=parent_name, abstract=abstract)


def _typed(entry: Mapping[str, Any], key: str, kind: type, where: str, default: Any) -> Any:
    value = entry.get(key, default)
    if value is not default and not isinstance(value, kind):
        raise ScenarioRegistryError(f"{where}：{key} 应为 {kind.__name__}，写的是 {value!r}")
    return value


def convert_field(section: str, key: str, value: Any, where: str) -> Any:
    """校验字段名并把 TOML 值转换成字段类型；取值有限的字段（`CHOICES`）再核取值。"""
    if key in RESERVED_FIELDS:
        raise ScenarioRegistryError(f"{where}：{key} 不能写进登记表（{RESERVED_FIELDS[key]}）")
    hints = _HINTS[section]
    if key not in hints:
        other = next(name for name in _SECTIONS if name != section)
        if key in _HINTS[other]:
            raise ScenarioRegistryError(f"{where}：{key} 是 {other} 节的字段，不是 {section} 节的")
        raise ScenarioRegistryError(f"{where}：{_SECTIONS[section].__name__} 没有字段 {key}{_suggest(key, hints)}")
    converted = _convert(value, hints[key], where)
    choices = CHOICES.get(key) if section == "scenario" else None
    if choices is not None and converted not in choices:
        hint = _suggest(converted, choices) if isinstance(converted, str) else ""
        raise ScenarioRegistryError(
            f"{where}：只能是 {'、'.join(repr(choice) for choice in choices)}，写的是 {converted!r}{hint}"
        )
    return converted


def _convert(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint)
    if origin is tuple:
        args = typing.get_args(hint)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise ScenarioRegistryError(f"{where}：登记表不支持 {hint} 类型的字段")
        if isinstance(value, (list, tuple)):
            return tuple(_convert(item, args[0], f"{where}[{i}]") for i, item in enumerate(value))
    elif origin is dict:
        key_type, value_type = typing.get_args(hint)
        if isinstance(value, dict) and key_type is str:
            return {str(k): _convert(v, value_type, f"{where}.{k}") for k, v in value.items()}
    elif hint is bool:
        if isinstance(value, bool):
            return value
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif hint is str:
        if isinstance(value, str):
            return value
    else:
        raise ScenarioRegistryError(f"{where}：登记表不支持 {hint} 类型的字段")
    label = getattr(hint, "__name__", None) if origin is None else None
    raise ScenarioRegistryError(f"{where}：应为 {label or hint}，写的是 {value!r}")


def _suggest(word: str, candidates: Iterable[str]) -> str:
    close = difflib.get_close_matches(word, list(candidates), n=3)
    return f"；是不是 {'、'.join(close)}？" if close else ""



def parse_set(text: str) -> tuple[str, str, Any]:
    """`节.字段=值` → (节, 字段, 已转换的值)。

    值按 TOML 读：数、`true` / `false`、`"字符串"`、`[数组]`、`{表}`；读不成 TOML 的原样当字符串，
    所以 `scenario.water_mode=grid_supply` 不用加引号。以引号、`[`、`{` 开头却读不成 TOML 的直接报错：
    多半是漏了引号或括号，原样收下会把 `"dry` 这样的值静默写进字符串字段。
    """
    key, sep, raw = text.partition("=")
    section, dot, name = key.strip().partition(".")
    if not sep or not dot or section not in _SECTIONS:
        raise ScenarioRegistryError(f"--set {text!r}：应写成 scenario.字段=值 或 assumptions.字段=值")
    raw = raw.strip()
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError as exc:
        if raw[:1] in ('"', "'", "[", "{"):
            raise ScenarioRegistryError(f"--set {text!r}：值按 TOML 读不成（漏了引号或括号？）：{exc}") from exc
        value = raw
    return section, name, convert_field(section, name, value, f"--set {key.strip()}")


def apply_sets(spec: ScenarioSpec, sets: Sequence[str]) -> tuple[ScenarioSpec, dict[str, Any]]:
    """把 `--set` 覆盖到情景上；返回新情景与实际覆盖的 {节.字段: 值}（后写的覆盖先写的）。"""
    scenario, assumptions = dict(spec.scenario), dict(spec.assumptions)
    applied: dict[str, Any] = {}
    for text in sets:
        section, name, value = parse_set(text)
        (scenario if section == "scenario" else assumptions)[name] = value
        applied[f"{section}.{name}"] = value
    return replace(spec, scenario=scenario, assumptions=assumptions), applied


def experiments_dict(registry: Registry) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """原 `scripts/run_single.py` 的 `EXPERIMENTS` 形状：{名: (scenario 覆盖, assumptions 覆盖)}，只含能求解的。"""
    return {
        name: (dict(registry.specs[name].scenario), dict(registry.specs[name].assumptions))
        for name in registry.runnable()
    }


def shown_path(path: Path) -> str:
    """仓库内的路径写相对仓库根的 posix 形式（如 `_indtree`），仓库外的写绝对路径。"""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


# 结果名，不是参数：两个不同情景的这两项总是不同，比参数时跳过。
_NAME_FIELDS = ("scenario.experiment_id", "scenario.description")


def flatten_resolved(resolved: Mapping[str, Any]) -> dict[str, Any]:
    """`resolved` 段 → {`tree` / `节.字段`: 值}，先按 JSON 规整（元组与列表视为同一个值）。"""
    normal = json.loads(json.dumps(dict(resolved), default=str))
    flat: dict[str, Any] = {"tree": normal.get("tree")}
    for section in _SECTIONS:
        for key, value in (normal.get(section) or {}).items():
            flat[f"{section}.{key}"] = value
    for key in _NAME_FIELDS:
        flat.pop(key, None)
    return flat


def diff_resolved(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[tuple[str, Any, Any]]:
    """两段 `resolved` 的参数差：[(`tree` / `节.字段`, a 的值, b 的值)]，按 a 的字段顺序，b 独有的排在后面。"""
    fa, fb = flatten_resolved(a), flatten_resolved(b)
    keys = [*fa, *(key for key in fb if key not in fa)]
    return [(key, fa.get(key), fb.get(key)) for key in keys if fa.get(key) != fb.get(key)]
