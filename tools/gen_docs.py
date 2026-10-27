#!/usr/bin/env python3
"""Axono 文档抽取器 (docs extractor)。

从仓库源码中抽取注释作为 axono.org/docs 的内容。

抽取约定 (extractor contract) — 文档内容即代码注释, 无需单独维护文档源:
- **模块级 docstring**: Python 文件顶部 `\"\"\"...\"\"` 作为该模块的文档正文;
- **API 文档**: 模块中公开的函数/类 docstring 作为 API 条目;
- **C++ 头文件**: `include/axono/**/*.h` 中每个公开类/函数前的 `//` 注释
  作为条目说明;
- 以 `# axono/...` 或 `// Axono/...` 开头的路径标注行被忽略;
- `tests/`、`benchmarks/`、`examples/vlm` 等目录不参与抽取。

输出: JSON ({"modules": [...]}), 供 docs 静态站生成器消费。

用法: python3 tools/gen_docs.py [repo_root] [out.json]
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

# 参与文档抽取的 Python 包路径
PY_DIRS = ["axono/python/axono"]
PY_EXCLUDE = ("/tests/", "/benchmarks/", "/examples/")

# 参与 API 抽取的 C++ 头文件
CPP_DIRS = ["include/axono"]


def _clean_doc(text: str) -> str:
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    out = []
    for ln in lines:
        # 路径标注行 / 纯分隔线不入文档
        if re.match(r"^(# |// )?(axono|Axono)/[\w./]+ —", ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def _sig_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    a = node.args
    args = []
    pos = [x.arg for x in a.posonlyargs + a.args]
    defaults = [None] * (len(pos) - len(a.defaults)) + [d for d in a.defaults]
    for name, d in zip(pos, defaults):
        args.append(name if d is None else f"{name}=...")
    if a.vararg:
        args.append(f"*{a.vararg.arg}")
    for x in a.kwonlyargs:
        args.append(x.arg)
    if a.kwarg:
        args.append(f"**{a.kwarg.arg}")
    return "(" + ", ".join(args) + ")"


def extract_python(root: Path) -> list:
    modules = []
    for d in PY_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            rel = f.relative_to(root).as_posix()
            if any(x in f"/{rel}" for x in PY_EXCLUDE):
                continue
            if f.name == "__init__.py" and "core" not in rel.split("/")[-2:-1]:
                pass
            try:
                tree = ast.parse(f.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            mod_doc = ast.get_docstring(tree) or ""
            mod_doc = _clean_doc(mod_doc)
            funcs, classes = [], []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if (
                        node.name.startswith("_")
                        and node.name != "__init__"
                        and not node.name.startswith("_tensor_")
                    ):
                        continue
                    doc = ast.get_docstring(node)
                    if doc:
                        funcs.append(
                            {
                                "name": node.name,
                                "signature": _sig_of(node),
                                "doc": _clean_doc(doc),
                            }
                        )
                elif isinstance(node, ast.ClassDef):
                    doc = _clean_doc(ast.get_docstring(node) or "")
                    methods = []
                    for sub in node.body:
                        if (
                            isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and not sub.name.startswith("_")
                            and ast.get_docstring(sub)
                        ):
                            methods.append(
                                {
                                    "name": sub.name,
                                    "signature": _sig_of(sub),
                                    "doc": _clean_doc(ast.get_docstring(sub)),
                                }
                            )
                    classes.append({"name": node.name, "doc": doc, "methods": methods})
            if not (mod_doc or funcs or classes):
                continue
            # Tensor 的便利方法以下划线 helper 形式实现再挂到绑定类上;
            # 文档中还原为公开 API 名 (from_numpy / to / reshape ...)。
            _rename = {"_tensor_to_device": "to", "_tensor_T": "T"}
            for f in funcs:
                if f["name"] in _rename:
                    f["name"] = _rename[f["name"]]
                elif f["name"].startswith("_tensor_"):
                    f["name"] = f["name"][len("_tensor_") :]
            modname = (
                rel.replace("axono/python/", "").replace("/", ".").removesuffix(".py")
            )
            modules.append(
                {
                    "id": modname,
                    "path": rel,
                    "lang": "python",
                    "doc": mod_doc,
                    "functions": funcs,
                    "classes": classes,
                }
            )
    return modules


# C++ 头文件: 抽取 namespace axono 内 class / 函数声明前的 // 注释
CPP_RE_METHOD = re.compile(
    r"(?P<comment>(?:^\s*//[^\n]*\n)*)"  # 前置 // 注释块
    r"^\s*(?:static\s+)?[\w:<>,&*\s]+?\b(?P<name>\w+)\s*\(",
    re.M,
)


def extract_cpp(root: Path) -> list:
    modules = []
    for d in CPP_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.h")):
            rel = f.relative_to(root).as_posix()
            text = f.read_text(encoding="utf-8", errors="replace")
            # 类级: "class Tensor {" 前的注释块
            entries = []
            for m in re.finditer(
                r"(?P<comment>(?:^[ \t]*//[^\n]*\n)*)"
                r"^[ \t]*class\s+(?P<name>\w+)[^;{]*\{",
                text,
                re.M,
            ):
                doc = _clean_doc(m.group("comment").replace("//", ""))
                if doc:
                    entries.append(
                        {
                            "name": "class " + m.group("name"),
                            "signature": "",
                            "doc": doc,
                        }
                    )
            for m in CPP_RE_METHOD.finditer(text):
                name = m.group("name")
                if name in ("if", "for", "while", "switch", "sizeof", "return"):
                    continue
                doc = _clean_doc(m.group("comment").replace("//", "").strip())
                if doc:
                    entries.append({"name": name, "signature": "", "doc": doc})
            if not entries:
                continue
            modname = (
                rel.replace("include/axono/", "cxx.")
                .replace("/", ".")
                .removesuffix(".h")
            )
            modules.append(
                {
                    "id": modname,
                    "path": rel,
                    "lang": "cpp",
                    "doc": "",
                    "functions": [],
                    "classes": [
                        {"name": rel.split("/")[-1], "doc": "", "methods": entries}
                    ],
                }
            )
    return modules


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "docs.json")
    modules = extract_python(root) + extract_cpp(root)
    doc = {
        "project": "Axono",
        "version": _read_version(root),
        "generated_by": "tools/gen_docs.py",
        "modules": modules,
    }
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"extracted {len(modules)} modules -> {out}")


def _read_version(root: Path) -> str:
    for cand in (root / "pyproject.toml", root / "axono/python/pyproject.toml"):
        if cand.is_file():
            m = re.search(r'version\s*=\s*"([^"]+)"', cand.read_text(encoding="utf-8"))
            if m:
                return m.group(1)
    return "unknown"


if __name__ == "__main__":
    main()
