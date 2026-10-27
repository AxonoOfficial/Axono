### 贡献准则
> Axono

## 代码风格

在仓库根目录安装与 CI 相同的工具并执行检查：

```bash
python -m pip install -r tools/requirements-style.txt
ruff check .
ruff format --check .
git ls-files -z '*.cpp' '*.hpp' '*.h' '*.cc' '*.cxx' '*.hxx' | xargs -0 -r clang-format --dry-run --Werror
```

自动修复可用 `ruff check --fix .`，然后运行 `ruff format .`。无法自动修复的 lint 问题需手动处理；C++ 格式修复将上述 `clang-format --dry-run --Werror` 换成 `clang-format -i`。提交前再次运行检查。
