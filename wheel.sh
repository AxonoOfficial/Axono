#!/usr/bin/env bash
# Axono wheel 打包脚本 (Linux) —— 与 install.sh 相同的 CUDA 检测逻辑,
# 产出 pip 可安装的 .whl 到 dist/。
#
# 用法:
#   bash wheel.sh                   # 自动检测 CUDA 兼容环境并编入 wheel
#   bash wheel.sh --cuda            # 强制 CUDA (不兼容则报错)
#   bash wheel.sh --no-cuda         # 强制纯 CPU
#   bash wheel.sh --no-cuda --outdir /tmp/wheels   # 额外参数透传给 pip wheel
#
# 安装产物:
#   pip install dist/axono-*.whl
#
# 注意: CUDA wheel 链接本机 CUDA 库, 目标机器需同代或更新驱动;
#       分发请按 CUDA 版本分目录 (参考 README 的 cu118/cu125/cu126 索引)。

set -euo pipefail

cd "$(dirname "$0")"

# 复用 install.sh 的 CUDA 检测 (只取函数与 NVCC_PATH 初始化, 不执行
# 安装逻辑); 兜底自行初始化避免 set -u 未绑定
NVCC_PATH=""
source <(sed -n '/^detect_cuda()/,/^}/p' "$(dirname "$0")/install.sh")

FORCE_CUDA=0
FORCE_CPU=0
OUTDIR=""
PASS_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cuda)    FORCE_CUDA=1 ;;
    --no-cuda) FORCE_CPU=1 ;;
    --outdir)  OUTDIR="$2"; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *)
      PASS_ARGS+=("$1") ;;
  esac
  shift
done

# ---------- 与 install.sh 一致的 CUDA 决策 ----------
USE_CUDA=0
if [ "$FORCE_CPU" = "1" ]; then
  echo "==> 按参数强制 CPU wheel"
elif [ "$FORCE_CUDA" = "1" ]; then
  det=$(detect_cuda) || { echo "错误: 强制 CUDA 但环境不兼容 ($det)"; exit 1; }
  echo "==> 按参数强制 CUDA wheel (nvcc $det)"
  USE_CUDA=1
else
  if det=$(detect_cuda); then
    echo "==> 检测到兼容 CUDA 环境 (nvcc $det) -> 打包 CUDA wheel"
    USE_CUDA=1
  else
    echo "==> 未检测到兼容 CUDA 环境 ($det) -> 打包 CPU wheel (如需 CUDA: bash wheel.sh --cuda)"
  fi
fi

export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

SETTINGS=(--config-settings=cmake.define.WITH_CUDA=OFF)
if [ "$USE_CUDA" = "1" ]; then
  # 重新定位 nvcc (detect_cuda 在 $( ) 子 shell 中运行, 其内部变量
  # 不会带回父 shell)
  if [ -z "$NVCC_PATH" ]; then
    for d in /usr/local/cuda*/bin /opt/cuda/bin; do
      [ -x "$d/nvcc" ] && NVCC_PATH="$d/nvcc" && break
    done
  fi
  [ -n "$NVCC_PATH" ] && export PATH="$(dirname "$NVCC_PATH"):$PATH"
  command -v nvcc >/dev/null 2>&1 || { echo "错误: 找不到 nvcc"; exit 1; }
  SETTINGS=(--config-settings=cmake.define.WITH_CUDA=ON)
fi

# pip 需要可用的构建后端
PIP="${PIP:-python3 -m pip}"
$PIP --version >/dev/null 2>&1 || { echo "错误: 找不到 pip (试: PIP='/path/python -m pip' bash wheel.sh)"; exit 1; }

OUTARGS=()
if [ -n "$OUTDIR" ]; then
  mkdir -p "$OUTDIR"
  OUTARGS=(-w "$OUTDIR")
fi

echo "==> pip wheel . (WITH_CUDA=$USE_CUDA, jobs=$CMAKE_BUILD_PARALLEL_LEVEL)"
$PIP wheel . "${OUTARGS[@]}" ${PIP_ARGS:-} "${PASS_ARGS[@]}" "${SETTINGS[@]}"

if [ -n "$OUTDIR" ]; then
  echo "==> 产物:"
  ls -lh "$OUTDIR"/axono-*.whl
else
  echo "==> 产物 (当前目录):"
  ls -lh ./axono-*.whl 2>/dev/null || ls -lh dist/axono-*.whl 2>/dev/null || true
fi
echo "==> 完成。安装: pip install <上面的 .whl 路径>"
