#!/usr/bin/env bash
# Axono Linux 安装脚本 (源码编译 → pip install)
#
# 用法:
#   bash install.sh                 # 自动检测: 有兼容 CUDA 环境则启用 CUDA
#   bash install.sh --cuda          # 强制启用 CUDA (无兼容环境则报错退出)
#   bash install.sh --no-cuda       # 强制纯 CPU
#
# 可选环境变量:
#   CMAKE_BUILD_PARALLEL_LEVEL=N    # 编译并行度 (默认取 nproc)
#   PIP="$PIP"                      # 指定 pip (默认 python3 -m pip)

set -euo pipefail

cd "$(dirname "$0")"

FORCE_CUDA=0
FORCE_CPU=0
for arg in "$@"; do
  case "$arg" in
    --cuda)    FORCE_CUDA=1 ;;
    --no-cuda) FORCE_CPU=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *)
      echo "未知参数: $arg (支持 --cuda / --no-cuda)"; exit 1 ;;
  esac
done

# ---------- Python / pip ----------
PIP="${PIP:-python3 -m pip}"
if ! $PIP --version >/dev/null 2>&1; then
  echo "错误: 找不到 pip (试: PIP='python3 -m pip -v' bash install.sh)"; exit 1
fi

# ---------- CUDA 兼容性自动检测 ----------
NVCC_PATH=""   # detect_cuda 找到的 nvcc 绝对路径 (供 cmake 使用)

detect_cuda() {
  # ① 编译器: nvcc
  if command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH="$(command -v nvcc)"
  else
    for d in /usr/local/cuda*/bin /opt/cuda/bin; do
      if [ -x "$d/nvcc" ]; then NVCC_PATH="$d/nvcc"; break; fi
    done
    if [ -z "$NVCC_PATH" ]; then echo "no-nvcc"; return 1; fi
    export PATH="$(dirname "$NVCC_PATH"):$PATH"
  fi
  local nvcc_ver
  nvcc_ver=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
  [ -n "$nvcc_ver" ] || { echo "no-nvcc"; return 1; }

  # ② 驱动 + GPU: nvidia-smi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "no-driver"; return 1
  fi
  local driver_ver
  driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  [ -n "$driver_ver" ] || { echo "no-driver"; return 1; }

  # ③ GPU 存在且驱动版本满足该 nvcc (CUDA 兼容矩阵: driver >= nvcc 要求)
  #    用 nvidia-smi 报告的 CUDA 版本与 nvcc 主版本比较即可 (驱动自带
  #    的 CUDA 运行时版本必须 >= 编译工具链版本)。
  local driver_cuda
  driver_cuda=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+\.[0-9]+')
  [ -n "$driver_cuda" ] || { echo "no-driver-cuda"; return 1; }

  local nvcc_major driver_major
  nvcc_major="${nvcc_ver%%.*}"
  driver_major="${driver_cuda%%.*}"
  if [ "$driver_major" -lt "$nvcc_major" ]; then
    echo "driver-too-old($driver_cuda < nvcc $nvcc_ver)"; return 1
  fi
  # 有 GPU 才值得编 CUDA (纯编译也行, 但这里按运行环境判断)
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q .; then
    echo "no-gpu"; return 1
  fi
  echo "cuda:$nvcc_ver"
  return 0
}

USE_CUDA=0
if [ "$FORCE_CPU" = "1" ]; then
  echo "==> 按参数强制 CPU 构建"
elif [ "$FORCE_CUDA" = "1" ]; then
  det=$(detect_cuda) || { echo "错误: 强制 CUDA 但环境不兼容 ($det)"; exit 1; }
  echo "==> 按参数强制 CUDA 构建 (nvcc $det)"
  USE_CUDA=1
else
  if det=$(detect_cuda); then
    echo "==> 检测到兼容 CUDA 环境 (nvcc $det) -> 启用 CUDA"
    USE_CUDA=1
  else
    echo "==> 未检测到兼容 CUDA 环境 ($det) -> 纯 CPU 构建 (如需 CUDA: bash install.sh --cuda)"
  fi
fi

# ---------- 构建 & 安装 ----------
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

SETTINGS=(--config-settings=cmake.define.WITH_CUDA=OFF)
if [ "$USE_CUDA" = "1" ]; then
  # 确保 nvcc 在 PATH (cmake 需要)
  # 重新定位 nvcc (detect_cuda 在 $( ) 子 shell 中运行, 其内部变量
  # 不会带回父 shell)
  if [ -z "$NVCC_PATH" ]; then
    for d in /usr/local/cuda*/bin /opt/cuda/bin; do
      [ -x "$d/nvcc" ] && NVCC_PATH="$d/nvcc" && break
    done
  fi
  command -v nvcc >/dev/null 2>&1 || { [ -n "$NVCC_PATH" ] && export PATH="$(dirname "$NVCC_PATH"):$PATH"; }
  command -v nvcc >/dev/null 2>&1 || { echo "错误: 找不到 nvcc"; exit 1; }
  SETTINGS=(--config-settings=cmake.define.WITH_CUDA=ON)
fi

echo "==> pip install . (WITH_CUDA=$USE_CUDA, jobs=$CMAKE_BUILD_PARALLEL_LEVEL)"
$PIP install --upgrade ${PIP_ARGS:-} "$@" . "${SETTINGS[@]}"

echo "==> 完成。验证:"
$PIP show axono 2>/dev/null | sed -n '1,2p'
python3 - <<'EOF'
import axono
print("axono", axono.__version__, "from", axono.__file__)
try:
    from libaxono import cached_bytes  # noqa: F401
    print("libaxono 扩展加载 OK")
except Exception as e:
    print("libaxono 扩展加载失败:", e)
EOF
