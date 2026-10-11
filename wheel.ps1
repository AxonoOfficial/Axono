<#
.SYNOPSIS
  Axono Windows wheel builder (source build -> pip wheel -> .whl in dist\).

.EXAMPLE
  .\wheel.ps1                                # auto-detect CUDA
  .\wheel.ps1 -Cuda                          # force CUDA (error if incompatible)
  .\wheel.ps1 -NoCuda                        # force CPU
  .\wheel.ps1 -Cuda -IgnoreDrivers           # CUDA: only require nvcc (CI / no-GPU build machine)
  .\wheel.ps1 -NoCuda -OutDir C:\tmp\wheels  # extra args are passed to `pip wheel`

.NOTES
  Requires: Python 3.10+ with pip, Visual Studio Build Tools (C++),
  and for CUDA builds: CUDA Toolkit (nvcc). Building CUDA does NOT
  require a GPU/driver — only the machine installing the wheel does.
#>
[CmdletBinding()]
param(
  [switch]$Cuda,
  [switch]$NoCuda,
  [switch]$IgnoreDrivers,
  [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ---------- Python / pip ----------
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
& $Python -m pip --version *> $null
if ($LASTEXITCODE -ne 0) {
  Write-Error "pip not found (try: `$env:PYTHON='C:\path\to\python.exe'; .\wheel.ps1)"
  exit 1
}

function Find-Nvcc {
  $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
  if ($nvcc) { return $nvcc.Source }
  $roots = @(
    "$env:CUDA_PATH\bin",
    "$env:SystemDrive\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin"
  ) | Where-Object { $_ }
  foreach ($pat in $roots) {
    $hit = Get-Item $pat -ErrorAction SilentlyContinue |
      ForEach-Object { Join-Path $_.FullName "nvcc.exe" } |
      Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($hit) { return $hit }
  }
  return $null
}

function Get-NvccVersion([string]$NvccPath) {
  $out = & $NvccPath --version 2>$null | Out-String
  if ($out -match 'release ([0-9]+\.[0-9]+)') { return $Matches[1] }
  return $null
}

function Test-CudaEnvironment {
  # 1) compiler
  $nvcc = Find-Nvcc
  if (-not $nvcc) { return $null }
  $ver = Get-NvccVersion $nvcc
  if (-not $ver) { return $null }

  # -IgnoreDrivers: compiling CUDA needs no driver/GPU; only nvcc matters.
  if ($IgnoreDrivers) {
    return @{ Nvcc = $nvcc; Version = $ver; Note = "(drivers ignored)" }
  }

  # 2) driver + GPU
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if (-not $smi) { return $null }
  $name = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
  if (-not $name) { return $null }

  # 3) driver CUDA version >= nvcc major
  $smiOut = (& nvidia-smi | Out-String)
  if ($smiOut -match 'CUDA Version:\s*([0-9]+\.[0-9]+)') {
    $driverMajor = [int]($Matches[1].Split('.')[0])
    $nvccMajor   = [int]($ver.Split('.')[0])
    if ($driverMajor -lt $nvccMajor) { return $null }
  }
  return @{ Nvcc = $nvcc; Version = $ver; Note = $null }
}

# ---------- CUDA decision (same rules as install.ps1) ----------
$UseCuda = $false
$NvccPath = $null
if ($NoCuda) {
  Write-Host "==> CPU wheel (forced by -NoCuda)"
} elseif ($Cuda) {
  $det = Test-CudaEnvironment
  if (-not $det) {
    Write-Error "Force CUDA but environment is incompatible (no usable nvcc found). Install the CUDA Toolkit or use -NoCuda."
    exit 1
  }
  Write-Host ("==> CUDA wheel (forced; nvcc {0} {1})" -f $det.Version, $det.Note)
  $UseCuda = $true; $NvccPath = $det.Nvcc
} else {
  $det = Test-CudaEnvironment
  if ($det) {
    Write-Host ("==> CUDA detected (nvcc {0}) -> CUDA wheel" -f $det.Version)
    $UseCuda = $true; $NvccPath = $det.Nvcc
  } else {
    Write-Host "==> No compatible CUDA environment -> CPU wheel (use -Cuda to force)"
  }
}

# ---------- build ----------
$jobs = $env:CMAKE_BUILD_PARALLEL_LEVEL
if (-not $jobs) { $jobs = [Environment]::ProcessorCount }
$env:CMAKE_BUILD_PARALLEL_LEVEL = "$jobs"

$OutArgs = @()
if ($OutDir) {
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  $OutArgs = @("-w", $OutDir)
}

if ($UseCuda) {
  $nvccDir = Split-Path $NvccPath
  if ((Get-Command nvcc -ErrorAction SilentlyContinue) -eq $null) {
    $env:Path = "$nvccDir;$env:Path"
  }
  & $Python -m pip wheel . @OutArgs --config-settings=cmake.define.WITH_CUDA=ON
} else {
  & $Python -m pip wheel . @OutArgs --config-settings=cmake.define.WITH_CUDA=OFF
}
if ($LASTEXITCODE -ne 0) { exit 1 }

if ($OutDir) {
  Write-Host "==> Artifacts:"
  Get-ChildItem (Join-Path $OutDir "axono-*.whl")
} else {
  Write-Host "==> Artifact (current dir):"
  Get-ChildItem "axono-*.whl" -ErrorAction SilentlyContinue
}
Write-Host "==> Done. Install: pip install <path to .whl>"
