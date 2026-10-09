# 第三方依赖 —— nanobind
# 1) 先用 wizard/FindNanoBind.cmake 探测 pip 安装的 nanobind
# 2) 仍找不到时回退到 FetchContent 自动下载
if(NOT TARGET nanobind::nanobind AND NOT nanobind_FOUND)
  include(FindNanoBind)
endif()

if(NOT TARGET nanobind::nanobind AND NOT nanobind_FOUND)
  showInfo("nanobind not found locally, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v2.4.0
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(nanobind)
endif()
