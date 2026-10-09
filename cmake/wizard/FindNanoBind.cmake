# 帮助寻找 NanoBind 库
# 优先使用 pip 安装的 nanobind；未安装则交由 FetchContent 回退 (third_party.cmake)
execute_process(
   COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
   RESULT_VARIABLE nanobind_PROBE_RESULT
   OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
   ERROR_QUIET)

if(nanobind_PROBE_RESULT EQUAL 0)
  showInfo("Found NanoBind: ${nanobind_ROOT}")
  list(APPEND CMAKE_PREFIX_PATH "${nanobind_ROOT}/..")
  set(nanobind_DIR "${nanobind_ROOT}" CACHE PATH "nanobind cmake dir")
else()
  showInfo("nanobind not installed for ${Python_EXECUTABLE}, will fetch")
endif()

find_package(nanobind CONFIG QUIET)
