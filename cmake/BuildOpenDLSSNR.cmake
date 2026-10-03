# Native OpenDLSS-NR integration.
#
# Compiles the vendored, Linux-patched OpenDLSS-NR host code into a static library and turns its 13 GLSL
# compute kernels into SPIR-V next to the runtime binary. See 3rdparty/OpenDLSS-NR/VENDORED.md for the local
# patches and why this is vendored rather than a submodule.
#
# Deliberately not built:
#   * the PTX route (VK_NV_cuda_kernel_launch) - its types live in Vulkan's beta headers, and the route also
#     throws at runtime when the .ptx files are absent (upstream ships none). NR_HAS_CUDA_LAUNCH=0 forces the
#     GLSL/SPIR-V kernels for every decision.
#   * src/main.cpp - the upstream `dlss5vk` CLI has its own main().

option(ENABLE_DLSS_NR "Enable native OpenDLSS-NR (DLSS 5 neural rendering) integration" ON)

if(NOT ENABLE_DLSS_NR)
  message(STATUS "OpenDLSS-NR: disabled (ENABLE_DLSS_NR=OFF)")
  return()
endif()

set(OPEN_DLSS_NR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/OpenDLSS-NR" CACHE PATH "OpenDLSS-NR source directory")

if(NOT EXISTS "${OPEN_DLSS_NR_DIR}/src/nr_model.cpp")
  message(WARNING "OpenDLSS-NR: sources not found at ${OPEN_DLSS_NR_DIR}; native DLSS-NR will be unavailable")
  return()
endif()

# ---------------------------------------------------------------------------------------
# Host library
# ---------------------------------------------------------------------------------------
file(GLOB OPEN_DLSS_NR_SOURCES "${OPEN_DLSS_NR_DIR}/src/*.cpp")
# main.cpp is the dlss5vk command line tool; the viewer owns main().
list(FILTER OPEN_DLSS_NR_SOURCES EXCLUDE REGEX "/main\\.cpp$")

add_library(opendlss_nr STATIC ${OPEN_DLSS_NR_SOURCES})
add_library(nvpro2::opendlss_nr ALIAS opendlss_nr)
set_property(TARGET opendlss_nr PROPERTY FOLDER "ThirdParty")

target_include_directories(opendlss_nr PUBLIC "${OPEN_DLSS_NR_DIR}/src")

# Force the GLSL/SPIR-V route: no PTX kernels, no counter chaining, no VK_NV_cuda_kernel_launch.
target_compile_definitions(opendlss_nr PUBLIC NR_HAS_CUDA_LAUNCH=0)

# volk provides both <volk.h> and the Vulkan headers (VOLK_PULL_IN_VULKAN), and it is the same volk the rest of
# the viewer uses. Do not add another copy of volk.c: it would duplicate the global function table.
target_link_libraries(opendlss_nr PUBLIC volk)

set_target_properties(opendlss_nr PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

if(MSVC)
  target_compile_options(opendlss_nr PRIVATE /W3)
else()
  # Upstream is MSVC-first; keep the noise down but do not hide real problems.
  target_compile_options(opendlss_nr PRIVATE -Wall -Wno-unused-parameter -Wno-sign-compare)
endif()

# ---------------------------------------------------------------------------------------
# Kernels: 13 GLSL compute shaders -> SPIR-V
#
# Kernels loads them eagerly as <dir>/<name>.spv, so they must sit beside the executable at runtime.
# ---------------------------------------------------------------------------------------
find_program(OPEN_DLSS_NR_GLSLANG
  NAMES glslangValidator
  HINTS "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}"
        "$ENV{VULKAN_SDK}/bin"
        "${VULKAN_SDK_PATH}/bin"
)

if(NOT OPEN_DLSS_NR_GLSLANG)
  message(WARNING
    "OpenDLSS-NR: glslangValidator not found, so the NR kernels cannot be built. "
    "Set VULKAN_SDK or install the Vulkan SDK; DLSS-NR will report the missing .spv files at runtime.")
  return()
endif()
set(OPEN_DLSS_NR_KERNELS
  gemm_fp8 gemm_f16 ops window_normalize window_attend global_normalize global_attend
  preprocess fused_block32 qkv_attention gemm_mlp global_attention gemm_reduce
)

# Same output layout as the rest of the viewer: _bin/<config>/.
if(CMAKE_BUILD_TYPE)
  set(_nr_runtime_dir "${CMAKE_SOURCE_DIR}/_bin/${CMAKE_BUILD_TYPE}")
else()
  set(_nr_runtime_dir "${CMAKE_SOURCE_DIR}/_bin/$<CONFIG>")
endif()
set(OPEN_DLSS_NR_SHADER_OUTPUT_DIR "${_nr_runtime_dir}/shaders_dlss_nr")
file(MAKE_DIRECTORY "${OPEN_DLSS_NR_SHADER_OUTPUT_DIR}")

# All shaders include the shared GLSL headers, so a change to any of them rebuilds every kernel.
file(GLOB OPEN_DLSS_NR_SHADER_INCLUDES "${OPEN_DLSS_NR_DIR}/shaders/*.glsl")

set(OPEN_DLSS_NR_SPV_FILES "")
foreach(kernel IN LISTS OPEN_DLSS_NR_KERNELS)
  set(_src "${OPEN_DLSS_NR_DIR}/shaders/${kernel}.comp")
  set(_spv "${OPEN_DLSS_NR_SHADER_OUTPUT_DIR}/${kernel}.spv")
  if(NOT EXISTS "${_src}")
    message(WARNING "OpenDLSS-NR: missing kernel ${_src}")
    continue()
  endif()
  add_custom_command(
    OUTPUT "${_spv}"
    COMMAND "${OPEN_DLSS_NR_GLSLANG}" -V --target-env vulkan1.3
            "-I${OPEN_DLSS_NR_DIR}/shaders" "${_src}" -o "${_spv}"
    DEPENDS "${_src}" ${OPEN_DLSS_NR_SHADER_INCLUDES}
    COMMENT "OpenDLSS-NR: glslang ${kernel}.comp"
    VERBATIM
  )
  list(APPEND OPEN_DLSS_NR_SPV_FILES "${_spv}")
endforeach()

add_custom_target(opendlss_nr_kernels ALL DEPENDS ${OPEN_DLSS_NR_SPV_FILES})
set_property(TARGET opendlss_nr_kernels PROPERTY FOLDER "ThirdParty")
add_dependencies(opendlss_nr opendlss_nr_kernels)

# ---------------------------------------------------------------------------------------
# The viewer's own NR passes.
#
# These are the two passes the viewer needs instead of the demo's three: display-referred proxy space (no HDR
# decode or display encode) and a motion texture that derives history validity from the reprojected uv. They
# include nr_common.glsl for the bit-exact roundF16/truncateHalf, so the demo shader directory is an include path.
# ---------------------------------------------------------------------------------------
set(OPEN_DLSS_NR_VIEWER_KERNELS dlss_nr_preprocess dlss_nr_composite)
set(OPEN_DLSS_NR_VIEWER_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders/dlss_nr")
set(OPEN_DLSS_NR_COMMON_INCLUDE "${OPEN_DLSS_NR_DIR}/demo/shaders")

set(OPEN_DLSS_NR_VIEWER_SPV_FILES "")
foreach(kernel IN LISTS OPEN_DLSS_NR_VIEWER_KERNELS)
  set(_src "${OPEN_DLSS_NR_VIEWER_SHADER_DIR}/${kernel}.comp")
  set(_spv "${OPEN_DLSS_NR_SHADER_OUTPUT_DIR}/${kernel}.spv")
  if(NOT EXISTS "${_src}")
    message(WARNING "OpenDLSS-NR: missing viewer kernel ${_src}")
    continue()
  endif()
  add_custom_command(
    OUTPUT "${_spv}"
    COMMAND "${OPEN_DLSS_NR_GLSLANG}" -V --target-env vulkan1.3
            "-I${OPEN_DLSS_NR_COMMON_INCLUDE}" "${_src}" -o "${_spv}"
    DEPENDS "${_src}" "${OPEN_DLSS_NR_COMMON_INCLUDE}/nr_common.glsl"
    COMMENT "OpenDLSS-NR: glslang ${kernel}.comp (viewer)"
    VERBATIM
  )
  list(APPEND OPEN_DLSS_NR_VIEWER_SPV_FILES "${_spv}")
endforeach()

add_custom_target(opendlss_nr_viewer_kernels ALL DEPENDS ${OPEN_DLSS_NR_VIEWER_SPV_FILES})
set_property(TARGET opendlss_nr_viewer_kernels PROPERTY FOLDER "ThirdParty")
add_dependencies(opendlss_nr opendlss_nr_viewer_kernels)

message(STATUS "OpenDLSS-NR: enabled (GLSL/SPIR-V route, no PTX)")
message(STATUS "  kernels: ${OPEN_DLSS_NR_SHADER_OUTPUT_DIR}")

# The viewer sources compile their DLSS-NR wrapper only when the library is actually available.
set(OPEN_DLSS_NR_AVAILABLE TRUE CACHE INTERNAL "OpenDLSS-NR library and kernels are built")
