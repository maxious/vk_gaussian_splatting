cmake_minimum_required(VERSION 3.15)

project(vk_radix_sort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED)

# We use SLANG from external resolution in ${Slang_SLANGC_EXECUTABLE}

# adds -fPIC, works for linux, when building shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/generated)

# build_shader(SHADER OUTPUT DEFINE...)
function(build_shader)
  list(POP_FRONT ARGV SHADER OUTPUT)
  list(TRANSFORM ARGV PREPEND "-D" OUTPUT_VARIABLE DEFINES)

  get_filename_component(SHADER ${SHADER} ABSOLUTE)

  if(UNIX)
     # Workaround: the Vulkan SDK sets LD_LIBRARY_PATH and
     # slangc may find a libslang.so there instead of its own
     set(_SLANGC env LD_LIBRARY_PATH= ${Slang_SLANGC_EXECUTABLE})
  else()
     set(_SLANGC ${Slang_SLANGC_EXECUTABLE})
  endif()

  add_custom_target(${OUTPUT} ALL
    COMMAND
      /usr/bin/python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/slangc_to_header.py
      -target spirv
      -fvk-t-shift 0 0
      -fvk-u-shift 0 0
      -o ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h
      ${DEFINES}
      ${SHADER}
    COMMENT "Compiling ${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${OUTPUT}.h"
  )
endfunction()

# header-only library
add_library(vk_radix_sort INTERFACE)

target_include_directories(vk_radix_sort INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

build_shader(src/shader/upsweep.slang upsweep_slang)
build_shader(src/shader/spine.slang spine_slang)
build_shader(src/shader/downsweep.slang downsweep_slang)
build_shader(src/shader/downsweep.slang downsweep_key_value_slang KEY_VALUE)
