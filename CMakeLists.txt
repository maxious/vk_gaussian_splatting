cmake_minimum_required(VERSION 3.6...3.31)

# Export compile commands for clangd/IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable compiler cache (sccache or ccache) if available
# sccache is preferred on Windows/MSVC, ccache works better on Linux
find_program(SCCACHE_PROGRAM sccache)
find_program(CCACHE_PROGRAM ccache)
if(SCCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
  message(STATUS "Using sccache: ${SCCACHE_PROGRAM}")
elseif(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
endif()

# get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(PROJECT_NAME "vk_gaussian_splatting")
Project(${PROJECT_NAME} LANGUAGES C CXX)

# Enable PDB generation for all build types on MSVC
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
endif()

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Override Slang version before NvproCore2 loads it
set(Slang_VERSION "2025.24.2" CACHE STRING "Slang version" FORCE)

find_package(NvproCore2 REQUIRED)

# set with -DDISABLE_DEFAULT_SCENE=ON on command line or use cmake-gui to set the parameter
# if default scene is not disabled at build using this option, one can still disable it by 
# providing "-loadDefaultScene 0" to the sample command line
option(DISABLE_DEFAULT_SCENE "If set to true the default scene is not downloaded by CMake, and the sample does not try to open it when starting up with no scene as parameter" OFF)

#####################################################################################
# DLSS-RR configuration
#
option(ENABLE_DLSS_RR "Enable DLSS Ray Reconstruction denoiser" ON)
option(DLSS_USE_DEVELOP_LIBRARIES "Use non-distributable DLSS libraries with debug overlay (Ctrl-Alt-F12 to enable, Ctrl-Alt-F11 to switch views)" OFF)

if(ENABLE_DLSS_RR)
  find_package(NGX REQUIRED)
endif()

#####################################################################################
# OpenXR support for HMD rendering
#
option(ENABLE_OPENXR "Enable OpenXR HMD support" ON)

if(ENABLE_OPENXR)
  find_package(OpenXR QUIET)
  if(OpenXR_FOUND)
    message(STATUS "OpenXR found, enabling HMD support")
  else()
    message(STATUS "OpenXR not found via find_package, trying to fetch from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
      OpenXR-SDK
      GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-SDK.git
      GIT_TAG        release-1.1.54
    )
    set(BUILD_ALL_EXTENSIONS ON CACHE BOOL "" FORCE)
    set(BUILD_LOADER ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(OpenXR-SDK)
    set(OpenXR_FOUND TRUE)
  endif()
endif()

# Download the default scene
if (NOT DISABLE_DEFAULT_SCENE)
  add_definitions(-DWITH_DEFAULT_SCENE_FEATURE)
  download_files(FILENAMES flowers_1.zip EXTRACT)
endif()

# set with -DDISABLE_MESH_ASSETS_DOWNLOAD=ON on command line or use cmake-gui to set the parameter
option(DISABLE_MESH_ASSETS_DOWNLOAD "If set to true the test mesh assets are not downloaded by CMake" OFF)

if (NOT DISABLE_MESH_ASSETS_DOWNLOAD)
  download_files(
    FILENAMES 
     armadillo.obj    
     xyzrgb_dragon.obj
     beast.obj        
     happy.obj        
     horse.obj        
     lucy.obj         
     nefertiti.obj    
     teapot.obj       
   URLS 
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/xyzrgb_dragon.obj
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/beast.obj
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/happy.obj
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/horse.obj
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/lucy.obj
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/nefertiti.obj
     https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/teapot.obj)
endif()

#####################################################################################
# additions of 3rdparty

#####
# libwebp for SOG format support
include(FetchContent)
FetchContent_Declare(
  libwebp
  GIT_REPOSITORY https://chromium.googlesource.com/webm/libwebp.git
  GIT_TAG        v1.4.0
)
set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_LIBWEBPMUX OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
set(WEBP_UNICODE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libwebp)
set_property(TARGET webp PROPERTY FOLDER "ThirdParty/libwebp")
set_property(TARGET webpdecode PROPERTY FOLDER "ThirdParty/libwebp")
set_property(TARGET webpdspdecode PROPERTY FOLDER "ThirdParty/libwebp")
set_property(TARGET webputilsdecode PROPERTY FOLDER "ThirdParty/libwebp")
message(STATUS "libwebp enabled for SOG format support")

#####
# WebSocket++ for ComfyUI integration
option(ENABLE_COMFYUI "Enable ComfyUI WebSocket integration" ON)

if(ENABLE_COMFYUI)
  include(FetchContent)
  
  # Standalone ASIO (no Boost dependency)
  FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG        asio-1-30-2
  )
  FetchContent_MakeAvailable(asio)
  
  # WebSocket++ (header-only)
  FetchContent_Declare(
    websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
    GIT_TAG        4dfe1be74e684acca19ac1cf96cce0df9eac2a2d
  )
  FetchContent_MakeAvailable(websocketpp)
  
  # Create interface library for websocketpp
  add_library(websocketpp INTERFACE)
  target_include_directories(websocketpp INTERFACE 
    ${websocketpp_SOURCE_DIR}
    ${asio_SOURCE_DIR}/asio/include
  )
  target_compile_definitions(websocketpp INTERFACE 
    ASIO_STANDALONE
    _WEBSOCKETPP_CPP11_STL_
  )
  
  message(STATUS "WebSocket++ enabled for ComfyUI integration")
endif()

#####
# Include Vulkan Radix Sort (header-only library)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx)

# vk_radix_sort is now an INTERFACE library, only set folder for shader targets
set_property(TARGET spine_slang PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET downsweep_slang PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET downsweep_key_value_slang PROPERTY FOLDER "ThirdParty/vrdx")
set_property(TARGET upsweep_slang PROPERTY FOLDER "ThirdParty/vrdx")

#####
# Include spz (uses zlib from nvpro_core2)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spz)
set_property(TARGET spz PROPERTY FOLDER "ThirdParty")

#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES src/*.*)
# Remove "unity build" partial source files that are #included by gaussian_splatting.cpp
list(FILTER SOURCE_FILES EXCLUDE REGEX "gaussian_splatting_(render|shaders|pipelines|rtx|postprocess|dlss_rr|frame_ubo|sorting)\\.cpp$")
file(GLOB SHADER_FILES shaders/*.slang shaders/*.h)
file(GLOB EXTERN_FILES 3rdparty/miniply/*.*)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src 
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders 
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/miniply
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx/include
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spz/src/cc
  ${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core2/third_party/meshoptimizer/src)

#####################################################################################
# Executable
#

add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${SHADER_FILES} ${EXTERN_FILES})

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_DEFINE_MATH_OPERATORS )

#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group("Shader Files" FILES ${SHADER_FILES})
source_group("Source Files" FILES ${SOURCE_FILES})
source_group("Extern Files" FILES ${EXTERN_FILES})

#####################################################################################
# Linkage
#

# Linking with other libraries
# Link libraries and include directories
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvvkglsl
  nvpro2::nvslang
  nvpro2::nvshaders_host
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvgpu_monitor
  vk_radix_sort
  spz::spz
  meshoptimizer
  webp
  zlib
)

target_include_directories(${PROJECT_NAME} PRIVATE ${libwebp_SOURCE_DIR}/src)

# WebSocket++ linkage for ComfyUI
if(ENABLE_COMFYUI)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_COMFYUI)
  target_link_libraries(${PROJECT_NAME} PRIVATE websocketpp)
endif()

# DLSS-RR linkage
if(ENABLE_DLSS_RR AND NGX_FOUND)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_DLSS_RR)
  target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
endif()

# OpenXR linkage
if(ENABLE_OPENXR AND OpenXR_FOUND)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_OPENXR XR_USE_PLATFORM_WIN32 XR_USE_GRAPHICS_API_VULKAN)
  if(TARGET OpenXR::openxr_loader)
    target_link_libraries(${PROJECT_NAME} PRIVATE OpenXR::openxr_loader)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE openxr_loader)
    target_include_directories(${PROJECT_NAME} PRIVATE ${openxr-sdk_SOURCE_DIR}/include)
  endif()
endif()

add_project_definitions(${PROJECT_NAME})

#####################################################################################
# copies files
#

# Copy shaders and required files
set(COPY_FILES ${Slang_GLSLANG})
if(ENABLE_DLSS_RR AND NGX_FOUND)
  list(APPEND COPY_FILES ${DLSS_DLLS})
endif()

copy_to_runtime_and_install(${PROJECT_NAME}
  DIRECTORIES "${NVSHADERS_DIR}/nvshaders"
  FILES ${COPY_FILES}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
  AUTO
)