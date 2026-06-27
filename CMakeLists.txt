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
set(PROJECT_NAME "vk_viewer")
Project(${PROJECT_NAME} LANGUAGES C CXX)

# Enable PDB generation for all build types on MSVC
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG /OPT:REF /OPT:ICF")
endif()

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Configure Vulkan SDK before anything else
include(cmake/VulkanSDK.cmake)

# Override Slang version before NvproCore2 loads it
set(Slang_VERSION "2025.24.3" CACHE STRING "Slang version" FORCE)

find_package(NvproCore2 REQUIRED)

#####################################################################################
# Unit Testing with doctest
#
enable_testing()

message(STATUS "doctest testing framework enabled (header-only)")

# Add tests subdirectory if not building with CMake's --install or similar
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  add_subdirectory(tests)
endif()

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
# cnpy for NPZ format support (header-only, uses zlib)
# FetchContent or use existing zlib from nvpro_core2
message(STATUS "cnpy NPZ loader enabled (uses zlib)")

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
set(WEBP_NEAR_LOSSLESS OFF CACHE BOOL "" FORCE)
set(WEBP_UNICODE OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(libwebp)
set_property(TARGET webp PROPERTY FOLDER "ThirdParty/libwebp")
set_property(TARGET webpdecode PROPERTY FOLDER "ThirdParty/libwebp")
set_property(TARGET webpdspdecode PROPERTY FOLDER "ThirdParty/libwebp")
set_property(TARGET webputilsdecode PROPERTY FOLDER "ThirdParty/libwebp")
  message(STATUS "libwebp enabled for SOG format support")



#####
# Network libraries for HTTP downloads, ComfyUI and Depth Streaming
option(ENABLE_COMFYUI "Enable ComfyUI integration" ON)
option(ENABLE_DEPTH_STREAMING "Enable VideoDepthViewer3D depth streaming" ON)

# Configure IXWebSocket options
set(USE_TLS OFF CACHE BOOL "" FORCE) # Disable TLS to avoid OpenSSL/MbedTLS issues for now
set(USE_OPENSSL OFF CACHE BOOL "" FORCE)
set(USE_MBED_TLS OFF CACHE BOOL "" FORCE)
set(USE_ZLIB OFF CACHE BOOL "" FORCE) 
set(USE_WS ON CACHE BOOL "" FORCE)
set(USE_TEST OFF CACHE BOOL "" FORCE)
set(IXWEBSOCKET_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ixwebsocket)

if(ENABLE_COMFYUI)
  message(STATUS "ComfyUI integration enabled")
endif()
if(ENABLE_DEPTH_STREAMING)
  message(STATUS "Depth streaming enabled")
endif()

#####
# Include Vulkan Radix Sort (header-only library)
# Ensure Slang is found before vrdx (vrdx needs Slang_SLANGC_EXECUTABLE)
find_package(Slang REQUIRED)
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

#####
# Include fastgltf for efficient GLTF/GLB loading
set(FASTGLTF_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(FASTGLTF_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/fastgltf)
set_property(TARGET fastgltf PROPERTY FOLDER "ThirdParty")

#####################################################################################
# FFmpeg support for video decoding

#
option(ENABLE_VIDEO_DECODER "Enable FFmpeg-based video decoder for VDZ support" ON)

if(ENABLE_VIDEO_DECODER)
  find_package(FFmpeg QUIET)
  if(FFmpeg_FOUND)
    message(STATUS "FFmpeg found - enabling video decoder support")
    add_definitions(-DWITH_VIDEO_DECODER)
  else()
    message(STATUS "FFmpeg not found - video decoder will be disabled. Install FFmpeg DLLs and set FFMPEG_DIR environment variable.")
  endif()
endif()

#####################################################################################
# Vulkan Video support for hardware-accelerated video decoding
#
option(ENABLE_VULKAN_VIDEO "Enable Vulkan Video hardware-accelerated decoder (requires Vulkan 1.3+)" ON)

if(ENABLE_VULKAN_VIDEO)
  # Vulkan Video extensions are part of Vulkan SDK 1.3.230+
  # We just need to check if the headers are available
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${Vulkan_INCLUDE_DIRS})
  check_cxx_source_compiles("
    #include <vulkan/vulkan.h>
    int main() {
      VkVideoSessionKHR session;
      VkVideoDecodeInfoKHR decodeInfo;
      return 0;
    }
  " VULKAN_VIDEO_HEADERS_AVAILABLE)
  
  if(VULKAN_VIDEO_HEADERS_AVAILABLE AND FFmpeg_FOUND)
    message(STATUS "Vulkan Video headers found - enabling hardware-accelerated video decoder")
    add_definitions(-DWITH_VULKAN_VIDEO)
    set(WITH_VULKAN_VIDEO_ENABLED TRUE CACHE INTERNAL "Vulkan Video is enabled")
  else()
    set(WITH_VULKAN_VIDEO_ENABLED FALSE CACHE INTERNAL "Vulkan Video is disabled")
    if(NOT FFmpeg_FOUND)
      message(STATUS "Vulkan Video requires FFmpeg for bitstream parsing - disabled")
    else()
      message(STATUS "Vulkan Video headers not found (requires Vulkan SDK 1.3.230+) - disabled")
    endif()
  endif()
endif()

#####################################################################################
# TCP Depth Processing Server
#
option(ENABLE_TCP_DEPTH "Enable TCP depth processing server using depth-anything.cpp" OFF)
option(DA_GGML_CUDA "Enable CUDA backend for depth-anything.cpp GPU inference" OFF)

if(ENABLE_TCP_DEPTH)
  message(STATUS "TCP depth processing enabled")
  set(DA_BUILD_CLI OFF CACHE BOOL "" FORCE)
  set(DA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/depth-anything-cpp)
  set_property(TARGET depthanything PROPERTY FOLDER "ThirdParty")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/depth_server)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND AND DA_GGML_CUDA)
    message(STATUS "CUDA found for depth-anything.cpp backend")
  endif()
endif()

#####################################################################################
# Free-Splatter (image-to-3DGS integration)
#
option(ENABLE_FREESPLATTER "Enable free-splatter.cpp integration (image-to-3DGS) for depth_server" OFF)

if(ENABLE_FREESPLATTER)
  message(STATUS "Free-splatter integration enabled")
  set(FREE_SPLATTER_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(FREE_SPLATTER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(FREE_SPLATTER_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(FREE_SPLATTER_CUDA OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/free-splatter)
  set_property(TARGET free_splatter PROPERTY FOLDER "ThirdParty")
endif()

#####################################################################################
# Source files for this project
#
# Note: GLOB is used here, so if you add new files, you must re-run CMake or touch this file.
file(GLOB SOURCE_FILES src/*.*)

# Exclude ComfyUI files if disabled
if (NOT ENABLE_COMFYUI)
    list(FILTER SOURCE_FILES EXCLUDE REGEX "comfyui_(client|loader)\\.cpp$")
endif()

# Exclude Depth Streaming files if disabled
if (NOT ENABLE_DEPTH_STREAMING)
     list(FILTER SOURCE_FILES EXCLUDE REGEX "depth_(stream_client|parser|buffer|to_vk)\\.cpp$")
     list(FILTER SOURCE_FILES EXCLUDE REGEX "backend_process_manager\\.cpp$")
endif()

# Exclude TCP depth client/manager files if disabled
if (NOT ENABLE_TCP_DEPTH)
     list(FILTER SOURCE_FILES EXCLUDE REGEX "tcp_depth_client\\.cpp$")
     list(FILTER SOURCE_FILES EXCLUDE REGEX "tcp_server_manager\\.cpp$")
     list(FILTER SOURCE_FILES EXCLUDE REGEX "local_depth_server_manager\\.cpp$")
else()
     list(APPEND SOURCE_FILES depth_server/protocol_parser.cpp)
endif()

# Exclude free-splatter client/manager files if disabled
if (NOT ENABLE_FREESPLATTER)
     list(FILTER SOURCE_FILES EXCLUDE REGEX "free_splatter_client\\.cpp$")
     list(FILTER SOURCE_FILES EXCLUDE REGEX "local_splat_server_manager\\.cpp$")
endif()

# Exclude Video Decoder files if disabled
if (NOT ENABLE_VIDEO_DECODER OR NOT FFmpeg_FOUND)
     list(FILTER SOURCE_FILES EXCLUDE REGEX "video_decoder\\.cpp$")
endif()

# Note: src/nvvk_swapchain_override.cpp implements the HDR support for the swapchain,
# overriding the default implementation in nvpro_core2 without modifying the submodule.
# Remove "unity build" partial source files that are #included by vk_viewer.cpp and vk_viewer_ui.cpp
list(FILTER SOURCE_FILES EXCLUDE REGEX "vk_viewer_(render|shaders|pipelines|rtx|postprocess|dlss_rr|frame_ubo|sorting|multiview|video|frame|chunk_culling|stochasticgs)\\.cpp$")
list(FILTER SOURCE_FILES EXCLUDE REGEX "vk_viewer_ui_(renderer|depth|vr|project)\\.cpp$")
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
  nvpro2::nvvkgltf
  vk_radix_sort
  spz::spz
  meshoptimizer
  webp
  zlib
  fastgltf
)

target_include_directories(${PROJECT_NAME} PRIVATE ${libwebp_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core2/third_party/stb)


# Network libraries linkage for ComfyUI and Depth Streaming
if(ENABLE_COMFYUI)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_COMFYUI)
endif()
if(ENABLE_DEPTH_STREAMING)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_DEPTH_STREAMING)
endif()
if(ENABLE_TCP_DEPTH)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_TCP_DEPTH)
endif()
if(ENABLE_FREESPLATTER)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_FREE_SPLATTER)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ixwebsocket)
target_include_directories(${PROJECT_NAME} PRIVATE ${ixwebsocket_SOURCE_DIR})

# DLSS-RR linkage
if(ENABLE_DLSS_RR AND NGX_FOUND)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_DLSS_RR)
  target_link_libraries(${PROJECT_NAME} PRIVATE ngx)
endif()

# FFmpeg linkage for video decoder
if(ENABLE_VIDEO_DECODER AND FFmpeg_FOUND)
  # Create an interface library to properly link FFmpeg with correct order
  add_library(ffmpeg_lib INTERFACE IMPORTED)
  set_target_properties(ffmpeg_lib PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${AVCODEC_INCLUDE_DIR};${AVFORMAT_INCLUDE_DIR};${AVUTIL_INCLUDE_DIR};${SWSCALE_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${AVCODEC_LIBRARY};${AVFORMAT_LIBRARY};${AVUTIL_LIBRARY};${SWSCALE_LIBRARY}"
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE ffmpeg_lib)
endif()

# OpenXR linkage
if(ENABLE_OPENXR AND OpenXR_FOUND)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_OPENXR XR_USE_GRAPHICS_API_VULKAN)
  if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE XR_USE_PLATFORM_WIN32)
  else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE XR_USE_PLATFORM_XLIB)
  endif()
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
