cmake_minimum_required(VERSION 3.6...3.31)

# get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
set(PROJECT_NAME "vk_gaussian_splatting")
Project(${PROJECT_NAME} LANGUAGES C CXX)

# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

# set with -DDISABLE_DEFAULT_SCENE=ON on command line or use cmake-gui to set the parameter
# if default scene is not disabled at build using this option, one can still disable it by 
# providing "-loadDefaultScene 0" to the sample command line
option(DISABLE_DEFAULT_SCENE "If set to true the default scene is not downloaded by CMake, and the sample does not try to open it when starting up with no scene as parameter" OFF)

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
# Include Vulkan Radix Sort
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx)

set_property(TARGET vk_radix_sort PROPERTY FOLDER "ThirdParty/vrdx")
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
list(FILTER SOURCE_FILES EXCLUDE REGEX "gaussian_splatting_(render|shaders|pipelines|rtx|postprocess)\\.cpp$")
file(GLOB SHADER_FILES shaders/*.slang shaders/*.h)
file(GLOB EXTERN_FILES 3rdparty/miniply/*.*)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src 
  ${CMAKE_CURRENT_SOURCE_DIR}/shaders 
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/miniply
  ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/vrdx
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
)

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

copy_to_runtime_and_install(${PROJECT_NAME}
  DIRECTORIES "${NVSHADERS_DIR}/nvshaders"
  FILES ${Slang_GLSLANG}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders"
  AUTO
)