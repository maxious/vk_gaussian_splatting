# Vulkan SDK Toolchain Configuration
# This ensures CMake uses the Vulkan SDK instead of system Vulkan headers

# Set Vulkan SDK paths if available
if(DEFINED ENV{VULKAN_SDK})
    set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
else()
    # Auto-detect latest Vulkan SDK under /opt/vulkan
    file(GLOB VULKAN_SDK_VERSIONS LIST_DIRECTORIES true "/opt/vulkan/1.*")
    list(SORT VULKAN_SDK_VERSIONS)
    list(REVERSE VULKAN_SDK_VERSIONS)
    foreach(_sdk_dir ${VULKAN_SDK_VERSIONS})
        if(EXISTS "${_sdk_dir}/x86_64")
            set(VULKAN_SDK_PATH "${_sdk_dir}/x86_64")
            break()
        elseif(EXISTS "${_sdk_dir}/setup-env.sh")
            set(VULKAN_SDK_PATH "${_sdk_dir}")
            break()
        endif()
    endforeach()
endif()

if(DEFINED VULKAN_SDK_PATH)
    message(STATUS "Using Vulkan SDK at: ${VULKAN_SDK_PATH}")

    # Set environment variables for Vulkan discovery
    set(ENV{VULKAN_SDK} "${VULKAN_SDK_PATH}")
    set(ENV{PATH} "${VULKAN_SDK_PATH}/bin:$ENV{PATH}")
    set(ENV{LD_LIBRARY_PATH} "${VULKAN_SDK_PATH}/lib:$ENV{LD_LIBRARY_PATH}")

    # Force CMake to find Vulkan SDK headers and libraries
    set(CMAKE_PREFIX_PATH "${VULKAN_SDK_PATH}" ${CMAKE_PREFIX_PATH})
    set(Vulkan_INCLUDE_DIRS "${VULKAN_SDK_PATH}/include" CACHE PATH "Vulkan include directory" FORCE)
    set(Vulkan_LIBRARY "${VULKAN_SDK_PATH}/lib/libvulkan.so" CACHE FILEPATH "Vulkan library" FORCE)

    # Additional Vulkan components
    set(Vulkan_GLSLANG_VALIDATOR_EXECUTABLE "${VULKAN_SDK_PATH}/bin/glslangValidator" CACHE FILEPATH "glslangValidator executable" FORCE)
    set(Vulkan_GLSLC_EXECUTABLE "${VULKAN_SDK_PATH}/bin/glslc" CACHE FILEPATH "glslc executable" FORCE)

    message(STATUS "Vulkan SDK configuration:")
    message(STATUS "  Include: ${Vulkan_INCLUDE_DIRS}")
    message(STATUS "  Library: ${Vulkan_LIBRARY}")
    message(STATUS "  glslangValidator: ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}")
    message(STATUS "  glslc: ${Vulkan_GLSLC_EXECUTABLE}")
else()
    message(WARNING "Vulkan SDK not found. Please install Vulkan SDK and set VULKAN_SDK environment variable.")
endif()