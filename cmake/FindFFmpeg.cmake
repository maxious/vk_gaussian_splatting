# FindFFmpeg.cmake
#
# Find the FFmpeg libraries (avcodec, avformat, avutil, swscale, swresample)
#
# This module defines:
#   FFmpeg_FOUND        - True if FFmpeg is found
#   FFmpeg_LIBRARIES    - List of FFmpeg libraries to link against
#   FFmpeg_INCLUDE_DIRS - Include directories for FFmpeg headers
#
# Additionally, the following variables are defined for each library:
#   AVUTIL_FOUND, AVUTIL_INCLUDE_DIR, AVUTIL_LIBRARY, etc.
#
# Usage:
#   find_package(FFmpeg REQUIRED)
#   target_link_libraries(my_target ${FFmpeg_LIBRARIES})
#   target_include_directories(my_target ${FFmpeg_INCLUDE_DIRS})

include(FindPackageHandleStandardArgs)

# Function to find a specific FFmpeg library
function(find_ffmpeg_library _lib_name _header_name)
    string(TOUPPER ${_lib_name} _upper_name)

    # Try to find the library
    find_library(${_upper_name}_LIBRARY
        NAMES ${_lib_name}
        PATHS
            ${CMAKE_SOURCE_DIR}/ffmpeg-8.0.1-full_build-shared/lib
            $ENV{FFMPEG_DIR}/lib
            $ENV{FFMPEG_ROOT}/lib
            /usr/local/lib
            /usr/lib
            $ENV{ProgramFiles}/FFmpeg/lib
            $ENV{ProgramFiles\(x86\)}/FFmpeg/lib
        DOC "Path to the FFmpeg ${_lib_name} library"
    )

    # Try to find the header
    find_path(${_upper_name}_INCLUDE_DIR
        NAMES lib${_lib_name}/${_header_name}
        PATHS
            ${CMAKE_SOURCE_DIR}/ffmpeg-8.0.1-full_build-shared/include
            $ENV{FFMPEG_DIR}/include
            $ENV{FFMPEG_ROOT}/include
            /usr/local/include
            /usr/include
            $ENV{ProgramFiles}/FFmpeg/include
            $ENV{ProgramFiles\(x86\)}/FFmpeg/include
        DOC "Path to the FFmpeg ${_lib_name} header"
    )

    # Set found variable
    if(${_upper_name}_LIBRARY AND ${_upper_name}_INCLUDE_DIR)
        set(${_upper_name}_FOUND TRUE PARENT_SCOPE)
    else()
        set(${_upper_name}_FOUND FALSE PARENT_SCOPE)
    endif()

    # Export variables
    set(${_upper_name}_LIBRARY ${${_upper_name}_LIBRARY} PARENT_SCOPE)
    set(${_upper_name}_INCLUDE_DIR ${${_upper_name}_INCLUDE_DIR} PARENT_SCOPE)
endfunction()

# Find individual FFmpeg libraries
find_ffmpeg_library(avutil avutil.h)
find_ffmpeg_library(avcodec avcodec.h)
find_ffmpeg_library(avformat avformat.h)
find_ffmpeg_library(swscale swscale.h)
find_ffmpeg_library(swresample swresample.h)

# Set FFmpeg variables
set(FFmpeg_LIBRARIES "")
set(FFmpeg_INCLUDE_DIRS "")

if(AVUTIL_FOUND)
    list(APPEND FFmpeg_LIBRARIES ${AVUTIL_LIBRARY})
    list(APPEND FFmpeg_INCLUDE_DIRS ${AVUTIL_INCLUDE_DIR})
endif()

if(AVCODEC_FOUND)
    list(APPEND FFmpeg_LIBRARIES ${AVCODEC_LIBRARY})
    list(APPEND FFmpeg_INCLUDE_DIRS ${AVCODEC_INCLUDE_DIR})
endif()

if(AVFORMAT_FOUND)
    list(APPEND FFmpeg_LIBRARIES ${AVFORMAT_LIBRARY})
    list(APPEND FFmpeg_INCLUDE_DIRS ${AVFORMAT_INCLUDE_DIR})
endif()

if(SWSCALE_FOUND)
    list(APPEND FFmpeg_LIBRARIES ${SWSCALE_LIBRARY})
    list(APPEND FFmpeg_INCLUDE_DIRS ${SWSCALE_INCLUDE_DIR})
endif()

if(SWRESAMPLE_FOUND)
    list(APPEND FFmpeg_LIBRARIES ${SWRESAMPLE_LIBRARY})
    list(APPEND FFmpeg_INCLUDE_DIRS ${SWRESAMPLE_INCLUDE_DIR})
endif()

# Remove duplicates from include directories
list(REMOVE_DUPLICATES FFmpeg_INCLUDE_DIRS)

# Handle the standard arguments
find_package_handle_standard_args(FFmpeg
    REQUIRED_VARS FFmpeg_LIBRARIES FFmpeg_INCLUDE_DIRS
    VERSION_VAR FFmpeg_VERSION
)

# Mark variables as advanced
mark_as_advanced(
    FFmpeg_LIBRARIES
    FFmpeg_INCLUDE_DIRS
    AVUTIL_LIBRARY AVUTIL_INCLUDE_DIR
    AVCODEC_LIBRARY AVCODEC_INCLUDE_DIR
    AVFORMAT_LIBRARY AVFORMAT_INCLUDE_DIR
    SWSCALE_LIBRARY SWSCALE_INCLUDE_DIR
    SWRESAMPLE_LIBRARY SWRESAMPLE_INCLUDE_DIR
)