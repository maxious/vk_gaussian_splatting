add_library(ovoxel STATIC
    ovoxel_unity.cpp
)

target_include_directories(ovoxel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
