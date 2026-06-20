add_executable(depth_server
    main.cpp
)

target_include_directories(depth_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/depth-anything-cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../nvpro_core2
)

target_link_libraries(depth_server PRIVATE depthanything nvpro2::nvutils)
target_compile_features(depth_server PRIVATE cxx_std_20)

if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(depth_server PRIVATE NDEBUG)
endif()

set_property(TARGET depth_server PROPERTY FOLDER "DepthServer")
