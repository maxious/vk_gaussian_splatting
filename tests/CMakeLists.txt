# CMakeLists.txt for unit tests using doctest
cmake_minimum_required(VERSION 3.22)

# Test executable
add_executable(unit_tests
    main.cpp
    test_depth.cpp
    test_tcp_protocol.cpp
    test_tcp_depth_client.cpp
    test_npz_loader.cpp
    test_lod_loader.cpp
    test_morton_order.cpp
    test_stochastic_rendering.cpp
    test_pbr_helpers.cpp
    test_ply_material.cpp
)

# Add source files we need to test
target_sources(unit_tests PRIVATE
    ../src/depth_parser.cpp
    ../depth_server/protocol_parser.cpp
    ../src/tcp_depth_client.cpp
    ../src/tcp_server_manager.cpp
    depth_buffer_stub.cpp
    ../src/npz_loader.cpp
    ../src/lod_loader.cpp
    ../src/sog_loader.cpp
)

# Include directories for src files and tests
target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../depth_server
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/miniply
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/spz/src/cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../nvpro_core2/third_party
)

# Link nvpro_core2 utilities for logger support
target_link_libraries(unit_tests PRIVATE nvpro2::nvutils)

# Link zlib for depth_parser (compression support)
# Use zlib from nvpro_core2, don't find_package again
target_link_libraries(unit_tests PRIVATE zlib)

# Link webp for sog_loader
target_link_libraries(unit_tests PRIVATE webp)

# C++20 required
target_compile_features(unit_tests PRIVATE cxx_std_20)

target_compile_definitions(unit_tests PRIVATE WITH_TCP_DEPTH)

# Disable assertions in Release builds to prevent SIGTRAP
if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(unit_tests PRIVATE NDEBUG)
endif()

# Register with CTest
add_test(NAME unit_tests COMMAND unit_tests)

# PBR screenshot test - requires GPU with ray tracing, disabled in CI
add_test(NAME pbr_screenshot_test
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_pbr_screenshot_test.sh $<CONFIG>
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(pbr_screenshot_test PROPERTIES DISABLED TRUE)

if(ENABLE_TCP_DEPTH)
    add_test(NAME tcp_depth_e2e 
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/integration/test_tcp_depth_e2e.sh $<TARGET_FILE_DIR:vk_viewer>)
endif()

if(ENABLE_FREESPLATTER)
    add_executable(test_splat_client
        ../src/free_splatter_client.cpp
        test_splat_client.cpp
    )
    target_link_libraries(test_splat_client PRIVATE pthread nvpro2::nvutils)
    target_include_directories(test_splat_client PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    target_compile_definitions(test_splat_client PRIVATE WITH_FREE_SPLATTER)
    target_compile_features(test_splat_client PRIVATE cxx_std_20)
    set_property(TARGET test_splat_client PROPERTY FOLDER "DepthServer")
endif()
