# CMakeLists.txt for unit tests using doctest
cmake_minimum_required(VERSION 3.22)

# Test executable
add_executable(unit_tests
    main.cpp
    test_depth.cpp
    test_vdz_sequence.cpp
)

# Add source files we need to test
target_sources(unit_tests PRIVATE
    ../src/depth_parser.cpp
    ../src/vdz_sequence_loader.cpp
)

# Include directories for src files and tests
target_include_directories(unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/miniply
    ${CMAKE_CURRENT_SOURCE_DIR}/../nvpro_core2/third_party
)

# Link nvpro_core2 utilities for logger support
target_link_libraries(unit_tests PRIVATE nvpro2::nvutils)

# Link zlib for depth_parser (compression support)
# Use zlib from nvpro_core2, don't find_package again
target_link_libraries(unit_tests PRIVATE zlib)

# C++20 required
target_compile_features(unit_tests PRIVATE cxx_std_20)

# Register with CTest
add_test(NAME unit_tests COMMAND unit_tests)
