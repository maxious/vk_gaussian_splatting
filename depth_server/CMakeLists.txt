add_executable(depth_server
    main.cpp
    benchmark.cpp
    da_capi_buffer.cpp
    model_downloader.cpp
    protocol_parser.cpp
    tcp_server.cpp
    worker_pool.cpp
    worker_monitor.cpp
    splat_worker_pool.cpp
    splat_worker.cpp
    freesplatter_capi_buffer.cpp
)

target_include_directories(depth_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/depth-anything-cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../nvpro_core2
)

target_link_libraries(depth_server PRIVATE depthanything nvpro2::nvutils)
target_compile_features(depth_server PRIVATE cxx_std_20)

# Free-splatter integration (image-to-3DGS)
if(TARGET free_splatter)
    target_link_libraries(depth_server PRIVATE free_splatter)
    target_include_directories(depth_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/free-splatter/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/free-splatter/third_party/stb
    )
    target_compile_definitions(depth_server PRIVATE WITH_FREE_SPLATTER)
    target_sources(depth_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/stb_resize_impl.cpp)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/stb_resize_impl.cpp
        PROPERTIES COMPILE_OPTIONS "-w")
endif()

add_executable(test_da_capi_buffer
    test_da_capi_buffer.cpp
    da_capi_buffer.cpp
)

target_include_directories(test_da_capi_buffer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/depth-anything-cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../nvpro_core2
)

target_link_libraries(test_da_capi_buffer PRIVATE depthanything nvpro2::nvutils)
target_compile_features(test_da_capi_buffer PRIVATE cxx_std_20)

if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_definitions(depth_server PRIVATE NDEBUG)
    target_compile_definitions(test_da_capi_buffer PRIVATE NDEBUG)
endif()

set_property(TARGET depth_server PROPERTY FOLDER "DepthServer")
set_property(TARGET test_da_capi_buffer PROPERTY FOLDER "DepthServer")
