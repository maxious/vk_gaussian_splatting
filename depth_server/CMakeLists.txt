cmake_minimum_required(VERSION 3.18)

file(GLOB DEPTH_SERVER_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

if(DEPTH_SERVER_SOURCES)
  add_executable(depth_server ${DEPTH_SERVER_SOURCES})
  target_link_libraries(depth_server PRIVATE depthanything)
  target_include_directories(depth_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  set_property(TARGET depth_server PROPERTY FOLDER "DepthServer")
else()
  message(STATUS "depth_server sources not present yet; integration scaffold only")
endif()
