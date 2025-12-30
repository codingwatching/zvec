set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable testing in yaml-cpp" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable parse tools in yaml-cpp" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable contrib stuff in yaml-cpp" FORCE)

add_subdirectory(yaml-cpp-0.6.3)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(yaml-cpp PRIVATE -Wno-shadow)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(yaml-cpp PRIVATE -Wno-effc++)
endif()