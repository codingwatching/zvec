include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

cc_proto_library(
    NAME zvec_proto STATIC
    SRCS proto/*.proto
    PROTOROOT ./
  )

cc_directory(common)
cc_directory(index)
cc_directory(sqlengine)

cc_library(
    NAME zvec STATIC STRICT
    SRCS *.cc
    LIBS zvec_common
         zvec_index
         zvec_sqlengine
    INCS .
    VERSION "${PROXIMA_ZVEC_VERSION}"
)