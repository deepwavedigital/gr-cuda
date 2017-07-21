INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_CUDA cuda)

FIND_PATH(
    CUDA_INCLUDE_DIRS
    NAMES cuda/api.h
    HINTS $ENV{CUDA_DIR}/include
        ${PC_CUDA_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    CUDA_LIBRARIES
    NAMES gnuradio-cuda
    HINTS $ENV{CUDA_DIR}/lib
        ${PC_CUDA_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUDA DEFAULT_MSG CUDA_LIBRARIES CUDA_INCLUDE_DIRS)
MARK_AS_ADVANCED(CUDA_LIBRARIES CUDA_INCLUDE_DIRS)

