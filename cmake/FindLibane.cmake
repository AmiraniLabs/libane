# FindLibane.cmake — find_package(libane) support
#
# Usage:
#   find_package(libane REQUIRED)
#   target_link_libraries(mytarget PRIVATE Libane::libane)
#
# Variables set:
#   LIBANE_FOUND
#   LIBANE_INCLUDE_DIRS
#   LIBANE_LIBRARIES
#   LIBANE_VERSION

find_path(LIBANE_INCLUDE_DIR libane.h
    PATH_SUFFIXES include
    HINTS ${LIBANE_ROOT} $ENV{LIBANE_ROOT}
    PATHS /usr/local /opt/homebrew
)

find_library(LIBANE_LIBRARY
    NAMES ane
    PATH_SUFFIXES lib
    HINTS ${LIBANE_ROOT} $ENV{LIBANE_ROOT}
    PATHS /usr/local /opt/homebrew
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(libane
    REQUIRED_VARS LIBANE_LIBRARY LIBANE_INCLUDE_DIR
    VERSION_VAR   LIBANE_VERSION
)

if(LIBANE_FOUND AND NOT TARGET Libane::libane)
    add_library(Libane::libane UNKNOWN IMPORTED)
    set_target_properties(Libane::libane PROPERTIES
        IMPORTED_LOCATION             "${LIBANE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBANE_INCLUDE_DIR}"
    )
    if(APPLE)
        set_property(TARGET Libane::libane APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES
            "-framework Foundation" "-framework IOSurface"
            "-framework CoreFoundation" "-framework Accelerate"
        )
    endif()
endif()

mark_as_advanced(LIBANE_INCLUDE_DIR LIBANE_LIBRARY)

set(LIBANE_INCLUDE_DIRS ${LIBANE_INCLUDE_DIR})
set(LIBANE_LIBRARIES    ${LIBANE_LIBRARY})
