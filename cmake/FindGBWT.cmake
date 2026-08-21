# Finds or builds GBWT (jltsiren/gbwt) plus the sdsl-lite fork it requires.
#
# gbwt needs sdsl/simple_sds.hpp, which exists only in the vgteam/sdsl-lite fork, not the
# simongog/sdsl-lite copy ODGI already vendors. First try to find an existing installation,
# otherwise vendor both packages from source using an out-of-process ExternalProject build for sdsl-lite
# (so its own CMakeLists.txt never shares this project's CMake/target namespace, avoiding a collision
# with this project's own FetchContent'd googletest), then compiles gbwt's library sources directly as
# a CMake target against it.
#
# Requires ExternalProject and FetchContent already include()'d, and Threads::Threads already available,
# by the including project (true for this repo's own CMakeLists.txt).
#
# Defines the GBWT::GBWT target (include dirs + link libraries) either way; sets GBWT_FOUND.

find_path(GBWT_INCLUDE_DIR
  NAMES gbwt/gbwt.h
)

# Link against a static library for GBWT, same reason and pattern as FindODGI.cmake: without this,
# find_library() would happily match a shared libgbwt.so over libgbwt.a on a system that has both
# (CMAKE_FIND_LIBRARY_SUFFIXES defaults to ".so;.a" on Linux, shared preferred), which would silently
# diverge from the vendored fallback below (always built STATIC) and from how every other native
# dependency in this project is linked.
if(DEFINED CMAKE_FIND_LIBRARY_SUFFIXES)
    set(_GBWT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
else()
    set(_GBWT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES .a)

find_library(GBWT_LIBRARY
  NAMES gbwt
)

# Restore the original find library ordering
if(DEFINED _GBWT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_GBWT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES}")
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES)
endif()
unset(_GBWT_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)

include(FindPackageHandleStandardArgs)
# Suppress REQUIRED-ness for this internal system probe: this module has its own source-build fallback
# below, so a missing system copy must not be treated as a hard failure here.
set(_GBWT_FIND_REQUIRED_SAVED ${GBWT_FIND_REQUIRED})
set(GBWT_FIND_REQUIRED FALSE)
find_package_handle_standard_args(GBWT
  DEFAULT_MSG
  GBWT_INCLUDE_DIR
  GBWT_LIBRARY
)
set(GBWT_FIND_REQUIRED ${_GBWT_FIND_REQUIRED_SAVED})
unset(_GBWT_FIND_REQUIRED_SAVED)
mark_as_advanced(GBWT_INCLUDE_DIR GBWT_LIBRARY)

find_package(OpenMP REQUIRED) # gbwt's parallel construction code needs it (-fopenmp -pthread)

if (GBWT_FOUND)
  set(GBWT_INCLUDE_DIRS ${GBWT_INCLUDE_DIR})
  set(GBWT_LIBRARIES ${GBWT_LIBRARY})
  if (NOT TARGET GBWT::GBWT)
    add_library(GBWT::GBWT UNKNOWN IMPORTED)
    set_target_properties(GBWT::GBWT PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GBWT_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${GBWT_LIBRARY}"
      INTERFACE_LINK_LIBRARIES "OpenMP::OpenMP_CXX;Threads::Threads"
    )
    # A system-installed gbwt is expected to already have its own sdsl-lite (the vgteam fork) and
    # divsufsort on the linker's search path (matching gbwt's own install layout); if that's ever not
    # the case for a given system install, link them explicitly here too.
  endif()
else()
  message(STATUS "GBWT not found, vendoring vgteam/sdsl-lite + jltsiren/gbwt v1.4...")

  set(SDSL_LITE_INSTALL_DIR "${CMAKE_BINARY_DIR}/lib/sdsl-lite-install")
  ExternalProject_Add(
    SdslLite
    GIT_REPOSITORY https://github.com/vgteam/sdsl-lite.git
    PREFIX "${CMAKE_BINARY_DIR}/lib/sdsl-lite"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    # sdsl-lite's own install.sh runs `cmake .. && make sdsl && make install` in its build/ directory;
    # passing CXXFLAGS=-fPIC (same reason as this project's Boost fallback's cxxflags="-fPIC") so the
    # resulting static libs can link into the _native_graph shared module. `make install` builds `all`
    # first (standard CMake/Make generator behavior), which is why libdivsufsort/libdivsufsort64 end up
    # installed into the prefix alongside libsdsl even though only libsdsl has an explicit install() rule
    # of its own -- confirmed by direct build-and-run during PR A development, not assumed.
    BUILD_COMMAND ${CMAKE_COMMAND} -E env CXXFLAGS=-fPIC CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
                  <SOURCE_DIR>/install.sh "${SDSL_LITE_INSTALL_DIR}"
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS
      "${SDSL_LITE_INSTALL_DIR}/lib/libsdsl.a"
      "${SDSL_LITE_INSTALL_DIR}/lib/libdivsufsort.a"
      "${SDSL_LITE_INSTALL_DIR}/lib/libdivsufsort64.a"
  )

  # INTERFACE_INCLUDE_DIRECTORIES must exist at configure time for an IMPORTED target, same requirement
  # this project's Boost fallback works around.
  file(MAKE_DIRECTORY "${SDSL_LITE_INSTALL_DIR}/include")

  add_library(SdslLite::sdsl STATIC IMPORTED)
  add_dependencies(SdslLite::sdsl SdslLite)
  set_target_properties(SdslLite::sdsl PROPERTIES
    IMPORTED_LOCATION "${SDSL_LITE_INSTALL_DIR}/lib/libsdsl.a"
    INTERFACE_INCLUDE_DIRECTORIES "${SDSL_LITE_INSTALL_DIR}/include"
  )
  add_library(SdslLite::divsufsort STATIC IMPORTED)
  add_dependencies(SdslLite::divsufsort SdslLite)
  set_target_properties(SdslLite::divsufsort PROPERTIES
    IMPORTED_LOCATION "${SDSL_LITE_INSTALL_DIR}/lib/libdivsufsort.a"
  )
  add_library(SdslLite::divsufsort64 STATIC IMPORTED)
  add_dependencies(SdslLite::divsufsort64 SdslLite)
  set_target_properties(SdslLite::divsufsort64 PROPERTIES
    IMPORTED_LOCATION "${SDSL_LITE_INSTALL_DIR}/lib/libdivsufsort64.a"
  )

  # Build gbwt library sources directly as a normal CMake target, sourced from a fetched (but never CMake-configured)
  # copy of its repository. Deliberately bypass gbwt's own Makefile/install.sh, which assume a sibling ../sdsl-lite
  # checkout laid out exactly like gbwt's own repo rather than an installed prefix. This exact 12-source file list
  # matches gbwt's Makefile (except test.cpp which is not needed).
  FetchContent_Declare(
    gbwt_source
    GIT_REPOSITORY https://github.com/jltsiren/gbwt.git
    GIT_TAG v1.4
  )
  FetchContent_Populate(gbwt_source)

  add_library(gbwt STATIC
    "${gbwt_source_SOURCE_DIR}/src/algorithms.cpp"
    "${gbwt_source_SOURCE_DIR}/src/bwtmerge.cpp"
    "${gbwt_source_SOURCE_DIR}/src/cached_gbwt.cpp"
    "${gbwt_source_SOURCE_DIR}/src/dynamic_gbwt.cpp"
    "${gbwt_source_SOURCE_DIR}/src/fast_locate.cpp"
    "${gbwt_source_SOURCE_DIR}/src/files.cpp"
    "${gbwt_source_SOURCE_DIR}/src/gbwt.cpp"
    "${gbwt_source_SOURCE_DIR}/src/internal.cpp"
    "${gbwt_source_SOURCE_DIR}/src/metadata.cpp"
    "${gbwt_source_SOURCE_DIR}/src/support.cpp"
    "${gbwt_source_SOURCE_DIR}/src/utils.cpp"
    "${gbwt_source_SOURCE_DIR}/src/variants.cpp"
  )
  set_target_properties(gbwt PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gbwt PUBLIC "${gbwt_source_SOURCE_DIR}/include")
  target_link_libraries(gbwt
    PUBLIC
    SdslLite::sdsl
    SdslLite::divsufsort
    SdslLite::divsufsort64
    OpenMP::OpenMP_CXX
    Threads::Threads
  )

  add_library(GBWT::GBWT ALIAS gbwt)
  set(GBWT_FOUND TRUE)
endif()
