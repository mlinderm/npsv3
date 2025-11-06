# To avoid library conflicts we use a static build of HTSLib, but need to modify the build to create
# a PIC static library that can be linked into a Python extension shared library. Thus we build HTSLib
# ourselves, even if it might be available system-wide.
# Adapted from: https://gitlab.com/robegan21/MetaBAT/-/blob/master/cmake/htslib.cmake
set(HTSLib_VERSION "1.21")

if (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    set(MAKE_COMMAND "$(MAKE)")
else()
    find_program(MAKE_COMMAND NAMES make gmake)
endif()

find_package(ZLIB REQUIRED)
find_package(BZip2)
find_package(LibLZMA)
find_package(CURL)
find_package(OpenSSL REQUIRED) # Not clear if actually required, but HTSLib uses it if available
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
    pkg_check_modules(deflate IMPORTED_TARGET libdeflate)
endif()

set(HTSLib_CONFIGURE_OPTIONS "")
set(HTSLib_DEPENDENCIES ZLIB::ZLIB OpenSSL::Crypto)
if (NOT BZip2_FOUND)
    set(HTSLib_CONFIGURE_OPTIONS "${HTSLib_CONFIGURE_OPTIONS} --disable-bz2")
else()
    list(APPEND HTSLib_DEPENDENCIES BZip2::BZip2)
endif()
if (NOT LibLZMA_FOUND)
    set(HTSLib_CONFIGURE_OPTIONS "${HTSLib_CONFIGURE_OPTIONS} --disable-lzma")
else()
    list(APPEND HTSLib_DEPENDENCIES LibLZMA::LibLZMA)
endif()
if (NOT CURL_FOUND)
    set(HTSLib_CONFIGURE_OPTIONS "${HTSLib_CONFIGURE_OPTIONS} --disable-libcurl")
else()
    list(APPEND HTSLib_DEPENDENCIES CURL::libcurl)
endif()
if (NOT TARGET PkgConfig::deflate)
    set(HTSLib_CONFIGURE_OPTIONS "${HTSLib_CONFIGURE_OPTIONS} --without-libdeflate")
else()
    list(APPEND HTSLib_DEPENDENCIES PkgConfig::deflate)
endif()

include(ExternalProject)
ExternalProject_Add(HTSLib
    # Downgrade to 1.21 due to this error in 1.22 (https://github.com/samtools/htslib/issues/1940)
    URL "https://github.com/samtools/htslib/releases/download/${HTSLib_VERSION}/htslib-${HTSLib_VERSION}.tar.bz2"
    PREFIX "${CMAKE_BINARY_DIR}/lib/htslib"
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ./configure "CFLAGS=-g -fPIC" --prefix=<INSTALL_DIR> ${HTSLib_CONFIGURE_OPTIONS}
    BUILD_COMMAND ${MAKE_COMMAND} lib-static
    INSTALL_COMMAND ${MAKE_COMMAND} install
    BUILD_IN_SOURCE true
    BUILD_BYPRODUCTS <INSTALL_DIR>/lib/libhts.a # Needed for ninja generator
)
ExternalProject_Get_Property(HTSLib install_dir)
set(HTSLib_INCLUDE_DIR "${install_dir}/include")
set(HTSLib_LIBRARY "${install_dir}/lib/libhts.a")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HTSLib
    REQUIRED_VARS HTSLib_INCLUDE_DIR HTSLib_LIBRARY
    VERSION_VAR HTSLib_VERSION
)
mark_as_advanced( HTSLib_INCLUDE_DIR HTSLib_LIBRARY )

# INTERFACE_INCLUDE_DIRECTORIES must be defined, so create the relevant directory if it doesn't exist
# https://stackoverflow.com/questions/45516209/cmake-how-to-use-interface-include-directories-with-externalproject
file(MAKE_DIRECTORY ${HTSLib_INCLUDE_DIR})
  
add_library(HTSLib::Static STATIC IMPORTED)
add_dependencies(HTSLib::Static HTSLib)
set_target_properties(HTSLib::Static PROPERTIES
    INCLUDE_DIRECTORIES "${HTSLib_INCLUDE_DIR}"
    INTERFACE_INCLUDE_DIRECTORIES "${HTSLib_INCLUDE_DIR}"
    IMPORTED_LOCATION "${HTSLib_LIBRARY}"
)
target_link_libraries(HTSLib::Static INTERFACE ${HTSLib_DEPENDENCIES})
