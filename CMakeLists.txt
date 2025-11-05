cmake_minimum_required(VERSION 3.15...3.27)
project(npsv3 CXX)
#set(CMAKE_FIND_DEBUG_MODE 1)
# Nanobind configuration adapted from: https://nanobind.readthedocs.io/en/latest/packaging.html
if (NOT SKBUILD)
  message(WARNING "\
  This CMake file is meant to be executed using 'scikit-build-core'.
  Running it directly will almost certainly not produce the desired
  result. If you are a user trying to install this package, use the
  command below, which will install all necessary build dependencies,
  compile the package in an isolated environment, and then install it.
  =====================================================================
   $ pip install .
  =====================================================================
  If you are a software developer, and this is your own package, then
  it is usually much more efficient to install the build dependencies
  in your environment once and use the following command that avoids
  a costly creation of a new virtual environment at every compilation:
  =====================================================================
   $ pip install nanobind scikit-build-core[pyproject]
   $ pip install --no-build-isolation -ve .
  =====================================================================
  You may optionally add -Ceditable.rebuild=true to auto-rebuild when
  the package is imported. Otherwise, you need to rerun the above
  after editing C++ files.")
endif()

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 REQUIRED COMPONENTS Interpreter ${DEV_MODULE} OPTIONAL_COMPONENTS Development.SABIModule)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(nanobind CONFIG REQUIRED)

include(FetchContent)
include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set global properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get external dependencies
# TODO: Move this into FindSeqLib.cmake
find_package(LibLZMA REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)

find_package(SeqLib REQUIRED) # This will also patch SeqLib as needed
find_package(HTSLib 1.17 REQUIRED)
find_package(ODGI REQUIRED)

# Find (or build) Boost 
find_package(Boost 1.88 HINTS "${CMAKE_BINARY_DIR}/lib/boost")
if (NOT Boost_FOUND)
  message(STATUS "Boost not found, downloading and building Boost...")
  ExternalProject_Add(
		Boost
    URL https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.bz2
    URL_HASH SHA256=85a33fa22621b4f314f8e85e1a5e2a9363d22e4f4992925d4bb3bc631b5a0c7a
    PREFIX "${CMAKE_BINARY_DIR}/lib/boost"
		UPDATE_COMMAND ""
		CONFIGURE_COMMAND ./bootstrap.sh --prefix=<INSTALL_DIR>
		BUILD_COMMAND ./b2 -d0 link=static --prefix=<INSTALL_DIR> --with-flyweight --with-hash2 install
		BUILD_IN_SOURCE true
		INSTALL_COMMAND ""
	)
  ExternalProject_Get_Property(Boost install_dir)
  set(Boost_FOUND true)
  set(Boost_INCLUDE_DIRS "${install_dir}/include")
  
  # INTERFACE_INCLUDE_DIRECTORIES must be defined, so create the relevant directory if it doesn't exist
  # https://stackoverflow.com/questions/45516209/cmake-how-to-use-interface-include-directories-with-externalproject
  file(MAKE_DIRECTORY ${Boost_INCLUDE_DIRS})

  add_library(Boost::boost INTERFACE IMPORTED)
  target_include_directories(Boost::boost INTERFACE ${Boost_INCLUDE_DIRS})
endif()

include(FetchContent)

if(DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
    set(_ORIG_CMAKE_POSITION_INDEPENDENT_CODE "${CMAKE_POSITION_INDEPENDENT_CODE}")
else()
    set(_ORIG_CMAKE_POSITION_INDEPENDENT_CODE)
endif()

# Build fmt with PIC enabled so it can be linked into the Python extension shared library
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        e424e3f2e607da02742f73db84873b8084fc714c # 12.0.0
)
FetchContent_MakeAvailable(fmt)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG        v1.16.0
)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt library instead of bundled") 
FetchContent_MakeAvailable(spdlog)

# Restore the original CMAKE_POSITION_INDEPENDENT_CODE ordering
if(DEFINED _ORIG_CMAKE_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE "${_ORIG_CMAKE_POSITION_INDEPENDENT_CODE}")
else()
    set(CMAKE_POSITION_INDEPENDENT_CODE)
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Set source directory
set(SOURCE_DIR "src/npsv3")
set(TEST_DIR "tests")

# We construct two separate modules because SeqLib and our own native code use inconsistent versions of HTSlib
nanobind_add_module(_native_realign STABLE_ABI 
  "${SOURCE_DIR}/native/realigner_bindings.cpp"
  "${SOURCE_DIR}/native/realigner.cpp"
)
target_include_directories(_native_realign
  PRIVATE
  "${SOURCE_DIR}/native"
  ${SEQLIB_INCLUDE_DIRS}
) 
target_link_libraries(_native_realign
  PRIVATE
  ${ZLIB_LIBRARIES}
  ${LIBLZMA_LIBRARIES}
  ${BZIP2_LIBRARIES}
  Threads::Threads
  ${SEQLIB_LIBRARIES}
)

add_library(graph STATIC
  "${SOURCE_DIR}/native/range.cpp"
  "${SOURCE_DIR}/native/fasta.cpp"
  "${SOURCE_DIR}/native/variant.cpp"
  "${SOURCE_DIR}/native/graph.cpp"
)
set_target_properties(graph PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(graph
  PUBLIC
  "${SOURCE_DIR}/native"
)
target_link_libraries(graph
  PUBLIC
  HTSLib::Static
  ODGI::ODGI
  Boost::boost
  fmt::fmt
  spdlog::spdlog
)

nanobind_add_module(_native_graph STABLE_ABI 
  "${SOURCE_DIR}/native/graph_bindings.cpp"
)
target_link_libraries(_native_graph
  PRIVATE
  graph
)

# Install directive for scikit-build-core, including installing additional scripts 
# (in place of using setup.py scripts)
install(TARGETS _native_realign LIBRARY DESTINATION npsv3)
install(TARGETS _native_graph LIBRARY DESTINATION npsv3)
install(PROGRAMS scripts/synthBAM DESTINATION ${SKBUILD_SCRIPTS_DIR})

# C++ unit tests using GoogleTest
FetchContent_Declare(
  googletest
  # Release 1.17.0
  URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
)

set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of googletest" FORCE)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(graph_test
  "${TEST_DIR}/native/test_variant.cpp"
  "${TEST_DIR}/native/test_graph.cpp"
)
target_link_libraries(graph_test
  GTest::gtest_main
  graph
  HTSLib::Static
  spdlog::spdlog
)

include(GoogleTest)
gtest_discover_tests(graph_test)