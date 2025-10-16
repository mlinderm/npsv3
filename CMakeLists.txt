cmake_minimum_required(VERSION 3.15...3.27)
project(npsv3 CXX)

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

include(ExternalProject)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Set global properties
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get external dependencies
find_package(LibLZMA REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)
find_package(SeqLib REQUIRED) # This will also patch SeqLib as needed

find_package(LibBDSG REQUIRED)
find_package(GBWT REQUIRED)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Set source directory
set(SOURCE_DIR "src/npsv3")

include_directories(${SOURCE_DIR} ${SEQLIB_INCLUDE_DIRS})
set(SOURCES "${SOURCE_DIR}/bindings.cpp" "${SOURCE_DIR}/realigner.cpp")

# Add bindings here
nanobind_add_module(_native STABLE_ABI ${SOURCES})

target_link_libraries(_native
  PRIVATE
  ${SEQLIB_LIBRARIES}
  ${ZLIB_LIBRARIES}
  ${LIBLZMA_LIBRARIES}
  ${BZIP2_LIBRARIES}
  Threads::Threads
  LibBDSG::LibBDSG
  GBWT::GBWT
)

# Install directive for scikit-build-core, including installing additional scripts 
# (in place of using setup.py scripts)
install(TARGETS _native LIBRARY DESTINATION npsv3)
install(PROGRAMS scripts/synthBAM DESTINATION ${SKBUILD_SCRIPTS_DIR})