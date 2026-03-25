if (CMAKE_GENERATOR STREQUAL "Unix Makefiles")
    set(MAKE_COMMAND "$(MAKE)")
else()
    find_program(MAKE_COMMAND NAMES make gmake)
endif()

find_package(ZLIB REQUIRED)

include(ExternalProject)
ExternalProject_Add(kmc
  GIT_REPOSITORY https://github.com/refresh-bio/KMC.git
  GIT_TAG v3.2.4
  GIT_SHALLOW TRUE
  PREFIX "${CMAKE_BINARY_DIR}/lib/kmc"
  PATCH_COMMAND patch --strip=1 --forward --reject-file=- --input=${CMAKE_SOURCE_DIR}/patches/kmc-v3.2.4.patch || true
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${MAKE_COMMAND} kmc
  BUILD_IN_SOURCE true
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(kmc source_dir binary_dir)
  
add_library(KMC::Static IMPORTED STATIC)
set_target_properties(KMC::Static PROPERTIES
  IMPORTED_LOCATION "${binary_dir}/bin/libkmc_core.a"
  INTERFACE_INCLUDE_DIRECTORIES "${source_dir}"
)

