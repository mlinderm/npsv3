find_path(ODGI_INCLUDE_DIR
    NAMES odgi.hpp
)

# Link against static libraries for ODGI (adapted from ZLib).
# https://github.com/Kitware/CMake/blob/cadbf8fe40b1c65ac2e3da4fcdd5764a736731d4/Modules/FindZLIB.cmake#L186
# We need to use static libraries in our use case because ODGI uses header-only libraries (e.g., flat_hash_map) that don't work
# if linked into multiple shared libraries (references static symbols).

if(DEFINED CMAKE_FIND_LIBRARY_SUFFIXES)
    set(_ODGI_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
else()
    set(_ODGI_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
endif()
set(CMAKE_FIND_LIBRARY_SUFFIXES .a)

find_library(ODGI_LIBRARY
    NAMES odgi
)

# Restore the original find library ordering
if(DEFINED _ODGI_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_ODGI_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES}")
else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ODGI
    DEFAULT_MSG
    ODGI_INCLUDE_DIR
    ODGI_LIBRARY
)
mark_as_advanced( ODGI_INCLUDE_DIR ODGI_LIBRARY )

if (ODGI_FOUND)
    find_package(OpenMP REQUIRED)

    # We also need to find the include directories (and some libraries) for all relevant odgi dependencies
    find_path(HandleGraph_INCLUDE_DIR
        NAMES handlegraph/handle_graph.hpp
        PATHS "${ODGI_INCLUDE_DIR}/../deps/libhandlegraph/src/include"
        REQUIRED
        NO_DEFAULT_PATH
    )
    # TODO: Make this more robust, the "build" portion is a choice during ODGI compilation. Here we assume the CMAKE_LIBRARY_PATH or other variable
    # is set appropriately to help find it.
    find_library(HandleGraph_LIBRARY
        NAMES libhandlegraph.a
        HINTS "${ODGI_INCLUDE_DIR}/../build/handlegraph-prefix"
        PATH_SUFFIXES "lib"
        REQUIRED
    )

    find_path(Dynamic_INCLUDE_DIR
        NAMES dynamic.hpp
        PATHS "${ODGI_INCLUDE_DIR}/../deps/DYNAMIC/include"
        REQUIRED
        NO_DEFAULT_PATH
    )
    find_path(Hopscotch_INCLUDE_DIR
        NAMES tsl/hopscotch_map.h
        PATHS "${ODGI_INCLUDE_DIR}/../deps/hopscotch-map/include"
        REQUIRED
        NO_DEFAULT_PATH
    )
    find_path(Sparsepp_INCLUDE_DIR
        NAMES spp.h
        PATHS "${ODGI_INCLUDE_DIR}/../deps/sparsepp/sparsepp"
        REQUIRED
        NO_DEFAULT_PATH
    )
    find_path(Ska_INCLUDE_DIR
        NAMES bytell_hash_map.hpp
        PATHS "${ODGI_INCLUDE_DIR}/../deps/flat_hash_map"
        REQUIRED
        NO_DEFAULT_PATH
    )
     find_path(AtomicBitVector_INCLUDE_DIR
        NAMES atomic_bitvector.hpp
        PATHS "${ODGI_INCLUDE_DIR}/../deps/atomicbitvector/include"
        REQUIRED
        NO_DEFAULT_PATH
    )

    set(ODGI_INCLUDE_DIRS ${ODGI_INCLUDE_DIR} ${HandleGraph_INCLUDE_DIR} ${Dynamic_INCLUDE_DIR} ${Hopscotch_INCLUDE_DIR} ${Sparsepp_INCLUDE_DIR} ${Ska_INCLUDE_DIR} ${AtomicBitVector_INCLUDE_DIR})
    set(ODGI_LIBRARIES ${ODGI_LIBRARY})
    if (NOT TARGET ODGI::ODGI)
        add_library(ODGI::ODGI STATIC IMPORTED)
        set_target_properties(ODGI::ODGI PROPERTIES
            INCLUDE_DIRECTORIES "${ODGI_INCLUDE_DIR}"
            INTERFACE_INCLUDE_DIRECTORIES "${ODGI_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${ODGI_LIBRARY}"
            INTERFACE_LINK_LIBRARIES "${HandleGraph_LIBRARY}"
        )
        target_link_libraries(ODGI::ODGI INTERFACE OpenMP::OpenMP_CXX atomic)
    endif()
endif()