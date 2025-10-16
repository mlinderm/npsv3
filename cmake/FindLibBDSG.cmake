find_path(LibBDSG_INCLUDE_DIR
  NAMES bdsg/hash_graph.hpp
)
find_library(LibBDSG_LIBRARY
  NAMES bdsg
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibBDSG
  DEFAULT_MSG
  LibBDSG_INCLUDE_DIR
  LibBDSG_LIBRARY
)
mark_as_advanced( LibBDSG_INCLUDE_DIR LibBDSG_LIBRARY )
if (LibBDSG_FOUND)
    set(LibBDSG_INCLUDE_DIRS ${LibBDSG_INCLUDE_DIR})
    set(LibBDSG_LIBRARIES ${LibBDSG_LIBRARY})
    if(NOT TARGET LibBDSG::LibBDSG)
        add_library(LibBDSG::LibBDSG UNKNOWN IMPORTED)
        set_target_properties(LibBDSG::LibBDSG PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${LibBDSG_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${LibBDSG_LIBRARY}"
        )
    endif()
endif()