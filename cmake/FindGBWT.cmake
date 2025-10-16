find_path(GBWT_INCLUDE_DIR
  NAMES gbwt/gbwt.h
)
find_library(GBWT_LIBRARY
  NAMES gbwt
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GBWT
  DEFAULT_MSG
  GBWT_INCLUDE_DIR
  GBWT_LIBRARY
)
mark_as_advanced( GBWT_INCLUDE_DIR GBWT_LIBRARY )

if (GBWT_FOUND)
    set(GBWT_INCLUDE_DIRS ${GBWT_INCLUDE_DIR})
    set(GBWT_LIBRARIES ${GBWT_LIBRARY})
    if (NOT TARGET GBWT::GBWT)
        add_library(GBWT::GBWT UNKNOWN IMPORTED)
        set_target_properties(GBWT::GBWT PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GBWT_INCLUDE_DIRS}"
            IMPORTED_LOCATION "${GBWT_LIBRARY}"
        )
        # TODO: Add dependency on libbdsg
    endif()
endif()