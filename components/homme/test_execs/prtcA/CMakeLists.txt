SET (FORTRAN_LOOPS TRUE)
SET (HOMME_USE_FLAT_ARRAYS FALSE)

PREQX_SETUP(${FORTRAN_LOOPS})

# Set the variables for this test executable
#                          NP  NC PLEV USE_PIO WITH_ENERGY QSIZE_D LINK_LANGUAGE
createTestExec(prtcA preqx 4 4 26 FALSE FALSE 4 Fortran)

# Copy the needed input files to the binary dir
CONFIGURE_FILE (${CMAKE_SOURCE_DIR}/test/reg_test/namelists/prtcA-r3-dry.nl
               ${CMAKE_CURRENT_BINARY_DIR}/namelist.nl
                COPYONLY)

FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vcoord)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/test/vcoord/camm-26.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/vcoord COPYONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/test/vcoord/cami-26.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/vcoord COPYONLY)
