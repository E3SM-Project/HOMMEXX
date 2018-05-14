SET (USE_KOKKOS_KERNELS FALSE)

PREQX_SETUP(${USE_KOKKOS_KERNELS})

# Set the variables for this test executable
#                          NP  NC PLEV USE_PIO WITH_ENERGY QSIZE_D LINK_LANGUAGE
createTestExec(prtcB preqx 4 4 72 FALSE FALSE 35 Fortran)

# Copy the needed input files to the binary dir
CONFIGURE_FILE (${CMAKE_SOURCE_DIR}/test/reg_test/namelists/prtcB-r3-dry.nl
                ${CMAKE_CURRENT_BINARY_DIR}/namelist.nl)

FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/movies)

FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vcoord)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/test/vcoord/acme-72m.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/vcoord COPYONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/test/vcoord/acme-72i.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/vcoord COPYONLY)
