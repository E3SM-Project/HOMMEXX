SET (USE_KOKKOS_KERNELS TRUE)

PREQX_SETUP(${USE_KOKKOS_KERNELS})

# Set the variables for this test executable
#                          NP  NC PLEV USE_PIO WITH_ENERGY QSIZE_D LINK_LANGUAGE


MESSAGE(STATUS " in test_execs B_c : Building with QSIZE_D = ${QSIZE_D}")
MESSAGE(STATUS " in test_execs B_c : Building with PREQX_NP = ${PREQX_NP}")
MESSAGE(STATUS " in test_execs B_c : Building with PREQX_PLEV = ${PREQX_PLEV}")
MESSAGE(STATUS " in test_execs B_c : Building with PREQX_USE_PIO = ${PREQX_USE_PIO}")
MESSAGE(STATUS " in test_execs B_c : Building with PREQX_USE_ENERGY = ${PREQX_USE_ENERGY}")

createTestExec(prtcB_c preqx ${PREQX_NP} 4 ${PREQX_PLEV} ${PREQX_USE_PIO}  ${PREQX_USE_ENERGY} ${QSIZE_D} Fortran)

INCLUDE_DIRECTORIES(${PREQX_INCLUDE_DIRS})

# Copy the needed input files to the binary dir
CONFIGURE_FILE (${CMAKE_SOURCE_DIR}/test/reg_test/namelists/prtcB-r3-dry.nl
                ${CMAKE_CURRENT_BINARY_DIR}/namelist.nl)

FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/movies)

FILE (MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/vcoord)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/test/vcoord/acme-72i.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/vcoord COPYONLY)
CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/test/vcoord/acme-72m.ascii
               ${CMAKE_CURRENT_BINARY_DIR}/vcoord COPYONLY)
