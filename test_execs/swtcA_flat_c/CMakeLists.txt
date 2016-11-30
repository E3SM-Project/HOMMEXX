sweqx_flat_setup(0)

# Set the variables for this test executable
#                          NP  NC PLEV USE_PIO WITH_ENERGY QSIZE_D LINK_LANGUAGE
createTestExec(swtcA_flat_c sweqx_flat 8 4 4 FALSE FALSE 4 Fortran)

INCLUDE_DIRECTORIES(${SRC_BASE} ${SRC_SHARE} ${SRC_BASE}/kokkos_kernels)
