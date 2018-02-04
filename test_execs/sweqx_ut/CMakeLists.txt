
sweqx_setup(FALSE)

SET(SWEQX_CXX_INCLUDE ${HOMME_SOURCE_DIR}/src/kokkos_kernels
  ${HOMME_SOURCE_DIR}/utils/timing)

SET(CONFIG_DEFINES PIO_INTERP NC=4 NP=4 PLEV=4 QSIZE_D=4 TIMING=1 _MPI=1 _PRESTART=1 _WK_GRAD=1 HOMME_QUAD_PREC=1 HOMME_USE_FLAT_ARRAYS=1)

cxx_unit_test(sweqx_flat_ut "${SWEQX_UT_SRCS}" "${SWEQX_CXX_INCLUDE}" "${CONFIG_DEFINES}")

SET(input_files
  divergence_sphere_np4.in
  divergence_sphere_np8.in
  gradient_sphere_np4.in
  gradient_sphere_np8.in
  vorticity_sphere_np4.in
  vorticity_sphere_np8.in
)

add_input_files(sweqx_flat_ut
  "${HOMME_SOURCE_DIR}/test/unit_tests/inputs"
  "${input_files}"
)
