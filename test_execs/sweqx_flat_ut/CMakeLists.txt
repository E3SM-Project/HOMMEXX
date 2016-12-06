
sweqx_flat_setup(FALSE)
cxx_unit_test(sweqx_flat_ut "${SWEQX_FLAT_UT_SRCS}")

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
