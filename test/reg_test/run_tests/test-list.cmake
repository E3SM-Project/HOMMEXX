# Lists of test files for the HOMME regression tests
SET(HOMME_TESTS
  preqx-nlev26-qsize4-r0-dry
  preqx-nlev26-qsize4-r3-dry
  preqx-nlev26-qsize4-r0-dry-kokkos
  preqx-nlev26-qsize4-r3-dry-kokkos
  preqx-nlev26-qsize4-r0-moist
  preqx-nlev26-qsize4-r3-moist
  preqx-nlev26-qsize4-r0-moist-kokkos
  preqx-nlev26-qsize4-r3-moist-kokkos
  preqx-nlev72-qsize4-r0-dry
  preqx-nlev72-qsize4-r3-dry
  preqx-nlev72-qsize4-r0-dry-kokkos
  preqx-nlev72-qsize4-r3-dry-kokkos
  preqx-nlev72-qsize4-r0-moist
  preqx-nlev72-qsize4-r3-moist
  preqx-nlev72-qsize4-r0-moist-kokkos
  preqx-nlev72-qsize4-r3-moist-kokkos
  preqx-nlev72-qsize4-r3-q6-dry
  preqx-nlev72-qsize4-r3-q6-dry-kokkos
  preqx-nlev72-qsize4-r3-tensorhv-dry
  preqx-nlev72-qsize4-r3-tensorhv-dry-kokkos
  preqx-nlev72-qsize4-r3-nudiv-dry
  preqx-nlev72-qsize4-r3-nudiv-dry-kokkos
  preqx-nlev72-qsize10-r3-lim9-dry
  preqx-nlev72-qsize10-r3-lim9-dry-kokkos
)

#This list (COMPARE_F_C_TEST) contains tests for which
#F vc C comparison will be run.
#Note: we run both a cprnc on the output nc files AND
#      a comparison of the values of diagnostic quantities
#      on the raw output files
SET (COMPARE_F_C_TEST
  preqx-nlev26-qsize4-r0-dry
  preqx-nlev26-qsize4-r3-dry
  preqx-nlev26-qsize4-r0-moist
  preqx-nlev26-qsize4-r3-moist
  preqx-nlev72-qsize4-r0-dry
  preqx-nlev72-qsize4-r3-dry
  preqx-nlev72-qsize4-r3-q6-dry
  preqx-nlev72-qsize4-r0-moist
  preqx-nlev72-qsize4-r3-moist
  preqx-nlev72-qsize4-r3-tensorhv-dry
  preqx-nlev72-qsize4-r3-nudiv-dry
  preqx-nlev72-qsize10-r3-lim9-dry
)
