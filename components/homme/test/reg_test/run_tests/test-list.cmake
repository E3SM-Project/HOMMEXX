# Lists of test files for the HOMME regression tests
SET(HOMME_TESTS
#  swirl.cmake
#  swtc1_flat.cmake
#  swtc1_flat_c.cmake
#  swtc2_flat.cmake
#  swtc2_flat_c.cmake
#  swtc5_flat.cmake
#  swtc5_flat_c.cmake
#  swtc6_flat.cmake
#  swtc6_flat_c.cmake
#  swirl_flat.cmake
#  swirl_flat_c.cmake
  prtcA_flat.cmake
  prtcA_flat_c.cmake
  prtcB_flat.cmake
  prtcB_flat_c.cmake
#  baro2b.cmake
#  baro2c.cmake
#  baro2d.cmake
#  baroCamMoist.cmake
#  baroCamMoist-SL.cmake
#  baroCamMoist-acc.cmake
#  baro2d-imp.cmake
#  templates.cmake
)

#This list (COMPARE_F_C_TEST) contains tests for which
#F vc C comparison will be run.
SET (COMPARE_F_C_TEST
  prtcA_flat
  prtcB_flat
)
