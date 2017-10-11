# Lists of test files for the HOMME regression tests
SET(HOMME_TESTS
  prtcA-r0-f.cmake
  prtcA-r3-f.cmake
  prtcB-r0-f.cmake
  prtcB-r3-f.cmake
  prtcA-flat-r0-f.cmake
  prtcA-flat-r3-f.cmake
  prtcA-flat-r0-c.cmake
  prtcA-flat-r3-c.cmake
  prtcB-flat-r0-f.cmake
  prtcB-flat-r3-f.cmake
  prtcB-flat-r0-c.cmake
  prtcB-flat-r3-c.cmake
)

#This list (COMPARE_F_C_TEST) contains tests for which
#F vc C comparison will be run.
SET (COMPARE_F_C_TEST
  prtcA-flat-r0
  prtcA-flat-r3
  prtcB-flat-r0
  prtcB-flat-r3
)
