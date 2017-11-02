# Lists of test files for the HOMME regression tests
SET(HOMME_TESTS
  prtcA-r0-f
  prtcA-r3-f
  prtcB-r0-f
  prtcB-r3-f
  prtcA-flat-r0-f
  prtcA-flat-r3-f
  prtcA-flat-r0-c
  prtcA-flat-r3-c
  prtcB-flat-r0-f
  prtcB-flat-r3-f
  prtcB-flat-r0-c
  prtcB-flat-r3-c
  prtcA-r0-moist-f
  prtcA-r3-moist-f
  prtcB-r0-moist-f
  prtcB-r3-moist-f
  prtcA-flat-r0-moist-f
  prtcA-flat-r3-moist-f
  prtcA-flat-r0-moist-c
  prtcA-flat-r3-moist-c
  prtcB-flat-r0-moist-f
  prtcB-flat-r3-moist-f
  prtcB-flat-r0-moist-c
  prtcB-flat-r3-moist-c
)

#This list (COMPARE_F_C_TEST) contains tests for which
#F vc C comparison will be run.
SET (COMPARE_F_C_TEST
  prtcA-flat-r0
  prtcA-flat-r3
  prtcB-flat-r0
  prtcB-flat-r3
  prtcA-flat-r0-moist
  prtcA-flat-r3-moist
  prtcB-flat-r0-moist
  prtcB-flat-r3-moist
)
