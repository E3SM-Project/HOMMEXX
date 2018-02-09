# Lists of test files for the HOMME regression tests
SET(HOMME_TESTS
  prtcA-r0-dry-f
  prtcA-r3-dry-f
  prtcB-r0-dry-f
  prtcB-r3-dry-f
  prtcA-flat-r0-dry-f
  prtcA-flat-r3-dry-f
  prtcA-flat-r0-dry-c
  prtcA-flat-r3-dry-c
  prtcB-flat-r0-dry-f
  prtcB-flat-r3-dry-f
  prtcB-flat-r0-dry-c
  prtcB-flat-r3-dry-c
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
  prtcA-flat-r0-dry
  prtcA-flat-r3-dry
  prtcB-flat-r0-dry
  prtcB-flat-r3-dry
  prtcA-flat-r0-moist
  prtcA-flat-r3-moist
  prtcB-flat-r0-moist
  prtcB-flat-r3-moist
)
