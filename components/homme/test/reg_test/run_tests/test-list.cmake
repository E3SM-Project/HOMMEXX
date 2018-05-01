# Lists of test files for the HOMME regression tests
SET(HOMME_TESTS
  prtcA-r0-dry-f
  prtcA-r3-dry-f
  prtcA-r0-dry-c
  prtcA-r3-dry-c
  prtcB-r0-dry-f
  prtcB-r3-dry-f
  prtcB-r0-dry-c
  prtcB-r3-dry-c
  prtcB-r3-q6-dry-f
  prtcB-r3-q6-dry-c
  prtcB-r3-tensorhv-dry-f
  prtcB-r3-tensorhv-dry-c
  prtcA-r0-moist-f
  prtcA-r3-moist-f
  prtcA-r0-moist-c
  prtcA-r3-moist-c
  prtcB-r0-moist-f
  prtcB-r3-moist-f
  prtcB-r0-moist-c
  prtcB-r3-moist-c
)

#This list (COMPARE_F_C_TEST) contains tests for which
#F vc C comparison will be run.
SET (COMPARE_F_C_TEST
  prtcA-r0-dry
  prtcA-r3-dry
  prtcB-r0-dry
  prtcB-r3-dry
  prtcB-r3-q6-dry
  prtcB-r3-tensorhv-dry
  prtcA-r0-moist
  prtcA-r3-moist
  prtcB-r0-moist
  prtcB-r3-moist
)
