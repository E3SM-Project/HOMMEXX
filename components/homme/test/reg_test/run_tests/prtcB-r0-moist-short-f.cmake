
# The name of this test (should be the basename of this file)
SET(TEST_NAME prtcB-r0-moist-short-f)
# The specifically compiled executable that this test uses
SET(EXEC_NAME prtcB)

SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/prtcB-r0-moist-short.nl)
SET(VCOORD_FILES ${HOMME_ROOT}/test/vcoord/acme-72*)

# compare all of these files against baselines:
SET(NC_OUTPUT_FILES
  jw_baroclinic1.nc
  jw_baroclinic2.nc)

