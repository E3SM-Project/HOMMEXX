
# The name of this test (should be the basename of this file)
SET(TEST_NAME prtcA-r3-f)
# The specifically compiled executable that this test uses
SET(EXEC_NAME prtcA)

SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/prtcA-r3.nl)
SET(VCOORD_FILES ${HOMME_ROOT}/test/vcoord/cam*-26.ascii)

# compare all of these files against baselines:
SET(NC_OUTPUT_FILES
  jw_baroclinic1.nc
  jw_baroclinic2.nc)
