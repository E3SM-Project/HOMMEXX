
SET(TEST_NAME prtcB-flat-r0-moist-c)
# The specifically compiled executable that this test uses
SET(EXEC_NAME prtcB_flat_c)

SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/prtcB-r0-moist.nl)
SET(VCOORD_FILES ${HOMME_ROOT}/test/vcoord/acme-72*)

# compare all of these files against baselines:
SET(NC_OUTPUT_FILES
  jw_baroclinic1.nc
  jw_baroclinic2.nc)

# For GPU testbeds, for now.
SET(TIMEOUT 240)
