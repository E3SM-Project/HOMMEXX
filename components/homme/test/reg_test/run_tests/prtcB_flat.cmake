###############################################################
# RK + PIO_INTERP
###############################################################
#
# Spectral Element -- 9 days of ASP baroclinic test
# (Jablonowski and Williamson test + 4 tracers)
# NE=15, dt=150, nu=1e16, filter_freq=0, NV=4, PLEV=26
# (explicit RK with subcycling)
#
###############################################################

# The name of this test (should be the basename of this file)
SET(TEST_NAME prtcB_flat)
# The specifically compiled executable that this test uses
SET(EXEC_NAME prtcB_flat)

SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/prtcB_flat.nl)
SET(VCOORD_FILES ${HOMME_ROOT}/test/vcoord/acme-72*)

# compare all of these files against baselines:
SET(NC_OUTPUT_FILES
  jw_baroclinic1.nc
  jw_baroclinic2.nc)
