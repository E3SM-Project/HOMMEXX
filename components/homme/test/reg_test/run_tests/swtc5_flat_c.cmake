###############################################################
# Explicit leapfrog default benchmark
###############################################################
#
# Spectral Element -- swtc5
# NE=30, dt=90, nu=1.5e15, limiter=0, filter_freq=0, NP=4
#
###############################################################

# The name of this test (should be the basename of this file)
SET(TEST_NAME swtc5_flat_c)
# The specifically compiled executable that this test uses
SET(EXEC_NAME swtcB_flat_c)

SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/swtc5.nl)
SET(NCL_FILES ${HOMME_ROOT}/test/reg_test/ncl/swtc5ref.ncl)

SET(NC_OUTPUT_FILES swtc51.nc)
