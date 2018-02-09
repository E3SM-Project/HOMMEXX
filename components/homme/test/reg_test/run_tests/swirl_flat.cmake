###############################################################
# RKSSP default benchmark (used to check nothing is broken)
###############################################################
#
# Spectral Element -- swtc1
# NE=10, dt=480, nu=0, limiter=4, filter_freq=0 NP=8
#
###############################################################

# The name of this test (should be the basename of this file)
SET(TEST_NAME swirl_flat)

# The specifically compiled executable that this test uses
SET(EXEC_NAME swtcA_flat)

SET(NUM_CPUS 16)

SET(NAMELIST_FILES ${HOMME_ROOT}/test/reg_test/namelists/swirl.nl)

SET(NC_OUTPUT_FILES swirl1.nc)
