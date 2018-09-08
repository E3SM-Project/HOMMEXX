#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/projects/hommexx/nightlyCDash/nightly_log_rideHOMMEXXcuda.txt -c`
TTTT=`grep "(Not Run)" /home/projects/hommexx/nightlyCDash/nightly_log_rideHOMMEXXcuda.txt -c`
TTTTT=`grep "(Timeout)" /home/projects/hommexx/nightlyCDash/nightly_log_rideHOMMEXXcuda.txt -c`
TT=`grep "...   Passed" /home/projects/hommexx/nightlyCDash/nightly_log_rideHOMMEXXcuda.txt -c`

/bin/mail -s "HOMMEXX (master, CUDA KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, onguba@sandia.gov, dsunder@sandia.gov, lbertag@sandia.gov, ambradl@sandia.gov" < /home/projects/hommexx/nightlyCDash/results_hommexx_cuda

