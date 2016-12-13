#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyRideHOMMEXXCDash/nightly_log_rideHOMMEXXcuda.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyRideHOMMEXXCDash/nightly_log_rideHOMMEXXcuda.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyRideHOMMEXXCDash/nightly_log_rideHOMMEXXcuda.txt -c`
TT=`grep "...   Passed" /home/ikalash/nightlyRideHOMMEXXCDash/nightly_log_rideHOMMEXXcuda.txt -c`

mail -s "HOMMEXX (master, CUDA KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, onguba@sandia.gov, mdeakin@sandia.gov, dsunder@sandia.gov, lbertag@sandia.gov" < /home/ikalash/nightlyRideHOMMEXXCDash/results_hommexx_cuda

