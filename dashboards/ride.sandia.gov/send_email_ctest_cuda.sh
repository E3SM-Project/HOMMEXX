#!/bin/bash

#source $1 

TTT=`grep "(Failed)" ${WORKSPACE}/nightly_log_rideHOMMEXXcuda.txt -c`
TTTT=`grep "(Not Run)" ${WORKSPACE}/nightly_log_rideHOMMEXXcuda.txt -c`
TTTTT=`grep "(Timeout)" ${WORKSPACE}/nightly_log_rideHOMMEXXcuda.txt -c`
TT=`grep "...   Passed" ${WORKSPACE}/nightly_log_rideHOMMEXXcuda.txt -c`

/bin/mail -s "HOMMEXX (master, CUDA KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, onguba@sandia.gov, mdeakin@sandia.gov, dsunder@sandia.gov, lbertag@sandia.gov" < ${WORKSPACE}/results_hommexx_cuda

