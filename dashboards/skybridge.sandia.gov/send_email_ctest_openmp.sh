#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`
TTTT=`grep "(Not Run)" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`
TTTTT=`grep "(Timeout)" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`
TT=`grep "...   Passed" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`

/bin/mail -s "HOMMEXX (master, OpenMP KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, onguba@sandia.gov, ambradl@sandia.gov" < /projects/hommexx/nightlyHOMMEXXCDash/results_hommexx_openmp

