#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`
TT=`grep "...   Passed" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXopenmp.txt -c`

/bin/mail -s "HOMMEXX (master, OpenMP KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, mdeakin@sandia.gov, onguba@sandia.gov, ambradl@sandia.gov" < /home/ikalash/nightlyHOMMEXXCDash/results_hommexx_openmp

