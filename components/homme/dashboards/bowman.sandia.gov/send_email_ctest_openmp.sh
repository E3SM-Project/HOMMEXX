#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TT=`grep "...   Passed" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`

/bin/mail -s "HOMMEXX (master, Bowman, OpenMP KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, mdeakin@sandia.gov, onguba@sandia.gov" < /home/ikalash/nightlyHOMMEXXCDash/results_hommexx_openmp

