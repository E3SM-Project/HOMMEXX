#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTT=`grep "(Not Run)" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTTT=`grep "(Timeout)" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TT=`grep "...   Passed" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`

mail -s " (master, OpenMP KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, mdeakin@sandia.gov, onguba@sandia.gov, ambradl@sandia.gov" < /home/projects/hommexx/nightlyCDash/results_hommexx

