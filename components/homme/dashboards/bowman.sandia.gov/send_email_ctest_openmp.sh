#!/bin/bash

#source $1 

TTT=`grep "(Failed)" nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTT=`grep "(Not Run)" nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTTT=`grep "(Timeout)" nightly_log_bowmanHOMMEXXopenmp.txt -c`
TT=`grep "...   Passed" nightly_log_bowmanHOMMEXXopenmp.txt -c`

mail -s "HOMMEXX (master, Bowman, OpenMP KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, mdeakin@sandia.gov, onguba@sandia.gov" < results_hommexx_openmp

#/bin/mail -s "HOMMEXX (master, Bowman, OpenMP KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" -r 'Irina Tezaur<ikalash@bowman.sandia.gov>' "ikalash@sandia.gov, agsalin@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, mdeakin@sandia.gov, onguba@sandia.gov" < results_hommexx_openmp
