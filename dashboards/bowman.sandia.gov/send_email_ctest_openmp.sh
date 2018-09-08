#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTT=`grep "(Not Run)" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TTTTT=`grep "(Timeout)" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`
TT=`grep "...   Passed" /home/projects/hommexx/nightlyCDash/nightly_log_bowmanHOMMEXXopenmp.txt -c`


echo "Subject: HOMMEXX (master, OpenMP KokkosNode, KNL): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" >& a
echo "" >& b 
cat a b >& c 
cat c results_hommexx >& d
mv d results_hommexx
rm a b c 
cat results_hommexx | /usr/lib/sendmail -F ikalash@bowman.sandia.gov -t "ikalash@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, onguba@sandia.gov, ambradl@sandia.gov"
