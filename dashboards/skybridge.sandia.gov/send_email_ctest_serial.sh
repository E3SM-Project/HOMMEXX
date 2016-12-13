#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`
TTTT=`grep "(Not Run)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`
TTTTT=`grep "(Timeout)" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`
TT=`grep "...   Passed" /home/ikalash/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`

mail -s "HOMMEXX (master, Serial KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov, agsalin@sandia.gov, lbertag@sandia.gov, dsunder@sandia.gov, mdeakin@sandia.gov, onguba@sandia.gov" < /home/ikalash/nightlyHOMMEXXCDash/results_hommexx_serial

