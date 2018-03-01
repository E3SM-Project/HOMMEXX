#!/bin/bash

#source $1 

TTT=`grep "(Failed)" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`
TTTT=`grep "(Not Run)" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`
TTTTT=`grep "(Timeout)" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`
TT=`grep "...   Passed" /projects/hommexx/nightlyHOMMEXXCDash/nightly_log_skybridgeHOMMEXXserial.txt -c`

/bin/mail -s "HOMMEXX (master, Serial KokkosNode): $TT tests passed, $TTT tests failed, $TTTT tests not run, $TTTTT timeouts" "ikalash@sandia.gov" < /projects/hommexx/nightlyHOMMEXXCDash/results_hommexx_serial

