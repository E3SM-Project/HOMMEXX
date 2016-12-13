#!/bin/csh

BASE_DIR=/ascldap/users/ikalash/nightlyHOMMEXXCDash
cd $BASE_DIR

unset http_proxy
unset https_proxy

cat hommexxCUDA ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_rideHOMMEXXcuda.txt

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR  ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

#bash process_results_ctest.sh
#bash send_email_ctest.sh

