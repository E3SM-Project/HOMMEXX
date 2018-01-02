#!/bin/csh

BASE_DIR=/projects/hommexx/nightlyHOMMEXXCDash
cd $BASE_DIR

unset http_proxy
unset https_proxy

cat hommexxSerial ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_skybridgeHOMMEXXserial.txt

eval "env TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR  ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

#rm -rf /projects/hommexx/hommexxCDash/SkybridgeHOMMEXXSerial/* 
#cp -r /projects/hommexx/nightlyHOMMEXXCDash/build/Testing/20* /projects/hommexx/hommexxCDash/SkybridgeHOMMEXXSerial
#rm -rf /projects/hommexx/nightlyHOMMEXXCDash/build/Testing/20* 

#bash process_results_ctest.sh
#bash send_email_ctest.sh

