#!/bin/csh

BASE_DIR=/home/ikalash/nightlyHOMMEXXCDash
cd $BASE_DIR

unset http_proxy
unset https_proxy

cat trilinosSerial ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_skybridgeTrilinosSerial.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

rm -rf /gpfs1/ikalash/hommexxCDash/SkybridgeTrilinosSerial/* 
cp -r /home/ikalash/nightlyHOMMEXXCDash/build/Testing/20* /gpfs1/ikalash/hommexxCDash/SkybridgeTrilinosSerial
rm -rf /home/ikalash/nightlyHOMMEXXCDash/build/Testing/20*
