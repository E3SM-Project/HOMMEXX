#!/bin/csh

BASE_DIR=/home/ikalash/nightlyHOMMEXXCDash
cd $BASE_DIR

unset http_proxy
unset https_proxy

source bowman_modules.sh

rm -rf $BASE_DIR/repos
rm -rf $BASE_DIR/build
rm -rf $BASE_DIR/ctest_nightly.cmake.work
rm -rf $BASE_DIR/nightly_log*
rm -rf $BASE_DIR/results*
rm -rf $BASE_DIR/test_summary.txt

cat trilinosOpenMP ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_bowmanTrilinosOpenMP.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

#rm -rf /gpfs1/ikalash/hommexxCDash/SkybridgeTrilinosOpenMP/* 
#cp -r /home/ikalash/nightlyHOMMEXXCDash/build/Testing/20* /gpfs1/ikalash/hommexxCDash/SkybridgeTrilinosOpenMP
#rm -rf /home/ikalash/nightlyHOMMEXXCDash/build/Testing/20*
