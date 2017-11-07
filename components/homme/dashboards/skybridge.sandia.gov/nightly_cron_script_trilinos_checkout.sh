#!/bin/csh

BASE_DIR=/projects/hommexx/nightlyHOMMEXXCDash
cd $BASE_DIR

unset http_proxy
unset https_proxy

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf slurm*
rm -rf modules*out
rm -rf build_kokkos*out 

cat trilinosCheckout ctest_nightly.cmake.frag >& ctest_nightly.cmake  

now=$(date +"%m_%d_%Y-%H_%M")
LOG_FILE=$BASE_DIR/nightly_log_skybridgeTrilinosCheckout.txt

eval "env  TEST_DIRECTORY=$BASE_DIR SCRIPT_DIRECTORY=$BASE_DIR ctest -VV -S $BASE_DIR/ctest_nightly.cmake" > $LOG_FILE 2>&1

