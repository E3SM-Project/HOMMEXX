#!/bin/bash                                           

#rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf modules*out
rm -rf *out 
rm -rf *err 

ulimit -c 0

#bash -c -l "source ride_modules_cuda.sh >& modules_trilinos.out; bash nightly_cron_script_trilinos_ride_cuda.sh"
bash -c -l "source ride_modules_cuda.sh >& modules_trilinos.out; bash build_kokkos.sh >& build_kokkos.out"
bash -c -l "source ride_modules_cuda.sh >& modules_hommexx.out; bash nightly_cron_script_hommexx_ride_cuda.sh"
bash process_results_ctest_cuda.sh
bash send_email_ctest_cuda.sh
