#!/bin/bash                                           
#BSUB -a "openmpi"                                    
#BSUB -n 32
#BSUB -W 01:00                                                           
#BSUB -R "span[ptile=16]"                             
#BSUB -o hommexxCDash.out                                 
#BSUB -e hommexxCDash.err                                
#BSUB -J hommexxCDash 

cd /ascldap/users/ikalash/nightlyHOMMEXXCDash

rm -rf repos
rm -rf build
rm -rf ctest_nightly.cmake.work
rm -rf nightly_log*
rm -rf results*
rm -rf modules*out
rm -rf *out 
rm -rf *err 

ulimit -c 0

bash -c -l "source ride_modules_cuda.sh >& modules_trilinos.out; bash nightly_cron_script_trilinos_ride_cuda.sh"
bash -c -l "source ride_modules_cuda.sh >& modules_trilinos.out; bash apply_kokkos_patch.sh >& apply_kokkos_patch.out"
bash -c -l "source ride_modules_cuda.sh >& modules_hommexx.out; bash nightly_cron_script_hommexx_ride_cuda.sh"
scp nightly_log_rideHOMMEXXcuda.txt ikalash@mockba.ca.sandia.gov:rideCDash/HOMMEXX
bash process_results_ctest_cuda.sh
scp results_hommexx_cuda ikalash@mockba.ca.sandia.gov:rideCDash/HOMMEXX
bash send_email_ctest_cuda.sh

# recursively resubmit this script
X=$( date -d "1 days 8 am" "+%m:%d:%H:%M" )
bsub -b $X -q rhel7G < /ascldap/users/ikalash/nightlyHOMMEXXCDash/cronjob_hack.batch
