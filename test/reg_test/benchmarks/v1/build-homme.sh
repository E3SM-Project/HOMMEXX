#!/bin/tcsh -f
#SBATCH --job-name baro-orig
#SBATCH --account=FY150001
#SBATCH -N 4
#SBATCH --time=00:01:00
#SBATCH -p ec
#SBATCH -o job%j

#  set paths to source code, build directory and run directory
#
set wdir =  `pwd`             # run directory
set HOMME = ~/acme-master/components/homme                # HOMME svn checkout     
echo $HOMME
set bld = $wdir/bld                      # cmake/build directory

set MACH = $HOMME/cmake/machineFiles/sandia-srn-sems.cmake
set nlev=72
set qsize=40

#
#  BUILD PREQX
#
mkdir -p $bld
cd $bld
set build = 1  # set to 1 to force build
set conf = 1
rm $bld/CMakeCache.txt    # remove this file to force re-configure
if ( $conf ) then
   rm -rf CMakeFiles CMakeCache.txt src
   #rm -rf utils/pio  utils/timing utils/cprnc   # may also need to do this
   echo "running CMAKE to configure the model"

   cmake -C $MACH -DQSIZE_D=$qsize -DPREQX_PLEV=$nlev -DPREQX_NP=4  \
   -DBUILD_HOMME_SWEQX=FALSE                     \
   -DPREQX_USE_ENERGY=FALSE  $HOMME

   make -j4 clean
   make -j8 preqx
   exit
endif

#if ( ! -f $exe) set build = 1   # no exe, force build
#if ( $build == 1 ) then
#make -j4 preqx
#if ($status) exit
#endif



