#!/bin/tcsh -f
set wdir =  `pwd`             # run directory
set HOMME = ~/acme-bench/components/homme                # HOMME svn checkout     
echo $HOMME
set bld = $wdir/bld                      # cmake/build directory

set MACH = $HOMME/cmake/machineFiles/edison-serial.cmake
set nlev=128
set qsize=10

mkdir -p $bld
cd $bld
set build = 1  # set to 1 to force build
set conf = 1
rm $bld/CMakeCache.txt    # remove this file to force re-configure
if ( $conf ) then
   rm -rf CMakeFiles CMakeCache.txt src
   echo "running CMAKE to configure the model"

   cmake -C $MACH -DQSIZE_D=$qsize -DPREQX_PLEV=$nlev -DPREQX_NP=4  \
   -DBUILD_HOMME_SWEQX=FALSE -DWITH_PNETCDF=FALSE \
   -DPREQX_USE_ENERGY=FALSE  $HOMME

   make -j4 clean
   make -j8 preqx
   exit
endif




