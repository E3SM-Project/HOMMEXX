#!/bin/tcsh -f
set wdir =  `pwd`             # run directory
set HOMME = ~/acme-bench/components/homme                # HOMME svn checkout     
echo $HOMME
set bld = $wdir/bld-hor-off-col-on                      # cmake/build directory

set MACH = $HOMME/cmake/machineFiles/cori-knl-nopnetcdf.cmake
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
   echo "running CMAKE to configure the model"

   cmake -C $MACH -DQSIZE_D=$qsize -DPREQX_PLEV=$nlev -DPREQX_NP=4  \
   -DENABLE_HORIZ_OPENMP=FALSE \
   -DENABLE_COLUMN_OPENMP=TRUE \
   -DBUILD_HOMME_SWEQX=FALSE                     \
   -DPREQX_USE_ENERGY=FALSE  $HOMME

   make -j4 clean
   make -j8 preqx
   exit
endif




