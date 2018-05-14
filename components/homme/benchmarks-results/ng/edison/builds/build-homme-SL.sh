#!/bin/tcsh -f
set wdir =  `pwd`             # run directory
set HOMME = ~/acme-sl/components/homme                # HOMME svn checkout     
echo $HOMME
set bld = $wdir/bld-SL                      # cmake/build directory

set MACH = $HOMME/cmake/machineFiles/edison-og.cmake
set nlev=128
set qsize=10

mkdir -p $bld
cd $bld
set build = 1  # set to 1 to force build
set conf = 1
rm $bld/CMakeCache.txt    # remove this file to force re-configure

#cmake -C ~/acmexx/components/homme/cmake/machineFiles/edison-serial.cmake -DHOMMEXX_FPMODEL=fast -DUSE_TRILINOS=OFF ~/acmexx/components/homme/
if ( $conf ) then
   rm -rf CMakeFiles CMakeCache.txt src
   echo "running CMAKE to configure the model"

   cmake -C $MACH \
   -USE_TRILINOS=OFF -DHOMMEXX_FPMODEL=fast \
   -DQSIZE_D=$qsize -DPREQX_PLEV=$nlev -DPREQX_NP=4  \
   -DBUILD_HOMME_SWEQX=FALSE -DWITH_PNETCDF=FALSE \
   -DPREQX_USE_ENERGY=FALSE  $HOMME


   make -j4 clean
   make -j8 preqx
   exit
endif




