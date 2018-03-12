#!/bin/tcsh -f
set wdir =  `pwd`             # run directory
set HOMME = ~/acmexx/components/homme                # HOMME svn checkout     
echo $HOMME
set bld = $wdir/bldxx                      # cmake/build directory

set MACH = $HOMME/cmake/machineFiles/edison-serial.cmake
set nlev=72
set qsize=40

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
   -USE_TRILINOS=OFF -DHOMMEXX_FPMODEL=fast $HOMME 

   make -j4 clean
   make -j8 prtcB_flat_c
   exit
endif




