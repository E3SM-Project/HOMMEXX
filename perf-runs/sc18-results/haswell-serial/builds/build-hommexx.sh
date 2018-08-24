#!/bin/tcsh -f

source ~/load-haswell

set wdir =  `pwd`             # run directory
set HOMME = ~/acmexx/components/homme                # HOMME svn checkout     
echo $HOMME
set bld = $wdir/bldxx                      # cmake/build directory

set MACH = $HOMME/cmake/machineFiles/cori-haswell-serial.cmake

mkdir -p $bld
cd $bld
set build = 1  # set to 1 to force build
set conf = 1
rm $bld/CMakeCache.txt    # remove this file to force re-configure

#fp-model strict is prob not needed for performance
if ( $conf ) then
   rm -rf CMakeFiles CMakeCache.txt src
   echo "running CMAKE to configure the model"

   cmake -C $MACH  \
   -DPREQX_USE_PIO=FALSE \
   -DBUILD_HOMME_SWEQX=FALSE                     \
   -DUSE_TRILINOS=OFF  \
   -DHOMMEXX_FPMODEL=fast  \
   -DPREQX_USE_ENERGY=FALSE  $HOMME

   make -j4 clean
   make -j8 prtcB_flat_c
   exit
endif



