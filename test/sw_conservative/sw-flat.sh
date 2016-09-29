#!/bin/tcsh -f
#
# 50 nodes, 1h, NE=80  dt=3
# 50 nodes, 2h, NE=160 dt=5  
# 100 nodes, 4h, NE=160 dt=1  
#
#SBATCH --job-name swirl
#SBATCH -N 4
#SBATCH --account=FY150001
#SBATCH -p ec
#SBATCH --time=0:10:00
#

set wdir = ~/runhomme
set HOMME = ~/HOMMEXX/components/homme
#set MACH = $HOMME/cmake/machineFiles/rhel5.cmake
set MACH = $HOMME/cmake/machineFiles/darwin.cmake
set input = $HOMME/test/sw_conservative

set builddir = $wdir/bld-acc
set rundir = $wdir/swirl-acc
mkdir -p $rundir
mkdir -p $wdir/bld-acc
cd $builddir

set NP = 4
set limiter = 8

set NCPU = 4


#configure the model
set conf = 0
set make = 1
#cmake:
cd $builddir
if ( $conf == 1 ) then
   rm -rf CMakeFiles CMakeCache.txt
   cmake -C $MACH -DSWEQX_FLAT_PLEV=6  -DSWEQX_FLAT_NP=$NP $HOMME
   make -j4 clean
   make -j4 sweqx_flat
   exit
endif
if ( $make == 1 ) then
   make -j4 sweqx_flat
    if ($status) exit
endif
set exe = $builddir/src/sweqx_flat/sweqx_flat
echo $exe

cd $rundir
mkdir movies


# NOTE:
# shallow water test was converted to use new/physical units 7/2015, and the
# case now runs for 12 days (instead of 5 days).  
# 
# settings below given in 'OLD' units need to be converted.
#   tstep should be INCREASED by 12/5
#   viscosity should be DECREASED by 12/5
# 


set NE = 10; set tstep = 50   ;  set nu=1e16   #l2errors: .120 .0458  .331


# output units: 0,1,2 = timesteps, days, hours
set OUTUNITS = 2  
set OUTFREQ =  12  # output 0,1.25,2.5,3.75,5.0 days

set test_case = swirl
set ndays = 5
set SUBCASE = "sub_case = 4"


# RKSSP settings - to mimic tracer advection used in 3D
set smooth = 0
set integration = runge_kutta
set LFTfreq = 0
set rk_stage = 3
set filter_freq = 0
set sfreq = 12
@ sfreq *= 3600
set sfreq = `echo "$sfreq / $tstep" | bc`


cd $rundir
mkdir movies


pwd

set nu_s = $nu
sed s/^ne.\*/"ne = $NE"/  $input/sw-flat.nl |\
sed s/tstep.\*/"tstep = $tstep"/  |\
sed s/limiter_option.\*/"limiter_option = $limiter"/  |\
sed s/smooth.\*/"smooth = $smooth"/  |\
sed s/test_case.\*/"test_case = \'$test_case\'   $SUBCASE"/  |\
sed s/explicit/$integration/  |\
sed s/rk_stage_user.\*/"rk_stage_user = $rk_stage"/  |\
sed s/LFTfreq.\*/"LFTfreq = $LFTfreq"/  |\
sed s/ndays.\*/"ndays = $ndays"/  |\
sed s/nu=.\*/"nu= $nu"/  |\
sed s/nu_s=.\*/"nu_s= $nu_s"/  |\
sed s/filter_freq.\*/"filter_freq = $filter_freq"/  |\
sed s/output_frequency.\*/"output_frequency = $OUTFREQ"/  |\
sed s/output_timeunits.\*/"output_timeunits = $OUTUNITS  interp_type=1"/  |\
sed s/statefreq.\*/"statefreq = $sfreq"/  \
> input.nl

date
$exe < input.nl | tee sweq.out
#valgrind --leak-check=full -v $exe < input.nl | tee sweq.out
#mpirun -np $NCPU $exe < input.nl | tee  sweq.out
date
# if timing was turned on, get sweq cost:
grep sweq HommeSWTime | sort | tail -1

#  ncl $input/geo.ncl




