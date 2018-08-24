#!/bin/bash

#partition=knl #knl haswell
mach=knl #knl haswell
#exehommeknl=${HOME}/runhomme/sc18/builds/bld/src/preqx/preqx
#exehommehsw=???
#exexxknl=${HOME}/runhomme/sc18/builds/bldxx/test_execs/prtcB_flat_c/prtcB_flat_c
#exexxhsw=???


################## these need to be hardcoded for each machine sepatately
################## KNL
#nearray=( 8 16 32 64 80 120 ) ;
#DEBUG
nearray=( 16 ) ;
declare -a ne8nodes=( 1 2 3 6 );
declare -a ne16nodes=( 1 2 3 4 6 8 12 24 );
########################################################################

## !!!!!!! ## should be this if we start with ne8
#nmax1=( 500 120 50 40 20 10 ) ;
nmax1=( 120 );

if [[ $mach == hsw ]]; then
   cflag=2
   nthr=1
#   exehomme=$exehommehsw
#   exexx=$exexxhsw
else
   cflag=4
   nthr=1
#   exehomme=$exehommeknl
#   exexx=$exexxknl
fi

tstep=1
count=0
submit=1
currfolder=`pwd`

for nume in ${nearray[@]} ; do

  echo "NE is ${nume}";

  nodearray=ne${nume}nodes ;
  #eval echo \${$nodearray[@]} ;

  for NN in $(eval echo \${$nodearray[@]}) ; do

    if [[ $mach == hsw ]]; then
      nrank=$(expr 32 \* $NN)
    else
      nrank=$(expr 64 \* $NN)
    fi

    nmax=$(expr ${nmax1[$count]} \* $NN)

    name="ne${nume}-mach${mach}-nnode${NN}-nmax${nmax}-th${nthr}"
    rundir=run/${name}
    rm -rf $rundir
    mkdir $rundir
    mkdir $rundir/movies
    cp -r vcoord $rundir

    #create a new script

    sed -e s/NE/${nume}/ -e s/NMAX/${nmax}/ -e s/TSTEP/${tstep}/ \
        -e s/NTHR/${nthr}/ \
        xx-template.nl > $rundir/xxinput.nl;

    #homme picks hor threads from environment
    sed -e s/NE/${nume}/ -e s/NMAX/${nmax}/ -e s/TSTEP/${tstep}/ \
        homme-template.nl > $rundir/hommeinput.nl;


### take care of TIME too
# for now, set time to 30 min for all
#    sed -e s/NAME/${name}/ -e s/NNODE/${NN}/ -e s/NRANK/${nrank}/   \
#        -e s/PARTITION/${mach}/ \
#        -e s/CFLAG/${cflag}/ -e s/NTHR/${nthr}/ \
#        -e s/EXEHOMME/${exehomme}/ \
#        -e s/EXEXX/${exexx}/ \
#        template.sh   >  run/${name}/job.sh

    sed -e s/NAME/${name}/ -e s/NNODE/${NN}/ -e s/NRANK/${nrank}/   \
        -e s/PARTITION/${mach}/ \
        -e s/CFLAG/${cflag}/ -e s/NTHR/${nthr}/ \
        template.sh   >  $rundir/job.sh

    if [[ $submit == 1 ]]; then
       cd $rundir
       sbatch job.sh
       cd $currfolder
    fi

  done

  count=$(( count +1));
  #echo "count = $count "
done





