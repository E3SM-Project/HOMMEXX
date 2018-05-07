#!/bin/bash

mach=edison #knl haswell
PERNODE=24

################## these need to be hardcoded for each machine sepatately
################## EDISON
#nearray=( 8 16 32 64 80 120 ) ;
nearray=( 120 );
#ne8nodes=( 1 2 4 8 16 );
#ne16nodes=( 1 2 4 8 16 32 64 );
#ne32nodes=( 1 2 4 8 16 32 64 128 256 );
#ne64nodes=( 1 2 4 8 16 32 64 128 256 512 1024 );
#ne80nodes=( 1 2 4 5 8 10 16 20 25 32 50 80 160 320 400 800 1600 );

#real ones
#ne120nodes=( 1 2 3 4 5 6 8 9 10 12 15 16 18 20 24 25 30 36 40 45 48 50 72 80 100 144 180 225 300 400 600 900 1200 1800 3600 );
#debug
ne120nodes=(10 20 30 100 300 600 900 1200  1800 3600 );
########################################################################

## !!!!!!! ## should be this if we start with ne8
#         8  16  32 64 80 120
#nmax1=( 500 120 50 40 20 10 ) ;
nmax1=( 6 ); #6 for ne120

tstep=75
count=0
submit=1
currfolder=`pwd`

for nume in ${nearray[@]} ; do

  echo "NE is ${nume}";

  nodearray=ne${nume}nodes ;
  #eval echo \${$nodearray[@]} ;

  for NN in $(eval echo \${$nodearray[@]}) ; do

    nrank=$(expr $PERNODE \* $NN)

    nmax=$(expr ${nmax1[$count]} \* $NN)

    name="ne${nume}-mach${mach}-nnode${NN}-nmax${nmax}"
    rundir=run/${name}
    rm -rf $rundir
    mkdir $rundir
    mkdir $rundir/movies
    cp -r vcoord $rundir

    #create a new script

    sed -e s/NE/${nume}/ -e s/NMAX/${nmax}/ -e s/TSTEP/${tstep}/ \
        xx-template.nl > $rundir/xxinput.nl;

    #homme picks hor threads from environment
    sed -e s/NE/${nume}/ -e s/NMAX/${nmax}/ -e s/TSTEP/${tstep}/ \
        homme-template.nl > $rundir/hommeinput.nl;

    #hommeSL picks hor threads from environment
    sed -e s/NE/${nume}/ -e s/NMAX/${nmax}/ -e s/TSTEP/${tstep}/ \
        homme-template-SL.nl > $rundir/hommeinput-SL.nl;


### take care of TIME too
# for now, set time to 30 min for all
#    sed -e s/NAME/${name}/ -e s/NNODE/${NN}/ -e s/NRANK/${nrank}/   \
#        -e s/PARTITION/${mach}/ \
#        -e s/CFLAG/${cflag}/ -e s/NTHR/${nthr}/ \
#        -e s/EXEHOMME/${exehomme}/ \
#        -e s/EXEXX/${exexx}/ \
#        template.sh   >  run/${name}/job.sh

    sed -e s/NAME/${name}/ -e s/NNODE/${NN}/ -e s/NRANK/${nrank}/   \
        -e s/NUME/${nume}/ \
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





