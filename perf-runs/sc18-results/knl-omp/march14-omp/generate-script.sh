#!/bin/bash

#partition=knl #knl haswell
mach=knl #knl haswell


################## these need to be hardcoded for each machine sepatately
################## KNL
nearray=( 32 64 80 120 ) ; # 8 16 32 64 80 120
declare -a ne8nodes=( 1 2 3 6 );
declare -a ne16nodes=( 1 2 3 4 6 8 12 24 );
ne32nodes=( 1 2 3 4 6 8 12 16 24 32 48 96 );
ne64nodes=( 1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 384 );
ne80nodes=( 1 2 3 4 5 6 8 10 12 15 20 24 25 30 40 60 100 150 200 300 600 );
ne120nodes=( 1 2 3 5 6 9 10 15 18 25 27 30 45 54 90 150 270 450 675 1350 );
########################################################################

## !!!!!!! ## should be this if we start with ne8
#         8   16 32 64 80 120
#nmax1=( 500 120 50 40 20 10 ) ;
nmax1ne8=500; nmax1ne16=120; nmax1ne32=50; nmax1ne64=40; nmax1ne80=20; nmax1ne120=10;

if [[ $mach == haswell ]]; then
   cflag=2
   nthr=1
   PERNODE=32
else
   cflag=4
   nthr=2
   PERNODE=64
fi

tstep=1
count=0
submit=1
currfolder=`pwd`

for nume in ${nearray[@]} ; do

  echo "NE is ${nume}";

  nodearray=ne${nume}nodes ;
  nmaxname=nmax1ne${nume};
  nmaxval=$( eval echo \${$nmaxname[0]} );
  echo "nmaxval = ${nmaxval} ";
  #eval echo \${$nodearray[@]} ;

  for NN in $(eval echo \${$nodearray[@]}) ; do

    nrank=$(expr $PERNODE \* $NN)

    nmax=$(expr ${nmaxval} \* $NN)

    name="ne${nume}-mach${mach}-nnode${NN}-nmax${nmax}-t${nthr}"

    echo "Submitting run with ne=${nume}, mach=${mach}, NNodes=${NN}, ranks=${nrank}, threads=${nthr}, nmax=${nmax}..."
    echo " ... and name ${name}."
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
        -e s/CFLAG/${cflag}/  \
        -e s/NUME/${nume}/ \
        -e s/NTHR/${nthr}/ \
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





