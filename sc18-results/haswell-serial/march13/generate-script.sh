#!/bin/bash

#partition=knl #knl haswell
mach=haswell #knl haswell


################## these need to be hardcoded for each machine sepatately
################## HSW
nearray=(8 16 32 64 80 120 ) ;
ne8nodes=( 1 2 3 4 6 12 );
ne16nodes=( 1 2 3 4 6 8 12 16 24 48 );
ne32nodes=( 1 2 3 4 6 8 12 16 24 32 48 64 96 192 );
ne64nodes=( 1 2 3 4 6 8 12 16 24 32 48 96 192 256 384 768 );
ne80nodes=( 1 2 3 4 5 6 8 10 12 15 16 20 24 25 30 48 60 80 120 200 300 400 600 1200 );
ne120nodes=( 1 2 3 4 5 6 9 10 12 15 18 20 25 27 30 36 50 60 90 108 150 225 300 540 675 900 1350 );
########################################################################

## !!!!!!! ## should be this if we start with ne8
#         8   16 32 64 80 120
#nmax1=( 500 120 50 40 20 10 ) ;
nmax1ne8=500; nmax1ne16=120; nmax1ne32=50; nmax1ne64=40; nmax1ne80=20; nmax1ne120=10;

if [[ $mach == haswell ]]; then
   cflag=2
   nthr=1
else
   cflag=4
   nthr=1
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

    if [[ $mach == haswell ]]; then
      nrank=$(expr 32 \* $NN)
    else
      nrank=$(expr 64 \* $NN)
    fi

    nmax=$(expr ${nmaxval} \* $NN)

    name="ne${nume}-mach${mach}-nnode${NN}-nmax${nmax}"

    echo "Submitting run with ne=${nume}, mach=${mach}, NNodes=${NN}, nmax=${nmax}..."
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
        -e s/CFLAG/${cflag}/  \
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





