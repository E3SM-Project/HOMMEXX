#!/bin/bash

#partition=knl #knl haswell
mach=knl #knl haswell


################## these need to be hardcoded for each machine sepatately
################## KNL
nearray=( 240 ) ; # 8 16 32 64 80 120
ne120nodes=( 1 2 3  5   6  9 10 15 18 25 27 30  45   54  90 150 270 450 675 1350 );
ne240nodes=(   6  9 10 15 18 27 45 54 270 450 );
#ne240nodes=(                25 30 90 150 675 1350  );
########################################################################

## !!!!!!! ## should be this if we start with ne8
#         8   16 32 64 80 120
#nmax1=( 500 120 50 40 20 10 ) ;
nmax1ne8=500; nmax1ne16=120; nmax1ne32=50; nmax1ne64=40; nmax1ne80=20; nmax1ne120=10;
nmax1ne240=10;

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

  for NN1 in $(eval echo \${$nodearray[@]}) ; do

    NN=$(( NN1 *4 ));

    #echo " NN is ${NN}";

    nrank=$(expr $PERNODE \* $NN)

    nmax=$(expr ${nmaxval} \* $NN )

    nmax=$(( nmax /4 ))

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





