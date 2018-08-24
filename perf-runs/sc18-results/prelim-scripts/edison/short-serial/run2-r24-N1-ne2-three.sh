#!/bin/bash

exech=./bld/src/preqx/preqx
execxx=${HOME}/runhomme-edison/bxx-serial/test_execs/prtcB_flat_c/prtcB_flat_c
execxx2=${HOME}/runhomme-edison/bxx-serial-old/test_execs/prtcB_flat_c/prtcB_flat_c
nlh=homme-ne2-v1.nl
nlxx=xx-ne2-v1.nl

PER_NODE=24
NN=1
NMPI=$((PER_NODE*NN))
comm="srun -N ${NN} -n ${NMPI} "

dd=`date +"%s"`


for i in 1 2 ; do

$comm $exech < $nlh > tmp.txt
tail -n 5 tmp.txt
mv HommeTime_stats time-homme-ne2-N${NN}-r${PER_NODE}-iter${i}-${dd}
$comm $execxx < $nlxx > tmp.txt
tail -n 5 tmp.txt
mv HommeTime_stats time-xx-ne2-N${NN}-r${PER_NODE}-iter${i}-${dd}

$comm $execxx2 < $nlxx > tmp.txt
tail -n 5 tmp.txt
mv HommeTime_stats time-xxOLD-ne2-N${NN}-r${PER_NODE}-iter${i}-${dd}



done

#now putput
for i in 1 2 ; do

#repetition
n1=time-xx-ne2-N${NN}-r${PER_NODE}-iter${i}-${dd}
n1OLD=time-xxOLD-ne2-N${NN}-r${PER_NODE}-iter${i}-${dd}
n2=time-homme-ne2-N${NN}-r${PER_NODE}-iter${i}-${dd}

echo "times from ${n1}" 
grep "tl-sc vertical\|prim_main\|_RK2\|U3\-5\|vis_dp" $n1

echo "times from ${n1OLD}" 
grep "tl-sc vertical\|prim_main\|_RK2\|U3\-5\|vis_dp" $n1OLD

echo "times from ${n2}"
grep "vertical_remap\"\|prim_main\|_rk2\|U3\-5\|vis_dp" $n2



done




