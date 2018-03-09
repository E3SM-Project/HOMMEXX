#!/bin/bash

exech=./bld/src/preqx/preqx
exech2=./bld-omp/src/preqx/preqx
nlh=homme-ne8-v1.nl

PER_NODE=24
NN=2
NMPI=$((PER_NODE*NN))
comm="srun -N ${NN} -n ${NMPI} "

dd=`date +"%s"`


for i in 1 2 3 4 ; do

$comm $exech < $nlh > tmp.txt
mv HommeTime_stats time-homme-serial-ne8-N${NN}-r${PER_NODE}-iter${i}-${dd}
$comm $exech2 < $nlh > tmp.txt
mv HommeTime_stats time-homme-omp-ne8-N${NN}-r${PER_NODE}-iter${i}-${dd}

done

#now putput
for i in 1 2 3 4 ; do

#repetition
n1=time-homme-serial-ne8-N${NN}-r${PER_NODE}-iter${i}-${dd}
n2=time-homme-omp-ne8-N${NN}-r${PER_NODE}-iter${i}-${dd}

echo "times from ${n1}" 
grep "vertical_remap\"\|prim_main\|_rk2\|U3\-5\|vis_dp" $n1
echo "times from ${n2}"
grep "vertical_remap\"\|prim_main\|_rk2\|U3\-5\|vis_dp" $n2

done




