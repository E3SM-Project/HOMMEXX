#!/bin/bash

exech=./bld-ne2/src/preqx/preqx
execxx=../bxx-master-omp/test_execs/prtcB_flat_c/prtcB_flat_c
nlh=homme-ne2-v1.nl
nlxx=xx-ne2-v1.nl

PER_NODE=24
NN=1
tt=8 #threads
NMPI=$((PER_NODE*NN))
comm="srun -N ${NN} -n ${NMPI} -c 8 --cpu_bind=cores "
export OMP_NUM_THREADS=${tt}
export KMP_AFFINITY=balanced

dd=`date +"%s"`

for i in 1 2 3 4 ; do

echo "running with iter = ${i}, N=${NN}, nmpi=${NMPI}, threads=${OMP_NUM_THREADS} "
$comm $exech < $nlh &> tmphomme.txt
tail -n 5 tmp.txt
#how to save these in some array?
mv HommeTime_stats time-homme-ne2-N${NN}-r${PER_NODE}-t${tt}-iter${i}-${dd}

$comm $execxx < $nlxx &> tmpxx.txt
tail -n 5 tmp.txt
mv HommeTime_stats time-xx-ne2-N${NN}-r${PER_NODE}-t${tt}-iter${i}-${dd}

done

#now putput
for i in 1 2 3 4 ; do

#repetition
n1=time-xx-ne2-N${NN}-r${PER_NODE}-t${tt}-iter${i}-${dd}
n2=time-homme-ne2-N${NN}-r${PER_NODE}-t${tt}-iter${i}-${dd}

echo "times from ${n1}" 
grep "tl-sc vertical\|prim_main\|_RK2\|U3\-5\|vis_dp" $n1
echo "times from ${n2}"
grep "vertical_remap\"\|prim_main\|_rk2\|U3\-5\|vis_dp" $n2

done




