#!/bin/sh

PRTCB_C_BUILD_DIR=${PWD}/build/test_execs/prtcB_flat_c
cd ${PRTCB_C_BUILD_DIR}

cp namelist.nl namelist.gpu_cmp.nl
sed -i 's:ndays *= *[0-9]*:nmax = 100:' namelist.gpu_cmp.nl
sed -i 's:ne *= [0-9]*:ne = 8:' namelist.gpu_cmp.nl
sed -i 's:qsize *= [0-9]*:qsize = 35:' namelist.gpu_cmp.nl
sed -i 's:output_timeunits *= [0-9]*,[0-9]*:output_timeunits = -1,-1:' namelist.gpu_cmp.nl
sed -i 's:output_frequency *= [0-9]*,[0-9]*:output_frequency = -1,-1:' namelist.gpu_cmp.nl

for i in $(seq 4); do
  mpiexec -n 32 --map-by ppr:32:node:pe=1 --bind-to core -x OMP_PROC_BIND=spread -x OMP_PLACES=threads -x OMP_NUM_THREADS=2 ./prtcB_flat_c < namelist.gpu_cmp.nl;
  mv HommeTime_stats HommeTime_stats.mpi.${i};
done
