#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # NOTE for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-NAME # do not use custom output file?
#SBATCH --output=slurm-NAME
#SBATCH --nodes=NNODE
#SBATCH -C PARTITION
#SBATCH -t 00:30:00  # who knows how much....
#SBATCH --job-name=NUMEkNNODE # format to shrink job name


hommedir=${HOME}/runhomme/benchng/builds/bld-hor-on-col-off/src/preqx/
hommeSLdir=${HOME}/runhomme/benchng/builds/bld-hor-on-col-off-SL/src/preqx/
xxdir=${HOME}/runhomme/benchng/builds/bldxx-omp/test_execs/prtcB_c/

exehomme=${hommedir}/preqx
exehommeSL=${hommeSLdir}/preqx
exexx=${xxdir}/prtcB_c

export OMP_NUM_THREADS=NTHR
export KMP_AFFINITY=balanced
export OMP_STACKSIZE=16M

#homme run
srun -N NNODE \
     -n NRANK \
     -c CFLAG \
     --cpu_bind=cores \
     $exehomme \
     < hommeinput.nl |& grep -v "OMP: Warning #239: KMP_AFFINITY: granularity=fine will be used."

hname=time-homme-NAME-${SLURM_JOB_ID}
grep COMMIT ${hommedir}/../../CMakeCache.txt > $hname
echo ${SLURM_JOBID} >> $hname
head -n 50 HommeTime_stats >> $hname

rm HommeTime_stats

#hommeSL run
srun -N NNODE \
     -n NRANK \
     -c CFLAG \
     --cpu_bind=cores \
     $exehommeSL \
     < hommeinput-SL.nl |& grep -v "OMP: Warning #239: KMP_AFFINITY: granularity=fine will be used."

hSLname=time-hommeSL-NAME-${SLURM_JOB_ID}
grep COMMIT ${hommeSLdir}/../../CMakeCache.txt > $hSLname
echo ${SLURM_JOBID} >> $hSLname
head -n 50 HommeTime_stats >> $hSLname

rm HommeTime_stats


#hommeXX run
# how to silence warnings
#srun -N 1 \
#     -n 24 \
#     -c 4 \
#     --cpu_bind=cores \
#     $exexx \
#     < xxinput.nl  |& grep -v "Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set" \
#|& grep -v "In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads" \
#|& grep -v "For best performance with OpenMP 3.1 set OMP_PROC_BIND=true" \
#|& grep -v "For unit testing set OMP_PROC_BIND=false" \
#|& grep -v "Kokkos::OpenMP thread_pool_topology\[ 1 x 2 x 1 \]"

#hommexx run
srun -N NNODE \
     -n NRANK \
     -c CFLAG \
     --cpu_bind=cores \
     $exexx \
     < xxinput.nl |& grep -v "Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set" \
|& grep -v "In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads" \
|& grep -v "For best performance with OpenMP 3.1 set OMP_PROC_BIND=true" \
|& grep -v "For unit testing set OMP_PROC_BIND=false" \
|& grep -v "OMP: Warning #239: KMP_AFFINITY: granularity=fine will be used." \
|& grep -v "Kokkos::OpenMP thread_pool_topology\[ 1 x 2 x 1 \]"


xname=time-xx-NAME-${SLURM_JOB_ID}
grep COMMIT ${xxdir}/../../CMakeCache.txt > $xname
echo ${SLURM_JOBID} >> $xname
head -n 50 HommeTime_stats >> $xname

rm HommeTime_stats


