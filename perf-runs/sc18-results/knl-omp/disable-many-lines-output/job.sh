#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # NOTE for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-ne240-machknl-nnode5400-nmax13500-t2 # do not use custom output file?
#SBATCH --output=slurm-ne240-machknl-nnode5400-nmax13500-t2
#SBATCH --nodes=5400
#SBATCH -C knl
#SBATCH -t 00:30:00  # who knows how much....
#SBATCH --job-name=k240-5400


hommedir=${HOME}/runhomme/sc18/builds/bld-hor-on-col-off/src/preqx/
xxdir=${HOME}/runhomme/sc18/builds/bldxx-omp/test_execs/prtcB_flat_c/

exehomme=${hommedir}/preqx
exexx=${xxdir}/prtcB_flat_c

export OMP_NUM_THREADS=2
export KMP_AFFINITY=balanced,granularity=fine
export OMP_STACKSIZE=16M


#hommeXX run
srun -N 5400 \
     -n 345600 \
     -c 4 \
     --cpu_bind=cores \
     $exexx \
     < xxinput.nl |& grep -v "Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set" \
|& grep -v "In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads" \
|& grep -v "For best performance with OpenMP 3.1 set OMP_PROC_BIND=true" \
|& grep -v "For unit testing set OMP_PROC_BIND=false" \
|& grep -v "Kokkos::OpenMP thread_pool_topology\[ 1 x 2 x 1 \]"

grep COMMIT ${xxdir}/../../CMakeCache.txt > time-xx-ne240-machknl-nnode5400-nmax13500-t2
echo ${SLURM_JOBID} >> time-xx-ne240-machknl-nnode5400-nmax13500-t2
head -n 50 HommeTime_stats >> time-xx-ne240-machknl-nnode5400-nmax13500-t2

rm HommeTime_stats


#homme run
srun -N 5400 \
     -n 345600 \
     -c 4 \
     --cpu_bind=cores \
     $exehomme \
     < hommeinput.nl

grep COMMIT ${hommedir}/../../CMakeCache.txt > time-homme-ne240-machknl-nnode5400-nmax13500-t2
echo ${SLURM_JOBID} >> time-homme-ne240-machknl-nnode5400-nmax13500-t2
head -n 50 HommeTime_stats >> time-homme-ne240-machknl-nnode5400-nmax13500-t2

rm HommeTime_stats





