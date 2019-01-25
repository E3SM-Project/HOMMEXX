#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # NOTE for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-NAME-%j
#SBATCH --output=slurm-NAME-%j
#SBATCH --nodes=NNODE
#SBATCH -C PARTITION
#SBATCH -t 00:30:00  # who knows how much....
#SBATCH --job-name=NUMEkNNODE # format to shrink job name


hommedir=${HOME}/runhomme/benchng/builds/bld-hor-on-col-off/src/preqx/
hommeSLdir=${HOME}/runhomme/benchng/builds/bld-hor-on-col-off-SL/src/preqx/
xxdir=${HOME}/runhomme/benchng/builds/bldxx-omp/src/preqx/

exehomme=${hommedir}/preqx
exehommeSL=${hommeSLdir}/preqx
exexx=${xxdir}/preqx

export OMP_NUM_THREADS=NTHR
export KMP_AFFINITY=balanced
export OMP_STACKSIZE=16M

#homme run
srun -N NNODE \
     -n NRANK \
     -c CFLAG \
     --cpu_bind=cores \
     $exehomme \
     < hommeinput.nl |& grep -v "KMP_AFFINITY:"

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
     < hommeinput-SL.nl |& grep -v "KMP_AFFINITY:" |& grep -v "Kokkos::OpenMP" |& grep -v "OMP_PROC_BIND"

hSLname=time-hommeSL-NAME-${SLURM_JOB_ID}
grep COMMIT ${hommeSLdir}/../../CMakeCache.txt > $hSLname
echo ${SLURM_JOBID} >> $hSLname
head -n 50 HommeTime_stats >> $hSLname

rm HommeTime_stats

#hommexx run
srun -N NNODE \
     -n NRANK \
     -c CFLAG \
     --cpu_bind=cores \
     $exexx \
     < xxinput.nl |& grep -v "KMP_AFFINITY:" |& grep -v "Kokkos::OpenMP" |& grep -v "OMP_PROC_BIND"

xname=time-xx-NAME-${SLURM_JOB_ID}
grep COMMIT ${xxdir}/../../CMakeCache.txt > $xname
echo ${SLURM_JOBID} >> $xname
head -n 50 HommeTime_stats >> $xname

rm HommeTime_stats


