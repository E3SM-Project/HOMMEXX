#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # NOTE for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-ne120-machknl-nnode2700-nmax13500-t2 # do not use custom output file?
#SBATCH --output=slurm-ne120-machknl-nnode2700-nmax13500-t2
#SBATCH --nodes=2700
#SBATCH -C knl
#SBATCH -t 00:30:00  # who knows how much....
#SBATCH --job-name=k120-2700


hommedir=${HOME}/runhomme/sc18/builds/bld-hor-off-col-on/src/preqx/
xxdir=${HOME}/runhomme/sc18/builds/bldxx-omp/test_execs/prtcB_flat_c/

exehomme=${hommedir}/preqx
exexx=${xxdir}/prtcB_flat_c

export OMP_NUM_THREADS=2
export KMP_AFFINITY=balanced
export OMP_STACKSIZE=16M

#homme run
srun -N 2700 \
     -n 86400 \
     -c 8 \
     --cpu_bind=cores \
     $exehomme \
     < hommeinput.nl

grep COMMIT ${hommedir}/../../CMakeCache.txt > time-homme-ne120-machknl-nnode2700-nmax13500-t2
echo ${SLURM_JOBID} >> time-homme-ne120-machknl-nnode2700-nmax13500-t2
head -n 50 HommeTime_stats >> time-homme-ne120-machknl-nnode2700-nmax13500-t2

rm HommeTime_stats

#homme run
srun -N 2700 \
     -n 86400 \
     -c 8 \
     --cpu_bind=cores \
     $exexx \
     < xxinput.nl

grep COMMIT ${xxdir}/../../CMakeCache.txt > time-xx-ne120-machknl-nnode2700-nmax13500-t2
echo ${SLURM_JOBID} >> time-xx-ne120-machknl-nnode2700-nmax13500-t2
head -n 50 HommeTime_stats >> time-xx-ne120-machknl-nnode2700-nmax13500-t2

rm HommeTime_stats


