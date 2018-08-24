#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # NOTE for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-ne120-machknl-nnode5400-nmax13500-t4 # do not use custom output file?
#SBATCH --output=slurm-ne120-machknl-nnode5400-nmax13500-t4
#SBATCH --nodes=5400
#SBATCH -C knl
#SBATCH -t 00:30:00  # who knows how much....
#SBATCH --job-name=k120-5400


hommedir=${HOME}/runhomme/sc18/builds/bld-hor-off-col-on/src/preqx/
xxdir=${HOME}/runhomme/sc18/builds/bldxx-omp/test_execs/prtcB_flat_c/

exehomme=${hommedir}/preqx
exexx=${xxdir}/prtcB_flat_c

export OMP_NUM_THREADS=4
export KMP_AFFINITY=balanced,granularity=fine
export OMP_STACKSIZE=16M

name=ne120-machknl-nnode5400-nmax13500-t4

#hommexx run
srun -N 5400 \
     -n 86400 \
     -c 16 \
     --cpu_bind=cores \
     $exexx \
     < xxinput.nl

grep COMMIT ${xxdir}/../../CMakeCache.txt > time-xx-${name}
echo ${SLURM_JOBID} >> time-xx-${name}
head -n 50 HommeTime_stats >> time-xx-${name}

rm HommeTime_stats

#homme run
srun -N 5400 \
     -n 86400 \
     -c 16 \
     --cpu_bind=cores \
     $exehomme \
     < hommeinput.nl

grep COMMIT ${hommedir}/../../CMakeCache.txt > time-homme-${name}
echo ${SLURM_JOBID} >> time-homme-${name}
head -n 50 HommeTime_stats >> time-homme-${name}

rm HommeTime_stats


