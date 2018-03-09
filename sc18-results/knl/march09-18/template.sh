#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # NOTE for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-NAME # do not use custom output file?
#SBATCH --output=slurm-NAME
#SBATCH --nodes=NNODE
#SBATCH -C PARTITION
#SBATCH -t 00:30:00  # who knows how much....


exehomme=${HOME}/runhomme/sc18/builds/bld/src/preqx/preqx
exexx=${HOME}/runhomme/sc18/builds/bldxx/test_execs/prtcB_flat_c/prtcB_flat_c


export OMP_NUM_THREADS=NTHR
export KMP_AFFINITY=balanced

#homme run
srun -N NNODE \
     -n NRANK \
     -c CFLAG \
     --cpu_bind=cores \
     $exehomme \
     < hommeinput.nl

mv HommeTime_stats time-homme-NAME

#homme run
srun -N NNODE \
     -n NRANK \
     -c CFLAG \
     --cpu_bind=cores \
     $exexx \
     < xxinput.nl

mv HommeTime_stats time-xx-NAME




