#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # note for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-NAME # do not use custom output file?
#SBATCH --output=slurm-NAME
#SBATCH --nodes=NNODE
#SBATCH -t 00:30:00  # who knows how much....
#SBATCH --job-name=NUMEeNNODE # format to shrink job name

hommedir=${HOME}/runhomme-edison/benchv1/builds/bld/src/preqx/
hommedirSL=${HOME}/runhomme-edison/benchv1/builds/bld-SL/src/preqx/
exehomme=${hommedir}/preqx
exehommeSL=${hommedirSL}/preqx

xxdir=${HOME}/runhomme-edison/benchv1/builds/bldxx/test_execs/prtcB_c/
exexx=${xxdir}/prtcB_c

#export KMP_AFFINITY=balanced

#homme run
srun -N NNODE \
     -n NRANK \
     $exehomme \
     < hommeinput.nl

grep COMMIT ${hommedir}/../../CMakeCache.txt > time-homme-NAME
echo ${SLURM_JOBID} >> time-homme-NAME
head -n 50 HommeTime_stats >> time-homme-NAME

rm HommeTime_stats

#hommeSL run
srun -N NNODE \
     -n NRANK \
     $exehommeSL \
     < hommeinput-SL.nl

grep COMMIT ${hommedir}/../../CMakeCache.txt > time-hommeSL-NAME
echo ${SLURM_JOBID} >> time-hommeSL-NAME
head -n 50 HommeTime_stats >> time-hommeSL-NAME

rm HommeTime_stats

#xx run
srun -N NNODE \
     -n NRANK \
     $exexx \
     < xxinput.nl

grep COMMIT ${xxdir}/../../CMakeCache.txt > time-xx-NAME
echo ${SLURM_JOBID} >> time-xx-NAME
head -n 50 HommeTime_stats >> time-xx-NAME

rm HommeTime_stats


