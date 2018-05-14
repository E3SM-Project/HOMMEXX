#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # note for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-NAME # do not use custom output file?
#SBATCH --output=slurm-NAME
#SBATCH --nodes=NNODE
#SBATCH -t 00:30:00  # who knows how much....
#SBATCH --job-name=NUMEeNNODE # format to shrink job name

hommedir=${HOME}/runhomme-edison/benchng/builds/bld/src/preqx/
hommedirSL=${HOME}/runhomme-edison/benchng/builds/bld-SL/src/preqx/
exehomme=${hommedir}/preqx
exehommeSL=${hommedirSL}/preqx

xxdir=${HOME}/runhomme-edison/benchng/builds/bldxx/test_execs/prtcB_c/
exexx=${xxdir}/prtcB_c

#export KMP_AFFINITY=balanced

#homme run
srun -N NNODE \
     -n NRANK \
     $exehomme \
     < hommeinput.nl

hname=time-homme-NAME-${SLURM_JOB_ID}
grep COMMIT ${hommedir}/../../CMakeCache.txt > $hname
echo ${SLURM_JOBID} >> $hname
head -n 50 HommeTime_stats >> $hname

rm HommeTime_stats

#hommeSL run
srun -N NNODE \
     -n NRANK \
     $exehommeSL \
     < hommeinput-SL.nl

hSLname=time-hommeSL-NAME-${SLURM_JOB_ID}
grep COMMIT ${hommedir}/../../CMakeCache.txt > $hSLname
echo ${SLURM_JOBID} >> $hSLname
head -n 50 HommeTime_stats >> $hSLname

rm HommeTime_stats

#xx run
srun -N NNODE \
     -n NRANK \
     $exexx \
     < xxinput.nl

xname=time-xx-NAME-${SLURM_JOB_ID}
grep COMMIT ${xxdir}/../../CMakeCache.txt > $xname
echo ${SLURM_JOBID} >> $xname
head -n 50 HommeTime_stats >> $xname

rm HommeTime_stats


