#!/bin/bash -f
#SBATCH --account=m2664
#SBATCH -p regular # note for debug que chaining (script to submit a script) is not allowed
#SBATCH --error=slurm-NAME # do not use custom output file?
#SBATCH --output=slurm-NAME
#SBATCH --nodes=NNODE
#SBATCH -t 00:30:00  # who knows how much....


hommedir=${HOME}/runhomme-edison/sc18/builds/bld/src/preqx/
exehomme=${hommedir}/preqx
xxdir=${HOME}/runhomme-edison/sc18/builds/bldxx/test_execs/prtcB_flat_c/
exexx=${xxdir}/prtcB_flat_c

#export KMP_AFFINITY=balanced

#homme run
srun -N NNODE \
     -n NRANK \
     $exehomme \
     < hommeinput.nl

grep COMMIT ${hommedir}/../../CMakeCache.txt > time-homme-NAME
head -n 50 HommeTime_stats >> time-homme-NAME

#homme run
srun -N NNODE \
     -n NRANK \
     $exexx \
     < xxinput.nl

grep COMMIT ${xxdir}/../../CMakeCache.txt > time-xx-NAME
head -n 50 HommeTime_stats >> time-xx-NAME

rm HommeTime_stats


