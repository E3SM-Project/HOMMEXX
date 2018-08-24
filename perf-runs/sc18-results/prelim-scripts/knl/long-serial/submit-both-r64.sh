#!/bin/bash -f
#SBATCH --job-name j
#SBATCH -C knl
#SBATCH -p regular # debug regular?
#SBATCH --time=00:15:00 # change to 15 min
#SBATCH --nodes=43 #for homme it takes 8 min for 1 day, we do 10 tsteps now
#SBATCH --output=slurm-ne120
#SBATCH --error=slurm-ne120
#SBATCH --account=m2664

ne=120
PER_NODE=64         #  MPI per node
#ht=4

#homme params
nlh=homme-ne${ne}-v1.nl
prefh=homme-ne${ne}-r${PER_NODE}-
outph=output-${prefh}

#xx params
nlxx=xx-ne${ne}-v1.nl
prefxx=xx-ne${ne}-r${PER_NODE}-
outpxx=output-${prefxx}

source /etc/profile.d/modules.sh
module load cmake cray-hdf5-parallel/1.10.0 cray-netcdf-hdf5parallel
module unload craype-haswell
module load craype-mic-knl

#next time, build homme in this folder
exech=../short-serial/bld/src/preqx/preqx
execxx=../bxx-master-serial/test_execs/prtcB_flat_c/prtcB_flat_c

#copied from mt
# NE30
# good nodes: 338  169    85  43    22  11  6
# elem/node    16   32    64  128 
# NE120
# good nodes:  5400 2700 1350  675  338      225  169      85          43          22
# ele/node       16   32   64  128  256/255  384  512/511  1024/1023  2048/2047   4096/4095
#set namelist = v1-ne120.nl ; set name = ne120

#export OMP_STACKSIZE=16M    # needed?
#export OMP_DISPLAY_ENV=TRUE
#export OMP_PROC_BIND=spread #ignored since kmp_affinity is set
#export KMP_AFFINITY="verbose"
#export OMP_PLACES=threads #ignored because of kmp_affinity
export KMP_AFFINITY=balanced
NNODES=${SLURM_NNODES}

#for ne120 and 43 nodes, used 2752=43*64 mpis
NMPI=$((PER_NODE*NNODES))
echo "nmpi = ${NMPI}"
nelem=$((6*ne*ne))
echo "nelem = ${nelem}"
NMPI=$(( NMPI < nelem ? NMPI : nelem ))
echo "nmpi after min = ${NMPI}"
#export OMP_NUM_THREADS=$ht
bind=cores
cflag=4

#nersc advice: check aff and bindings with lightweight check-hybrid.intel.cori
#comm="srun -N ${NNODES} -n ${NMPI}  --threads-per-core=4 --cpus-per-task=${nthr} --cpu_bind=${bind} -C knl -c ${cflag} ${exec}"

#rm -f input.nl
#sed s/NThreads.\*/NThreads=$ht/ ${nlh} > input.nl

commh="srun -N ${NNODES} -n ${NMPI} --cpu_bind=${bind} -C knl -c ${cflag} ${exech}"
echo "homme command is ${commh}" > $outph
${commh} < ${nlh} &>> $outph   
#grep "tl-sc vertical\|prim_main\|_RK2\|U3\-5\|vis_dp" HommeTime_stats
mv HommeTime_stats time-${prefh}

commxx="srun -N ${NNODES} -n ${NMPI} --cpu_bind=${bind} -C knl -c ${cflag} ${execxx}"
echo "xx command is ${commxx}" > $outpxx
${commxx} < ${nlxx} &>> $outpxx    
#grep "tl-sc vertical\|prim_main\|_RK2\|U3\-5\|vis_dp" HommeTime_stats
mv HommeTime_stats time-${prefxx}





