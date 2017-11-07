source /projects/sems/modulefiles/utils/sems-modules-init.sh
module load intel/16.0
module load openmpi-intel/1.10
module load sems-env 
module add sems-hdf5/1.8.12/base sems-netcdf/4.4.1/exo
module list 
export OMP_NUM_THREADS=2
