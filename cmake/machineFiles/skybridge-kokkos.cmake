#OG my profile on skybridge:
#module purge
#source /projects/sems/modulefiles/utils/sems-modules-init.sh
#module load sems-intel/17.0.0
#module load sems-openmpi/1.10.5
#module load sems-python/2.7.9
#module load sems-cmake/3.5.2
#module load sems-git
#module load sems-hdf5/1.8.12/base
#module load sems-netcdf/4.4.1/exo
####for homme
#export NETCDF_PATH=${SEMS_NETCDF_ROOT}
#export PATH=${NETCDF_PATH}/bin/:${PATH}
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$NETCDF_PATH/lib

SET (CMAKE_Fortran_COMPILER mpif90 CACHE FILEPATH "")
SET (CMAKE_C_COMPILER mpicc CACHE FILEPATH "")
SET (CMAKE_CXX_COMPILER mpicc CACHE FILEPATH "")


#this way opt. flags won't be overwritten
set(CMAKE_BUILD_TYPE "" CACHE STRING "")

set (CMAKE_Fortran_FLAGS "-g -O1" CACHE STRING "")
set (CMAKE_C_FLAGS "-g -O3" CACHE STRING "")
set (CMAKE_CXX_FLAGS "-g -O3" CACHE STRING "")


SET (WITH_PNETCDF FALSE CACHE FILEPATH "")
SET (NETCDF_DIR $ENV{SEMS_NETCDF_ROOT} CACHE FILEPATH "")
SET (PNETCDF_DIR $ENV{SEMS_NETCDF_ROOT} CACHE FILEPATH "")
SET (HDF5_DIR $ENV{SEMS_HDF5_ROOT} CACHE FILEPATH "")
SET (ZLIB_DIR $ENV{SEMS_ZLIB_ROOT} CACHE FILEPATH "")

SET(USE_TRILINOS OFF CACHE BOOL "")
SET(KOKKOS_PATH "/ascldap/users/onguba/kokkos/build-single-nodebug/install/" CACHE STRING "")

SET (USE_QUEUING FALSE CACHE BOOL "")
SET (HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")
