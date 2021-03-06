# 
# CMake initial cache file for Sandia's redsky
# configurd for redsky's netcdf-intel/4.1 module 
#
SET (CMAKE_Fortran_COMPILER mpif90 CACHE FILEPATH "")
SET (CMAKE_C_COMPILER mpicc CACHE FILEPATH "")
SET (CMAKE_CXX_COMPILER mpicc CACHE FILEPATH "")

# Openmpi 1.8 only
#SET (USE_MPI_OPTIONS "--map-by node:SPAN" CACHE FILEPATH "")

# Openmpi 1.6
# IKT, 7/28/17: commented out the following option, which only works with omp1.6 (not opm1.8)
#SET (USE_MPI_OPTIONS "-loadbalance" CACHE FILEPATH "")

# this is ignored if we use FORCE_Fortran_FLAGS
SET (ADD_Fortran_FLAGS "-traceback" CACHE STRING "")

#this way opt. flags won't be overwritten
set(CMAKE_BUILD_TYPE "" CACHE STRING "")

set (CMAKE_Fortran_FLAGS "-g -O1" CACHE STRING "")
set (CMAKE_C_FLAGS "-g -O3" CACHE STRING "")
set (CMAKE_CXX_FLAGS "-g -O3" CACHE STRING "")


SET (NETCDF_DIR $ENV{SEMS_NETCDF_ROOT} CACHE FILEPATH "")
SET (HDF5_DIR $ENV{SEMS_HDF5_ROOT} CACHE FILEPATH "")

SET (USE_QUEUING FALSE CACHE BOOL "")
SET (HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

SET (WHICH_MACHINE "skybridge" CACHE STRING "")



