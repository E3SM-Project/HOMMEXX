
SET (CMAKE_Fortran_COMPILER mpif90 CACHE FILEPATH "")
SET (CMAKE_C_COMPILER mpicc CACHE FILEPATH "") 
SET (CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "")

# I'll send a tar file with these TPLs.
SET (WITH_PNETCDF FALSE CACHE FILEPATH "") 
SET (NETCDF_DIR /home/projects/hommexx/TPL/ellis-netcdf4 CACHE FILEPATH "") 
SET (HDF5_DIR /home/projects/hommexx/TPL/ellis-netcdf4 CACHE FILEPATH "")
SET (ZLIB_DIR $ENV{ZLIB_ROOT} CACHE FILEPATH "")

SET (USE_QUEUING FALSE CACHE BOOL "")
# 24 is big enough to be useful but small enough to get ne=2 case right. At some # point we should improve things for more flexibility.
set (USE_NUM_PROCS 24 CACHE STRING "")

SET (HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

# Don't use Trilinos.
set (USE_TRILINOS FALSE CACHE BOOL "")
# Use standalone Kokkos.
set (KOKKOS_PATH "/home/ambradl/lib/kokkos/knl" CACHE FILEPATH "") # To trigger KNL vectorization.
set (ENABLE_INTEL_PHI TRUE CACHE BOOL "")

set (CMAKE_Fortran_FLAGS "-g -O1" CACHE STRING "") 
set (CMAKE_C_FLAGS "-g -O3" CACHE STRING "") 
set (CMAKE_CXX_FLAGS "-g -O3" CACHE STRING "") 
set (CMAKE_EXE_LINKER_FLAGS "-ldl" CACHE STRING "")

# These don't matter for the C++ code, only the Fortran. Just pick something # reasonable.
set (ENABLE_HORIZ_OPENMP TRUE CACHE BOOL "") 
set (ENABLE_COLUMN_OPENMP FALSE CACHE BOOL "")

# Need this to get MPI + threads to have reasonable affinities.
SET (USE_MPI_OPTIONS "--bind-to core" CACHE FILEPATH "")


