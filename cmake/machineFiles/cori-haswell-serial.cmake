
#SET (HOMME_FIND_BLASLAPACK "TRUE" CACHE FILEPATH "")
SET (HOMME_USE_MKL "TRUE" CACHE FILEPATH "") # for Intel

SET (CMAKE_Fortran_COMPILER ftn CACHE FILEPATH "")
SET (CMAKE_C_COMPILER cc CACHE FILEPATH "")
SET (CMAKE_CXX_COMPILER CC CACHE FILEPATH "")

SET (CMAKE_CXX_FLAGS "-g -fopenmp -craype-verbose" CACHE STRING "")
SET (CMAKE_Fortran_FLAGS "-craype-verbose" CACHE STRING "")

SET (NETCDF_DIR $ENV{NETCDF_DIR} CACHE FILEPATH "")
SET (HDF5_DIR $ENV{HDF5_DIR} CACHE FILEPATH "")

set (ENABLE_HORIZ_OPENMP FALSE CACHE BOOL "")
set (ENABLE_COLUMN_OPENMP FALSE CACHE BOOL "")

SET (CMAKE_SYSTEM_NAME Catamount CACHE FILEPATH "")
SET (USE_TRILINOS OFF CACHE BOOL "")
SET (KOKKOS_PATH "/global/homes/o/onguba/kokkos/build-serial-nodebug-haswell/" CACHE STRING "")
SET (USE_QUEUING FALSE CACHE BOOL "")
SET (USE_MPIEXEC "srun" CACHE STRING "")


