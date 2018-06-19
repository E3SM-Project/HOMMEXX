# other prep:
#
# ./kokkos/generate_makefile.bash --with-openmp --with-options=aggressive_vectorization --prefix=/home/ambradl/lib/kokkos/arm --with_options=disable_profiling --arch=ARMv8-TX2
#
# module load netcdf-fortran/4.4.4 openblas/0.2.20/gcc/7.2.0

SET (WITH_PNETCDF FALSE CACHE FILEPATH "")
SET (NETCDFF_DIR $ENV{NETCDF_FORTRAN_DIR} CACHE FILEPATH "")
SET (NETCDF_DIR $ENV{NETCDF_DIR} CACHE FILEPATH "")
SET (HDF5_DIR $ENV{HDF5_DIR} CACHE FILEPATH "")
SET (NETCDF_Fortran_FOUND TRUE CACHE BOOL "")

SET (USE_QUEUING FALSE CACHE BOOL "")
set (USE_NUM_PROCS 24 CACHE STRING "")

SET (HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

set (USE_TRILINOS FALSE CACHE BOOL "")
set (KOKKOS_PATH "/home/ambradl/lib/kokkos/arm" CACHE FILEPATH "")

set (CMAKE_Fortran_FLAGS "-g" CACHE STRING "")
set (CMAKE_C_FLAGS "-g" CACHE STRING "")
set (CMAKE_CXX_FLAGS "-g" CACHE STRING "")
set (CMAKE_EXE_LINKER_FLAGS "-ldl" CACHE STRING "")

set (ENABLE_HORIZ_OPENMP FALSE CACHE BOOL "")
set (ENABLE_COLUMN_OPENMP TRUE CACHE BOOL "")

SET (USE_MPI_OPTIONS "--bind-to core" CACHE FILEPATH "")
