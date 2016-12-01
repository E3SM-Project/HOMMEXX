# CMake initial cache file for Sandia's bowman
# tested with stock intel & openmpi

SET (CMAKE_Fortran_COMPILER mpif90 CACHE FILEPATH "")
SET (CMAKE_C_COMPILER mpicc CACHE FILEPATH "")
SET (CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "")

SET (FORCE_Fortran_FLAGS "-traceback -fp-model precise -ftz -g -O2" CACHE STRING "")

# Michael's prefix directory which contains HDF5, PNetCDF, NetCDF, and NetCDF Fortran
SET (MDEAKIN_PREFIX "/home/mdeakin/prefix" FILEPATH "")

SET (PNETCDF_DIR ${MDEAKIN_PREFIX} CACHE FILEPATH "")
SET (NETCDF_DIR ${MDEAKIN_PREFIX} CACHE FILEPATH "")
SET (HDF5_DIR ${MDEAKIN_PREFIX} CACHE FILEPATH "")
SET (ZLIB_DIR $ENV{ZLIB_ROOT} CACHE FILEPATH "")

SET (USE_QUEUING FALSE CACHE BOOL "")
SET (HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")
