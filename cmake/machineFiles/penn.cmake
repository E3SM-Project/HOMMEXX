# CMake initial cache file for Linux 64bit RHEL6/CENTOS6
# tested with stock gcc/gfortran & openmpi
#

message("--In penn.cmake ")

SET (CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "")

SET (CMAKE_Fortran_COMPILER mpif90 CACHE FILEPATH "")
SET (CMAKE_C_COMPILER mpicc CACHE FILEPATH "")
SET (CMAKE_CXX_COMPILER mpicxx CACHE FILEPATH "")

SET (WITH_PNETCDF TRUE CACHE FILEPATH "")
SET (NETCDF_DIR $ENV{SEMS_NETCDF_ROOT} CACHE FILEPATH "")
SET (PNETCDF_DIR $ENV{SEMS_NETCDF_ROOT} CACHE FILEPATH "")
SET (HDF5_DIR $ENV{SEMS_HDF5_ROOT} CACHE FILEPATH "")
SET (ZLIB_DIR $ENV{SEMS_ZLIB_ROOT} CACHE FILEPATH "")

# hack until findnetcdf is updated to look for netcdf.mod
# but this is ignored by cprnc
# SET (ADD_Fortran_FLAGS "-I/usr/lib64/gfortran/modules" CACHE STRING "")

SET (USE_QUEUING FALSE CACHE BOOL "")
SET (HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

#SET(USE_TRILINOS OFF CACHE BOOL "")
SET(USE_TRILINOS ON CACHE BOOL "")
#SET (TRILINOS_INSTALL_PATH "/home/agsalin/Trilinos/build/install" CACHE PATH "Path to Trilinos install")

#SET (KOKKOS_PATH "/home/agsalin/kokkos/build/install" CACHE PATH "Path to Kokkos install")

SET (BUILD_HOMME_PREQX_FLAT ON CACHE BOOL "")
SET (BUILD_HOMME_SWEQX_FLAT OFF CACHE BOOL "")
