SET(NETCDF_DIR $ENV{NETCDF_ROOT} CACHE FILEPATH "")
SET(NETCDF_Fortran_DIR $ENV{NETCDFF_ROOT} CACHE FILEPATH "")
SET(NETCDFF_DIR $ENV{NETCDFF_ROOT} CACHE FILEPATH "")
SET(PNETCDF_DIR $ENV{PNETCDF_ROOT} CACHE FILEPATH "")

#SET(HDF5_C_LIBRARY_hdf5 -lhdf5_hl -lhdf5 CACHE STRING "")
SET(NetcdfF_LIBRARY -lnetcdff -lhdf5_hl -lhdf5 CACHE LIST "")

SET(HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

SET(USE_QUEUING FALSE CACHE BOOL "")

SET(CMAKE_C_COMPILER "mpicc" CACHE STRING "")
SET(CMAKE_CXX_COMPILER "mpicxx" CACHE STRING "")
SET(CMAKE_Fortran_COMPILER "mpifort" CACHE STRING "")

SET(ENABLE_CUDA FALSE CACHE BOOL "")

SET(PIO_USE_MPIMOD OFF CACHE BOOL "")

SET(USE_TRILINOS OFF CACHE BOOL "")
SET(KOKKOS_PATH "/home/mdeakin/prefix" CACHE STRING "")
