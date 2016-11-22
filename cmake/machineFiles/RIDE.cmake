# Note: CMAKE_CXX_COMPILER needs to be set to the path of nvcc_wrapper
# nvcc_wrapper will choose either the Nvidia Cuda compiler or the OpenMP compiler depending on what's being compiled

SET(NETCDF_DIR $ENV{NETCDF_ROOT} CACHE FILEPATH "")
SET(PNETCDF_DIR $ENV{PNETCDF_ROOT} CACHE FILEPATH "")

SET(NETCDF_PREFIX /home/mdeakin/prefix)

# Hacky way of ensuring needed libraries are linked in the proper order
SET(NetcdfF_LIBRARY ${NETCDF_PREFIX}/lib64/libnetcdff.so -lhdf5_hl -lhdf5 -ldl CACHE LIST "")
SET(NetcdfF_INCLUDE_DIR ${NETCDF_PREFIX}/include CACHE FILEPATH "")

SET(HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

SET(USE_QUEUING FALSE CACHE BOOL "")

SET(CMAKE_C_COMPILER "mpicc" CACHE STRING "")
SET(CMAKE_Fortran_COMPILER "mpifort" CACHE STRING "")

SET(ENABLE_CUDA TRUE CACHE BOOL "")
