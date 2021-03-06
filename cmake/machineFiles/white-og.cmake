#for modules, see .bashrc file

#for BFB testing, after cloning kokkos, go to bin/nvcc_wrapper and change:
# Cuda (NVCC) only arguments
# -cuda_args=""
# +cuda_args="-fmad=false"

# building kokkos for q -rhel7G (pascal60)
#../generate_makefile.bash --with-serial --with-cuda --with-options=aggressive_vectorization --prefix=/home/onguba/kokkos/build-nodebug-white --arch="Power8,Pascal60" --with-cuda-options=enable_lambda --compiler=/home/onguba/kokkos/bin/nvcc_wrapper
#for -rhel7F (Kepler37) it would be "Power8,Kepler37"

#
# Note: CMAKE_CXX_COMPILER needs to be set to the path of nvcc_wrapper
# nvcc_wrapper will choose either the Nvidia Cuda compiler or the OpenMP compiler depending on what's being compiled

SET(NETCDF_DIR $ENV{NETCDF_ROOT} CACHE FILEPATH "")
SET(PNETCDF_DIR $ENV{PNETCDF_ROOT} CACHE FILEPATH "")

SET(NETCDF_Fortran_DIR $ENV{NETCDFF_ROOT} CACHE FILEPATH "")
SET(NETCDFF_DIR $ENV{NETCDFF_ROOT} CACHE FILEPATH "")

# Hacky way of ensuring needed libraries are linked in the proper order
SET(NetcdfF_LIBRARY $ENV{NETCDFF_ROOT}/lib/libnetcdff.a -lnetcdf -lhdf5_hl -lhdf5 -ldl -lz CACHE LIST "")
SET(ZLIB_DIR $ENV{ZLIB_ROOT} CACHE FILEPATH "")

SET(HOMME_FIND_BLASLAPACK TRUE CACHE BOOL "")

SET(USE_QUEUING FALSE CACHE BOOL "")

SET(ENABLE_CUDA TRUE CACHE BOOL "")

SET(USE_TRILINOS OFF CACHE BOOL "")
#SET(KOKKOS_PATH "/home/mdeakin/prefix" CACHE STRING "")
SET(KOKKOS_PATH "/ascldap/users/onguba/kokkos/build-debug-white" CACHE STRING "")

SET(CMAKE_C_COMPILER "mpicc" CACHE STRING "")
SET(CMAKE_CXX_COMPILER "/ascldap/users/onguba/kokkos/bin/nvcc_wrapper" CACHE STRING "")
SET(CMAKE_Fortran_COMPILER "mpifort" CACHE STRING "")
