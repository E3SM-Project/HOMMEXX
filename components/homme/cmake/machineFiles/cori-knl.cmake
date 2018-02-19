### Cori setup

###modules, get rid of haswell 
#source /etc/profile.d/modules.sh
#module load cmake cray-hdf5-parallel/1.10.0 cray-netcdf-hdf5parallel
#module load PrgEnv-intel ; module unload craype-haswell ; module load craype-mic-knl

###build kokkos
#../generate_makefile.bash --prefix=~/kokkos/build-omp-nodebug/ --with-options=aggressive_vectorization --with-options=disable_profiling --arch=KNL --with-openmp --compiler=icpc
#make; make install

###config hommexx
#~/runhomme/bxx-master> cmake -C ~/acmexx/components/homme/cmake/machineFiles/cori-knl.cmake -DUSE_TRILINOS=OFF ~/acmexx/components/homme/

###make allocation
#salloc --account=m2664 --qos=interactive -t 10 -C knl -N 1

###running, NO BINDING< AFFINITY, OR ANYTHING for now
#bash:~/runhomme/bxx-master/test_execs/prtcA_flat_c> srun -n 8 ./prtcA_flat_c < namelist.nl




#SET (HOMME_FIND_BLASLAPACK "TRUE" CACHE FILEPATH "")
SET (HOMME_USE_MKL "TRUE" CACHE FILEPATH "") # for Intel

set (ENABLE_INTEL_PHI TRUE CACHE BOOL "")
SET (CMAKE_Fortran_COMPILER ftn CACHE FILEPATH "")
SET (CMAKE_C_COMPILER cc CACHE FILEPATH "")
SET (CMAKE_CXX_COMPILER CC CACHE FILEPATH "")
SET (CMAKE_CXX_FLAGS "-craype-verbose" CACHE STRING "")
SET (CMAKE_Fortran_FLAGS "-craype-verbose" CACHE STRING "")
SET (NETCDF_DIR $ENV{NETCDF_DIR} CACHE FILEPATH "")

#example with module cray-netcdf-hdf5parallel/4.3.3.1:
# NETCDF_DIR=/opt/cray/netcdf-hdf5parallel/4.3.3.1/INTEL/14.0
#ndk SET (PNETCDF_DIR $ENV{PARALLEL_NETCDF_DIR} CACHE FILEPATH "")
# this env var is not set with module cray-netcdf-hdf5parallel/4.3.3.1
SET (HDF5_DIR $ENV{HDF5_DIR} CACHE FILEPATH "")
#example with module cray-hdf5-parallel/1.8.14: 
# HDF5_DIR=/opt/cray/hdf5-parallel/1.8.14/INTEL/14.0
SET (CMAKE_SYSTEM_NAME Catamount CACHE FILEPATH "")
SET(USE_TRILINOS OFF CACHE BOOL "")
SET(KOKKOS_PATH "/global/homes/o/onguba/kokkos/build-omp-nodebug/" CACHE STRING "")
SET (USE_QUEUING FALSE CACHE BOOL "")
SET (USE_MPIEXEC "srun" CACHE STRING "")


