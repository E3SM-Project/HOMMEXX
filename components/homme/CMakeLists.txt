CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

SET (CMAKE_BUILD_TYPE RELEASE CACHE STRING "Select build type.")

PROJECT(HOMME C CXX Fortran)

# Print the sha of the last commit (useful to double check which version was tested on CDash)
EXECUTE_PROCESS (COMMAND git rev-parse HEAD
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                 OUTPUT_VARIABLE LAST_GIT_COMMIT_SHA
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
SET (LAST_GIT_COMMIT_SHA ${LAST_GIT_COMMIT_SHA} CACHE STRING "The sha of the last git commit.")

MESSAGE (STATUS "The sha of the last commit is ${LAST_GIT_COMMIT_SHA}")

INCLUDE(FortranCInterface)

# CTest For Nightly/Experimental Builds
OPTION(HOMME_USE_CTEST "Use CTest Dashboard" FALSE)
IF (${HOMME_USE_CTEST})
  INCLUDE(CTest)
ENDIF ()

# For testing
ENABLE_TESTING()
SET (HOMME_TESTING_PROFILE "nightly" CACHE STRING "Determine how long/pervasive the testing is. Currently available options: 'dev', 'short', 'nightly'.")

# Add the cmake/CMakeLists.txt file
SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# need to get the cmake_fortran_c_interface.h header
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
FortranCInterface_HEADER(cmake_fortran_c_interface.h
                         MACRO_NAMESPACE  "FCI_")

# Prefer shared libraries (Yellowstone etc.)
OPTION(PREFER_SHARED "Prefer linking with shared libraries." FALSE)

# need to tell the source to use this capability
ADD_DEFINITIONS(-DINCLUDE_CMAKE_FCI)

SET (PERFORMANCE_PROFILE "" CACHE STRING "Whether to build and link with various profiler libraries")

# Set up some directories to facilitate finding netcdf, pnetcdf
INCLUDE(SetHostPaths)
INCLUDE(HommeMacros)

# Debug statement, please leave here
#MESSAGE(STATUS "CMAKE_IMP_LIBS=${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES}")

OPTION(IS_BIG_ENDIAN "Whether the machine is big endian" FALSE)
#TRY_RUN(IS_BIG_ENDIAN BIG_ENDIAN_COMPILED
#  ${CMAKE_BINARY_DIR}/tests/compilerTests/
#  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/compilerTests/isBigEndian.c)

# Default use quad precision
OPTION(HOMME_QUAD_PREC "Default to try to use quadruple precision" TRUE)

IF (${HOMME_QUAD_PREC})
  testQuadPrec(HOMME_QUAD_SUPPORTED)
  IF (NOT ${HOMME_QUAD_SUPPORTED})
    SET(HOMME_QUAD_PREC FALSE)
    MESSAGE(STATUS "Quadruple-precision requested but unavailable on this
                    system with this compiler")
  ENDIF()
ENDIF ()

# Some options for conditional builds
OPTION(BUILD_HOMME_SWEQX "Shallow water equations FEM" OFF)
OPTION(BUILD_HOMME_SWEQX_FLAT "Shallow water equations FEM with flattened arrays" OFF)
OPTION(BUILD_HOMME_PREQX "Primitive equations FEM" ON)
OPTION(BUILD_HOMME_PREQX_FLAT "Primitive equations FEM" ON)
OPTION(BUILD_HOMME_PREQX_ACC "Primitive equations FEM with OpenACC" OFF)
OPTION(BUILD_HOMME_SWIM  "Shallow water equations implicit" OFF)
OPTION(BUILD_HOMME_PRIM  "Primitive equations implicit" OFF)

# Cristoph's "standalone" executables
OPTION(BUILD_HOMME_FVM  "Standalone FVM tracers" OFF)

OPTION(REFSOLN "Whether to read/write reference soln data" FALSE)
# Reference solution data
IF(REFSOLN)
  MESSAGE(STATUS "Including option to read or write reference soln data...")
  ADD_DEFINITIONS(-D_REFSOLN)
ENDIF()

############################################
# Selection of Kokkos execution space
############################################

SET (HOMMEXX_EXEC_SPACE "Default" CACHE STRING "Select the kokkos exec space")

STRING (TOUPPER ${HOMMEXX_EXEC_SPACE} HOMMEXX_EXEC_SPACE_UPPER)
IF (${HOMMEXX_EXEC_SPACE_UPPER} STREQUAL "CUDA")
  SET (HOMMEXX_CUDA_SPACE ON)
ELSEIF (${HOMMEXX_EXEC_SPACE_UPPER} STREQUAL "OPENMP")
  SET (HOMMEXX_OPENMP_SPACE ON)
ELSEIF (${HOMMEXX_EXEC_SPACE_UPPER} STREQUAL "THREADS")
  SET (HOMMEXX_THREADS_SPACE ON)
ELSEIF (${HOMMEXX_EXEC_SPACE_UPPER} STREQUAL "SERIAL")
  SET (HOMMEXX_SERIAL_SPACE ON)
ELSEIF (${HOMMEXX_EXEC_SPACE_UPPER} STREQUAL "DEFAULT")
  SET (HOMMEXX_DEFAULT_SPACE ON)
ELSE()
  MESSAGE (ABORT "Invalid choice for 'HOMMEXX_EXEC_SPACE'. Valid options (case insensitive) are 'Cuda', 'OpenMP', 'Threads', 'Serial', 'Default'")
ENDIF()

# Execution space parameters. Ideally, these will go away because they can be
# queried at runtime.
SET (HOMMEXX_CUDA_MIN_WARP_PER_TEAM 4 CACHE INT "Minimum number of warps to get 100% occoupancy on GPU")

# An option to allow to use GPU pointers for MPI calls. The value of this option is irrelevant for CPU/KNL builds.
OPTION (HOMMEXX_MPI_ON_DEVICE "Whether we want to use device pointers for MPI calls (relevant only for GPU builds)" ON)

##############################################################################
# Options for floating point model.
##############################################################################

SET (HOMMEXX_FPMODEL "precise" CACHE STRING "Floating point model for HOMMEXX")
SET (HOMMEXX_FPMODEL_UT "strict" CACHE STRING "Floating point model for HOMMEXX unit tests; overrides HOMMEXX_FPMODEL")

##############################################################################
# Compiler options set in cmake/SetCompiler.cmake
##############################################################################

INCLUDE(SetCompilerFlags)

if (${CUDA_FOUND})
  add_definitions(-DHOMMEXX_GPU_BFB_WITH_CPU)
endif ()

# Print the compiler flags to be used
STRING (TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
MESSAGE(STATUS "Fortran Flags ${CMAKE_BUILD_TYPE} = ${CMAKE_Fortran_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
MESSAGE(STATUS "C Flags ${CMAKE_BUILD_TYPE} = ${CMAKE_C_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
MESSAGE(STATUS "CXX Flags ${CMAKE_BUILD_TYPE} = ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")
MESSAGE(STATUS "Fortran Flags = ${CMAKE_Fortran_FLAGS}")
MESSAGE(STATUS "C Flags = ${CMAKE_C_FLAGS}")
MESSAGE(STATUS "CXX Flags = ${CMAKE_CXX_FLAGS}")
MESSAGE(STATUS "Linker Flags = ${CMAKE_EXE_LINKER_FLAGS}")

##############################################################################
# Trilinos Configuration depends on available compiler options
##############################################################################
INCLUDE(Trilinos)

##############################################################################

OPTION(ENABLE_PERSISTENT_MPI "Enable persistent MPI" OFF)
IF (${ENABLE_PERSISTENT_MPI})
  SET(MPI_PERSISTENT TRUE)
ENDIF ()

OPTION(HOMME_USE_MKL "Whether to use Intel's MKL instead of blas/lapack" FALSE)
MESSAGE(STATUS "HOMME_USE_MKL=${HOMME_USE_MKL}")
IF(${HOMME_USE_MKL})
  MESSAGE(STATUS "For Edison/Cori at NERSC, MKL included with Intel compiler")
ELSE()
  OPTION(HOMME_FIND_BLASLAPACK "Whether to use system blas/lapack" FALSE)
  MESSAGE(STATUS "HOMME_FIND_BLASLAPACK=${HOMME_FIND_BLASLAPACK}")

  IF(${HOMME_FIND_BLASLAPACK})
    FIND_PACKAGE(BLAS REQUIRED)
    FIND_PACKAGE(LAPACK REQUIRED)
  ELSE()
    SET(HOMME_FIND_BLASLAPACK FALSE)
    # Build the blas and lapack provided with homme
    ADD_SUBDIRECTORY(libs/blas)
    ADD_SUBDIRECTORY(libs/lapack)
  ENDIF()
ENDIF()

# Zoltan partitioning library
IF (DEFINED ZOLTAN_DIR)
  MESSAGE(STATUS "Building with Zoltan")
  FIND_PACKAGE(Zoltan REQUIRED)
  SET(HAVE_ZOLTAN TRUE)
ENDIF ()

IF (DEFINED EXTRAE_DIR)
  MESSAGE(STATUS "Building with Extrae")
  FIND_PACKAGE(Extrae REQUIRED)
  SET(HAVE_EXTRAE TRUE)
ENDIF ()

###########################################
# Definitions for timing library
###########################################
# This is required in timing to set HAVE_MPI
ADD_DEFINITIONS(-DSPMD)
OPTION(ENABLE_NANOTIMERS "Use nano timers in timing library" FALSE)
OPTION(USE_BIT64 "Set BIT64 (for 64 bit arch) in timing library when ENABLE_NANOTIME" FALSE)
IF (${ENABLE_NANOTIMERS})
  ADD_DEFINITIONS(-DHAVE_NANOTIME)
  IF (${USE_BIT64})
    ADD_DEFINITIONS(-DBIT64)
  ENDIF ()
ENDIF ()

# Use ccsm_utils FindNETCDF path
SET (CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cime/externals/CMake" ${CMAKE_MODULE_PATH})
FIND_PACKAGE(NETCDF)
# For pio - allow the user to set HDF5_DIR to find libhdf5 etc.
IF (HDF5_DIR)
  SET(ENV{HDF5_ROOT} ${HDF5_DIR})
ENDIF ()
ADD_SUBDIRECTORY(utils/pio)

# CPRNC locations
SET (CPRNC_INSTALL_DIR ${CMAKE_BINARY_DIR}/utils/cprnc)
SET (CPRNC_BINARY ${CMAKE_BINARY_DIR}/utils/cprnc/cprnc)
ADD_SUBDIRECTORY(utils/cprnc)

ADD_SUBDIRECTORY(utils/timing)

##########################################
# Processing executables subdirectories
##########################################

IF (CMAKE_BUILD_TYPE_UPPER MATCHES "DEBUG")
  SET (HOMMEXX_DEBUG ON)
ENDIF()

CONFIGURE_FILE (${CMAKE_CURRENT_SOURCE_DIR}/src/Hommexx_config.h.in ${CMAKE_BINARY_DIR}/src/Hommexx_config.h)

ADD_SUBDIRECTORY(test/unit_tests)

SET(EXEC_LIST ""  CACHE INTERNAL "List of configured executables")
IF(${BUILD_HOMME_SWEQX})
  ADD_SUBDIRECTORY(src/sweqx)
ENDIF()

IF(${BUILD_HOMME_SWEQX_FLAT})
  ADD_SUBDIRECTORY(src/sweqx_flat)
ENDIF()

IF(${BUILD_HOMME_PREQX})
  ADD_SUBDIRECTORY(src/preqx)
ENDIF()

IF(${BUILD_HOMME_PREQX_FLAT})
  ADD_SUBDIRECTORY(src/preqx_flat)
ENDIF()

IF(${BUILD_HOMME_PREQX_ACC})
  ADD_SUBDIRECTORY(src/preqx_acc)
ENDIF()

IF(${BUILD_HOMME_FVM})
  ADD_SUBDIRECTORY(src/fvm)
ENDIF()

IF(${BUILD_HOMME_SWIM})
  ADD_SUBDIRECTORY(src/swim)
ENDIF()

IF(${BUILD_HOMME_PRIM})
  ADD_SUBDIRECTORY(src/prim)
ENDIF()

ADD_SUBDIRECTORY(test_execs)

# Create a "make clean-all" target to delete CMake cache and the cprnc build
ADD_CUSTOM_TARGET(clean-all
   COMMAND ${CMAKE_BUILD_TOOL} clean
   COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/clean-cmake-files.cmake"
)

ADD_SUBDIRECTORY(dcmip_tests)
