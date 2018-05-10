##############################################################################
# Compiler specific options
##############################################################################

# Small function to set the compiler flag '-fp model' given the name of the model
# Note: this is an Intel-only flag
function (HOMMEXX_set_fpmodel_flags fpmodel_string flags)
  string(TOLOWER "${fpmodel_string}" fpmodel_string_lower)
  if (("${fpmodel_string_lower}" STREQUAL "precise") OR
      ("${fpmodel_string_lower}" STREQUAL "strict") OR
      ("${fpmodel_string_lower}" STREQUAL "fast") OR
      ("${fpmodel_string_lower}" STREQUAL "fast=1") OR
      ("${fpmodel_string_lower}" STREQUAL "fast=2"))
    if (CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
      set (${flags} "-fp-model ${fpmodel_string_lower}" PARENT_SCOPE)
    elseif (CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
      if ("${fpmodel_string_lower}" STREQUAL "strict")
        set (${flags} "-ffp-contract=off" PARENT_SCOPE)
      endif ()
    endif ()
  elseif ("${fpmodel_string_lower}" STREQUAL "")
    set (${flags} "" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "FP_MODEL_FLAG string '${fpmodel_string}' is not recognized.")
  endif()
endfunction()

set (FP_MODEL_FLAG "")
set (UT_FP_MODEL_FLAG "")
IF (DEFINED HOMMEXX_FPMODEL)
  HOMMEXX_set_fpmodel_flags("${HOMMEXX_FPMODEL}" FP_MODEL_FLAG)
  HOMMEXX_set_fpmodel_flags("${HOMMEXX_FPMODEL_UT}" UT_FP_MODEL_FLAG)
ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
  SET(${FP_MODEL_FLAG} "-fp-model precise")
  SET(${UT_FP_MODEL} "-fp-model precise")
ENDIF ()

# Need this for a fix in repro_sum_mod
IF (${CMAKE_Fortran_COMPILER_ID} STREQUAL XL)
  ADD_DEFINITIONS(-DnoI8)
ENDIF ()

# enable all warning
SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
IF (CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
  SET (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -warn all")
ELSE()
  SET (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Wall")
ENDIF()

IF (DEFINED BASE_FFLAGS)
  SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${BASE_FFLAGS}")
ELSE ()
  IF (CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffree-line-length-none ${FP_MODEL_FLAG}")
  ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL PGI)
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Mextend -Mflushz")
    # Needed by csm_share
    ADD_DEFINITIONS(-DCPRPGI)
  ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL PathScale)
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -extend-source")
  ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -assume byterecl")
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${FP_MODEL_FLAG}")
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ftz")
    #SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fp-model fast -qopt-report=5 -ftz")
    #SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mP2OPT_hpo_matrix_opt_framework=0 -fp-model fast -qopt-report=5 -ftz")

    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -diag-disable 8291")

    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FP_MODEL_FLAG}")

    # remark #8291: Recommended relationship between field width 'W' and the number of fractional digits 'D' in this edit descriptor is 'W>=D+7'.

    # Needed by csm_share
    ADD_DEFINITIONS(-DCPRINTEL)
  ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL XL)
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -WF,-C! -qstrict -qnosave")
    # Needed by csm_share
    ADD_DEFINITIONS(-DCPRIBM)
  ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL NAG)
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -kind=byte -wmismatch=mpi_send,mpi_recv,mpi_bcast,mpi_allreduce,mpi_reduce,mpi_isend,mpi_irecv,mpi_irsend,mpi_rsend,mpi_gatherv,mpi_gather,mpi_scatterv,mpi_allgather,mpi_alltoallv,mpi_file_read_all,mpi_file_write_all,mpi_file_read_at")
#    SET(OPT_FFLAGS "${OPT_FFLAGS} -ieee=full -O2")
    SET(DEBUG_FFLAGS "${DEBUG_FFLAGS} -g -time -f2003 -ieee=stop")
    ADD_DEFINITIONS(-DHAVE_F2003_PTR_BND_REMAP)
    # Needed by both PIO and csm_share
    ADD_DEFINITIONS(-DCPRNAG)
  ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL Cray)
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DHAVE_F2003_PTR_BND_REMAP")
    # Needed by csm_share
    ADD_DEFINITIONS(-DCPRCRAY)
 ENDIF ()
ENDIF ()

IF (DEFINED BASE_CPPFLAGS)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_CPPFLAGS}")
ELSE ()
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FP_MODEL_FLAG}")
ENDIF ()

# C++ Flags

SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" CXX11_SUPPORTED)
IF (${CXX11_SUPPORTED})
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSE ()
  MESSAGE (FATAL_ERROR "The C++ compiler does not support C++11")
ENDIF ()

CHECK_CXX_COMPILER_FLAG("-cxxlib" CXXLIB_SUPPORTED)

STRING(TOUPPER "${PERFORMANCE_PROFILE}" PERF_PROF_UPPER)
IF ("${PERF_PROF_UPPER}" STREQUAL "VTUNE")
  ADD_DEFINITIONS(-DVTUNE_PROFILE)
ELSEIF ("${PERF_PROF_UPPER}" STREQUAL "CUDA")
  ADD_DEFINITIONS(-DCUDA_PROFILE)
ELSEIF ("${PERF_PROF_UPPER}" STREQUAL "GPROF")
  ADD_DEFINITIONS(-DGPROF_PROFILE -pg)
  SET (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
ENDIF ()

# Handle Cuda.
#<<<<<<< Updated upstream
#find_package(CUDA QUIET)
#if (${CUDA_FOUND})
#  string (FIND ${CMAKE_CXX_COMPILER} "nvcc" pos)
#  if (${pos} GREATER -1)
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda -DCUDA_BUILD")
#  else ()
#    message ("Cuda was found, but the C++ compiler is not nvcc_wrapper, so building without Cuda support.")
#  endif ()
#endif ()
#=======
##### EDISON hack
#find_package(CUDA QUIET)
#if (${CUDA_FOUND})
#  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda -DCUDA_BUILD")
#endif ()
#>>>>>>> Stashed changes

##############################################################################
# Optimization flags
# 1) OPT_FLAGS if specified sets the Fortran,C, and CXX optimization flags
# 2) OPT_FFLAGS if specified sets the Fortran optimization flags
# 3) OPT_CFLAGS if specified sets the C optimization flags
# 4) OPT_CXXFLAGS if specified sets the CXX optimization flags
##############################################################################
IF (OPT_FLAGS)
  # Flags for Fortran C and CXX
  SET (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} ${OPT_FLAGS}")
  SET (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPT_FLAGS}")
  SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")

ELSE ()

  IF (OPT_FFLAGS)
    # User specified optimization flags
    SET (CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} ${OPT_FFLAGS}")
  ELSE ()
    # Defaults
    IF (CMAKE_Fortran_COMPILER_ID STREQUAL GNU)
      SET(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O3")
    ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL PGI)
      SET(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O2")
    ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL PathScale)
    ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL Intel)
      SET(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O3")
      #SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -mavx -DTEMP_INTEL_COMPILER_WORKAROUND_001")
    ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL XL)
      SET(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O2 -qmaxmem=-1")
    ELSEIF (CMAKE_Fortran_COMPILER_ID STREQUAL Cray)
      SET(CMAKE_Fortran_FLAGS_RELEASE "${CMAKE_Fortran_FLAGS_RELEASE} -O2")
    ENDIF ()
  ENDIF ()

  IF (OPT_CFLAGS)
    SET (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPT_CFLAGS}")
  ELSE ()
    IF (CMAKE_C_COMPILER_ID STREQUAL GNU)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
    ELSEIF (CMAKE_C_COMPILER_ID STREQUAL PGI)
    ELSEIF (CMAKE_C_COMPILER_ID STREQUAL PathScale)
    ELSEIF (CMAKE_C_COMPILER_ID STREQUAL Intel)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
      #SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx -DTEMP_INTEL_COMPILER_WORKAROUND_001")
    ELSEIF (CMAKE_C_COMPILER_ID STREQUAL XL)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2 -qmaxmem=-1")
    ELSEIF (CMAKE_C_COMPILER_ID STREQUAL Cray)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O2")
    ENDIF ()
  ENDIF ()

  IF (OPT_CXXFLAGS)
    SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_CXXFLAGS}")
  ELSE ()
    IF (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL PGI)
    ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL PathScale)
    ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL Intel)
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
      #SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -DTEMP_INTEL_COMPILER_WORKAROUND_001")
    ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL XL)
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 -qmaxmem=-1")
    ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL Cray)
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
    ENDIF ()
  ENDIF ()
ENDIF ()

##############################################################################
# DEBUG flags
# 1) DEBUG_FLAGS if specified sets the Fortran,C, and CXX debug flags
# 2) DEBUG_FFLAGS if specified sets the Fortran debugflags
# 3) DEBUG_CFLAGS if specified sets the C debug flags
# 4) DEBUG_CXXFLAGS if specified sets the CXX debug flags
##############################################################################
IF (DEBUG_FLAGS)
  SET (CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  SET (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
ELSE ()
  IF (DEBUG_FFLAGS)
    SET (CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${DEBUG_FFLAGS}")
  ELSE ()
    IF(${CMAKE_Fortran_COMPILER_ID} STREQUAL PGI)
      SET (CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -gopt")
    ELSEIF(NOT ${CMAKE_Fortran_COMPILER_ID} STREQUAL Cray)
      SET (CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -g")
    ENDIF ()
  ENDIF ()

  IF (DEBUG_CFLAGS)
    SET (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_CFLAGS}")
  ELSE ()
    IF(${CMAKE_Fortran_COMPILER_ID} STREQUAL PGI)
      SET (CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -gopt")
    ELSE()
      SET (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
    ENDIF()
  ENDIF ()

  IF (DEBUG_CXXFLAGS)
    SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_CXXFLAGS}")
  ELSE ()
    IF(${CMAKE_Fortran_COMPILER_ID} STREQUAL PGI)
      SET (CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -gopt")
    ELSE()
      SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    ENDIF ()
  ENDIF ()

ENDIF ()

OPTION(DEBUG_TRACE "Enables TRACE level debugging checks. Very slow" FALSE)
IF (${DEBUG_TRACE})
  SET (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG_TRACE")
ENDIF ()

##############################################################################
# OpenMP
# Two flavors:
#   1) HORIZ_OPENMP OpenMP over elements (standard OPENMP)
#   2) COLUMN_OPENMP OpenMP within an element (previously called ELEMENT_OPENMP)
# COLUMN_OPENMP will be disabled by the openACC exectuables.
#
# HOMMEXX does not distinguish between the two because Kokkos does not use
# nested OpenMP. Nested OpenMP is the reason the two are distinguished in the
# Fortran code.
##############################################################################
OPTION(ENABLE_OPENMP "OpenMP across elements" TRUE)
OPTION(ENABLE_HORIZ_OPENMP "OpenMP across elements" TRUE)
OPTION(ENABLE_COLUMN_OPENMP "OpenMP within an element" TRUE)

# If OpenMP is turned off also turn off ENABLE_HORIZ_OPENMP
IF (NOT ${ENABLE_OPENMP})
  SET(ENABLE_HORIZ_OPENMP FALSE)
  SET(ENABLE_COLUMN_OPENMP FALSE)
ENDIF ()

##############################################################################
IF (ENABLE_HORIZ_OPENMP OR ENABLE_COLUMN_OPENMP)
  IF(NOT ${CMAKE_Fortran_COMPILER_ID} STREQUAL Cray)
    FIND_PACKAGE(OpenMP)
    IF(OPENMP_FOUND)
      MESSAGE(STATUS "Found OpenMP Flags")
      IF (CMAKE_Fortran_COMPILER_ID STREQUAL XL)
        SET(OpenMP_C_FLAGS "-qsmp=omp")
        IF (ENABLE_COLUMN_OPENMP)
          SET(OpenMP_C_FLAGS "-qsmp=omp:nested_par")
        ENDIF ()
      ENDIF ()
      # This file is needed for the timing library - this is currently
      # inaccessible from the timing CMake script
      SET(OpenMP_Fortran_FLAGS "${OpenMP_C_FLAGS}")
      MESSAGE(STATUS "OpenMP_Fortran_FLAGS: ${OpenMP_Fortran_FLAGS}")
      MESSAGE(STATUS "OpenMP_C_FLAGS: ${OpenMP_C_FLAGS}")
      MESSAGE(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")
      MESSAGE(STATUS "OpenMP_EXE_LINKER_FLAGS: ${OpenMP_EXE_LINKER_FLAGS}")
      # The fortran openmp flag should be the same as the C Flag
      SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_C_FLAGS}")
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    ELSE ()
      MESSAGE(FATAL_ERROR "Unable to find OpenMP")
    ENDIF()
  ENDIF()
 IF (${ENABLE_HORIZ_OPENMP})
   # Set this as global so it can be picked up by all executables
#   SET(HORIZ_OPENMP TRUE CACHE BOOL "Threading in the horizontal direction")
   SET(HORIZ_OPENMP TRUE BOOL "Threading in the horizontal direction")
   MESSAGE(STATUS "  Using HORIZ_OPENMP")
 ENDIF ()

 IF (${ENABLE_COLUMN_OPENMP})
   # Set this as global so it can be picked up by all executables
#   SET(COLUMN_OPENMP TRUE CACHE BOOL "Threading in the vertical direction")
   SET(COLUMN_OPENMP TRUE BOOL "Threading in the vertical direction")
   MESSAGE(STATUS "  Using COLUMN_OPENMP")
 ENDIF ()
ENDIF ()
##############################################################################

##############################################################################
# Intel Phi (MIC) specific flags - only supporting the Intel compiler
##############################################################################
OPTION(ENABLE_INTEL_PHI "Whether to build with Intel Xeon Phi (MIC) support" FALSE)

IF (ENABLE_INTEL_PHI)
  IF (NOT ${CMAKE_Fortran_COMPILER_ID} STREQUAL Intel)
    MESSAGE(FATAL_ERROR "Intel Phi acceleration only supported through the Intel compiler")
  ELSE ()
    STRING(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_ci)
    IF (${CMAKE_BUILD_TYPE_ci} STREQUAL release)
      SET(INTEL_PHI_FLAGS "-xMIC-AVX512")
      SET(AVX_VERSION "512")
    ELSE ()
      SET(AVX_VERSION "0")
    ENDIF ()
    SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${INTEL_PHI_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  ${INTEL_PHI_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${INTEL_PHI_FLAGS}")
    SET(IS_ACCELERATOR TRUE)
    # CMake magic for cross-compilation
    SET(CMAKE_SYSTEM_NAME Linux)
    SET(CMAKE_SYSTEM_PROCESSOR k1om)
    SET(CMAKE_SYSTEM_VERSION 1)
    SET(_CMAKE_TOOLCHAIN_PREFIX  x86_64-k1om-linux-)
    # Specify the location of the target environment
    IF (TARGET_ROOT_PATH)
      SET(CMAKE_FIND_ROOT_PATH ${TARGET_ROOT_PATH})
    ELSE ()
      SET(CMAKE_FIND_ROOT_PATH /usr/linux-k1om-4.7)
    ENDIF ()
  ENDIF ()
ENDIF ()
##############################################################################
# Compiler FLAGS for AVX1 and AVX2 (CXX compiler only)
##############################################################################
IF (NOT DEFINED AVX_VERSION)
  INCLUDE(FindAVX)
  FindAVX()
  IF (AVX512_FOUND)
    SET(AVX_VERSION "512")
  ELSEIF (AVX2_FOUND)
    SET(AVX_VERSION "2")
  ELSEIF (AVX_FOUND)
    SET(AVX_VERSION "1")
  ELSE ()
    SET(AVX_VERSION "0")
  ENDIF ()
ENDIF ()

IF (AVX_VERSION STREQUAL "2")
  IF (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
  ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL Intel)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xCORE-AVX2")
  ENDIF()
ELSEIF (AVX_VERSION STREQUAL "1")
  IF (CMAKE_CXX_COMPILER_ID STREQUAL GNU)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
  ELSEIF (CMAKE_CXX_COMPILER_ID STREQUAL Intel)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xAVX")
  ENDIF()
ENDIF ()

##############################################################################
# Allow the option to add compiler flags to those provided
##############################################################################
SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${ADD_Fortran_FLAGS}")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ADD_C_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ADD_CXX_FLAGS}")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ADD_LINKER_FLAGS}")

##############################################################################
# Allow the option to override compiler flags
##############################################################################
IF (FORCE_Fortran_FLAGS)
  SET(CMAKE_Fortran_FLAGS ${FORCE_Fortran_FLAGS})
ENDIF ()
IF (FORCE_C_FLAGS)
  SET(CMAKE_C_FLAGS ${FORCE_C_FLAGS})
ENDIF ()
IF (FORCE_CXX_FLAGS)
  SET(CMAKE_CXX_FLAGS ${FORCE_CXX_FLAGS})
ENDIF ()
IF (FORCE_LINKER_FLAGS)
  SET(CMAKE_EXE_LINKER_FLAGS ${FORCE_LINKER_FLAGS})
ENDIF ()
