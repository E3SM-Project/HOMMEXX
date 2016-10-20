
# Set up Trilinos as an external project
SET(TRILINOS_REPO "git@github.com:trilinos/Trilinos")
SET(TRILINOS_SRCDIR "${CMAKE_SOURCE_DIR}/../../cime/externals/trilinos")

SET(EXECUTION_SPACES "-DTPL_ENABLE_MPI=ON")

FIND_PACKAGE(CUDA QUIET)
IF(${CUDA_FOUND})
  OPTION(ENABLE_CUDA "Whether or not to enable CUDA" ON)
  IF(${ENABLE_CUDA})
    SET(EXECUTION_SPACES "${EXECUTION_SPACES} -DTPL_ENABLE_CUDA=ON")
  ENDIF()
ENDIF()

SET(TRILINOS_INSTALL_DIR "~/prefix" CACHE FILEPATH "Where to install Trilinos")

FIND_PACKAGE(Trilinos QUIET PATHS ${TRILINOS_INSTALL_DIR}/lib/cmake/Trilinos)

IF(NOT Trilinos_FOUND OR NOT "${Trilinos_PACKAGE_LIST}" MATCHES "Kokkos")
  SET(TRILINOS_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${TRILINOS_INSTALL_DIR} -DTrilinos_ENABLE_Kokkos=ON ${EXECUTION_SPACES})

  INCLUDE(ExternalProject)

  EXTERNALPROJECT_ADD(Trilinos

    GIT_REPOSITORY ${TRILINOS_REPO}
    GIT_TAG master
    UPDATE_COMMAND ""
    PATCH_COMMAND ""

    SOURCE_DIR ${TRILINOS_SRCDIR}

    CMAKE_ARGS ${TRILINOS_CMAKE_ARGS}
  )
ELSE()
  MESSAGE("\nFound Trilinos!  Here are the details: ")
  MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
  MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
  MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
  MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
  MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
  MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
  MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
  MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
  MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
  MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
  MESSAGE("   Trilinos_C_COMPILER = ${Trilinos_C_COMPILER}")
  MESSAGE("   Trilinos_Fortran_COMPILER = ${Trilinos_Fortran_COMPILER}")
  MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
  MESSAGE("   Trilinos_C_COMPILER_FLAGS = ${Trilinos_C_COMPILER_FLAGS}")
  MESSAGE("   Trilinos_Fortran_COMPILER_FLAGS = ${Trilinos_Fortran_COMPILER_FLAGS}")
  MESSAGE("   Trilinos_LINKER = ${Trilinos_LINKER}")
  MESSAGE("   Trilinos_EXTRA_LD_FLAGS = ${Trilinos_EXTRA_LD_FLAGS}")
  MESSAGE("   Trilinos_AR = ${Trilinos_AR}")
  MESSAGE("End of Trilinos details\n")
ENDIF()


