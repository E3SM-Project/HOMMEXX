
SET(TRILINOS_BUILD_TYPE RELEASE CACHE STRING "Build type of trilinos (defaults to build type of Hommexx)")

SET(TRILINOS_INSTALL_DIR "~/prefix/trilinos/${TRILINOS_BUILD_TYPE}" CACHE FILEPATH "The base directory where to download, build and install Trilinos")

FIND_PACKAGE(Trilinos QUIET PATHS ${TRILINOS_INSTALL_DIR}/lib/cmake/Trilinos)

IF(NOT Trilinos_FOUND OR NOT "${Trilinos_PACKAGE_LIST}" MATCHES "Kokkos")

  MESSAGE (STATUS "Trilinos not found. A fresh repo will be cloned and installed in ${TRILINOS_INSTALL_DIR}")
  MESSAGE (STATUS "If you already have an installation of trilinos, you can specify it in the variable 'TRILINOS_INSTALL_DIR'.")

  SET(PACKAGES -DTrilinos_ENABLE_Kokkos=ON
               -DTrilinos_ENABLE_KokkosAlgorithms=ON
               -DTrilinos_ENABLE_KokkosContainers=ON
               -DTrilinos_ENABLE_KokkosCore=ON
               -DTrilinos_ENABLE_KokkosExample=OFF)
  SET(EXECUTION_SPACES -DTPL_ENABLE_MPI=ON
                       -DKokkos_ENABLE_MPI=ON)

  SET(Kokkos_LIBRARIES "kokkosalgorithms;kokkoscontainers;kokkoscore")
  SET(Kokkos_TPL_LIBRARIES "dl")

  IF(${OPENMP_FOUND})
    MESSAGE(STATUS "Enabling Trilinos' OpenMP")
    SET(EXECUTION_SPACES ${EXECUTION_SPACES}
        -DTrilinos_ENABLE_OpenMP=ON
        -DKokkos_ENABLE_OpenMP=ON
        -DTPL_ENABLE_Pthread=OFF
        -DKokkos_ENABLE_Pthread=OFF)
  ELSE()
    MESSAGE(STATUS "Enabling Trilinos' Pthread")
    SET(EXECUTION_SPACES ${EXECUTION_SPACES}
        -DTrilinos_ENABLE_OpenMP=OFF
        -DKokkos_ENABLE_OpenMP=OFF
        -DTPL_ENABLE_Pthread=ON
        -DKokkos_ENABLE_Pthread=ON)
  ENDIF()

  FIND_PACKAGE(CUDA QUIET)
  IF(${CUDA_FOUND})
    OPTION(ENABLE_CUDA "Whether or not to enable CUDA" ON)
    IF(${ENABLE_CUDA})
      SET(EXECUTION_SPACES ${EXECUTION_SPACES}
          -DTPL_ENABLE_CUDA=ON
          -DKokkos_ENABLE_CUDA=ON
          -DKokkos_ENABLE_CUDA_UVM=ON)
      SET(Kokkos_TPL_LIBRARIES "${Kokkos_TPL_LIBRARIES};cudart;cublas;cufft")
    ENDIF()
  ENDIF()

  # Set up Trilinos as an external project
  SET(TRILINOS_REPO "git@github.com:trilinos/Trilinos")

  SET(TRILINOS_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${TRILINOS_BUILD_TYPE} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -DCMAKE_INSTALL_PREFIX=${TRILINOS_INSTALL_DIR} ${PACKAGES} ${EXECUTION_SPACES})

  INCLUDE(ExternalProject)

  EXTERNALPROJECT_ADD(Trilinos

    GIT_REPOSITORY ${TRILINOS_REPO}
    GIT_TAG master
    UPDATE_COMMAND ""
    PATCH_COMMAND ""

    PREFIX ${TRILINOS_INSTALL_DIR}
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

macro(link_to_trilinos targetName)
  TARGET_INCLUDE_DIRECTORIES(${targetName} PUBLIC "${TRILINOS_INSTALL_DIR}/include")
  TARGET_LINK_LIBRARIES(${targetName} ${Kokkos_TPL_LIBRARIES} ${Kokkos_LIBRARIES} -L${TRILINOS_INSTALL_DIR}/lib)

  IF(TARGET Trilinos)
    # In case we are building Trilinos with ExternalProject, we need to compile this after the fact
    ADD_DEPENDENCIES(${targetName} Trilinos)
  ENDIF()
endmacro(link_to_trilinos)

