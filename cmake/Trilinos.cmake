OPTION (USE_TRILINOS "Whether to use Kokkos as a part of Trilinos" ON)

IF(USE_TRILINOS)
  MESSAGE("Using Trilinos")
  SET (TRILINOS_INSTALL_DIR "~/prefix" CACHE FILEPATH "Where to install Trilinos")
  SET (TRILINOS_BUILD_TYPE "RELEASE" CACHE STRING "Build type for trilinos")

  FIND_PACKAGE(Trilinos QUIET PATHS ${TRILINOS_INSTALL_DIR}/lib/cmake/Trilinos)

  # This can be defined true in machine file or cmake script with -D USE_SACADO_MP_VECTOR=True
  SET (USE_SACADO_MP_VECTOR FALSE CACHE BOOL "Flag to turn on use of Sadaco_MP_Vector")

  IF (${USE_SACADO_MP_VECTOR})
    MESSAGE("** Using SACADO_MP_VECTOR **")
    ADD_DEFINITIONS(-DUSE_SACADO_MP_VECTOR)
    SET(SACADO_PACKAGES -DTrilinos_ENABLE_Sacado=ON -DTrilinos_ENABLE_Stokhos=ON)
  ELSE()
    MESSAGE("** NOT Using SACADO_MP_VECTOR **")
    SET(SACADO_PACKAGES "")
  ENDIF()

  IF(NOT Trilinos_FOUND OR NOT "${Trilinos_PACKAGE_LIST}" MATCHES "Kokkos")

    SET(PACKAGES -DTrilinos_ENABLE_Kokkos=ON
                 -DTrilinos_ENABLE_KokkosAlgorithms=ON
                 -DTrilinos_ENABLE_KokkosContainers=ON
                 -DTrilinos_ENABLE_KokkosCore=ON
                 -DTrilinos_ENABLE_KokkosExample=OFF
                 -DTrilinos_ENABLE_ALL_OPTIONAL_PACKAGES=OFF
                 ${SACADO_PACKAGES}
    )

    SET(EXECUTION_SPACES -DTPL_ENABLE_MPI=ON
                         -DKokkos_ENABLE_MPI=ON)
  
    SET(Kokkos_LIBRARIES "kokkosalgorithms;kokkoscore;kokkoscontainers")
    SET(Kokkos_TPL_LIBRARIES "dl")

    IF (${USE_SACADO_MP_VECTOR})
      SET(Trilinos_LIBRARIES "stokhos_sacado;stokhos;sacado;teuchosremainder;teuchosnumerics;teuchoscomm;teuchosparameterlist;teuchoscore")
    ENDIF()
  
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
  
    SET(TRILINOS_SRCDIR "${CMAKE_SOURCE_DIR}/../../cime/externals/trilinos")
  
    FIND_PACKAGE(CUDA QUIET)
    IF(${CUDA_FOUND})
      OPTION(ENABLE_CUDA "Whether or not to enable CUDA" ON)
      IF(${ENABLE_CUDA})
        SET(NVCC_WRAPPER ${TRILINOS_SRCDIR}/packages/kokkos/config/nvcc_wrapper)
        SET(ENV{OMPI_CXX} ${NVCC_WRAPPER})
        SET(ENV{NVCC_WRAPPER_DEFAULT_COMPILER} ${CMAKE_CXX_COMPILER})
        SET(EXECUTION_SPACES ${EXECUTION_SPACES}
            -DTPL_ENABLE_CUDA=ON
            -DKokkos_ENABLE_CUDA=ON
            -DKokkos_ENABLE_CUDA_UVM=ON
            -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
            -DCMAKE_CXX_COMPILER=${NVCC_WRAPPER})
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -expt-extended-lambda -expt-relaxed-constexpr")
        MESSAGE("CUDA Enabled")
        SET(Kokkos_TPL_LIBRARIES "${Kokkos_TPL_LIBRARIES};cudart;cublas;cufft")
      ENDIF()
    ENDIF()
  
    # Set up Trilinos as an external project
    SET(TRILINOS_REPO "git@github.com:trilinos/Trilinos")
  
    SET(TRILINOS_CMAKE_ARGS -DCMAKE_BUILD_TYPE=${TRILINOS_BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${TRILINOS_INSTALL_DIR} ${PACKAGES} ${EXECUTION_SPACES})
  
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
  
    IF(";${Trilinos_TPL_LIST};" MATCHES ";CUDA;")
      MESSAGE("CUDA Enabled")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -expt-extended-lambda")
    ENDIF()
  ENDIF()

ELSE(USE_TRILINOS)
  MESSAGE(STATUS "Trilinos turned off")
  IF(KOKKOS_PATH)
    SET(TRILINOS_INSTALL_DIR ${KOKKOS_PATH})
    SET(Kokkos_LIBRARIES "kokkos")
    SET(Kokkos_TPL_LIBRARIES "dl")
  ELSE()
    MESSAGE(FATAL_ERROR "Kokkos is required; either specify KOKKOS_PATH or enable finding/installing Trilinos with USE_TRILINOS and TRILINOS_INSTALL_DIR")
  ENDIF()
ENDIF()

macro(link_to_trilinos targetName)

  TARGET_INCLUDE_DIRECTORIES(${targetName} PUBLIC "${TRILINOS_INSTALL_DIR}/include")
  TARGET_LINK_LIBRARIES(${targetName} -L${TRILINOS_INSTALL_DIR}/lib ${Trilinos_LIBRARIES} ${Kokkos_TPL_LIBRARIES} ${Kokkos_LIBRARIES})

  IF("${ENABLE_CUDA}")
    TARGET_COMPILE_OPTIONS(${targetName} PUBLIC -expt-extended-lambda -DCUDA_BUILD)
  ENDIF()

  IF(TARGET Trilinos)
    # In case we are building Trilinos with ExternalProject, we need to compile this after the fact
    ADD_DEPENDENCIES(${targetName} Trilinos)
  ENDIF()
endmacro(link_to_trilinos)
