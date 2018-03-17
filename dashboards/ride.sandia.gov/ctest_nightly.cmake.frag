
# Begin User inputs:
set (CTEST_SITE "ride.sandia.gov" ) # generally the output of hostname
set (CTEST_DASHBOARD_ROOT "$ENV{TEST_DIRECTORY}" ) # writable path
set (CTEST_SCRIPT_DIRECTORY "$ENV{SCRIPT_DIRECTORY}" ) # where the scripts live
set (CTEST_CMAKE_GENERATOR "Unix Makefiles" ) # What is your compilation apps ?
set (CTEST_CONFIGURATION  Release) # What type of build do you want ?

set (INITIAL_LD_LIBRARY_PATH $ENV{LD_LIBRARY_PATH})

set (CTEST_PROJECT_NAME "HOMMEXX" )
set (CTEST_SOURCE_NAME repos)
set (CTEST_NAME "linux-gcc-${CTEST_BUILD_CONFIGURATION}")
set (CTEST_BINARY_NAME build)


set (CTEST_SOURCE_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_SOURCE_NAME}")
set (CTEST_BINARY_DIRECTORY "${CTEST_DASHBOARD_ROOT}/${CTEST_BINARY_NAME}")

if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_SOURCE_DIRECTORY}")
endif ()
if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}")
  file (MAKE_DIRECTORY "${CTEST_BINARY_DIRECTORY}")
endif ()

configure_file (${CTEST_SCRIPT_DIRECTORY}/CTestConfig.cmake
  ${CTEST_SOURCE_DIRECTORY}/CTestConfig.cmake COPYONLY)

set (CTEST_NIGHTLY_START_TIME "00:00:00 UTC")
set (CTEST_CMAKE_COMMAND "cmake")
set (CTEST_COMMAND "ctest -D ${CTEST_TEST_TYPE}")
set (CTEST_FLAGS "-j32")
SET (CTEST_BUILD_FLAGS "-j32")

set (CTEST_DROP_METHOD "http")

if (CTEST_DROP_METHOD STREQUAL "http")
  set (CTEST_DROP_SITE "cdash.sandia.gov")
  set (CTEST_PROJECT_NAME "HOMMEXX")
  set (CTEST_DROP_LOCATION "/CDash-2-3-0/submit.php?project=HOMMEXX")
  set (CTEST_TRIGGER_SITE "")
  set (CTEST_DROP_SITE_CDASH TRUE)
endif ()

find_program (CTEST_GIT_COMMAND NAMES git)

set (HOMMEXX_REPOSITORY_LOCATION git@github.com:ACME-Climate/HOMMEXX.git)
set (Trilinos_REPOSITORY_LOCATION git@github.com:trilinos/Trilinos.git)
set (CUDA_ROOT /home/projects/pwr8-rhel73-lsf/cuda/8.0.44)
set (NVCC_WRAPPER $ENV{jenkins_trilinos_dir}/packages/kokkos/bin/nvcc_wrapper) 

if (CLEAN_BUILD)
  # Initial cache info
  set (CACHE_CONTENTS "
  SITE:STRING=${CTEST_SITE}
  CMAKE_TYPE:STRING=Release
  CMAKE_GENERATOR:INTERNAL=${CTEST_CMAKE_GENERATOR}
  TESTING:BOOL=OFF
  PRODUCT_REPO:STRING=${HOMMEXX_REPOSITORY_LOCATION}
  " )

  ctest_empty_binary_directory( "${CTEST_BINARY_DIRECTORY}" )
  file(WRITE "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" "${CACHE_CONTENTS}")
endif ()

if (DOWNLOAD_TRILINOS)

  set (CTEST_CHECKOUT_COMMAND)
 
  #
  # Get Trilinos
  #
  
  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/Trilinos")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${Trilinos_REPOSITORY_LOCATION} -b master ${CTEST_SOURCE_DIRECTORY}/Trilinos
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone Trilinos repository!")
    endif ()
  endif ()

endif()


if (DOWNLOAD_HOMMEXX)

  set (CTEST_CHECKOUT_COMMAND)
  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")
  
  #
  # Get HOMMEXX
  #

  if (NOT EXISTS "${CTEST_SOURCE_DIRECTORY}/HOMMEXX")
    execute_process (COMMAND "${CTEST_GIT_COMMAND}" 
      clone ${HOMMEXX_REPOSITORY_LOCATION} -b master ${CTEST_SOURCE_DIRECTORY}/HOMMEXX
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err
      RESULT_VARIABLE HAD_ERROR)
    
    message(STATUS "out: ${_out}")
    message(STATUS "err: ${_err}")
    message(STATUS "res: ${HAD_ERROR}")
    if (HAD_ERROR)
      message(FATAL_ERROR "Cannot clone HOMMEXX repository!")
    endif ()
  endif ()

  set (CTEST_UPDATE_COMMAND "${CTEST_GIT_COMMAND}")


endif ()


ctest_start(${CTEST_TEST_TYPE})

# 
# Set the common Trilinos config options & build Trilinos
# 

if (BUILD_TRILINOS_CUDA) 
  message ("ctest state: BUILD_TRILINOS_CUDA")
  #
  # Configure the Trilinos build
  #
  set_property (GLOBAL PROPERTY SubProject RideTrilinosCuda)
  set_property (GLOBAL PROPERTY Label RideTrilinosCuda)

  set (CONFIGURE_OPTIONS
    "-DCMAKE_INSTALL_PREFIX:PATH=${CTEST_BINARY_DIRECTORY}/TrilinosInstall"
    "-DTrilinos_ENABLE_Kokkos=ON"
    "-DTrilinos_ENABLE_KokkosAlgorithms=ON"
    "-DTrilinos_ENABLE_KokkosContainers=ON"
    "-DTrilinos_ENABLE_KokkosCore=ON"
    "-DTrilinos_ENABLE_KokkosExample=OFF"
    "-DTPL_ENABLE_MPI=ON"
    "-DKokkos_ENABLE_MPI=ON"
    "-DTrilinos_ENABLE_OpenMP=ON"
    "-DKokkos_ENABLE_OpenMP=ON"
    "-DTPL_ENABLE_Pthread=OFF"
    "-DKokkos_ENABLE_Pthread=OFF"
    "-DTPL_ENABLE_CUDA=ON"
    "-DKokkos_ENABLE_Cuda=ON"
    "-DKokkos_ENABLE_Cuda_UVM=ON"
    "-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_ROOT}"
    "-DCMAKE_CXX_COMPILER=${NVCC_WRAPPER}"
  )

  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/TriBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/TriBuild)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    SOURCE "$ENV{jenkins_trilinos_dir}"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure Trilinos build!")
  endif ()

  #
  # Build the rest of Trilinos and install everything
  #

  set_property (GLOBAL PROPERTY SubProject RideTrilinosCuda)
  set_property (GLOBAL PROPERTY Label RideTrilinosCuda)
  #set (CTEST_BUILD_TARGET all)
  set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/TriBuild"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit Trilinos build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build Trilinos!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in Trilinos build. Exiting!")
  endif ()

endif()

if (BUILD_HOMMEXX_CUDA)

  # Configure the HOMMEXX build 
  #

  set_property (GLOBAL PROPERTY SubProject RideHOMMEXXCuda)
  set_property (GLOBAL PROPERTY Label RideHOMMEXXCuda)
  
  set (CONFIGURE_OPTIONS
    "-C$ENV{jenkins_hommexx_dir}/components/homme/cmake/machineFiles/RIDE.cmake"
    "-DUSE_NUM_PROCS=16"
    "-DUSE_TRILINOS=FALSE"
    "-DHOMMEXX_FPMODEL=strict"
    "-DHOMME_BASELINE_DIR=/home/projects/hommexx/baselines/HOMMEXX_baseline_P100/build" 
    "-DCMAKE_CXX_COMPILER=$ENV{jenkins_trilinos_dir}/packages/kokkos/bin/nvcc_wrapper"
    "-DKOKKOS_PATH=${CTEST_BINARY_DIRECTORY}/KokkosInstall"
    )
  
  if (NOT EXISTS "${CTEST_BINARY_DIRECTORY}/HOMMEXXBuild")
    file (MAKE_DIRECTORY ${CTEST_BINARY_DIRECTORY}/HOMMEXXBuild)
  endif ()

  CTEST_CONFIGURE(
    BUILD "${CTEST_BINARY_DIRECTORY}/HOMMEXXBuild"
    SOURCE "$ENV{jenkins_hommexx_dir}/components/homme"
    OPTIONS "${CONFIGURE_OPTIONS}"
    RETURN_VALUE HAD_ERROR
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Configure
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit HOMMEXX configure results!")
    endif ()
  endif ()

  if (HAD_ERROR)
    message ("Cannot configure HOMMEXX build!")
  endif ()

  #
  # Build the rest of HOMMEXX and install everything
  #

  set_property (GLOBAL PROPERTY SubProject RideHOMMEXXCuda)
  set_property (GLOBAL PROPERTY Label RideHOMMEXXCuda)
  set (CTEST_BUILD_TARGET all)
  #set (CTEST_BUILD_TARGET install)

  MESSAGE("\nBuilding target: '${CTEST_BUILD_TARGET}' ...\n")

  CTEST_BUILD(
    BUILD "${CTEST_BINARY_DIRECTORY}/HOMMEXXBuild"
    RETURN_VALUE  HAD_ERROR
    NUMBER_ERRORS  BUILD_LIBS_NUM_ERRORS
    APPEND
    )

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Build
      RETURN_VALUE  S_HAD_ERROR
      )

    if (S_HAD_ERROR)
      message ("Cannot submit HOMMEXX build results!")
    endif ()

  endif ()

  if (HAD_ERROR)
    message ("Cannot build HOMMEXX!")
  endif ()

  if (BUILD_LIBS_NUM_ERRORS GREATER 0)
    message ("Encountered build errors in HOMMEXX build. Exiting!")
  endif ()

  #
  # Run HOMMEXX tests
  #

  set (CTEST_TEST_TIMEOUT 1200)
  CTEST_TEST (
    BUILD "${CTEST_BINARY_DIRECTORY}/HOMMEXXBuild"
    RETURN_VALUE HAD_ERROR)

  if (CTEST_DO_SUBMIT)
    ctest_submit (PARTS Test RETURN_VALUE S_HAD_ERROR)

    if (S_HAD_ERROR)
      message ("Cannot submit HOMMEXX test results!")
    endif ()
  endif ()


endif ()

