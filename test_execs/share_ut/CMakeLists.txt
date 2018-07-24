SET(CONFIG_DIR       ${CMAKE_BINARY_DIR}/src)
SET(SRC_SHARE_DIR    ${HOMME_SOURCE_DIR}/src/share)
SET(SRC_PREQX_DIR    ${HOMME_SOURCE_DIR}/src/preqx)
SET(SHARE_UT_DIR     ${CMAKE_CURRENT_SOURCE_DIR})
SET(UTILS_TIMING_DIR ${HOMME_SOURCE_DIR}/utils/timing)

### Infrastructure unit stest

SET (INFRASTRUCTURE_UT_CXX_SRCS
  ${SRC_SHARE_DIR}/cxx/VerticalRemapManager.cpp
  ${SRC_SHARE_DIR}/cxx/ErrorDefs.cpp
  ${SRC_SHARE_DIR}/cxx/CaarFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/EulerStepFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/Diagnostics.cpp
  ${SRC_SHARE_DIR}/cxx/Elements.cpp
  ${SRC_SHARE_DIR}/cxx/Tracers.cpp
  ${SRC_SHARE_DIR}/cxx/ExecSpaceDefs.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctorImpl.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BoundaryExchange.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BuffersManager.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Comm.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Connectivity.cpp
  ${SHARE_UT_DIR}/infrastructure_ut.cpp
)
SET (INFRASTRUCTURE_UT_F90_SRCS)

SET (CONFIG_DEFINES PIO_INTERP NP=4 QSIZE_D=1 PLEV=8)
SET (INFRASTRUCTURE_UT_INCLUDE_DIRS
  ${CONFIG_DIR}
  ${SRC_SHARE_DIR}
  ${SRC_SHARE_DIR}/cxx
  ${SRC_SHARE_DIR}/cxx/mpi
  ${SHARE_UT_DIR}
  ${UTILS_TIMING_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

SET (NUM_CPUS 1)
cxx_unit_test (infrastructure_ut "${INFRASTRUCTURE_UT_F90_SRCS}" "${INFRASTRUCTURE_UT_CXX_SRCS}" "${INFRASTRUCTURE_UT_INCLUDE_DIRS}" "${CONFIG_DEFINES}" ${NUM_CPUS})

### Random initialization unit stest

SET (RANDOM_INIT_UT_CXX_SRCS
  ${SRC_SHARE_DIR}/cxx/CaarFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/EulerStepFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/VerticalRemapManager.cpp
  ${SRC_SHARE_DIR}/cxx/ErrorDefs.cpp
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/Diagnostics.cpp
  ${SRC_SHARE_DIR}/cxx/Elements.cpp
  ${SRC_SHARE_DIR}/cxx/Tracers.cpp
  ${SRC_SHARE_DIR}/cxx/ExecSpaceDefs.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctorImpl.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BoundaryExchange.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BuffersManager.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Comm.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Connectivity.cpp
  ${SHARE_UT_DIR}/random_init_ut.cpp
)
SET (RANDOM_INIT_UT_F90_SRCS)

SET (CONFIG_DEFINES PIO_INTERP NP=4 PLEV=8 QSIZE_D=4)
SET (RANDOM_INIT_UT_INCLUDE_DIRS
  ${CONFIG_DIR}
  ${SRC_SHARE_DIR}
  ${SRC_SHARE_DIR}/cxx
  ${SRC_SHARE_DIR}/cxx/mpi
  ${SHARE_UT_DIR}
  ${UTILS_TIMING_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

SET (NUM_CPUS 1)
cxx_unit_test (random_init_ut "${RANDOM_INIT_UT_F90_SRCS}" "${RANDOM_INIT_UT_CXX_SRCS}" "${RANDOM_INIT_UT_INCLUDE_DIRS}" "${CONFIG_DEFINES}" ${NUM_CPUS})

### Boundary exchange unit test ###

SET (BOUNDARY_EXCHANGE_UT_F90_SRCS
  ${SRC_SHARE_DIR}/bndry_mod_base.F90
  ${SRC_SHARE_DIR}/control_mod.F90
  ${SRC_SHARE_DIR}/coordinate_systems_mod.F90
  ${SRC_SHARE_DIR}/cube_mod.F90
  ${SRC_SHARE_DIR}/derivative_mod_base.F90
  ${SRC_SHARE_DIR}/dimensions_mod.F90
  ${SRC_SHARE_DIR}/edge_mod_base.F90
  ${SRC_SHARE_DIR}/edgetype_mod.F90
  ${SRC_SHARE_DIR}/element_mod.F90
  ${SRC_SHARE_DIR}/gridgraph_mod.F90
  ${SRC_SHARE_DIR}/hybrid_mod.F90
  ${SRC_SHARE_DIR}/kinds.F90
  ${SRC_SHARE_DIR}/ll_mod.F90
  ${SRC_SHARE_DIR}/metagraph_mod.F90
  ${SRC_SHARE_DIR}/parallel_mod.F90
  ${SRC_SHARE_DIR}/params_mod.F90
  ${SRC_SHARE_DIR}/physical_constants.F90
  ${SRC_SHARE_DIR}/quadrature_mod.F90
  ${SRC_SHARE_DIR}/schedtype_mod.F90
  ${SRC_SHARE_DIR}/schedule_mod.F90
  ${SRC_SHARE_DIR}/spacecurve_mod.F90
  ${SRC_SHARE_DIR}/thread_mod.F90
  ${SRC_SHARE_DIR}/viscosity_mod_base.F90
  ${SRC_PREQX_DIR}/bndry_mod.F90
  ${SRC_PREQX_DIR}/derivative_mod.F90
  ${SRC_PREQX_DIR}/edge_mod.F90
  ${SHARE_UT_DIR}/boundary_exchange_ut.F90
  ${UTILS_TIMING_DIR}/perf_mod.F90
  ${UTILS_TIMING_DIR}/perf_utils.F90
)
SET (BOUNDARY_EXCHANGE_UT_CXX_SRCS
  ${SRC_SHARE_DIR}/cxx/VerticalRemapManager.cpp
  ${SRC_SHARE_DIR}/cxx/ErrorDefs.cpp
  ${SRC_SHARE_DIR}/cxx/CaarFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/Tracers.cpp
  ${SRC_SHARE_DIR}/cxx/EulerStepFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/ExecSpaceDefs.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctorImpl.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BoundaryExchange.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BuffersManager.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Comm.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Connectivity.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/mpi_cxx_f90_interface.cpp
  ${SHARE_UT_DIR}/boundary_exchange_ut.cpp
)

SET (CONFIG_DEFINES NP=4 NC=4 PLEV=16 QSIZE_D=4 _MPI=1 _PRIM)
SET (BOUNDARY_EXCHANGE_UT_INCLUDE_DIRS
  ${CONFIG_DIR}
  ${SRC_SHARE_DIR}
  ${SRC_SHARE_DIR}/cxx
  ${SRC_SHARE_DIR}/cxx/mpi
  ${SHARE_UT_DIR}
  ${UTILS_TIMING_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

IF (USE_NUM_PROCS)
  SET (NUM_CPUS ${USE_NUM_PROCS})
ELSE()
  SET (NUM_CPUS 1)
ENDIF()
cxx_unit_test (boundary_exchange_ut "${BOUNDARY_EXCHANGE_UT_F90_SRCS}" "${BOUNDARY_EXCHANGE_UT_CXX_SRCS}" "${BOUNDARY_EXCHANGE_UT_INCLUDE_DIRS}" "${CONFIG_DEFINES}" ${NUM_CPUS})

### CAM Forcing unit stest

SET (CAM_FORCING_UT_CXX_SRCS
  ${SHARE_UT_DIR}/cam_forcing_ut.cpp
  ${SRC_SHARE_DIR}/cxx/CaarFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/CamForcing.cpp
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/Diagnostics.cpp
  ${SRC_SHARE_DIR}/cxx/Elements.cpp
  ${SRC_SHARE_DIR}/cxx/ErrorDefs.cpp
  ${SRC_SHARE_DIR}/cxx/EulerStepFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/ExecSpaceDefs.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/HybridVCoord.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctorImpl.cpp
  ${SRC_SHARE_DIR}/cxx/Tracers.cpp
  ${SRC_SHARE_DIR}/cxx/VerticalRemapManager.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BoundaryExchange.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BuffersManager.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Comm.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Connectivity.cpp
)

SET (CAM_FORCING_UT_F90_SRCS
  ${SRC_PREQX_DIR}/bndry_mod.F90
  ${SRC_PREQX_DIR}/derivative_mod.F90
  ${SRC_PREQX_DIR}/edge_mod.F90
  ${SRC_SHARE_DIR}/bndry_mod_base.F90
  ${SRC_SHARE_DIR}/control_mod.F90
  ${SRC_SHARE_DIR}/coordinate_systems_mod.F90
  ${SRC_SHARE_DIR}/cube_mod.F90
  ${SRC_SHARE_DIR}/derivative_mod_base.F90
  ${SRC_SHARE_DIR}/dimensions_mod.F90
  ${SRC_SHARE_DIR}/edge_mod_base.F90
  ${SRC_SHARE_DIR}/edgetype_mod.F90
  ${SRC_SHARE_DIR}/element_mod.F90
  ${SRC_SHARE_DIR}/gridgraph_mod.F90
  ${SRC_SHARE_DIR}/hybrid_mod.F90
  ${SRC_SHARE_DIR}/hybvcoord_mod.F90
  ${SRC_SHARE_DIR}/kinds.F90
  ${SRC_SHARE_DIR}/ll_mod.F90
  ${SRC_SHARE_DIR}/metagraph_mod.F90
  ${SRC_SHARE_DIR}/parallel_mod.F90
  ${SRC_SHARE_DIR}/params_mod.F90
  ${SRC_SHARE_DIR}/physical_constants.F90
  ${SRC_SHARE_DIR}/prim_forcing_mod.F90
  ${SRC_SHARE_DIR}/quadrature_mod.F90
  ${SRC_SHARE_DIR}/schedtype_mod.F90
  ${SRC_SHARE_DIR}/schedule_mod.F90
  ${SRC_SHARE_DIR}/spacecurve_mod.F90
  ${SRC_SHARE_DIR}/thread_mod.F90
  ${SRC_SHARE_DIR}/viscosity_mod_base.F90
  ${UTILS_TIMING_DIR}/perf_mod.F90
  ${UTILS_TIMING_DIR}/perf_utils.F90
)

SET (CONFIG_DEFINES PIO_INTERP NP=4 NC=4 QSIZE_D=1 PLEV=8 _PRIM _MPI)
SET (CAM_FORCING_UT_INCLUDE_DIRS
  ${CONFIG_DIR}
  ${SRC_SHARE_DIR}
  ${SRC_SHARE_DIR}/cxx
  ${SRC_SHARE_DIR}/cxx/mpi
  ${SHARE_UT_DIR}
  ${UTILS_TIMING_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

SET (NUM_CPUS 1)
cxx_unit_test (cam_forcing_ut "${CAM_FORCING_UT_F90_SRCS}" "${CAM_FORCING_UT_CXX_SRCS}" "${CAM_FORCING_UT_INCLUDE_DIRS}" "${CONFIG_DEFINES}" ${NUM_CPUS})
