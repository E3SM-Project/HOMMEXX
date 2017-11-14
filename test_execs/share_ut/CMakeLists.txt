SET(CONFIG_DIR       ${CMAKE_BINARY_DIR}/src)
SET(SRC_SHARE_DIR    ${HOMME_SOURCE_DIR}/src/share)
SET(SRC_PREQX_DIR    ${HOMME_SOURCE_DIR}/src/preqx_flat)
SET(SHARE_UT_DIR     ${CMAKE_CURRENT_SOURCE_DIR})
SET(TIMING_DIR       ${HOMME_SOURCE_DIR}/utils/timing)

### Sphere operators unit test ###

SET (SPHERE_OPERATORS_UT_SRCS
  ${SRC_SHARE_DIR}/kinds.F90
  ${SRC_SHARE_DIR}/control_mod.F90
  ${SRC_SHARE_DIR}/coordinate_systems_mod.F90
  ${SRC_SHARE_DIR}/derivative_mod_base.F90
  ${SRC_SHARE_DIR}/dimensions_mod.F90
  ${SRC_SHARE_DIR}/edgetype_mod.F90
  ${SRC_SHARE_DIR}/element_mod.F90
  ${SRC_SHARE_DIR}/gridgraph_mod.F90
  ${SRC_SHARE_DIR}/parallel_mod.F90
  ${SRC_SHARE_DIR}/params_mod.F90
  ${SRC_SHARE_DIR}/physical_constants.F90
  ${SRC_SHARE_DIR}/quadrature_mod.F90
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BoundaryExchange.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Comm.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Connectivity.cpp
  ${SHARE_UT_DIR}/sphere_operators_ut.cpp
  ${SHARE_UT_DIR}/sphere_operators_ut.F90
)

SET (CONFIG_DEFINES PIO_INTERP NC=4 NP=4 PLEV=8 QSIZE_D=4 _PRIM TIMING=1 _MPI=1 _PRESTART=1 _WK_GRAD=1 HOMME_QUAD_PREC=1)
SET (SPHERE_OPERATORS_UT_INCLUDE_DIRS
  ${CONFIG_DIR}
  ${SRC_SHARE_DIR}
  ${SRC_SHARE_DIR}/cxx
  ${SRC_SHARE_DIR}/cxx/mpi
  ${SHARE_UT_DIR}
  ${TIMING_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
cxx_unit_test (sphere_operators_ut "${SPHERE_OPERATORS_UT_SRCS}" "${SPHERE_OPERATORS_UT_INCLUDE_DIRS}" "${CONFIG_DEFINES}")

### Boundary exchange unit test ###

SET (BOUNDARY_EXCHANGE_UT_SRCS
  ${SRC_SHARE_DIR}/bndry_mod_base.F90
  ${SRC_SHARE_DIR}/control_mod.F90
  ${SRC_SHARE_DIR}/coordinate_systems_mod.F90
  ${SRC_SHARE_DIR}/cube_mod.F90
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
  ${SRC_SHARE_DIR}/schedtype_mod.F90
  ${SRC_SHARE_DIR}/schedule_mod.F90
  ${SRC_SHARE_DIR}/spacecurve_mod.F90
  ${SRC_SHARE_DIR}/thread_mod.F90
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/Control.cpp
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BoundaryExchange.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Comm.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Connectivity.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/mpi_cxx_f90_interface.cpp
  ${SRC_PREQX_DIR}/bndry_mod.F90
  ${SRC_PREQX_DIR}/edge_mod.F90
  ${SHARE_UT_DIR}/boundary_exchange_ut.F90
  ${SHARE_UT_DIR}/boundary_exchange_ut.cpp
  ${TIMING_DIR}/perf_mod.F90
  ${TIMING_DIR}/perf_utils.F90
)

SET (CONFIG_DEFINES NP=4 NC=4 PLEV=16 QSIZE_D=4 _MPI=1 HOMMEXX_MPI_TEST)
SET (BOUNDARY_EXCHANGE_UT_INCLUDE_DIRS
  ${CONFIG_DIR}
  ${SRC_SHARE_DIR}
  ${SRC_SHARE_DIR}/cxx
  ${SRC_SHARE_DIR}/cxx/mpi
  ${SHARE_UT_DIR}
  ${TIMING_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
cxx_unit_test (boundary_exchange_ut "${BOUNDARY_EXCHANGE_UT_SRCS}" "${BOUNDARY_EXCHANGE_UT_INCLUDE_DIRS}" "${CONFIG_DEFINES}")
