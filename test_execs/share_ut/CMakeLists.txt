SET(CONFIG_DIR       ${CMAKE_BINARY_DIR}/src)
SET(SRC_SHARE_DIR    ${HOMME_SOURCE_DIR}/src/share)
SET(SHARE_UT_DIR     ${CMAKE_CURRENT_SOURCE_DIR})
SET(UTILS_TIMING_DIR ${HOMME_SOURCE_DIR}/utils/timing)

SET (SHARE_UT_SRCS
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
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/ExecSpaceDefs.cpp
  ${SHARE_UT_DIR}/share_ut.cpp
  ${SHARE_UT_DIR}/share_ut.F90
)

SET (CONFIG_DEFINES PIO_INTERP NC=4 NP=4 PLEV=8 QSIZE_D=4 _PRIM TIMING=1 _MPI=1 _PRESTART=1 _WK_GRAD=1 HOMME_QUAD_PREC=1)
SET (SHARE_UT_INCLUDE_DIRS ${CONFIG_DIR} ${SRC_SHARE_DIR} ${SRC_SHARE_DIR}/cxx ${SHARE_UT_DIR} ${UTILS_TIMING_DIR} ${CMAKE_CURRENT_BINARY_DIR})

cxx_unit_test (share_ut "${SHARE_UT_SRCS}" "${SHARE_UT_INCLUDE_DIRS}" "${CONFIG_DEFINES}")
