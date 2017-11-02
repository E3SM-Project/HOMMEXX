# Relevant Directories within the HOMME repository
# holding source files needed for building the library
# and executable sweqx_flat.

macro(sweqx_flat_setup FORTRAN_LOOPS)

MESSAGE(STATUS "Executing sweqx_flat_setup -------------")

SET(TARGET_DIR        ${HOMME_SOURCE_DIR}/src/sweqx_flat)
SET(SRC_BASE          ${HOMME_SOURCE_DIR}/src)
SET(SRC_SHARE         ${HOMME_SOURCE_DIR}/src/share)
SET(SRC_UTILS         ${HOMME_SOURCE_DIR}/utils/csm_share)
SET(UTILS_TIMING_DIR  ${CMAKE_BINARY_DIR}/utils/timing)
SET(UTILS_TIMING_SRC_DIR ${HOMME_SOURCE_DIR}/utils/timing)
SET(SRC_KERNELS_DIR   ${SRC_BASE}/kokkos_kernels)
SET(UNIT_TEST_DIR     ${TARGET_DIR}/unit_tests)
SET(HOMME_CONFIG_DIR  ${CMAKE_BINARY_DIR}/src)  # where Hommexx_config.h is located

# Find F90 files in target directory
SET (TARGET_F90
  ${TARGET_DIR}/advance_mod.F90
  ${TARGET_DIR}/bndry_mod.F90
  ${TARGET_DIR}/derivative_mod.F90
  ${TARGET_DIR}/edge_mod.F90
  ${TARGET_DIR}/init_mod.F90
  ${TARGET_DIR}/main.F90
  ${TARGET_DIR}/restart_mod.F90
  ${TARGET_DIR}/rk_mod.F90
  ${TARGET_DIR}/shallow_water_mod.F90
  ${TARGET_DIR}/shal_movie_mod.F90
  ${TARGET_DIR}/solver_init_mod.F90
  ${TARGET_DIR}/state_mod.F90
  ${TARGET_DIR}/sweq_mod.F90
  ${TARGET_DIR}/types_mod.F90
  ${TARGET_DIR}/viscosity_mod.F90
)

# Make INCLUDE_DIRS global so the tests can access it
SET (EXEC_INCLUDE_DIRS ${HOMME_CONFIG_DIR} ${PIO_INCLUDE_DIRS} ${UTILS_TIMING_DIR} ${UTILS_TIMING_SRC_DIR})

SET(SWEQX_FLAT_DEPS
  ${SRC_BASE}/checksum_mod.F90
  ${SRC_BASE}/common_io_mod.F90
  ${SRC_BASE}/common_movie_mod.F90
  ${SRC_BASE}/interp_movie_mod.F90
  ${SRC_BASE}/netcdf_io_mod.F90
  ${SRC_BASE}/pio_io_mod.F90
  ${SRC_BASE}/repro_sum_mod.F90
  ${SRC_BASE}/restart_io_mod.F90
  ${SRC_SHARE}/bndry_mod_base.F90
  ${SRC_SHARE}/cg_mod.F90
  ${SRC_SHARE}/control_mod.F90
  ${SRC_SHARE}/coordinate_systems_mod.F90
  ${SRC_SHARE}/cube_mod.F90
  ${SRC_SHARE}/derivative_mod_base.F90
  ${SRC_SHARE}/dimensions_mod.F90
  ${SRC_SHARE}/dof_mod.F90
  ${SRC_SHARE}/domain_mod.F90
  ${SRC_SHARE}/edge_mod_base.F90
  ${SRC_SHARE}/element_mod.F90
  ${SRC_SHARE}/filter_mod.F90
  ${SRC_SHARE}/fvm_bsp_mod.F90
  ${SRC_SHARE}/fvm_transformation_mod.F90
  ${SRC_SHARE}/fvm_analytic_mod.F90
  ${SRC_SHARE}/fvm_control_volume_mod.F90
  ${SRC_SHARE}/fvm_filter_mod.F90
  ${SRC_SHARE}/fvm_line_integrals_mod.F90
  ${SRC_SHARE}/fvm_line_integrals_flux_mod.F90
  ${SRC_SHARE}/fvm_mod.F90
  ${SRC_SHARE}/fvm_reconstruction_mod.F90
  ${SRC_SHARE}/global_norms_mod.F90
  ${SRC_SHARE}/gridgraph_mod.F90
  ${SRC_SHARE}/hybrid_mod.F90
  ${SRC_SHARE}/interpolate_mod.F90
  ${SRC_SHARE}/kinds.F90
  ${SRC_SHARE}/linear_algebra_mod.F90
  ${SRC_SHARE}/ll_mod.F90
  ${SRC_SHARE}/mass_matrix_mod.F90
  ${SRC_SHARE}/mesh_mod.F90
  ${SRC_SHARE}/metagraph_mod.F90
  ${SRC_SHARE}/metis_mod.F90
  ${SRC_SHARE}/namelist_mod.F90
  ${SRC_SHARE}/parallel_mod.F90
  ${SRC_SHARE}/params_mod.F90
  ${SRC_SHARE}/physical_constants.F90
  ${SRC_SHARE}/quadrature_mod.F90
  ${SRC_SHARE}/reduction_mod.F90
  ${SRC_SHARE}/schedule_mod.F90
  ${SRC_SHARE}/schedtype_mod.F90
  ${SRC_SHARE}/edgetype_mod.F90
  ${SRC_SHARE}/solver_init_mod_base.F90
  ${SRC_SHARE}/solver_mod.F90
  ${SRC_SHARE}/spacecurve_mod.F90
  ${SRC_SHARE}/thread_mod.F90
  ${SRC_SHARE}/time_mod.F90
  ${SRC_SHARE}/unit_tests_mod.F90
  ${SRC_SHARE}/viscosity_mod_base.F90
  ${SRC_UTILS}/shr_file_mod.F90
  ${SRC_UTILS}/shr_kind_mod.F90
  ${SRC_UTILS}/shr_mpi_mod.F90
  ${SRC_UTILS}/shr_sys_mod.F90
  ${SRC_UTILS}/shr_vmath_mod.F90
  ${SRC_UTILS}/shr_const_mod.F90
  ${SRC_UTILS}/shr_spfn_mod.F90
  ${SRC_BASE}/ref_state_mod.F90
  ${SRC_BASE}/jrio.c
  ${SRC_KERNELS_DIR}/recover_q.cpp
  ${SRC_KERNELS_DIR}/team_parallel.cpp
  ${SRC_SHARE}/cxx/Hommexx_Session.cpp
)

SET(SWEQX_FLAT_SRCS
  ${TARGET_F90}
  ${SWEQX_FLAT_DEPS}
)

SET(SWEQX_FLAT_UT_SRCS
  ${UNIT_TEST_DIR}/advance_rk.cpp
  ${TARGET_DIR}/advance_mod.F90
  ${TARGET_DIR}/shallow_water_mod.F90
  ${TARGET_DIR}/edge_mod.F90
  ${TARGET_DIR}/bndry_mod.F90
  ${TARGET_DIR}/derivative_mod.F90
  ${TARGET_DIR}/viscosity_mod.F90
  ${TARGET_DIR}/types_mod.F90
  ${SWEQX_FLAT_DEPS}
)

SET(EXEC_SOURCES ${SWEQX_FLAT_SRCS} )

SET(HOMME_USE_FLAT_ARRAYS TRUE)

IF (${FORTRAN_LOOPS})
  SET (DONT_USE_KOKKOS TRUE)
ELSE ()
  SET (DONT_USE_KOKKOS FALSE)
ENDIF ()

MESSAGE(STATUS " Building with DONT_USE_KOKKOS = ${DONT_USE_KOKKOS}")
MESSAGE(STATUS "End of executing sweqx_flat_setup ------------")

endmacro(sweqx_flat_setup)
