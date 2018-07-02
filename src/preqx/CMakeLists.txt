# macro for building PREQx primitive equations executable

MACRO(PREQX_SETUP USE_KOKKOS_KERNELS)

IF (${USE_KOKKOS_KERNELS})
  SET (HOMME_USE_FLAT_ARRAYS TRUE)
ELSE ()
  SET (HOMME_USE_FLAT_ARRAYS FALSE)
ENDIF ()

MESSAGE (STATUS " +---------- Beginning execution of PREQX_SETUP ------------+")
MESSAGE (STATUS "  --> Building with USE_KOKKOS_KERNELS = ${USE_KOKKOS_KERNELS}")
MESSAGE (STATUS "  --> Building with HOMME_USE_FLAT_ARRAYS = ${HOMME_USE_FLAT_ARRAYS}")


SET(TARGET_DIR          ${HOMME_SOURCE_DIR}/src/preqx)
SET(UTILS_SHARE_DIR     ${HOMME_SOURCE_DIR}/utils/csm_share)
SET(SRC_BASE_DIR        ${HOMME_SOURCE_DIR}/src)
SET(SRC_SHARE_DIR       ${HOMME_SOURCE_DIR}/src/share)
SET(PHYSICS_DIR         ${HOMME_SOURCE_DIR}/physics)
SET(UTILS_TIMING_SRC_DIR ${HOMME_SOURCE_DIR}/utils/timing)
SET(UTILS_TIMING_DIR    ${CMAKE_BINARY_DIR}/utils/timing)
SET(UNIT_TEST_DIR       ${TARGET_DIR}/unit_tests)
SET(HOMMEXX_CONFIG_DIR  ${CMAKE_BINARY_DIR}/src)  # where Hommexx_config.h is located

# Make INCLUDE_DIRS global so the tests can access it
SET (EXEC_INCLUDE_DIRS ${PIO_INCLUDE_DIRS} ${UTILS_TIMING_DIR} )

SET (TARGET_F90
  ${TARGET_DIR}/bndry_mod.F90
  ${TARGET_DIR}/dcmip_tests.F90
  ${TARGET_DIR}/derivative_mod.F90
  ${TARGET_DIR}/edge_mod.F90
  ${TARGET_DIR}/prim_advection_mod.F90
  ${TARGET_DIR}/solver_init_mod.F90
  ${TARGET_DIR}/vertremap_mod.F90
  ${TARGET_DIR}/viscosity_mod.F90
)

SET(PREQX_DEPS_F90
  ${TARGET_F90}
  ${SRC_BASE_DIR}/test_src/asp_tests.F90
  ${SRC_BASE_DIR}/test_src/baroclinic_inst_mod.F90
  ${SRC_BASE_DIR}/test_src/dcmip2012_test1_2_3.F90
  ${SRC_BASE_DIR}/test_src/dcmip2012_test4.F90
  ${SRC_BASE_DIR}/test_src/dcmip2012_test5.F90
  ${SRC_BASE_DIR}/checksum_mod.F90
  ${SRC_BASE_DIR}/common_io_mod.F90
  ${SRC_BASE_DIR}/common_movie_mod.F90
  ${SRC_BASE_DIR}/forcing_mod.F90
  ${SRC_BASE_DIR}/interpolate_driver_mod.F90
  ${SRC_BASE_DIR}/interp_movie_mod.F90
  ${SRC_BASE_DIR}/netcdf_io_mod.F90
  ${SRC_BASE_DIR}/pio_io_mod.F90
  ${SRC_BASE_DIR}/prim_movie_mod.F90
  ${SRC_BASE_DIR}/prim_restart_mod.F90
  ${SRC_BASE_DIR}/ref_state_mod.F90
  ${SRC_BASE_DIR}/repro_sum_mod.F90
  ${SRC_BASE_DIR}/restart_io_mod.F90
  ${SRC_BASE_DIR}/surfaces_mod.F90
  ${SRC_BASE_DIR}/vertical_mod.F90
  ${SRC_BASE_DIR}/test_mod.F90
  ${SRC_SHARE_DIR}/utils_mod.F90
  ${SRC_SHARE_DIR}/Manager.F90
  ${SRC_SHARE_DIR}/bndry_mod_base.F90
  ${SRC_SHARE_DIR}/cg_mod.F90
  ${SRC_SHARE_DIR}/control_mod.F90
  ${SRC_SHARE_DIR}/coordinate_systems_mod.F90
  ${SRC_SHARE_DIR}/cube_mod.F90
  ${SRC_SHARE_DIR}/derivative_mod_base.F90
  ${SRC_SHARE_DIR}/diffusion_mod.F90
  ${SRC_SHARE_DIR}/dimensions_mod.F90
  ${SRC_SHARE_DIR}/dof_mod.F90
  ${SRC_SHARE_DIR}/domain_mod.F90
  ${SRC_SHARE_DIR}/edge_mod_base.F90
  ${SRC_SHARE_DIR}/edgetype_mod.F90
  ${SRC_SHARE_DIR}/element_mod.F90
  ${SRC_SHARE_DIR}/filter_mod.F90
  ${SRC_SHARE_DIR}/global_norms_mod.F90
  ${SRC_SHARE_DIR}/gridgraph_mod.F90
  ${SRC_SHARE_DIR}/hybrid_mod.F90
  ${SRC_SHARE_DIR}/hybvcoord_mod.F90
  ${SRC_SHARE_DIR}/interpolate_mod.F90
  ${SRC_SHARE_DIR}/kinds.F90
  ${SRC_SHARE_DIR}/linear_algebra_mod.F90
  ${SRC_SHARE_DIR}/ll_mod.F90
  ${SRC_SHARE_DIR}/mass_matrix_mod.F90
  ${SRC_SHARE_DIR}/mesh_mod.F90
  ${SRC_SHARE_DIR}/metagraph_mod.F90
  ${SRC_SHARE_DIR}/metis_mod.F90
  ${SRC_SHARE_DIR}/namelist_mod.F90
  ${SRC_SHARE_DIR}/parallel_mod.F90
  ${SRC_SHARE_DIR}/params_mod.F90
  ${SRC_SHARE_DIR}/physical_constants.F90
  ${SRC_SHARE_DIR}/physics_mod.F90
  ${SRC_SHARE_DIR}/prim_advection_mod_base.F90
  ${SRC_SHARE_DIR}/prim_driver_mod.F90
  ${SRC_SHARE_DIR}/prim_si_mod.F90
  ${SRC_SHARE_DIR}/prim_si_ref_mod.F90
  ${SRC_SHARE_DIR}/prim_state_mod.F90
  ${SRC_SHARE_DIR}/quadrature_mod.F90
  ${SRC_SHARE_DIR}/reduction_mod.F90
  ${SRC_SHARE_DIR}/schedtype_mod.F90
  ${SRC_SHARE_DIR}/schedule_mod.F90
  ${SRC_SHARE_DIR}/solver_init_mod_base.F90
  ${SRC_SHARE_DIR}/solver_mod.F90
  ${SRC_SHARE_DIR}/spacecurve_mod.F90
  ${SRC_SHARE_DIR}/thread_mod.F90
  ${SRC_SHARE_DIR}/time_mod.F90
  ${SRC_SHARE_DIR}/unit_tests_mod.F90
  ${SRC_SHARE_DIR}/vertremap_mod_base.F90
  ${SRC_SHARE_DIR}/viscosity_mod_base.F90
  ${SRC_SHARE_DIR}/prim_smooth_mod.F90
  ${UTILS_SHARE_DIR}/shr_const_mod.F90
  ${UTILS_SHARE_DIR}/shr_file_mod.F90
  ${UTILS_SHARE_DIR}/shr_kind_mod.F90
  ${UTILS_SHARE_DIR}/shr_mpi_mod.F90
  ${UTILS_SHARE_DIR}/shr_spfn_mod.F90
  ${UTILS_SHARE_DIR}/shr_sys_mod.F90
  ${UTILS_SHARE_DIR}/shr_vmath_mod.F90
  ${PHYSICS_DIR}/column_model_mod.F90
  ${PHYSICS_DIR}/column_types_mod.F90
  ${PHYSICS_DIR}/heldsuarez/held_suarez_mod.F90
  ${PHYSICS_DIR}/physics_io_mod.F90
  ${PHYSICS_DIR}/physics_types_mod.F90
)

IF (NOT ${USE_KOKKOS_KERNELS})
  SET (PREQX_DEPS_F90
    ${PREQX_DEPS_F90}
    ${SRC_SHARE_DIR}/caar_pre_exchange_driver_mod.F90
    ${SRC_SHARE_DIR}/prim_advance_caar_mod.F90
    ${SRC_SHARE_DIR}/prim_advance_exp_mod.F90
    ${SRC_SHARE_DIR}/prim_advance_hypervis_mod.F90
    ${SRC_SHARE_DIR}/prim_advance_mod.F90
  )
ELSE()
  SET (PREQX_DEPS_F90
    ${PREQX_DEPS_F90}
    ${SRC_SHARE_DIR}/prim_cxx_driver_mod.F90
  )
ENDIF()

SET(PREQX_DEPS_C
  ${UTILS_SHARE_DIR}/shr_vmath_fwrap.c
  ${SRC_BASE_DIR}/jrio.c
)

SET(PREQX_DEPS_CXX
  ${SRC_SHARE_DIR}/cxx/cxx_f90_interface.cpp
  ${SRC_SHARE_DIR}/cxx/prim_advance_exp.cpp
  ${SRC_SHARE_DIR}/cxx/prim_advec_tracers_remap.cpp
  ${SRC_SHARE_DIR}/cxx/prim_driver.cpp
  ${SRC_SHARE_DIR}/cxx/prim_step.cpp
  ${SRC_SHARE_DIR}/cxx/vertical_remap.cpp
  ${SRC_SHARE_DIR}/cxx/CaarFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/Diagnostics.cpp
  ${SRC_SHARE_DIR}/cxx/Elements.cpp
  ${SRC_SHARE_DIR}/cxx/Tracers.cpp
  ${SRC_SHARE_DIR}/cxx/Derivative.cpp
  ${SRC_SHARE_DIR}/cxx/HybridVCoord.cpp
  ${SRC_SHARE_DIR}/cxx/Context.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/HyperviscosityFunctorImpl.cpp
  ${SRC_SHARE_DIR}/cxx/EulerStepFunctor.cpp
  ${SRC_SHARE_DIR}/cxx/VerticalRemapManager.cpp
  ${SRC_SHARE_DIR}/cxx/Hommexx_Session.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Comm.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/Connectivity.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BuffersManager.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/BoundaryExchange.cpp
  ${SRC_SHARE_DIR}/cxx/ErrorDefs.cpp
  ${SRC_SHARE_DIR}/cxx/mpi/mpi_cxx_f90_interface.cpp
  ${SRC_SHARE_DIR}/cxx/ExecSpaceDefs.cpp
)

# If the user specified a file for custom compiler options use those
IF (DEFINED PREQX_CUSTOM_FLAGS_FILE)
  setCustomCompilerFlags(PREQX_CUSTOM_FLAGS_FILE PREQX_SRCS_F90)
ENDIF ()

# We pulled prim_main off the deps cause it would create a main function
# clash when building the unit test (a main in prim_main.F90 and one in tester.cpp)
SET(PREQX_DEPS ${TARGET_F90} ${PREQX_DEPS_F90} ${PREQX_DEPS_C})
IF (${USE_KOKKOS_KERNELS})
  SET (PREQX_DEPS
    ${PREQX_DEPS}
    ${PREQX_DEPS_CXX}
  )
ENDIF ()
SET(PREQX_SRCS ${PREQX_DEPS} ${SRC_BASE_DIR}/prim_main.F90)

SET(PREQX_UT_F90_SRCS
  ${TARGET_F90}
  ${PREQX_DEPS_F90}
  ${SRC_SHARE_DIR}/caar_pre_exchange_driver_mod.F90
  ${SRC_SHARE_DIR}/prim_advance_caar_mod.F90
  ${SRC_SHARE_DIR}/prim_advance_exp_mod.F90
  ${SRC_SHARE_DIR}/prim_advance_hypervis_mod.F90
  ${SRC_SHARE_DIR}/prim_advance_mod.F90
)

SET(PREQX_UT_CXX_SRCS
  ${UNIT_TEST_DIR}/preqx_ut.cpp
  ${UNIT_TEST_DIR}/preqx_ut_sphere_op_sl.cpp
  ${UNIT_TEST_DIR}/preqx_ut_sphere_op_ml.cpp
  ${PREQX_DEPS_C}
  ${PREQX_DEPS_CXX}
)

SET(PREQX_REMAP_UT_CXX_SRCS
  ${UNIT_TEST_DIR}/preqx_ut_remap.cpp
  ${PREQX_DEPS_C}
  ${PREQX_DEPS_CXX}
)

SET(PREQX_INCLUDE_DIRS
  ${SRC_SHARE_DIR}
  ${SRC_BASE_DIR}
  ${PHYSICS_DIR}
  ${TARGET_DIR}
  ${HOMMEXX_CONFIG_DIR}
)

IF (${USE_KOKKOS_KERNELS})
  SET (PREQX_INCLUDE_DIRS
    ${PREQX_INCLUDE_DIRS}
    ${SRC_SHARE_DIR}/cxx
    ${SRC_SHARE_DIR}/cxx/mpi
    ${UTILS_TIMING_SRC_DIR}
  )
ENDIF ()

INCLUDE_DIRECTORIES(${PREQX_INCLUDE_DIRS})

# Make SRCS global so the tests can access it
SET(EXEC_SOURCES ${PREQX_SRCS})


# Set up defaults if they are not given
IF (NOT PREQX_NP)
  SET (PREQX_NP 4)
ENDIF ()
IF (NOT PREQX_PLEV)
  SET (PREQX_PLEV 72)
ENDIF ()
IF (NOT PREQX_USE_PIO)
  SET (PREQX_USE_PIO FALSE)
ENDIF ()
IF (NOT PREQX_USE_ENERGY)
  SET (PREQX_USE_ENERGY FALSE)
ENDIF ()
#this should be removed later
IF (NOT PREQX_NC)
  SET (PREQX_NC 4)
ENDIF ()
IF (NOT QSIZE_D)
  SET (QSIZE_D 4)
ENDIF ()

MESSAGE(STATUS " in preqx_setup(): Building with QSIZE_D = ${QSIZE_D}")
MESSAGE(STATUS " in preqx_setup(): Building with PREQX_NP = ${PREQX_NP}")
MESSAGE(STATUS " in preqx_setup(): Building with PREQX_PLEV = ${PREQX_PLEV}")
MESSAGE(STATUS " in preqx_setup(): Building with PREQX_USE_PIO = ${PREQX_USE_PIO}")
MESSAGE(STATUS " in preqx_setup(): Building with PREQX_USE_ENERGY = ${PREQX_USE_ENERGY}")

IF (${ENABLE_HORIZ_OPENMP} AND ${ENABLE_COLUMN_OPENMP})
  SET(NESTED_OPENMP TRUE CACHE BOOL "Both horizontal and column threading")
ENDIF ()

MESSAGE (STATUS " +------------ Ending execution of PREQX_SETUP -------------+")

ENDMACRO(PREQX_SETUP)

#configure this for _c exec:
SET (USE_KOKKOS_KERNELS TRUE)
preqx_setup(USE_KOKKOS_KERNELS)
createTestExec(preqx preqx ${PREQX_NP} ${PREQX_NC} ${PREQX_PLEV} ${PREQX_USE_PIO}  ${PREQX_USE_ENERGY} ${QSIZE_D} Fortran)



