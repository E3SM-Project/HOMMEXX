
SET(UNITTESTER_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

macro(cxx_unit_test target_name target_srcs include_dirs config_defines)
  ADD_EXECUTABLE(${target_name} ${UNITTESTER_DIR}/tester.cpp ${target_srcs})
  ADD_TEST(${target_name}_test ${target_name})

  TARGET_LINK_LIBRARIES(${target_name} pio timing ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

  STRING(TOUPPER "${PERFORMANCE_PROFILE}" PERF_PROF_UPPER)
  IF ("${PERF_PROF_UPPER}" STREQUAL "VTUNE")
    TARGET_LINK_LIBRARIES(${target_name} ittnotify)
  ENDIF()

  IF(UNIX AND NOT APPLE)
    TARGET_LINK_LIBRARIES(${target_name} rt)
  ENDIF()

  link_to_trilinos(${target_name})

  # Fortran modules
  SET(MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_modules)
  SET_TARGET_PROPERTIES(${target_name} PROPERTIES Fortran_MODULE_DIRECTORY ${MODULE_DIR})

  # Compile definitions and flags
  SET_TARGET_PROPERTIES(${target_name} PROPERTIES COMPILE_DEFINITIONS "${config_defines}")
  SET_TARGET_PROPERTIES(${target_name} PROPERTIES COMPILE_OPTIONS     "${CPP_UT_FPMODEL}")

  TARGET_INCLUDE_DIRECTORIES(${target_name} PUBLIC "${include_dirs}")
  TARGET_INCLUDE_DIRECTORIES(${target_name} PUBLIC "${SRC_SHARE};${SRC_BASE};${UNITTESTER_DIR}")
  TARGET_INCLUDE_DIRECTORIES(${target_name} PUBLIC "${PIO_INCLUDE_DIRS};${UTILS_TIMING_DIR}")
  TARGET_INCLUDE_DIRECTORIES(${target_name} PUBLIC "${TRILINOS_INSTALL_DIR}/include")
  TARGET_INCLUDE_DIRECTORIES(${target_name} PUBLIC "${CMAKE_BINARY_DIR}/src")

endmacro(cxx_unit_test)

macro(add_input_files target srcdir inputfiles)
  FOREACH(f ${inputfiles})
    MESSAGE("Installing ${f} as a custom target")
    ADD_CUSTOM_TARGET(
      "${f}"
      COMMAND ${CMAKE_COMMAND} -E copy "${srcdir}/${f}" "${f}"
      COMMENT "Copying ${f} for ${target}"
      )
    ADD_DEPENDENCIES("${target}" "${f}")
  ENDFOREACH()
endmacro(add_input_files)
