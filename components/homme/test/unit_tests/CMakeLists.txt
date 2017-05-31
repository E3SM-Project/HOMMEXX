MACRO (CREATE_UNIT_TEST
        UT_NAME
        UT_SRCS
        UT_INCLUDE_DIRS)

  MESSAGE (STATUS "Configuring unit test ${UT_NAME}")

  # Checking if we passed in extra compile definitions
  SET (CONFIG_DEFINES "")
  IF (${ARGC} EQUAL 4)
    SET (CONFIG_DEFINES ${ARGV3})
  ENDIF ()

  # Creating executable and test
  SET(UNIT_TESTER_DIR ${CMAKE_SOURCE_DIR}/test/unit_tests)
  ADD_EXECUTABLE (${UT_NAME} ${UNIT_TESTER_DIR}/tester.cpp ${UT_SRCS})
  ADD_TEST (${UT_NAME}_test ${UT_NAME})

  # Linking
  TARGET_LINK_LIBRARIES(${UT_NAME} pio timing ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
  IF(UNIX AND NOT APPLE)
    TARGET_LINK_LIBRARIES(${UT_NAME} rt)
  ENDIF()
  LINK_TO_TRILINOS (${UT_NAME})

  # Fortran modules
  SET (MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${UT_NAME}_modules)
  SET_TARGET_PROPERTIES(${UT_NAME} PROPERTIES Fortran_MODULE_DIRECTORY ${MODULE_DIR})

  # Compile definitions and flags
  SET_TARGET_PROPERTIES(${UT_NAME} PROPERTIES COMPILE_DEFINITIONS "${CONFIG_DEFINES}")

  # Include directories
  TARGET_INCLUDE_DIRECTORIES (${UT_NAME} PUBLIC "${UT_INCLUDE_DIRS}")
  TARGET_INCLUDE_DIRECTORIES (${UT_NAME} PUBLIC "${PIO_INCLUDE_DIRS};${UTILS_TIMING_DIR}")
  IF (${USE_TRILINOS})
    TARGET_INCLUDE_DIRECTORIES (${UT_NAME} PUBLIC "${TRILINOS_INSTALL_DIR}/include")
  ENDIF ()
  TARGET_INCLUDE_DIRECTORIES (${UT_NAME} PUBLIC "${UNIT_TESTER_DIR}")

  MESSAGE (STATUS "----- End of configuration of unit test ${UT_NAME} ---------")

ENDMACRO (CREATE_UNIT_TEST)

SET(UNITTESTER_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

macro(cxx_unit_test targetname targetsrcs)

  ADD_EXECUTABLE(${targetname} ${UNITTESTER_DIR}/tester.cpp ${targetsrcs})
  ADD_TEST(${targetname}_test ${targetname})

  TARGET_LINK_LIBRARIES(${targetname} pio timing ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

  IF(UNIX AND NOT APPLE)
    TARGET_LINK_LIBRARIES(${targetname} rt)
  ENDIF()

  link_to_trilinos(${targetname})

  # Fortran modules
  SET (MODULE_DIR ${CMAKE_CURRENT_BINARY_DIR}/${targetname}_modules)
  SET_TARGET_PROPERTIES(${targetname} PROPERTIES Fortran_MODULE_DIRECTORY ${MODULE_DIR})

  # Compile definitions and flags
  SET(CONFIG_DEFINES PIO_INTERP NC=4 NP=4 PLEV=4 QSIZE_D=4 TIMING=1 _MPI=1 _PRESTART=1 _WK_GRAD=1 HOMME_QUAD_PREC=1 HOMME_USE_FLAT_ARRAYS=1)
  SET_TARGET_PROPERTIES(${targetname} PROPERTIES COMPILE_DEFINITIONS "${CONFIG_DEFINES}")

  TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${SRC_SHARE};${SRC_BASE};${SRC_KERNELS_DIR};${UNITTESTER_DIR}")
  TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${PIO_INCLUDE_DIRS};${UTILS_TIMING_DIR}")
  TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${TRILINOS_INSTALL_DIR}/include")
  TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${CMAKE_BINARY_DIR}/src")

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
