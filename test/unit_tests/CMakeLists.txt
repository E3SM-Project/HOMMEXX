
SET(UNITTESTER_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

macro(cxx_unit_test targetname targetsrcs)

ADD_EXECUTABLE(${targetname} ${UNITTESTER_DIR}/tester.cpp ${targetsrcs})
ADD_TEST(${targetname}_test ${targetname})

TARGET_LINK_LIBRARIES(${targetname} pio timing ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

IF(UNIX AND NOT APPLE)
  TARGET_LINK_LIBRARIES(${targetname} rt)
ENDIF()

TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${SRC_SHARE};${SRC_BASE};${SRC_KERNELS_DIR};;${UNITTESTER_DIR}")
TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${PIO_INCLUDE_DIRS};${UTILS_TIMING_DIR}")
TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${TRILINOS_INSTALL_DIR}/include")

# Variables required for the configuration
SET(CONFIG_DEFINES -DPIO_INTERP -DNC=4 -DNP=4 -DPLEV=4 -DQSIZE_D=4 -DTIMING=1 -D_MPI=1 -D_PRESTART=1 -D_WK_GRAD=1 -DHOMME_QUAD_PREC=1)

TARGET_COMPILE_OPTIONS(${targetname} PUBLIC -Wall -g -O0 ${CONFIG_DEFINES})

link_to_trilinos(${targetname})

endmacro(cxx_unit_test)
