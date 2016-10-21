
SET(UNITTESTER_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

macro(cxx_unit_test targetname useomp usecuda targetsrcs)

ENABLE_LANGUAGE(CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

ADD_EXECUTABLE(${targetname} ${UNITTESTER_DIR}/tester.cpp ${targetsrcs})
TARGET_LINK_LIBRARIES(${targetname} dl pthread rt -parallel kokkoscore pio timing ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
ADD_TEST(${targetname}_test ${targetname})

TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${UNITTESTER_DIR}")
TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${TRILINOS_INSTALL_DIR}/include")

IF(${useomp})
  TARGET_COMPILE_OPTIONS(${targetname} PUBLIC -DKOKKOS_HAVE_OPENMP=1)
  TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${TRILINOS_INSTALL_DIR}/include/OpenMP")
ENDIF()

IF(${usecuda})
  TARGET_COMPILE_OPTIONS(${targetname} PUBLIC -DKOKKOS_HAVE_CUDA=1)
  TARGET_INCLUDE_DIRECTORIES(${targetname} PUBLIC "${TRILINOS_INSTALL_DIR}/include/Cuda")
ENDIF()

endmacro(cxx_unit_test)
