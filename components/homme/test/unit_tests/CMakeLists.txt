
SET(UNITTESTER_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

macro(cxx_unit_test targetname useomp usecuda targetsrcs)

ENABLE_LANGUAGE(CXX)

INCLUDE_DIRECTORIES("${TRILINOS_INSTALL_DIR}/include" "${UNITTESTER_DIR}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

IF(${useomp})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_HAVE_OPENMP=1")
  INCLUDE_DIRECTORIES("${TRILINOS_INSTALL_DIR}/include/OpenMP")
ENDIF()

IF(${usecuda})
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DKOKKOS_HAVE_CUDA=1")
  INCLUDE_DIRECTORIES("${TRILINOS_INSTALL_DIR}/include/Cuda")
ENDIF()

ADD_EXECUTABLE(${targetname} ${UNITTESTER_DIR}/tester.cpp ${targetsrcs})
TARGET_LINK_LIBRARIES(${targetname} dl pthread rt -parallel kokkoscore)
endmacro(cxx_unit_test)
