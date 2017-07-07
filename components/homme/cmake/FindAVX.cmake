# Check if AVX instructions are available on the machine where
# the project is compiled.

MACRO (FindAVX)

  SET(AVX_FOUND    false CACHE BOOL "AVX available on host")
  SET(AVX2_FOUND   false CACHE BOOL "AVX2 available on host")
  SET(AVX512_FOUND false CACHE BOOL "AVX512 available on host")

  MESSAGE (STATUS "Looking for AVX on host...")
  EXECUTE_PROCESS (COMMAND cat /proc/cpuinfo
                   COMMAND grep avx
                   OUTPUT_VARIABLE OUTPUT_VARIABLE_CPUINFO)

  IF (NOT OUTPUT_VARIABLE_CPUINFO STREQUAL "")
    SET (AVX_FOUND TRUE)

    # Found AVX...let's see if it is a higher version
    EXECUTE_PROCESS (COMMAND cat /proc/cpuinfo
                     COMMAND grep avx2
                     OUTPUT_VARIABLE OUTPUT_VARIABLE_CPUINFO)

    IF (NOT OUTPUT_VARIABLE_CPUINFO STREQUAL "")
      SET (AVX2_FOUND TRUE)
    ENDIF()

    EXECUTE_PROCESS (COMMAND cat /proc/cpuinfo
                     COMMAND grep avx512
                     OUTPUT_VARIABLE OUTPUT_VARIABLE_CPUINFO)

    IF (NOT OUTPUT_VARIABLE_CPUINFO STREQUAL "")
      SET (AVX512_FOUND TRUE)
    ENDIF()

  ENDIF()

  IF (${AVX512_FOUND})
    MESSAGE (STATUS "Found AVX512.")
    SET (AVX_VERSION_AUTO "512")
  ELSEIF(${AVX2_FOUND})
    MESSAGE (STATUS "Found AVX2.")
    SET (AVX_VERSION_AUTO "2")
  ELSEIF(${AVX_FOUND})
    MESSAGE (STATUS "Found AVX.")
    SET (AVX_VERSION_AUTO "1")
  ELSE()
    SET (AVX_VERSION_AUTO "0")
    MESSAGE (STATUS "AVX not found")
  ENDIF()

ENDMACRO()
