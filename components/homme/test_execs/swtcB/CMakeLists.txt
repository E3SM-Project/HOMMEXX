IF (${BUILD_HOMME_SWEQX})
   sweqx_setup()

   # Set the variables for this test executable
   #                          NP  NC PLEV USE_PIO WITH_ENERGY  QSIZE_D LINK_LANGUAGE
   createTestExec(swtcB sweqx  4   4    1   FALSE      FALSE   4       Fortran)
ENDIF()

IF (${BUILD_HOMME_SWEQX_FLAT})
   sweqx_flat_setup()

   createTestExec(swtcB_flat sweqx_flat 4 4 1 FALSE FALSE 4 Fortran)
ENDIF()
