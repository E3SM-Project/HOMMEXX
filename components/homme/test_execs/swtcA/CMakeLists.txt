IF (${HOMME_BUILD_SWEQX})
   sweqx_setup()

   # Set the variables for this test executable
   #                          NP  NC PLEV USE_PIO WITH_ENERGY QSIZE_D
   createTestExec(swtcA sweqx  8   4    4   FALSE        FALSE   4 )
ENDIF()

IF (${BUILD_HOMME_SWEQX_FLAT})
   sweqx_flat_setup()

   createTestExec(swtcA_flat sweqx_flat 8 4 4 FALSE FALSE 4)
ENDIF()
