prim_setup()
#                         NP  NC PLEV USE_PIO  WITH_ENERGY   QSIZE_D Language
createTestExec(primA prim  4   4   26   FALSE        TRUE     4      Fortran)

# we are doing somethign wrong here.  createTestExec should set these
# and they should not have to be repeated for every single test executable

TARGET_LINK_LIBRARIES(primA ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})
