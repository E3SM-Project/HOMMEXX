
#define CATCH_CONFIG_RUNNER

#include "catch/catch.hpp"

#include <Kokkos_Core.hpp>

#include "Hommexx_Session.hpp"
#ifdef HOMMEXX_MPI_TEST
#include <mpi.h>
#endif

int main(int argc, char **argv) {

  Homme::initialize_hommexx_session();

#ifdef HOMMEXX_MPI_TEST
  // Initialize mpi
  MPI_Init(&argc,&argv);
#endif

  int result = Catch::Session().run(argc, argv);

  Homme::finalize_hommexx_session();

#ifdef HOMMEXX_MPI_TEST
  // Finalize mpi
  MPI_Finalize();
#endif

  return result;
}
