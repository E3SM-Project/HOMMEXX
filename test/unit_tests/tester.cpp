
#define CATCH_CONFIG_RUNNER

#include "catch/catch.hpp"

#include <Kokkos_Core.hpp>

#include "Hommexx_Session.hpp"
#include <mpi.h>

int main(int argc, char **argv) {

  Homme::initialize_hommexx_session();

  // Initialize mpi
  MPI_Init(&argc,&argv);

  int result = Catch::Session().run(argc, argv);

  Homme::finalize_hommexx_session();

  // Finalize mpi
  MPI_Finalize();

  return result;
}
