
#define CATCH_CONFIG_RUNNER

#include "catch/catch.hpp"

#include <Kokkos_Core.hpp>

#include "Hommexx_Session.hpp"

int main(int argc, char **argv) {

  Homme::initialize_hommexx_session();

  int result = Catch::Session().run(argc, argv);

  Homme::finalize_hommexx_session();

  return result;
}
